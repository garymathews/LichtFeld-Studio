/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "vk_phase_timing.hpp"

#include "core/logger.hpp"
#include "core/vulkan_phase_timing.hpp"
#include "vk_context.hpp"
#include "vk_recorder.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace lfs::core::internal {
    namespace {
        /// Commands recorded since the last begin(), sampled at every announced boundary.
        std::atomic<std::uint64_t> g_recorded_commands{0};

        struct State {
            std::mutex mutex;
            VkDevice device = VK_NULL_HANDLE;
            VkQueryPool pool = VK_NULL_HANDLE;
            std::uint32_t budget = 0;
            std::uint32_t used = 0;
            float period_ns = 1.0f;
            bool supported = false;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> tags; // tag, query index
            std::vector<VulkanPhaseCommandStamp> command_stamps;
        };

        State& state() {
            static State instance;
            return instance;
        }

        /// Boundaries announced by the training thread but not yet timestamped.
        thread_local std::vector<std::uint32_t> tls_pending;

        bool queue_supports_timestamps(const VulkanContext& context) {
            std::uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, families.data());
            if (context.queue_family() >= families.size())
                return false;
            const auto& family = families[context.queue_family()];
            return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && family.timestampValidBits > 0;
        }
    } // namespace

    bool vk_phase_timing_supported() {
        State& s = state();
        std::lock_guard lock(s.mutex);
        return s.supported;
    }

    void vk_phase_timing_begin(const std::uint32_t query_budget) {
        tls_pending.clear();
        const auto context = try_live_vulkan_context();
        State& s = state();
        std::lock_guard lock(s.mutex);
        if (s.pool != VK_NULL_HANDLE || !context || query_budget == 0)
            return;
        if (!queue_supports_timestamps(*context))
            return;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
        if (!(properties.limits.timestampPeriod > 0.0f))
            return;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = query_budget;
        if (vkCreateQueryPool(context->device(), &info, nullptr, &s.pool) != VK_SUCCESS) {
            s.pool = VK_NULL_HANDLE;
            return;
        }
        s.device = context->device();
        s.budget = query_budget;
        s.used = 0;
        s.period_ns = properties.limits.timestampPeriod;
        s.tags.clear();
        s.command_stamps.clear();
        g_recorded_commands.store(0, std::memory_order_relaxed);
        s.supported = true;
        LOG_DEBUG("Vulkan phase timing enabled: {} queries at {:.3f} ns/tick", query_budget, s.period_ns);
    }

    void vk_phase_timing_end() {
        tls_pending.clear();
        g_recorded_commands.store(0, std::memory_order_relaxed);
        const auto context = try_live_vulkan_context();
        State& s = state();
        std::lock_guard lock(s.mutex);
        if (s.pool == VK_NULL_HANDLE)
            return;
        // The pool must not be destroyed while queries are still in flight.
        if (context && context->accepting_work())
            context->recorders().wait_all();
        vkDestroyQueryPool(s.device, s.pool, nullptr);
        s.pool = VK_NULL_HANDLE;
        s.device = VK_NULL_HANDLE;
        s.used = 0;
        s.budget = 0;
        s.tags.clear();
        s.supported = false;
    }

    void vk_phase_timing_announce(const std::uint32_t tag) {
        State& s = state();
        if (!s.supported)
            return;
        tls_pending.push_back(tag);
        std::lock_guard lock(s.mutex);
        s.command_stamps.push_back({tag, g_recorded_commands.load(std::memory_order_relaxed)});
    }

    void vk_phase_timing_note_command() {
        g_recorded_commands.fetch_add(1, std::memory_order_relaxed);
    }

    std::vector<VulkanPhaseCommandStamp> vk_phase_command_stamps() {
        State& s = state();
        std::lock_guard lock(s.mutex);
        return s.command_stamps;
    }

    void vk_phase_timing_write_pending(const VkCommandBuffer command) {
        if (tls_pending.empty())
            return;
        State& s = state();
        std::lock_guard lock(s.mutex);
        for (const std::uint32_t tag : tls_pending) {
            if (s.pool == VK_NULL_HANDLE || s.used >= s.budget)
                break;
            const std::uint32_t index = s.used++;
            // Reset immediately before the write: the pool spans many submissions, so each
            // slot is cleared in the same command buffer that fills it.
            vkCmdResetQueryPool(command, s.pool, index, 1);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, s.pool, index);
            s.tags.emplace_back(tag, index);
        }
        tls_pending.clear();
    }

    std::vector<VulkanPhaseStamp> vk_phase_timestamps() {
        State& s = state();
        std::lock_guard lock(s.mutex);
        std::vector<VulkanPhaseStamp> stamps;
        if (s.pool == VK_NULL_HANDLE || s.tags.empty())
            return stamps;
        std::vector<std::uint64_t> ticks(s.used, 0);
        const VkResult result = vkGetQueryPoolResults(
            s.device, s.pool, 0, s.used, ticks.size() * sizeof(std::uint64_t), ticks.data(),
            sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS && result != VK_NOT_READY)
            return stamps;
        stamps.reserve(s.tags.size());
        for (const auto& [tag, index] : s.tags) {
            if (index >= ticks.size() || ticks[index] == 0)
                continue;
            stamps.push_back({tag, static_cast<double>(ticks[index]) * double(s.period_ns) / 1.0e6});
        }
        return stamps;
    }
} // namespace lfs::core::internal

namespace lfs::core {
    void vulkan_phase_timing_begin(const std::uint32_t query_budget) {
        internal::vk_phase_timing_begin(query_budget);
    }
    void vulkan_phase_timing_end() { internal::vk_phase_timing_end(); }
    void vulkan_phase_timing_announce(const std::uint32_t tag) {
        internal::vk_phase_timing_announce(tag);
    }
    std::vector<VulkanPhaseStamp> vulkan_phase_timestamps() {
        return internal::vk_phase_timestamps();
    }
    void vulkan_phase_timing_note_command() { internal::vk_phase_timing_note_command(); }
    std::vector<VulkanPhaseCommandStamp> vulkan_phase_command_stamps() {
        return internal::vk_phase_command_stamps();
    }
    bool vulkan_phase_timing_supported() { return internal::vk_phase_timing_supported(); }

    namespace {
        /// Total host nanoseconds spent blocked on the device timeline.
        std::atomic<std::uint64_t> g_host_wait_ns{0};
        /// Off by default so training pays one relaxed load per wait, not two clock reads.
        std::atomic<bool> g_host_wait_enabled{false};
    } // namespace

    void vulkan_host_wait_note_ns(const std::uint64_t nanoseconds) {
        g_host_wait_ns.fetch_add(nanoseconds, std::memory_order_relaxed);
    }

    std::uint64_t vulkan_host_wait_total_ns() {
        return g_host_wait_ns.load(std::memory_order_relaxed);
    }

    void vulkan_host_wait_set_enabled(const bool enabled) {
        g_host_wait_enabled.store(enabled, std::memory_order_relaxed);
    }

    bool vulkan_host_wait_enabled() {
        return g_host_wait_enabled.load(std::memory_order_relaxed);
    }
} // namespace lfs::core
