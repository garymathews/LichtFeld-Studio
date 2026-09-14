/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_recorder.hpp"

#include "core/assert.hpp"
#include "core/tensor_storage.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::core::internal {
    namespace {
        constexpr uint32_t kCommandLimit = 64;

        struct ThreadRecorderToken {
            uint64_t context_id = 0;
            uint64_t recorder_id = 0;

            ~ThreadRecorderToken() {
                if (recorder_id == 0) {
                    return;
                }
                if (const auto context = try_live_vulkan_context();
                    context && context->context_id() == context_id && context->accepting_work()) {
                    try {
                        context->recorders().release_thread(recorder_id);
                    } catch (const std::exception& error) {
                        context->quarantine();
                        std::fprintf(stderr, "Vulkan thread recorder release failed: %s\n", error.what());
                    } catch (...) {
                        context->quarantine();
                        std::fputs("Vulkan thread recorder release failed\n", stderr);
                    }
                }
            }
        };

        thread_local ThreadRecorderToken tls_recorder;
        thread_local bool tls_external_work = false;
        thread_local bool tls_retiring_resources = false;

        void require_tensor_access() {
            if (tls_external_work)
                throw std::logic_error("Tensor/backend calls are forbidden inside with_idle_vulkan_device");
        }

        void global_barrier(const VkCommandBuffer command) {
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }
    } // namespace

    struct VulkanRecorderRegistry::Recorder {
        struct Submitted {
            VkCommandBuffer command = VK_NULL_HANDLE;
            uint64_t value = 0;
        };

        uint64_t id = 0;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        // Submitted batches whose completion the host has not observed yet,
        // oldest first. Recording continues while they run; the host waits
        // only when kInFlightLimit batches are outstanding.
        std::deque<Submitted> in_flight;
        uint64_t reserved_value = 0;
        uint64_t submitted_value = 0;
        uint32_t command_count = 0;
        bool owner_alive = true;
    };

    namespace {
        constexpr size_t kInFlightLimit = 4;
    } // namespace

    void VulkanRecorderRegistry::retire_completed_locked(Recorder& recorder,
                                                         const uint64_t completed) {
        while (!recorder.in_flight.empty() && recorder.in_flight.front().value <= completed) {
            vkFreeCommandBuffers(context_.device(), recorder.pool, 1,
                                 &recorder.in_flight.front().command);
            recorder.in_flight.pop_front();
        }
        if (recorder.in_flight.empty() && recorder.command == VK_NULL_HANDLE &&
            recorder.submitted_value != 0) {
            vk_check(&context_, vkResetCommandPool(context_.device(), recorder.pool, 0),
                     "vkResetCommandPool");
        }
    }

    VulkanRecorderRegistry::VulkanRecorderRegistry(VulkanContext& context)
        : context_(context) {}

    VulkanRecorderRegistry::~VulkanRecorderRegistry() {
        try { shutdown(); }
        catch (...) {
            // Do not destroy command pools whose completion is unknown.
            for (auto& [id, recorder] : recorders_) (void)recorder.release();
        }
    }

    VulkanRecorderRegistry::Recorder& VulkanRecorderRegistry::current_locked() {
        context_.require_work();
        LFS_ASSERT_MSG(!shutting_down_,
                       "Vulkan backend is shutting down; new recording is rejected");
        if (tls_recorder.context_id == context_.context_id() &&
            tls_recorder.recorder_id != 0) {
            return *recorders_.at(tls_recorder.recorder_id);
        }

        auto recorder = std::make_unique<Recorder>();
        recorder->id = next_recorder_id_++;
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                          VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = context_.queue_family();
        vk_check(&context_, vkCreateCommandPool(context_.device(), &pool_info, nullptr, &recorder->pool),
                 "vkCreateCommandPool");
        const uint64_t id = recorder->id;
        recorders_.emplace(id, std::move(recorder));
        tls_recorder.context_id = context_.context_id();
        tls_recorder.recorder_id = id;
        return *recorders_.at(id);
    }

    void VulkanRecorderRegistry::begin_locked(Recorder& recorder) {
        if (recorder.command != VK_NULL_HANDLE) {
            return;
        }
        retire_completed_locked(recorder, context_.completed_timeline());
        if (recorder.in_flight.size() >= kInFlightLimit) {
            context_.wait(recorder.in_flight.front().value);
            retire_completed_locked(recorder, context_.completed_timeline());
        }
        VkCommandBufferAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate_info.commandPool = recorder.pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        vk_check(&context_, vkAllocateCommandBuffers(context_.device(), &allocate_info, &recorder.command),
                 "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try {
            vk_check(&context_, vkBeginCommandBuffer(recorder.command, &begin_info),
                     "vkBeginCommandBuffer");
            recorder.reserved_value = context_.reserve_timeline_value();
        } catch (...) {
            vkFreeCommandBuffers(context_.device(), recorder.pool, 1, &recorder.command);
            recorder.command = VK_NULL_HANDLE;
            throw;
        }
        recorder.command_count = 0;
    }

    void VulkanRecorderRegistry::stamp(const StorageRef storage,
                                       const uint64_t recorder_id,
                                       const uint64_t value) {
        if (storage.meta == nullptr || storage.backend != GpuBackend::Vulkan) {
            return;
        }
        auto* const meta = const_cast<StorageMeta*>(storage.meta);
        meta->pending_recorder.store(recorder_id, std::memory_order_release);
        meta->pending_value.store(value, std::memory_order_release);
    }

    void VulkanRecorderRegistry::ensure_submitted_locked(const StorageRef storage) {
        if (storage.meta == nullptr || storage.backend != GpuBackend::Vulkan) {
            return;
        }
        const uint64_t value = storage.meta->pending_value.load(std::memory_order_acquire);
        const uint64_t recorder_id =
            storage.meta->pending_recorder.load(std::memory_order_acquire);
        if (value == 0 || recorder_id == 0) {
            return;
        }
        const auto iterator = recorders_.find(recorder_id);
        if (iterator != recorders_.end() && iterator->second->reserved_value == value &&
            iterator->second->command != VK_NULL_HANDLE) {
            flush_through_locked(value);
        }
    }

    uint64_t VulkanRecorderRegistry::record(
        const std::span<const StorageRef> reads,
        const std::span<const StorageRef> writes,
        const std::function<void(VkCommandBuffer)>& command,
        const bool writes_fault) {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        collect_completed_locked(context_.completed_timeline());
        Recorder& recorder = current_locked();
        const auto flush_foreign_access = [&](const StorageRef storage) {
            if (storage.meta != nullptr &&
                storage.meta->pending_recorder.load(std::memory_order_acquire) != recorder.id) {
                ensure_submitted_locked(storage);
            }
        };
        for (const StorageRef storage : reads) {
            flush_foreign_access(storage);
        }
        for (const StorageRef storage : writes) {
            flush_foreign_access(storage);
        }
        begin_locked(recorder);
        global_barrier(recorder.command);
        command(recorder.command);
        if (writes_fault) {
            const VkMemoryBarrier2 host_read{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
            };
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &host_read,
            };
            vkCmdPipelineBarrier2(recorder.command, &dependency);
            fault_value_ = std::max(fault_value_, recorder.reserved_value);
        }
        ++recorder.command_count;
        // A later writer on another thread must submit these readers first.
        for (const StorageRef storage : reads) {
            stamp(storage, recorder.id, recorder.reserved_value);
        }
        for (const StorageRef storage : writes) {
            stamp(storage, recorder.id, recorder.reserved_value);
        }
        const uint64_t value = recorder.reserved_value;
        context_.memory().mark_used(reads, writes, value);
        if (recorder.command_count >= kCommandLimit) {
            flush_through_locked(value);
        }
        return value;
    }

    void VulkanRecorderRegistry::submit_locked(Recorder& recorder) {
        if (recorder.command == VK_NULL_HANDLE) {
            return;
        }
        // Allocate host bookkeeping before submitting. Keep the command owned by
        // its pool and retain the first failure so a later timeline cannot hide it.
        try {
            recorder.in_flight.push_back({recorder.command, recorder.reserved_value});
            vk_check(&context_, vkEndCommandBuffer(recorder.command), "vkEndCommandBuffer");
            context_.submit(recorder.command, recorder.reserved_value);
        } catch (...) {
            submission_failure_ = std::current_exception();
            context_.quarantine();
            throw;
        }
        recorder.submitted_value = std::exchange(recorder.reserved_value, uint64_t{0});
        recorder.command = VK_NULL_HANDLE;
        recorder.command_count = 0;
    }

    uint64_t VulkanRecorderRegistry::flush_through_locked(const uint64_t value) {
        if (submission_failure_)
            std::rethrow_exception(submission_failure_);
        std::vector<Recorder*> pending;
        for (auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->command != VK_NULL_HANDLE && recorder->reserved_value <= value) {
                pending.push_back(recorder.get());
            }
        }
        std::ranges::sort(pending, {}, &Recorder::reserved_value);
        uint64_t submitted = 0;
        for (Recorder* const recorder : pending) {
            submitted = recorder->reserved_value;
            submit_locked(*recorder);
        }
        return submitted;
    }

    void VulkanRecorderRegistry::collect_completed_locked(const uint64_t completed) {
        std::erase_if(recorders_, [&](const auto& entry) {
            Recorder& recorder = *entry.second;
            if (recorder.owner_alive || recorder.command != VK_NULL_HANDLE ||
                recorder.submitted_value > completed) {
                return false;
            }
            retire_completed_locked(recorder, completed);
            vkDestroyCommandPool(context_.device(), recorder.pool, nullptr);
            return true;
        });
    }

    void VulkanRecorderRegistry::flush_storage(const StorageRef storage) {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        ensure_submitted_locked(storage);
    }

    uint64_t VulkanRecorderRegistry::flush_current() {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        if (tls_recorder.context_id != context_.context_id() ||
            tls_recorder.recorder_id == 0) {
            return 0;
        }
        Recorder& recorder = *recorders_.at(tls_recorder.recorder_id);
        return recorder.command == VK_NULL_HANDLE
                   ? recorder.submitted_value
                   : flush_through_locked(recorder.reserved_value);
    }

    uint64_t VulkanRecorderRegistry::flush_all() {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        uint64_t submitted =
            flush_through_locked(std::numeric_limits<uint64_t>::max());
        for (const auto& [id, recorder] : recorders_) {
            (void)id;
            submitted = std::max(submitted, recorder->submitted_value);
        }
        collect_completed_locked(context_.completed_timeline());
        return submitted;
    }

    void VulkanRecorderRegistry::wait_all() {
        const uint64_t value = flush_all();
        if (value != 0) {
            context_.wait(value);
        }
        std::lock_guard lock(mutex_);
        collect_completed_locked(context_.completed_timeline());
    }

    void VulkanRecorderRegistry::check_fault_buffer() {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        if (fault_value_ == 0)
            return;
        // No writer can record or submit between this wait and the host clear.
        flush_through_locked(fault_value_);
        context_.wait(fault_value_);
        fault_value_ = 0;
        context_.check_fault_buffer();
    }

    void VulkanRecorderRegistry::run_external(const std::function<void()>& work) {
        require_tensor_access();
        std::lock_guard lock(mutex_);
        flush_through_locked(std::numeric_limits<uint64_t>::max());
        for (const auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->submitted_value)
                context_.wait(recorder->submitted_value);
        }
        collect_completed_locked(context_.completed_timeline());
        tls_external_work = true;
        try {
            context_.run_external(work);
        } catch (...) {
            tls_external_work = false;
            throw;
        }
        tls_external_work = false;
    }

    void VulkanRecorderRegistry::retire_resources(const std::function<void()>& work) {
        if (tls_retiring_resources) {
            work();
            return;
        }
        const auto retire = [&] {
            tls_retiring_resources = true;
            try {
                context_.retire_resources(work);
            } catch (...) {
                tls_retiring_resources = false;
                throw;
            }
            tls_retiring_resources = false;
        };
        if (tls_external_work)
            retire();
        else
            run_external(retire);
    }

    void VulkanRecorderRegistry::trim_memory() {
        require_tensor_access();
        // Block new recording/submission until physical buffer release finishes.
        // A plain wait_all() followed by a separate allocator lock leaves a
        // window for another thread to submit buffers about to be destroyed.
        std::lock_guard lock(mutex_);
        flush_through_locked(std::numeric_limits<uint64_t>::max());
        for (const auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->submitted_value != 0)
                context_.wait(recorder->submitted_value);
        }
        const uint64_t completed = context_.completed_timeline();
        collect_completed_locked(completed);
        context_.retire_resources([&] { context_.memory().trim_completed(completed); });
    }

    void VulkanRecorderRegistry::release_thread(const uint64_t recorder_id) {
        std::lock_guard lock(mutex_);
        const auto iterator = recorders_.find(recorder_id);
        if (iterator == recorders_.end()) {
            return;
        }
        Recorder& recorder = *iterator->second;
        if (recorder.command != VK_NULL_HANDLE) {
            if (context_.dead()) {
                // Nothing reaches a lost device; the buffer goes with the pool at
                // shutdown. Submitting here would throw out of a thread-exit
                // destructor.
                recorder.command = VK_NULL_HANDLE;
                recorder.reserved_value = 0;
                recorder.command_count = 0;
            } else {
                flush_through_locked(recorder.reserved_value);
            }
        }
        recorder.owner_alive = false;
    }

    void VulkanRecorderRegistry::shutdown() {
        std::lock_guard lock(mutex_);
        if (shutting_down_) {
            return;
        }
        shutting_down_ = true;
        if (!context_.dead()) {
            try {
                const uint64_t submitted =
                    flush_through_locked(std::numeric_limits<uint64_t>::max());
                if (submitted != 0) {
                    context_.wait(submitted);
                }
                for (auto& [id, recorder] : recorders_) {
                    (void)id;
                    if (recorder->submitted_value != 0) {
                        context_.wait(recorder->submitted_value);
                    }
                }
            } catch (...) {
                if (!context_.dead()) {
                    shutting_down_ = false;
                    throw;
                }
            }
        }
        if (context_.dead()) {
            static_cast<void>(vkDeviceWaitIdle(context_.device()));
        }
        for (auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(context_.device(), recorder->pool, nullptr);
            }
        }
        recorders_.clear();
        if (tls_recorder.context_id == context_.context_id()) {
            tls_recorder = {};
        }
    }

} // namespace lfs::core::internal
