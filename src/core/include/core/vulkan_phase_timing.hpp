/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <chrono>
#include <cstdint>
#include <vector>

namespace lfs::core {
    /// One GPU timestamp: `tag` is the caller's boundary label and `gpu_ms` is the device
    /// clock reading when every command submitted before it had completed.
    struct VulkanPhaseStamp {
        std::uint32_t tag;
        double gpu_ms;
    };

    /// One recorded-command boundary sample: `tag` is the caller's boundary label and
    /// `commands` is the number of commands recorded when the boundary was announced.
    ///
    /// This exists because a phase duration cannot say *why* a phase is long: the device may
    /// be busy, or the host may still be submitting. Differencing consecutive command counts
    /// attributes recorded work per phase, so submission volume and waiting can be told apart.
    struct VulkanPhaseCommandStamp {
        std::uint32_t tag;
        std::uint64_t commands;
    };

    /// GPU-side phase timing for the Vulkan training loop (plan §4).
    ///
    /// The training loop announces a phase boundary; the next command recorded afterwards
    /// carries a timestamp query. A timestamp write is ordered after all previously
    /// submitted work on the queue, so it marks where the named phase actually ends in the
    /// command stream rather than when the host reached the boundary. Difference the stamps
    /// of consecutive tags within an iteration to get phase durations.
    ///
    /// Host-side marks cannot do this: work recorded by an earlier phase is still in flight,
    /// so which phase drains it is arbitrary.
    LFS_CORE_API void vulkan_phase_timing_begin(std::uint32_t query_budget);
    LFS_CORE_API void vulkan_phase_timing_end();
    LFS_CORE_API void vulkan_phase_timing_announce(std::uint32_t tag);
    /// Absolute timestamps in recording order. Call only after the device has drained.
    /// Empty when the device cannot write timestamps.
    LFS_CORE_API std::vector<VulkanPhaseStamp> vulkan_phase_timestamps();
    /// Count one recorded command. Called by the recorder for every command.
    LFS_CORE_API void vulkan_phase_timing_note_command();
    /// Recorded-command totals at each announced boundary, in announcement order.
    LFS_CORE_API std::vector<VulkanPhaseCommandStamp> vulkan_phase_command_stamps();
    LFS_CORE_API bool vulkan_phase_timing_supported();

    /// Host-side wait accounting for the Vulkan backend (plan §81, phase 0).
    ///
    /// A host interval says how long a phase took, not what the host spent that time on. Every
    /// blocking wait on the device — allocator recycling, the recorder's in-flight limit, explicit
    /// flushes — goes through the timeline wait, so accumulating that time separates submission
    /// from stalls and shows whether a phase is host-bound because it is recording or because it
    /// is waiting.
    LFS_CORE_API void vulkan_host_wait_note_ns(std::uint64_t nanoseconds);
    /// Monotonic total of waited nanoseconds since process start.
    LFS_CORE_API std::uint64_t vulkan_host_wait_total_ns();
    /// Accounting is off until something asks for it, so the shipped path pays only one relaxed load
    /// per wait rather than two clock reads. The perf bench turns it on for the run it reports.
    LFS_CORE_API void vulkan_host_wait_set_enabled(bool enabled);
    LFS_CORE_API bool vulkan_host_wait_enabled();

    /// Adds one wait's duration to the total when the scope exits; free when accounting is off.
    class VulkanHostWaitScope {
    public:
        VulkanHostWaitScope() noexcept : enabled_(vulkan_host_wait_enabled()) {
            if (enabled_)
                start_ = std::chrono::steady_clock::now();
        }
        ~VulkanHostWaitScope() {
            if (!enabled_)
                return;
            const auto elapsed = std::chrono::steady_clock::now() - start_;
            vulkan_host_wait_note_ns(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
        }
        VulkanHostWaitScope(const VulkanHostWaitScope&) = delete;
        VulkanHostWaitScope& operator=(const VulkanHostWaitScope&) = delete;
        VulkanHostWaitScope(VulkanHostWaitScope&&) = delete;
        VulkanHostWaitScope& operator=(VulkanHostWaitScope&&) = delete;

    private:
        bool enabled_;
        std::chrono::steady_clock::time_point start_;
    };
} // namespace lfs::core
