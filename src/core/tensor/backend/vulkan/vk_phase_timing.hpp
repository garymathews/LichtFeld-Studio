/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/vulkan_phase_timing.hpp"
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core::internal {
    /// Write a timestamp for every announced boundary that has not been timestamped yet.
    /// Called by the recorder before it appends a command: the write is ordered after all
    /// previously submitted work, which is exactly where the announced phase ended.
    /// Count one recorded command, so boundaries can attribute submission volume rather
    /// than only elapsed device time.
    void vk_phase_timing_note_command();
    [[nodiscard]] std::vector<VulkanPhaseCommandStamp> vk_phase_command_stamps();
    void vk_phase_timing_write_pending(VkCommandBuffer command);
    void vk_phase_timing_begin(std::uint32_t query_budget);
    void vk_phase_timing_end();
    void vk_phase_timing_announce(std::uint32_t tag);
    std::vector<lfs::core::VulkanPhaseStamp> vk_phase_timestamps();
    bool vk_phase_timing_supported();
} // namespace lfs::core::internal
