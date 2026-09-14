/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "vk_ops_common.hpp"

#include <span>

namespace lfs::core::internal::vk {

    // Stable radix keys and original positions stay alive until the consuming
    // dispatch is recorded. ScopedAllocation retires scratch on the GPU timeline.
    class RadixSort {
    public:
        enum KeyKind : uint32_t { FloatKeys = 0,
                                  IndexKeys = 1 };

        RadixSort(VulkanContext& context, StorageRef values,
                  const SortProgram& program, KeyKind key_kind);
        [[nodiscard]] StorageRef keys() const { return scratch_[0].storage(); }
        [[nodiscard]] StorageRef positions() const { return scratch_[2].storage(); }
        void gather(StorageRef values, StorageRef indices);

    private:
        struct RadixPush {
            uint64_t values_address;
            uint64_t indices_address;
            uint64_t keys_a_address;
            uint64_t keys_b_address;
            uint64_t positions_a_address;
            uint64_t positions_b_address;
            uint64_t histogram_address;
            uint32_t lines;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t blocks_per_line;
            uint32_t shift;
            uint32_t parity;
            uint32_t descending;
            uint32_t total;
        };
        static_assert(sizeof(RadixPush) == 88);
        void dispatch(uint32_t phase, uint32_t groups,
                      std::span<const StorageRef> reads, std::span<const StorageRef> writes);

        VulkanContext& context_;
        KeyKind key_kind_;
        RadixPush push_;
        std::array<ScopedAllocation, 5> scratch_;
    };

} // namespace lfs::core::internal::vk
