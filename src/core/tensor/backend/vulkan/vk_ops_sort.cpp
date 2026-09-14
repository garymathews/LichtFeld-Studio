/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_backend_ops.hpp"

#include "core/assert.hpp"
#include "core/tensor_storage.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_radix_sort.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <span>

namespace lfs::core::internal {
    namespace vk {
        namespace {
            constexpr size_t kRadixBlockElements = kLocalSize * 8;
            constexpr uint32_t kRadixDigits = 16;
            constexpr uint32_t kRadixPasses = 8;

            // Phases of radix.slang.
            constexpr uint32_t kExtractPhase = 0;
            constexpr uint32_t kHistogramPhase = 1;
            constexpr uint32_t kScanPhase = 2;
            constexpr uint32_t kScatterPhase = 3;
            constexpr uint32_t kGatherPhase = 4;
            constexpr uint32_t kWritePhase = 5;
        } // namespace

        RadixSort::RadixSort(VulkanContext& context, const StorageRef values,
                             const SortProgram& program, const KeyKind key_kind)
            : context_(context), key_kind_(key_kind),
              push_{
                  .values_address = address(values),
                  .lines = checked_u32(program.outer_size * program.inner_size, "Vulkan sort line count exceeds uint32"),
                  .dim_size = checked_u32(program.dim_size, "Vulkan sort size exceeds uint32"),
                  .inner = checked_u32(program.inner_size, "Vulkan sort inner size exceeds uint32"),
                  .blocks_per_line = checked_u32((program.dim_size + kRadixBlockElements - 1) / kRadixBlockElements,
                                                "Vulkan sort block count exceeds uint32"),
                  .descending = program.descending ? 1u : 0u,
                  .total = checked_u32(program.outer_size * program.inner_size * program.dim_size,
                                       "Vulkan sort element count exceeds uint32"),
              },
              scratch_{
                  ScopedAllocation(context, size_t(push_.total) * sizeof(uint32_t)),
                  ScopedAllocation(context, size_t(push_.total) * sizeof(uint32_t)),
                  ScopedAllocation(context, size_t(push_.total) * sizeof(uint32_t)),
                  ScopedAllocation(context, size_t(push_.total) * sizeof(uint32_t)),
                  ScopedAllocation(context, size_t(push_.lines) * kRadixDigits * push_.blocks_per_line * sizeof(uint32_t)),
              } {
            const StorageRef keys_a = keys();
            const StorageRef keys_b = scratch_[1].storage();
            const StorageRef positions_a = positions();
            const StorageRef positions_b = scratch_[3].storage();
            const StorageRef histogram = scratch_[4].storage();
            push_.keys_a_address = address(keys_a);
            push_.keys_b_address = address(keys_b);
            push_.positions_a_address = address(positions_a);
            push_.positions_b_address = address(positions_b);
            push_.histogram_address = address(histogram);
            const uint32_t element_groups = dispatch_groups(context, push_.total);
            const uint32_t block_groups = static_cast<uint32_t>(
                std::min<size_t>(size_t(push_.lines) * push_.blocks_per_line, context.caps().max_workgroup_count[0]));
            const uint32_t line_groups = std::min(push_.lines, context.caps().max_workgroup_count[0]);
            {
                const std::array reads{values};
                const std::array writes{keys_a, positions_a};
                dispatch(kExtractPhase, element_groups, reads, writes);
            }
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
                push_.shift = pass * 4;
                push_.parity = pass & 1u;
                const StorageRef source_keys = push_.parity == 0 ? keys_a : keys_b;
                const StorageRef source_positions = push_.parity == 0 ? positions_a : positions_b;
                const StorageRef target_keys = push_.parity == 0 ? keys_b : keys_a;
                const StorageRef target_positions = push_.parity == 0 ? positions_b : positions_a;
                {
                    const std::array reads{source_keys};
                    const std::array writes{histogram};
                    dispatch(kHistogramPhase, block_groups, reads, writes);
                }
                {
                    const std::array reads{histogram};
                    const std::array writes{histogram};
                    dispatch(kScanPhase, line_groups, reads, writes);
                }
                {
                    const std::array reads{source_keys, source_positions, histogram};
                    const std::array writes{target_keys, target_positions};
                    dispatch(kScatterPhase, block_groups, reads, writes);
                }
            }
        }

        void RadixSort::dispatch(const uint32_t phase, const uint32_t groups,
                                 const std::span<const StorageRef> reads,
                                 const std::span<const StorageRef> writes) {
            const std::array constants{phase, static_cast<uint32_t>(key_kind_)};
            const VulkanPipeline& pipeline =
                context_.pipelines().specialized("radix", sizeof(RadixPush), constants);
            vk::record_dispatch(context_, pipeline, push_, reads, writes, groups, 1, 1);
        }

        void RadixSort::gather(const StorageRef values, const StorageRef indices) {
            push_.indices_address = address(indices);
            const StorageRef keys_b = scratch_[1].storage();
            const StorageRef positions_a = positions();
            const uint32_t element_groups = dispatch_groups(context_, push_.total);
            {
                const std::array reads{values, positions_a};
                const std::array writes{keys_b};
                dispatch(kGatherPhase, element_groups, reads, writes);
            }
            {
                const std::array reads{keys_b, positions_a};
                const std::array writes{values, indices};
                dispatch(kWritePhase, element_groups, reads, writes);
            }
        }
    } // namespace vk

    namespace {
        using vk::address;
        using vk::checked_u32;

        // Lines up to this length sort in one workgroup's shared memory
        // (sort.slang); longer lines take the multi-block radix sort (radix.slang).
        constexpr size_t kBitonicCapacity = 2048;

        struct SortPush {
            uint64_t values_address;
            uint64_t indices_address;
            uint32_t lines;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t descending;
            uint32_t pad0;
            uint32_t pad1;
        };
        static_assert(sizeof(SortPush) == 40);

        void sort_in_shared(VulkanContext& context, const StorageRef values,
                            const StorageRef indices, const SortProgram& program) {
            const SortPush push{
                .values_address = address(values),
                .indices_address = address(indices),
                .lines = checked_u32(program.outer_size * program.inner_size, "Vulkan sort line count exceeds uint32"),
                .dim_size = checked_u32(program.dim_size, "Vulkan sort size exceeds uint32"),
                .inner = checked_u32(program.inner_size, "Vulkan sort inner size exceeds uint32"),
                .descending = program.descending ? 1u : 0u,
            };
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("sort", sizeof(SortPush), {});
            const uint32_t groups = static_cast<uint32_t>(
                std::min<size_t>(program.outer_size * program.inner_size, context.caps().max_workgroup_count[0]));
            const std::array reads{values};
            const std::array writes{values, indices};
            vk::record_dispatch(context, pipeline, push, reads, writes, groups, 1, 1);
        }

        void sort_lines(const StorageRef values, const StorageRef indices, const SortProgram& program) {
            LFS_ASSERT_MSG(values.dtype == DataType::Float32 && indices.dtype == DataType::Int64,
                           "Vulkan sort requires Float32 values and Int64 indices");
            if (program.outer_size * program.inner_size == 0 || program.dim_size == 0) {
                return;
            }
            const auto context = acquire_vulkan_context();
            if (program.dim_size <= kBitonicCapacity) {
                sort_in_shared(*context, values, indices, program);
            } else {
                vk::RadixSort sorted(*context, values, program, vk::RadixSort::FloatKeys);
                sorted.gather(values, indices);
            }
        }
    } // namespace

    void VulkanBackendOps::sort_1d(
        const StorageRef values, const StorageRef indices, const size_t count,
        const SortProgram& program, ExecContext) {

        sort_lines(values, indices,
                   SortProgram{.dim_size = count, .descending = program.descending});
    }

    void VulkanBackendOps::sort_2d(
        const StorageRef values, const StorageRef indices, const SortProgram& program,
        ExecContext) {

        sort_lines(values, indices, program);
    }

} // namespace lfs::core::internal
