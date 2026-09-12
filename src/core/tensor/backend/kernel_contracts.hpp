/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <cstdint>
#include <cstddef>
// ============= CPU Helpers (Generic, Header-Only) =============
namespace lfs::core {
    // CPU helper for unary operations
    template <typename T, typename OutT, typename Op>
    void apply_unary_cpu(const T* input, OutT* output, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = op(input[i]);
        }
    }

    // CPU helper for binary operations
    template <typename T, typename OutputT, typename Op>
    void apply_binary_cpu(const T* a, const T* b, OutputT* c, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = op(a[i], b[i]);
        }
    }
} // namespace lfs::core


namespace lfs::core::tensor_ops {
    // Host heuristic: true → prefer strided_fast over permute+contiguous.
    [[nodiscard]] LFS_CORE_API bool should_prefer_strided_over_transpose(
        size_t outer_size, size_t reduce_size, size_t inner_size) noexcept;

    // Test/debug hooks for path selection
    enum class ReducePathForTesting : int {
        None = 0,
        StridedFast,
        Transpose,
        Column,
        Default,
    };
    LFS_CORE_API void set_reduce_path_override_for_testing(ReducePathForTesting path) noexcept;
    [[nodiscard]] LFS_CORE_API ReducePathForTesting reduce_path_override_for_testing() noexcept;
    [[nodiscard]] LFS_CORE_API ReducePathForTesting reduce_last_path_for_testing() noexcept;
    LFS_CORE_API void set_reduce_last_path_for_testing(ReducePathForTesting path) noexcept;


    // ============= Fused Pointwise Chain =============
    static constexpr int FUSED_POINTWISE_MAX_OPS = 16;

    struct FusedPointwiseOp {
        uint8_t kind = 0;
        float scalar = 0.0f;
        // Device pointer for tensor-binary stages (kinds 4-7). Null for scalar/unary.
        const float* rhs = nullptr;
    };

    struct FusedPointwiseOpChain {
        FusedPointwiseOp ops[FUSED_POINTWISE_MAX_OPS];
        int num_ops = 0;
    };

    // Optional test/diagnostic counter of tensor-lib kernel launches (fused + binary).
    LFS_CORE_API void reset_tensor_kernel_launch_count() noexcept;
    LFS_CORE_API uint64_t tensor_kernel_launch_count() noexcept;
    LFS_CORE_API void record_tensor_kernel_launch(uint64_t n = 1) noexcept;

}
