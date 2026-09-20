/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * Periodic 3D Morton reordering of the live Gaussian training set.
 *
 * Densification appends rows in arbitrary order; reordering by Morton code of
 * the current AABB restores spatial locality for depth-sorted raster gathers.
 * q16 SH codes and joint Adam moments use per-256-splat block bounds, so a
 * raw row gather is invalid — this path gather-decodes per dest 256-block,
 * re-encodes with fresh dest bounds, and copies back into the live region.
 */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include "core/cuda_stream_fwd.hpp"
#include <cstddef>
#include <functional>

namespace lfs::training {
    class AdamOptimizer;
}

namespace lfs::training::morton {

    [[nodiscard]] inline bool should_reorder(
        const int iter,
        const std::size_t interval,
        const std::size_t stop_refine) noexcept {
        if (interval == 0 || iter <= 0) {
            return false;
        }
        if (static_cast<std::size_t>(iter) > stop_refine) {
            return false;
        }
        return static_cast<std::size_t>(iter) % interval == 0;
    }

    struct ReorderResult {
        bool applied = false;
        lfs::core::Tensor permutation; // Int64 [N], dest i <- src perm[i]
    };

    /// Permute the specified primitive axis (default 0; densification_info uses 1).
    /// Tensors longer than perm.n on dim 0 preserve the unused tail.
    void permute_row_tensor(lfs::core::Tensor& tensor, const lfs::core::Tensor& perm, int axis = 0);

    /// dest[i] <- src[perm[i]] for shN. q16 gather-decodes each dest 256-block
    /// in one launch (no full fp32 expansion). fp32 / IEEE-f16 keep the
    /// swizzled gather path.
    void permute_shN(
        core::SplatData& splat,
        const lfs::core::Tensor& perm,
        cudaStream_t stream = nullptr);

    /// Apply Morton ordering to all per-Gaussian model parameters, optimizer
    /// moments, and row-indexed aux tensors on `splat`. Skips (and logs at
    /// debug) when FrozenRange spans are active.
    [[nodiscard]] ReorderResult apply_morton_reorder(
        lfs::core::SplatData& splat,
        AdamOptimizer* optimizer = nullptr,
        cudaStream_t stream = nullptr
#if !LFS_TENSOR_CUDA
        , const std::function<void(const core::Tensor&)>& before_commit = {}
#endif
    );

} // namespace lfs::training::morton
