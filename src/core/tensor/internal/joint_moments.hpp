/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core::internal {
    struct AdamUpdateConfig {
        float beta1, beta2, bias_correction2, eps, learning_rate;
    };
    // Inputs are canonical Float32 rows. Optional masks/scales broadcast by row.
    LFS_CORE_API void vulkan_adam_update(Tensor& first, Tensor& second, Tensor& delta,
                                         const Tensor& gradient, const Tensor& enabled,
                                         const Tensor& lr_scale, AdamUpdateConfig config);

    // Storage is validated by the tensor codec. False selects its portable fallback.
    LFS_CORE_API bool try_decode_joint_moments(const Tensor& packed, const Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               Tensor& first, Tensor& second);
    LFS_CORE_API bool try_encode_joint_moments(const Tensor& first, const Tensor& second,
                                               Tensor& packed, Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               const Tensor* valid);
} // namespace lfs::core::internal
