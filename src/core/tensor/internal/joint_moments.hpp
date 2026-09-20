/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core::internal {
    struct AdamUpdateConfig {
        float beta1, beta2, bias_correction2, eps, learning_rate;
    };
    // Optional joint-codec source for the moment inputs. When `packed` is set the
    // kernel decodes the moments itself instead of reading `first`/`second`, which
    // removes a dispatch and the fp32 moment round trip per chunk. `slots` is the SH
    // float4 slot count (0 for row-major parameters) and `block_cells` is the codec's
    // bounds block width, 256 rows times the row width.
    struct AdamMomentSource {
        const Tensor* packed = nullptr;
        const Tensor* bounds = nullptr;
        uint32_t slots = 0;
        uint32_t block_cells = 0;
        int bits = 0;
        float eps = 0.0f;
    };
    // Inputs are canonical Float32 rows. Optional masks/scales broadcast by row.
    LFS_CORE_API void vulkan_adam_update(Tensor& first, Tensor& second, Tensor& delta,
                                         const Tensor& gradient, const Tensor& enabled,
                                         const Tensor& lr_scale, AdamUpdateConfig config,
                                         const AdamMomentSource& source = {},
                                         bool swizzled_grad = false);

    // Storage is validated by the tensor codec. False selects its portable fallback.
    LFS_CORE_API bool try_decode_joint_moments(const Tensor& packed, const Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               Tensor& first, Tensor& second);
    LFS_CORE_API bool try_encode_joint_moments(const Tensor& first, const Tensor& second,
                                               Tensor& packed, Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               const Tensor* valid);
} // namespace lfs::core::internal
