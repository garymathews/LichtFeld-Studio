/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core::internal {
    // Storage is validated by the tensor codec. False selects its portable fallback.
    LFS_CORE_API bool try_decode_joint_moments(const Tensor& packed, const Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               Tensor& first, Tensor& second);
    LFS_CORE_API bool try_encode_joint_moments(const Tensor& first, const Tensor& second,
                                               Tensor& packed, Tensor& bounds,
                                               int bits, size_t block_cells, float eps,
                                               const Tensor* valid);
} // namespace lfs::core::internal
