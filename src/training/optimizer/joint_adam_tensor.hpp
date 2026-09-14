/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"

namespace lfs::training::joint_adam {
    struct TensorMoments {
        lfs::core::Tensor first;
        lfs::core::Tensor second;
    };

    // Storage order is preserved, including swizzled SH cells. Each consecutive
    // cells_per_block cells shares the existing four-float joint-codec bounds.
    TensorMoments decode_tensor(const lfs::core::Tensor& packed,
                                const lfs::core::Tensor& bounds,
                                int bits, size_t cells_per_block,
                                const lfs::core::Tensor* cell_indices = nullptr);
    void encode_tensor(const TensorMoments& moments,
                       lfs::core::Tensor& packed, lfs::core::Tensor& bounds,
                       int bits, size_t cells_per_block,
                       const lfs::core::Tensor* valid_cells = nullptr);
} // namespace lfs::training::joint_adam
