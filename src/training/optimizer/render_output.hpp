/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::core {
    class Camera;
}

namespace lfs::training {

    struct RenderOutput {
        lfs::core::Tensor image;        // [..., channels, H, W]
        lfs::core::Tensor target_image; // Current GT image [C, H, W], when available
        lfs::core::Tensor alpha;        // [..., C, H, W, 1]
        lfs::core::Tensor depth;        // [..., C, H, W, 1] - accumulated or expected depth
        lfs::core::Tensor normal;       // [3, H, W] - accumulated camera-space normals, empty unless rendered
        lfs::core::Tensor means2d;      // [..., C, N, 2]
        lfs::core::Tensor depths;       // [..., N] - per-gaussian depths
        lfs::core::Tensor radii;        // [..., N]
        lfs::core::Tensor visibility;   // [..., N]
        lfs::core::Tensor edges_score;
        lfs::core::Camera* camera = nullptr; // Current training camera, when available
        int width = 0;
        int height = 0;

        // Replace the solid background already included by the rasterizer.
        void composite_background(const core::Tensor& pixels, const core::Tensor& solid = {}) {
            if (!pixels.is_valid()) return;
            const auto difference = solid.is_valid() ? pixels - solid.reshape({3, 1, 1}) : pixels;
            image = image + difference * (alpha.reshape(core::TensorShape{1, image.shape()[1], image.shape()[2]}).neg() + 1.f);
        }
    };

    enum class RenderMode {
        RGB = 0
    };

} // namespace lfs::training
