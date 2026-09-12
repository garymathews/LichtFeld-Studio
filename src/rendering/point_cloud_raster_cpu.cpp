/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "point_cloud_raster_common.hpp"
#include <stdexcept>
#include <vector>
namespace lfs::rendering::pcraster {
    void rasterizePointCloudCpu(const LaunchParams& p) {
        if (p.width <= 0 || p.height <= 0 || (p.channels != 3 && p.channels != 4) ||
            !p.image || !p.depth || (p.n_points && (!p.positions || !p.colors)))
            throw std::invalid_argument("Invalid point cloud raster buffers or dimensions");
        const size_t pixels = static_cast<size_t>(p.width) * p.height;
        constexpr uint64_t empty = ~uint64_t{0};
        std::vector<uint64_t> packed(pixels, empty);
        for (size_t i = 0; i < p.n_points; ++i)
            rasterizePoint(p, static_cast<int>(i), [&](int pixel, uint64_t value) {
                packed[pixel] = std::min(packed[pixel], value);
            });
        for (size_t i = 0; i < pixels; ++i) {
            if (packed[i] == empty) {
                p.image[i] = p.bg_r;
                p.image[pixels + i] = p.bg_g;
                p.image[2 * pixels + i] = p.bg_b;
                if (p.channels == 4) p.image[3 * pixels + i] = p.transparent_background ? 0.f : p.bg_a;
                p.depth[i] = p.far_plane;
            } else {
                const uint32_t color = static_cast<uint32_t>(packed[i]);
                for (size_t c = 0; c < 3; ++c) p.image[c * pixels + i] = float((color >> (8 * c)) & 255) / 255.f;
                if (p.channels == 4) p.image[3 * pixels + i] = 1.f;
                p.depth[i] = std::bit_cast<float>(static_cast<uint32_t>(packed[i] >> 32));
            }
        }
    }
}
