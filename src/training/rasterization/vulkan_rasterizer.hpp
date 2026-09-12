/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <array>
#include <filesystem>
#include <memory>

namespace lfs::training {
    class AdamOptimizer;
    // One training camera at a time. The model must remain unchanged between
    // forward and backward. Native resources retire before their Vulkan device,
    // even if this consumer outlives backend shutdown.
    // Large image regions are subdivided, then recomputed during backward to
    // bound retained raster scratch independently of the total instance count.
    class VulkanTrainingRasterizer {
    public:
        // Zero selects the renderer's native per-region instance budget.
        explicit VulkanTrainingRasterizer(const std::filesystem::path& shader_directory = {},
                                            uint32_t max_tile_instances = 0);
        ~VulkanTrainingRasterizer();
        VulkanTrainingRasterizer(const VulkanTrainingRasterizer&) = delete;
        VulkanTrainingRasterizer& operator=(const VulkanTrainingRasterizer&) = delete;
        RenderOutput forward(const core::Camera& camera, core::SplatData& model,
                             std::array<float, 3> background = {0.f, 0.f, 0.f}, bool mip_filter = false);
        void backward(const core::Tensor& image_gradient, AdamOptimizer& optimizer,
                      const core::Tensor& alpha_gradient = {}, const core::Tensor& error_map = {});

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::training
