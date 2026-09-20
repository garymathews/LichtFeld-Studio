/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        struct ScreenAxes {
            Tensor x;
            Tensor y;
            Tensor valid;
        };

        ScreenAxes screen_axes(const Tensor& screen_positions) {
            ScreenAxes axes{
                .x = screen_positions.slice(1, 0, 1),
                .y = screen_positions.slice(1, 1, 2),
            };
            axes.valid = (axes.x >= kInvalidScreenPositionThreshold)
                             .logical_and(axes.y >= kInvalidScreenPositionThreshold);
            return axes;
        }
    } // namespace

    void rect_select_tensor(
        const Tensor& screen_positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor inside = axes.valid
                                  .logical_and(axes.x >= x0)
                                  .logical_and(axes.x <= x1)
                                  .logical_and(axes.y >= y0)
                                  .logical_and(axes.y <= y1)
                                  .flatten();
        selection_out.masked_fill_(inside, 1.0f);
    }

    int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        const float x,
        const float y,
        const float radius) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return -1;
        }
        if (screen_positions.device() != Device::CUDA ||
            screen_positions.dtype() != DataType::Float32 ||
            screen_positions.ndim() != 2 ||
            screen_positions.size(1) != 2) {
            throw std::runtime_error("pick_projected_gaussian_tensor expects a CUDA Float32 [N, 2] tensor");
        }
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor finite = axes.valid.logical_and(axes.x.isfinite()).logical_and(axes.y.isfinite());
        const Tensor dx = axes.x - x;
        const Tensor dy = axes.y - y;
        const Tensor dist_sq = (dx * dx + dy * dy)
                                   .masked_fill(finite.logical_not(), std::numeric_limits<float>::infinity())
                                   .flatten();
        const float best = dist_sq.min_scalar();
        if (!(best <= radius * radius)) {
            return -1;
        }
        // Equal distances pick the largest index, as the kernel did.
        const auto candidates = (dist_sq == best).nonzero().flatten().to_vector_int();
        return *std::max_element(candidates.begin(), candidates.end());
    }

    void set_selection_element(Tensor& selection, const int index, const bool value) {
        if (!selection.is_valid() || index < 0 ||
            static_cast<size_t>(index) >= selection.numel()) {
            return;
        }
#if LFS_TENSOR_CUDA
        if (lfs::core::gpu_backend_of(selection) == lfs::core::GpuBackend::CUDA) {
            set_selection_element(selection.ptr<bool>(), index, value);
            return;
        }
#endif
        selection.flatten().slice(0, index, index + 1).fill_(value ? 1.f : 0.f);
    }

    void brush_select_tensor(const Tensor& positions, float x, float y, float radius, Tensor& output) {
        if (!positions.is_valid() || !positions.numel())
            return;
#if LFS_TENSOR_CUDA
        if (core::gpu_backend_of(positions) == core::GpuBackend::CUDA &&
            core::gpu_backend_of(output) == core::GpuBackend::CUDA) {
            if (positions.size(0) > size_t(std::numeric_limits<int>::max()))
                throw std::overflow_error("n_primitives exceeds int range");
            brush_select(reinterpret_cast<const float2*>(positions.ptr<float>()), x, y, radius,
                         reinterpret_cast<uint8_t*>(output.ptr<bool>()), static_cast<int>(positions.size(0)));
            return;
        }
#endif
        const auto px = positions.slice(1, 0, 1).squeeze(1);
        const auto py = positions.slice(1, 1, 2).squeeze(1);
        const auto hit = ((px - x).square() + (py - y).square() <= radius * radius)
                             .logical_and(px >= kInvalidScreenPositionThreshold)
                             .logical_and(py >= kInvalidScreenPositionThreshold);
        output.copy_(output.logical_or(hit));
    }
    void polygon_select_tensor(const Tensor& positions, const Tensor& vertices, Tensor& output) {
        if (!positions.is_valid() || !positions.numel() || !vertices.is_valid() || vertices.shape()[0] < 3)
            return;
#if LFS_TENSOR_CUDA
        if (core::gpu_backend_of(positions) == core::GpuBackend::CUDA &&
            core::gpu_backend_of(vertices) == core::GpuBackend::CUDA &&
            core::gpu_backend_of(output) == core::GpuBackend::CUDA) {
            if (positions.size(0) > size_t(std::numeric_limits<int>::max()) ||
                vertices.size(0) > size_t(std::numeric_limits<int>::max()))
                throw std::overflow_error("polygon selection count exceeds int range");
            polygon_select(reinterpret_cast<const float2*>(positions.ptr<float>()),
                           reinterpret_cast<const float2*>(vertices.ptr<float>()), static_cast<int>(vertices.size(0)),
                           output.ptr<bool>(), static_cast<int>(positions.size(0)));
            return;
        }
#endif
        const auto polygon = vertices.cpu().to(core::DataType::Float32).to_vector();
        const auto x = positions.slice(1, 0, 1).squeeze(1);
        const auto y = positions.slice(1, 1, 2).squeeze(1);
        auto inside = Tensor::zeros_like(output);
        const size_t n = polygon.size() / 2;
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const float xi = polygon[2 * i], yi = polygon[2 * i + 1], xj = polygon[2 * j], yj = polygon[2 * j + 1];
            if (yi == yj)
                continue;
            const auto crossing = (y < yi).logical_xor(y < yj).logical_and(x < (y - yi) * ((xj - xi) / (yj - yi)) + xi);
            inside = inside.logical_xor(crossing);
        }
        inside = inside.logical_and(x >= kInvalidScreenPositionThreshold).logical_and(y >= kInvalidScreenPositionThreshold);
        output.copy_(output.logical_or(inside));
    }

} // namespace lfs::rendering
