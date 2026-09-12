/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <climits>
#include <cmath>

namespace lfs::core {
#if LFS_TENSOR_CUDA
    Tensor resize_depth_prior_cuda(const Tensor&, int, int, cudaStream_t);
    Tensor resize_normal_prior_cuda(const Tensor&, int, int, cudaStream_t);
#endif
    namespace {
        template <int Channels>
        Tensor resize_prior(const Tensor& input, const int height, const int width, cudaStream_t stream) {
            if (!input.is_valid() || input.dtype() != DataType::Float32 ||
                input.ndim() != (Channels == 1 ? 2 : 3) ||
                (Channels == 3 && input.size(0) != 3) || height <= 0 || width <= 0 ||
                input.size(input.ndim() - 2) == 0 || input.size(input.ndim() - 1) == 0 ||
                input.size(input.ndim() - 2) > INT_MAX || input.size(input.ndim() - 1) > INT_MAX) {
                throw std::invalid_argument("Prior resize requires float depth [H,W] or normals [3,H,W] and positive dimensions");
            }
#if LFS_TENSOR_CUDA
            if (gpu_backend_of(input) == GpuBackend::CUDA) {
                if constexpr (Channels == 1)
                    return resize_depth_prior_cuda(input.contiguous(), height, width, stream);
                else
                    return resize_normal_prior_cuda(input.contiguous(), height, width, stream);
            }
#endif
            if (stream)
                throw TensorError("A CUDA stream cannot be used for CPU/Vulkan prior preprocessing");
            const auto source = input.cpu().contiguous();
            const int sh = static_cast<int>(input.size(input.ndim() - 2));
            const int sw = static_cast<int>(input.size(input.ndim() - 1));
            const size_t sp = static_cast<size_t>(sw) * sh, dp = static_cast<size_t>(width) * height;
            const auto shape = Channels == 1 ? TensorShape{size_t(height), size_t(width)}
                                             : TensorShape{3, size_t(height), size_t(width)};
            auto output = Tensor::empty(shape, Device::CPU);
            const float* src = source.ptr<float>();
            float* dst = output.ptr<float>();
            const auto valid = [&](size_t i) {
                if constexpr (Channels == 1)
                    return std::isfinite(src[i]) && src[i] > 0.f;
                else {
                    const float x = src[i], y = src[sp + i], z = src[2 * sp + i];
                    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x * x + y * y + z * z >= 0.25f;
                }
            };
            // Dataset preprocessing mirrors the CUDA prior kernel, including its
            // nearest validity gate and renormalization over valid neighbors.
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const float sx = std::clamp((x + 0.5f) * sw / width - 0.5f, 0.f, sw - 1.f);
                    const float sy = std::clamp((y + 0.5f) * sh / height - 0.5f, 0.f, sh - 1.f);
                    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
                    const size_t nearest = size_t(static_cast<int>(sy + 0.5f)) * sw + static_cast<int>(sx + 0.5f);
                    float value[Channels] = {}, weight = 0.f;
                    if (valid(nearest))
                        for (int j = 0; j < 2; ++j)
                            for (int i = 0; i < 2; ++i) {
                                const size_t index = size_t(std::min(y0 + j, sh - 1)) * sw + std::min(x0 + i, sw - 1);
                                if (!valid(index))
                                    continue;
                                const float w = (i ? sx - x0 : 1.f - (sx - x0)) * (j ? sy - y0 : 1.f - (sy - y0));
                                weight += w;
                                for (int c = 0; c < Channels; ++c)
                                    value[c] += w * src[c * sp + index];
                            }
                    if constexpr (Channels == 3)
                        weight = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
                    for (int c = 0; c < Channels; ++c)
                        dst[c * dp + size_t(y) * width + x] = weight > 1e-8f ? value[c] / weight : 0.f;
                }
            if (const auto backend = gpu_backend_of(input)) {
                GpuBackendScope scope(*backend);
                return output.to(input.device());
            }
            return output;
        }
    } // namespace
    Tensor resize_depth_prior(const Tensor& input, int height, int width, cudaStream_t stream) {
        return resize_prior<1>(input, height, width, stream);
    }
    Tensor resize_normal_prior(const Tensor& input, int height, int width, cudaStream_t stream) {
        return resize_prior<3>(input, height, width, stream);
    }
} // namespace lfs::core
