/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_kernels.hpp"
#include "core/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lfs::training::kernels {
    core::Tensor normalized_canny_edge_weights(const core::Tensor& image) {
        if (image.ndim() != 3 || image.shape()[0] != 3 || image.numel() == 0)
            throw std::invalid_argument("Edge weights require a nonempty RGB CHW image");
        // This once-per-camera CPU fallback preserves the CUDA filter's halo:
        // replicate RGB before blur, rather than replicate the blurred image.
        auto input = image.cpu().to(core::DataType::Float32);
        if (image.dtype() == core::DataType::UInt8) input = input / 255.f;
        const auto rgb = input.to_vector();
        const int h = static_cast<int>(image.shape()[1]), w = static_cast<int>(image.shape()[2]);
        const size_t pixels = static_cast<size_t>(h) * w;
        constexpr float gaussian[25] = {2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2};
        constexpr float sobel[9] = {-1,0,1,-2,0,2,-1,0,1};
        std::vector<float> blur(static_cast<size_t>(h + 4) * (w + 4));
        for (int y = -2; y < h + 2; ++y) for (int x = -2; x < w + 2; ++x) {
            float sum = 0.f;
            for (int ky = -2; ky <= 2; ++ky) for (int kx = -2; kx <= 2; ++kx) {
                const size_t i = static_cast<size_t>(std::clamp(y + ky, 0, h - 1)) * w + std::clamp(x + kx, 0, w - 1);
                sum += (gaussian[(ky + 2) * 5 + kx + 2] / 159.f) *
                       (.299f * rgb[i] + .587f * rgb[pixels + i] + .114f * rgb[2 * pixels + i]);
            }
            blur[static_cast<size_t>(y + 2) * (w + 4) + x + 2] = sum;
        }
        const int stride = w + 2;
        std::vector<float> gx(static_cast<size_t>(h + 2) * stride), gy(gx.size()), magnitude(gx.size());
        for (int y = -1; y < h + 1; ++y) for (int x = -1; x < w + 1; ++x) {
            const size_t i = static_cast<size_t>(y + 1) * stride + x + 1;
            for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx) {
                const float value = blur[static_cast<size_t>(y + ky + 2) * (w + 4) + x + kx + 2];
                gx[i] += sobel[(ky + 1) * 3 + kx + 1] * value;
                gy[i] += sobel[(kx + 1) * 3 + ky + 1] * value;
            }
            magnitude[i] = std::hypot(gx[i], gy[i]);
        }
        std::vector<float> output(pixels), positive;
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y + 1) * stride + x + 1;
            float value = magnitude[i];
            if (value > 0.f) {
                const int dx = std::clamp(static_cast<int>(std::round(gx[i] / value)), -1, 1);
                const int dy = std::clamp(static_cast<int>(std::round(gy[i] / value)), -1, 1);
                const auto offset = static_cast<ptrdiff_t>(dy) * stride + dx;
                if (value < magnitude[i + offset] || value < magnitude[i - offset]) value = 0.f;
            }
            output[static_cast<size_t>(y) * w + x] = value;
            if (value > 0.f) positive.push_back(value);
        }
        if (!positive.empty()) {
            auto median = positive.begin() + positive.size() / 2;
            std::nth_element(positive.begin(), median, positive.end());
            const float divisor = std::max(*median, 1e-9f);
            for (auto& value : output) value /= divisor;
        }
        return core::Tensor::from_vector(output, {static_cast<size_t>(h), static_cast<size_t>(w)}, image.device());
    }
}
