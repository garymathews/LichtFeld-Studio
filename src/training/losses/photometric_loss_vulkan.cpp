/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nn/ops.hpp"
#include "photometric_loss.hpp"
#include <cmath>
#if LFS_TENSOR_VULKAN
#include "core/nn/vulkan_ops.hpp"
#endif

namespace lfs::training::losses {

    PhotometricLoss::Context PhotometricLoss::forward_vulkan(
        const core::Tensor& prediction, const core::Tensor& target, const float weight,
        const core::Tensor& pixel_weight) {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        core::GpuBackendScope backend(core::GpuBackend::Vulkan);
        const Tensor gt = target.dtype() == DataType::UInt8
                              ? target.to(DataType::Float32) * (1.0f / 255.0f)
                              : target;
        const Tensor delta = prediction - gt;
        Tensor weights;
        float denominator = static_cast<float>(prediction.numel());
        if (pixel_weight.is_valid()) {
            weights = pixel_weight.dtype() == DataType::Float32 ? pixel_weight : pixel_weight.ne(0).to(DataType::Float32);
            // The existing masked-loss contract sums batches and normalizes by
            // channels * spatial mask sum, with a finite zero-mask denominator.
            denominator = weights.sum().cpu().item<float>() * prediction.shape()[1] + kernels::SSIM_EPSILON;
            if (!std::isfinite(denominator) || denominator <= 0.f)
                throw std::invalid_argument("Photometric mask has invalid total weight");
            weights = weights.unsqueeze(0).unsqueeze(0);
        }
        const float l1_scale = (1.f - weight) / denominator;
        Tensor loss = (weights.is_valid() ? delta.abs() * weights : delta.abs()).sum() * l1_scale;
        Tensor grad = delta.sign() * l1_scale;
        if (weights.is_valid()) grad = grad * weights;
        if (weight == 0.0f)
            return {loss.reshape({1}), grad};

        const size_t channels = prediction.shape()[1];
        if (!gaussian_horizontal_.is_valid() || gaussian_horizontal_.shape()[0] != channels) {
            // Same radius=5, sigma=1.5 Gaussian as the CUDA SSIM implementation.
            std::vector<float> coefficients(channels * 11);
            float total = 0.0f;
            for (int i = 0; i < 11; ++i) {
                coefficients[i] = std::exp(-static_cast<float>((i - 5) * (i - 5)) / 4.5f);
                total += coefficients[i];
            }
            for (int i = 0; i < 11; ++i)
                coefficients[i] /= total;
            for (size_t c = 1; c < channels; ++c)
                std::copy_n(coefficients.data(), 11, coefficients.data() + c * 11);
            gaussian_horizontal_ = Tensor::from_vector(coefficients, {channels, 1, 1, 11}, Device::GPU);
#if !LFS_TENSOR_VULKAN
            gaussian_vertical_ = gaussian_horizontal_.reshape({static_cast<int>(channels), 1, 11, 1});
#endif
        }
        const auto blur = [&](const Tensor& image) {
#if LFS_TENSOR_VULKAN
            return core::nn::vulkan::gaussian_blur_11(image, gaussian_horizontal_);
#else
            core::nn::Conv2dParams horizontal;
            horizontal.pad_w = 5;
            horizontal.groups = static_cast<int>(channels);
            core::nn::Conv2dParams vertical;
            vertical.pad_h = 5;
            vertical.groups = static_cast<int>(channels);
            return core::nn::conv2d(core::nn::conv2d(image, gaussian_horizontal_, nullptr, horizontal),
                                    gaussian_vertical_, nullptr, vertical);
#endif
        };
        const Tensor mu_x = blur(prediction);
        const Tensor mu_y = blur(gt);
        const Tensor var_x = blur(prediction * prediction) - mu_x * mu_x;
        const Tensor var_y = blur(gt * gt) - mu_y * mu_y;
        const Tensor covariance = blur(prediction * gt) - mu_x * mu_y;
        const Tensor a = mu_x * mu_x + mu_y * mu_y + 0.0001f;
        const Tensor b = var_x + var_y + 0.0009f;
        const Tensor c = mu_x * mu_y * 2.0f + 0.0001f;
        const Tensor d = covariance * 2.0f + 0.0009f;
        const Tensor ssim = c * d / (a * b);
        arena_.pure_ssim().ssim_map = ssim;
        arena_.pure_ssim().cs_map = d / b;
        Tensor mean_map = ssim;
        const size_t height = prediction.shape()[2], width = prediction.shape()[3];
        const bool valid_padding = !weights.is_valid() && height > 10 && width > 10;
        if (valid_padding)
            mean_map = ssim.slice(2, 5, height - 5).slice(3, 5, width - 5);
        loss = loss + (weights.is_valid() ? ((ssim.neg() + 1.f) * weights).sum() * (weight / denominator)
                                         : (mean_map.mean() * -1.f + 1.f) * weight);

        Tensor dmap = Tensor::zeros(prediction.shape(), Device::GPU);
        const float map_scale = -weight / static_cast<float>(mean_map.numel());
        if (weights.is_valid())
            dmap = weights.expand({static_cast<int>(prediction.shape()[0]), static_cast<int>(channels),
                                   static_cast<int>(height), static_cast<int>(width)}) * (-weight / denominator);
        else if (valid_padding)
            dmap.slice(2, 5, height - 5).slice(3, 5, width - 5).fill_(map_scale);
        else
            dmap.fill_(map_scale);
        // Differentiate through E[x], E[x^2], E[xy]. The symmetric zero-padded
        // Gaussian's transpose is the same blur, including the image boundary.
        const Tensor dvar = (ssim / b) * -1.0f;
        const Tensor dcov = c * 2.0f / (a * b);
        const Tensor dmu = mu_y * 2.0f * d / (a * b) - mu_x * 2.0f * ssim / a - mu_x * 2.0f * dvar - mu_y * dcov;
        grad = grad + blur(dmap * dmu) + prediction * 2.0f * blur(dmap * dvar) + gt * blur(dmap * dcov);
        return {loss.reshape({1}), grad};
    }
} // namespace lfs::training::losses
