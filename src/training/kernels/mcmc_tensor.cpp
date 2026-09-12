/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "mcmc_tensor.hpp"
#include "core/tensor_backend.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace lfs::training::mcmc {
    using namespace core;

    std::pair<Tensor, Tensor> relocate(const Tensor& opacities, const Tensor& scales,
                                       const Tensor& ratios, const float min_opacity) {
        const size_t n = opacities.numel();
        if (!opacities.is_valid() || opacities.ndim() != 1 || opacities.dtype() != DataType::Float32 ||
            scales.shape() != TensorShape{n, 3} || scales.dtype() != DataType::Float32 ||
            ratios.shape() != TensorShape{n} || ratios.dtype() != DataType::Int32 ||
            scales.device() != opacities.device() || ratios.device() != opacities.device() ||
            gpu_backend_of(scales) != gpu_backend_of(opacities) || gpu_backend_of(ratios) != gpu_backend_of(opacities) ||
            !std::isfinite(min_opacity) || min_opacity < 0 || min_opacity >= 1)
            throw std::invalid_argument("MCMC relocation requires matching Float32 opacity/scales and Int32 ratios");
        if (n == 0)
            return {opacities.clone(), scales.clone()};
        if (ratios.min().cpu().item<int32_t>() < 1 || ratios.max().cpu().item<int32_t>() > 51)
            throw std::invalid_argument("MCMC relocation ratios must be in [1,51]");
        auto opacity = opacities.clamp(1e-6f, 1.f - 1e-6f);
        auto p = ((((opacity.neg() + 1.f).log() / ratios.to(DataType::Float32)).exp()).neg() + 1.f)
                     .clamp(std::max(1e-6f, min_opacity), 1.f - 1e-6f);
        // Sum_i C(i-1,k-1) = C(n,k). Horner evaluation avoids the quadratic
        // number of powers in the CUDA expression for the same denominator.
        static const auto coefficients = [] {
            std::vector<float> values(52 * 51, 0.f);
            for (int ratio = 1; ratio <= 51; ++ratio) {
                double binomial = 1;
                for (int k = 1; k <= ratio; ++k) {
                    binomial *= double(ratio - k + 1) / k;
                    values[ratio * 51 + k - 1] = static_cast<float>(binomial * (k % 2 ? 1 : -1) / std::sqrt(double(k)));
                }
            }
            return values;
        }();
        // Only the small constant coefficient table originates on the CPU.
        GpuBackendScope backend(gpu_backend_of(opacities).value_or(default_gpu_backend()));
        auto table = Tensor::from_vector(coefficients, {52, 51}, opacities.device());
        auto denominator = Tensor::zeros({n}, opacities.device());
        for (int k = 50; k >= 0; --k) {
            auto coefficient = table.slice(1, k, k + 1).squeeze(1).index_select(0, ratios);
            denominator = denominator * p + coefficient;
        }
        denominator = denominator * p;
        auto safe = denominator.abs().clamp_min(1e-8f);
        safe = Tensor::where(denominator.lt(0.f), -safe, safe);
        auto factor = (opacity / safe).clamp(-1e6f, 1e6f);
        return {p, (scales * factor.unsqueeze(1)).abs().clamp_min(1e-10f)};
    }

    void add_noise(const Tensor& raw_opacities, const Tensor& raw_scales, const Tensor& raw_quats,
                   const Tensor& noise, Tensor& means, const Tensor& frozen_mask, const float learning_rate) {
        const size_t n = means.ndim() == 2 ? means.shape()[0] : 0;
        if (!means.is_valid() || means.shape() != TensorShape{n, 3} || !std::isfinite(learning_rate) ||
            raw_opacities.numel() != n || raw_scales.shape() != means.shape() ||
            raw_quats.shape() != TensorShape{n, 4} || noise.shape() != means.shape())
            throw std::invalid_argument("MCMC noise tensor shapes do not match");
        for (const Tensor* tensor : {&raw_opacities, &raw_scales, &raw_quats, &noise, static_cast<const Tensor*>(&means)}) {
            if (tensor->dtype() != DataType::Float32 || tensor->device() != means.device() ||
                gpu_backend_of(*tensor) != gpu_backend_of(means))
                throw std::invalid_argument("MCMC noise requires Float32 tensors on one backend");
        }
        if (frozen_mask.is_valid() && (frozen_mask.shape() != TensorShape{n} || frozen_mask.dtype() != DataType::Bool ||
                                       frozen_mask.device() != means.device() || gpu_backend_of(frozen_mask) != gpu_backend_of(means)))
            throw std::invalid_argument("MCMC frozen mask must match means");
        if (n == 0)
            return;
        auto q = raw_quats * raw_quats.square().sum(1, true).rsqrt().clamp_max(1e12f);
        auto w = q.slice(1, 0, 1).squeeze(1), x = q.slice(1, 1, 2).squeeze(1), y = q.slice(1, 2, 3).squeeze(1), z = q.slice(1, 3, 4).squeeze(1);
        auto rotation = Tensor::stack({(y.square() + z.square()) * -2.f + 1.f, (x * y - w * z) * 2.f, (x * z + w * y) * 2.f,
                                       (x * y + w * z) * 2.f, (x.square() + z.square()) * -2.f + 1.f, (y * z - w * x) * 2.f,
                                       (x * z - w * y) * 2.f, (y * z + w * x) * 2.f, (x.square() + y.square()) * -2.f + 1.f},
                                      1)
                            .reshape(TensorShape{n, 3, 3});
        auto local_noise = rotation.transpose(1, 2).bmm(noise.unsqueeze(2));
        auto transformed = rotation.bmm(local_noise * (raw_scales * 2.f).exp().unsqueeze(2)).squeeze(2);
        auto factor = (raw_opacities.reshape(TensorShape{n, 1}).sigmoid() * -100.f + .5f).sigmoid() * learning_rate;
        auto updated = means + transformed * factor;
        if (frozen_mask.is_valid())
            updated = Tensor::where(frozen_mask.unsqueeze(1), means, updated);
        means.copy_from(updated);
    }

    void inject_noise(const Tensor& raw_opacities, const Tensor& raw_scales, const Tensor& raw_quats,
                      Tensor& means, const Tensor& frozen_mask, const float learning_rate, const uint64_t seed) {
        GpuBackendScope backend(gpu_backend_of(means).value_or(default_gpu_backend()));
        auto draws = Tensor::empty({2, means.numel()}, means.device());
        draws.uniform_(0.f, 1.f, seed);
        // Box-Muller from the existing explicitly seeded Philox uniform kernel.
        auto noise = ((draws.slice(0, 0, 1).squeeze(0).clamp_min(1e-7f).log() * -2.f).sqrt() *
                      (draws.slice(0, 1, 2).squeeze(0) * (2.f * std::numbers::pi_v<float>)).cos())
                         .reshape(means.shape());
        add_noise(raw_opacities, raw_scales, raw_quats, noise, means, frozen_mask, learning_rate);
    }
} // namespace lfs::training::mcmc
