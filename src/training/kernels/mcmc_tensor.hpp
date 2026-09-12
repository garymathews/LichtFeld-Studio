/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"

namespace lfs::training::mcmc {
    // Tensor equivalents of the MCMC kernels. All arithmetic stays on the input backend.
    std::pair<core::Tensor, core::Tensor> relocate(
        const core::Tensor& opacities, const core::Tensor& scales,
        const core::Tensor& ratios, float min_opacity);
    void add_noise(const core::Tensor& raw_opacities, const core::Tensor& raw_scales,
                   const core::Tensor& raw_quats, const core::Tensor& noise,
                   core::Tensor& means, const core::Tensor& frozen_mask, float learning_rate);
    void inject_noise(const core::Tensor& raw_opacities, const core::Tensor& raw_scales,
                      const core::Tensor& raw_quats, core::Tensor& means,
                      const core::Tensor& frozen_mask, float learning_rate, uint64_t seed);
} // namespace lfs::training::mcmc
