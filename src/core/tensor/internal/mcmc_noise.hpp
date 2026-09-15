// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <cstdint>

namespace lfs::core::internal {
    LFS_CORE_API void vulkan_mcmc_noise(const Tensor& opacity, const Tensor& scales,
                                       const Tensor& quaternions, Tensor& means,
                                       const Tensor& frozen, float learning_rate, uint64_t seed);
}
