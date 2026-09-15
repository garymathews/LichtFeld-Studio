// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/tensor/internal/mcmc_noise.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#if LFS_TENSOR_VULKAN
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_pipelines.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#endif

namespace lfs::core::internal {
    void vulkan_mcmc_noise(const Tensor& opacity, const Tensor& scales, const Tensor& quaternions,
                           Tensor& means, const Tensor& frozen, float learning_rate, uint64_t seed) {
#if LFS_TENSOR_VULKAN
        const size_t n = means.ndim() == 2 ? means.shape()[0] : 0;
        if (!means.is_valid() || means.shape() != TensorShape{n, 3} || opacity.numel() != n ||
            scales.shape() != means.shape() || quaternions.shape() != TensorShape{n, 4} || !std::isfinite(learning_rate))
            throw std::invalid_argument("MCMC noise tensor shapes do not match");
        for (const Tensor* input : {&opacity, &scales, &quaternions, static_cast<const Tensor*>(&means)}) {
            if (input->dtype() != DataType::Float32 || gpu_backend_of(*input) != GpuBackend::Vulkan)
                throw std::invalid_argument("MCMC noise requires Float32 Vulkan tensors");
        }
        if (frozen.is_valid() && (frozen.shape() != TensorShape{n} || frozen.dtype() != DataType::Bool ||
                                  gpu_backend_of(frozen) != GpuBackend::Vulkan))
            throw std::invalid_argument("MCMC frozen mask must match means");
        if (n == 0) return;
        const std::array inputs{opacity.contiguous(), scales.contiguous(), quaternions.contiguous(),
                                means.contiguous(), frozen.is_valid() ? frozen.contiguous() : Tensor{}};
        auto next = allocate_like(means, means.shape(), DataType::Float32);
        const std::array reads{storage_ref(inputs[0]), storage_ref(inputs[1]), storage_ref(inputs[2]),
                               storage_ref(inputs[3]), frozen.is_valid() ? storage_ref(inputs[4]) : StorageRef{}};
        const std::array writes{storage_ref(next)};
        struct Push {
            uint64_t opacity, scales, quaternions, means, frozen, output, seed;
            uint32_t count;
            float learning_rate;
        };
        static_assert(sizeof(Push) == 64);
        const Push push{vk::address(reads[0]), vk::address(reads[1]), vk::address(reads[2]), vk::address(reads[3]),
                        frozen.is_valid() ? vk::address(reads[4]) : 0, vk::address(writes[0]), seed,
                        vk::checked_u32(quaternions.numel(), "MCMC noise exceeds uint32") / 4u, learning_rate};
        const auto context = acquire_vulkan_context();
        const auto& pipeline = context->pipelines().specialized("mcmc_noise", sizeof(Push), {});
        vk::record_dispatch(*context, pipeline, push, reads, writes, vk::dispatch_groups(*context, n), 1, 1);
        means.copy_from(next);
#else
        throw std::runtime_error("Vulkan MCMC noise requires the Vulkan backend");
#endif
    }
}
