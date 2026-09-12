/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/kernels/regularization.cuh" // LibTorch-free CUDA kernels
#include <cmath>
#include <format>

namespace lfs::training::losses {

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward(
        const lfs::core::Tensor& scaling_raw,
        lfs::core::Tensor& scaling_raw_grad,
        const Params& params) {
        try {
            if (!std::isfinite(params.weight))
                return std::unexpected("Regularization weight must be finite");
            core::GpuBackendScope backend(core::gpu_backend_of(scaling_raw).value_or(core::default_gpu_backend()));
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Validate inputs
            if (scaling_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw must be on a GPU device");
            }
            if (scaling_raw_grad.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw_grad must be on a GPU device");
            }
            if (scaling_raw.shape() != scaling_raw_grad.shape()) {
                return std::unexpected("scaling_raw and scaling_raw_grad must have same shape");
            }

            if (scaling_raw.dtype() != core::DataType::Float32 || !scaling_raw.is_contiguous())
                return std::unexpected("Regularization parameters must be contiguous Float32 tensors");
            if (scaling_raw_grad.dtype() != core::DataType::Float32 || !scaling_raw_grad.is_contiguous() ||
                core::gpu_backend_of(scaling_raw) != core::gpu_backend_of(scaling_raw_grad))
                return std::unexpected("Regularization gradient must match parameter dtype, layout and backend");
            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            if (core::gpu_backend_of(scaling_raw) == core::GpuBackend::Vulkan) {
                const auto activated = scaling_raw.exp();
                scaling_raw_grad.add_((activated) * (params.weight / static_cast<float>(n)));
                return (activated.mean() * params.weight).reshape({1});
            }
#if LFS_TENSOR_CUDA
            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_scale_regularization(
                scaling_raw.ptr<float>(),
                scaling_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

#else
            return std::unexpected("This regularization build requires Vulkan");
#endif
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward_loss_only(
        const lfs::core::Tensor& scaling_raw,
        const Params& params) {
        try {
            if (!std::isfinite(params.weight))
                return std::unexpected("Regularization weight must be finite");
            core::GpuBackendScope backend(core::gpu_backend_of(scaling_raw).value_or(core::default_gpu_backend()));
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }
            if (scaling_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("scaling_raw must be on a GPU device");
            }

            if (scaling_raw.dtype() != core::DataType::Float32 || !scaling_raw.is_contiguous())
                return std::unexpected("Regularization parameters must be contiguous Float32 tensors");
            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            if (core::gpu_backend_of(scaling_raw) == core::GpuBackend::Vulkan) {
                const auto activated = scaling_raw.exp();
                return (activated.mean() * params.weight).reshape({1});
            }
#if LFS_TENSOR_CUDA
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            lfs::training::kernels::launch_fused_scale_regularization(
                scaling_raw.ptr<float>(),
                nullptr,
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            return loss_tensor;
#else
            return std::unexpected("This regularization build requires Vulkan");
#endif
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward_loss_only: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward(
        const lfs::core::Tensor& opacity_raw,
        lfs::core::Tensor& opacity_raw_grad,
        const Params& params) {
        try {
            if (!std::isfinite(params.weight))
                return std::unexpected("Regularization weight must be finite");
            core::GpuBackendScope backend(core::gpu_backend_of(opacity_raw).value_or(core::default_gpu_backend()));
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            // Validate inputs
            if (opacity_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw must be on a GPU device");
            }
            if (opacity_raw_grad.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw_grad must be on a GPU device");
            }
            if (opacity_raw.shape() != opacity_raw_grad.shape()) {
                return std::unexpected("opacity_raw and opacity_raw_grad must have same shape");
            }

            if (opacity_raw.dtype() != core::DataType::Float32 || !opacity_raw.is_contiguous())
                return std::unexpected("Regularization parameters must be contiguous Float32 tensors");
            if (opacity_raw_grad.dtype() != core::DataType::Float32 || !opacity_raw_grad.is_contiguous() ||
                core::gpu_backend_of(opacity_raw) != core::gpu_backend_of(opacity_raw_grad))
                return std::unexpected("Regularization gradient must match parameter dtype, layout and backend");
            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            if (core::gpu_backend_of(opacity_raw) == core::GpuBackend::Vulkan) {
                const auto activated = opacity_raw.sigmoid();
                opacity_raw_grad.add_((activated * (activated * -1.0f + 1.0f)) * (params.weight / static_cast<float>(n)));
                return (activated.mean() * params.weight).reshape({1});
            }
#if LFS_TENSOR_CUDA
            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_opacity_regularization(
                opacity_raw.ptr<float>(),
                opacity_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

#else
            return std::unexpected("This regularization build requires Vulkan");
#endif
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward_loss_only(
        const lfs::core::Tensor& opacity_raw,
        const Params& params) {
        try {
            if (!std::isfinite(params.weight))
                return std::unexpected("Regularization weight must be finite");
            core::GpuBackendScope backend(core::gpu_backend_of(opacity_raw).value_or(core::default_gpu_backend()));
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }
            if (opacity_raw.device() != lfs::core::Device::GPU) {
                return std::unexpected("opacity_raw must be on a GPU device");
            }

            if (opacity_raw.dtype() != core::DataType::Float32 || !opacity_raw.is_contiguous())
                return std::unexpected("Regularization parameters must be contiguous Float32 tensors");
            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
            }

            if (core::gpu_backend_of(opacity_raw) == core::GpuBackend::Vulkan) {
                const auto activated = opacity_raw.sigmoid();
                return (activated.mean() * params.weight).reshape({1});
            }
#if LFS_TENSOR_CUDA
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::GPU);

            lfs::training::kernels::launch_fused_opacity_regularization(
                opacity_raw.ptr<float>(),
                nullptr,
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            return loss_tensor;
#else
            return std::unexpected("This regularization build requires Vulkan");
#endif
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward_loss_only: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
