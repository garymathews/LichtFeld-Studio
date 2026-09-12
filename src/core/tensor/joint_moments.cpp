/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor/internal/joint_moments.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <stdexcept>

#if LFS_TENSOR_VULKAN
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_pipelines.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <array>
#include <climits>
#endif

namespace lfs::core::internal {
#if LFS_TENSOR_VULKAN
    namespace {
        bool supported(const Tensor& packed, size_t block_cells, int bits) {
            return gpu_backend_of(packed) == GpuBackend::Vulkan && packed.numel() <= INT_MAX &&
                   block_cells <= INT_MAX && storage_ref(packed).byte_offset % (bits / 4) == 0;
        }

        void dispatch(const Tensor& first, const Tensor& second, const Tensor& packed, const Tensor& bounds,
                      int bits, size_t block_cells, float eps, const Tensor* valid, bool encode) {
            struct Push {
                uint64_t first, second, packed, bounds, valid;
                uint32_t cells, block_cells, bits;
                float eps;
            };
            static_assert(sizeof(Push) == 56);
            const auto a = storage_ref(first), b = storage_ref(second);
            const auto codes = storage_ref(packed), limits = storage_ref(bounds);
            const auto mask = valid ? storage_ref(*valid) : StorageRef{};
            const Push push{vk::address(a), vk::address(b), vk::address(codes), vk::address(limits),
                            valid ? vk::address(mask) : 0, static_cast<uint32_t>(packed.numel() / (bits / 4)),
                            static_cast<uint32_t>(block_cells), static_cast<uint32_t>(bits), eps};
            const auto context = acquire_vulkan_context();
            const std::array constants{encode ? 1u : 0u};
            const auto& pipeline = context->pipelines().specialized("joint_moments", sizeof(Push), constants);
            const std::array reads{encode ? a : codes, encode ? b : limits, mask};
            const std::array writes{encode ? codes : a, encode ? limits : b};
            const auto work = encode ? bounds.size(0) * 256 : push.cells;
            const auto groups = vk::dispatch_groups(*context, work);
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, groups, 1, 1);
            });
        }
    } // namespace
#endif

    void vulkan_adam_update(Tensor& first, Tensor& second, Tensor& delta,
                            const Tensor& gradient, const Tensor& enabled,
                            const Tensor& lr_scale, const AdamUpdateConfig config) {
#if LFS_TENSOR_VULKAN
        LFS_ASSERT_MSG(first.ndim() == 2 && first.numel() && first.shape() == second.shape() && first.shape() == gradient.shape(),
                       "Adam update requires matching nonempty row tensors");
        for (const Tensor* input : std::array<const Tensor*, 5>{&first, &second, &gradient, &enabled, &lr_scale}) {
            if (!input->is_valid()) continue;
            const auto dtype = input == &enabled ? DataType::Bool : DataType::Float32;
            LFS_ASSERT_MSG(input->dtype() == dtype && gpu_backend_of(*input) == GpuBackend::Vulkan &&
                               (input->shape() == first.shape() || input->shape() == TensorShape{first.shape()[0], 1}),
                           "Adam update requires Vulkan Float32 rows and a matching Bool mask");
        }
        const Tensor a = first.contiguous(), b = second.contiguous(), g = gradient.contiguous();
        const Tensor mask = enabled.is_valid() ? enabled.contiguous() : Tensor{};
        const Tensor scale = lr_scale.is_valid() ? lr_scale.contiguous() : Tensor{};
        Tensor next_first = allocate_like(a, a.shape(), DataType::Float32);
        Tensor next_second = allocate_like(a, a.shape(), DataType::Float32);
        delta = allocate_like(a, a.shape(), DataType::Float32);
        const std::array reads{storage_ref(a), storage_ref(b), storage_ref(g),
                               mask.is_valid() ? storage_ref(mask) : StorageRef{},
                               scale.is_valid() ? storage_ref(scale) : StorageRef{}};
        const std::array writes{storage_ref(next_first), storage_ref(next_second), storage_ref(delta)};
        struct Push {
            uint64_t first, second, gradient, enabled, lr_scale, next_first, next_second, delta;
            uint32_t cells, mask_stride, scale_stride;
            AdamUpdateConfig config;
        };
        static_assert(sizeof(Push) == 96);
        const Push push{vk::address(reads[0]), vk::address(reads[1]), vk::address(reads[2]),
                        mask.is_valid() ? vk::address(reads[3]) : 0,
                        scale.is_valid() ? vk::address(reads[4]) : 0,
                        vk::address(writes[0]), vk::address(writes[1]), vk::address(writes[2]),
                        vk::checked_u32(a.numel(), "Adam update exceeds uint32"),
                        mask.is_valid() ? uint32_t(a.numel() / mask.numel()) : 1u,
                        scale.is_valid() ? uint32_t(a.numel() / scale.numel()) : 1u, config};
        const auto context = acquire_vulkan_context();
        const auto& pipeline = context->pipelines().specialized("adam", sizeof(Push), {});
        context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, vk::dispatch_groups(*context, a.numel()), 1, 1);
        });
        first = std::move(next_first);
        second = std::move(next_second);
#else
        throw std::runtime_error("Vulkan Adam requires the Vulkan backend");
#endif
    }

    bool try_decode_joint_moments(const Tensor& packed, const Tensor& bounds, int bits, size_t block_cells,
                                  float eps, Tensor& first, Tensor& second) {
#if LFS_TENSOR_VULKAN
        if (supported(packed, block_cells, bits)) {
            first = allocate_like(packed, TensorShape{packed.numel() / (bits / 4)}, DataType::Float32);
            second = allocate_like(packed, first.shape(), DataType::Float32);
            dispatch(first, second, packed, bounds, bits, block_cells, eps, nullptr, false);
            return true;
        }
#endif
        return false;
    }

    bool try_encode_joint_moments(const Tensor& first, const Tensor& second, Tensor& packed, Tensor& bounds,
                                  int bits, size_t block_cells, float eps, const Tensor* valid) {
#if LFS_TENSOR_VULKAN
        if (supported(packed, block_cells, bits)) {
            // The portable codec snapshots inputs before overwriting aliased storage.
            for (const Tensor* source : {&first, &second, valid}) {
                if (source && (storage_ref(*source).meta == storage_ref(packed).meta ||
                               storage_ref(*source).meta == storage_ref(bounds).meta))
                    return false;
            }
            const Tensor a = first.contiguous(), b = second.contiguous();
            const Tensor mask = valid ? valid->contiguous() : Tensor{};
            preserve_lazy_snapshots_before_write(packed);
            preserve_lazy_snapshots_before_write(bounds);
            dispatch(a, b, packed, bounds, bits, block_cells, eps, valid ? &mask : nullptr, true);
            return true;
        }
#endif
        return false;
    }
} // namespace lfs::core::internal
