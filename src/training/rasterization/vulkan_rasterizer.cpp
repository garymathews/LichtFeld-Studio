/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/rasterizer/vulkan/shader_paths.hpp"
#include "vulkan_rasterizer.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/rasterizer/vulkan/src/viewport_scratch_bucket.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace lfs::training {
    using namespace core;
    struct VulkanTrainingRasterizer::Impl {
        struct Region {
            uint32_t x, y, width, height;
        };
        struct NativeResources {
            std::unique_ptr<VulkanGSRenderer> renderer = std::make_unique<VulkanGSRenderer>();
            VulkanGSPipelineBuffers forward_buffers, gradient_buffers;
            void cleanup() noexcept {
                if (!renderer)
                    return;
                renderer->cancelCommandBatch();
                for (auto* buffers : {&forward_buffers, &gradient_buffers}) {
                    try {
                        // The backend has drained submissions or declared device
                        // loss. Both cases permit native resource destruction.
                        renderer->cleanupBuffers(*buffers, false);
                    } catch (const std::exception& error) {
                        LOG_ERROR("Vulkan training buffer cleanup failed: {}", error.what());
                    }
                }
                renderer.reset();
            }
        };
        std::shared_ptr<NativeResources> native = std::make_shared<NativeResources>();
        std::unique_ptr<VulkanGSRenderer>& renderer = native->renderer;
        VulkanGSPipelineBuffers& forward_buffers = native->forward_buffers;
        VulkanGSPipelineBuffers& gradient_buffers = native->gradient_buffers;
        VulkanGSRendererUniforms uniforms{};
        SplatData* model = nullptr;
        uint64_t context_id = 0;
        uint64_t model_generation = 0;
        std::array<const void*, 6> model_storage{};
        std::vector<Region> regions;
        std::array<Tensor, 6> parameters, raw_gradients;
        std::filesystem::path shader_directory;
        Tensor background, pixels, visibility, screen_share;
        std::unordered_set<VkBuffer> external_parents;
        bool pending_backward = false;
        VulkanGSRenderer::TileInstanceGate last_gate{};
        uint32_t max_tile_instances;
        std::shared_ptr<void> native_lifetime;

        explicit Impl(const std::filesystem::path& directory, uint32_t instance_budget)
            : max_tile_instances(instance_budget ? std::min(instance_budget, uint32_t(HIGS_DEPTH_WAVE_INSTANCES)) : HIGS_DEPTH_WAVE_INSTANCES) {
            shader_directory = directory;
            std::map<std::string, std::string> shaders;
            for (const auto* subdirectory : {"generated", "radix_sort"}) {
                for (const auto& entry : std::filesystem::directory_iterator(directory / subdirectory)) {
                    if (entry.path().extension() != ".spv")
                        continue;
                    const auto key = std::string(subdirectory) == "radix_sort" ? "radix_sort/" + entry.path().stem().string() : entry.path().stem().string();
                    shaders.emplace(key, entry.path().string());
                }
            }
            native_lifetime = register_vulkan_external_resource([&](const VulkanExternalDevice& device) {
                context_id = device.context_id;
                renderer->initializeExternal(shaders, static_cast<VkInstance>(device.handles.instance),
                                             static_cast<VkPhysicalDevice>(device.handles.physical_device), static_cast<VkDevice>(device.handles.device),
                                             static_cast<VkQueue>(device.handles.queue), device.handles.queue_family,
                                             static_cast<VmaAllocator>(device.allocator));
            }, [resources = native] { resources->cleanup(); });
        }
        void bind(Buffer<float>& buffer, const TensorVulkanBuffer& view) {
            buffer.deviceBuffer = {};
            auto& target = buffer.deviceBuffer;
            target.buffer = reinterpret_cast<VkBuffer>(view.buffer);
            target.offset = view.offset;
            target.allocSize = view.offset + view.bytes;
            target.capacity = target.size = view.bytes;
            if (external_parents.insert(target.buffer).second)
                renderer->trackExternalParent(target.buffer);
        }
        VulkanGSRendererUniforms region_uniforms(const Region& region) const {
            auto u = uniforms;
            u.image_width = region.width;
            u.image_height = region.height;
            u.grid_width = (region.width + TILE_WIDTH - 1) / TILE_WIDTH;
            u.grid_height = (region.height + TILE_HEIGHT - 1) / TILE_HEIGHT;
            u.render_origin_x = region.x;
            u.render_origin_y = region.y;
            return u;
        }
        VulkanGSRenderer::TileInstanceGate render(const Region& region) {
            auto u = region_uniforms(region);
            const auto bucket = lfs::rendering::vulkan::viewportScratchBucket(region.width, region.height);
            pixels = Tensor::empty(TensorShape{bucket.alloc_pixels * 4}, Device::GPU);
            visibility = Tensor::empty(TensorShape{u.num_splats}, Device::GPU, DataType::Int32);
            const auto pixel_view = tensor_vulkan_write_buffer(pixels).value();
            const auto visible_view = tensor_vulkan_write_buffer(visibility).value();
            std::array<TensorVulkanBuffer, 6> views;
            for (size_t i = 0; i < views.size(); ++i)
                views[i] = tensor_vulkan_buffer(parameters[i]).value();
            VulkanGSRenderer::TileInstanceGate gate{};
            with_idle_vulkan_device([&](const VulkanExternalDevice& device) {
                if (device.context_id != context_id)
                    throw std::runtime_error("Training Vulkan device changed");
                for (const auto parent : external_parents)
                    renderer->untrackExternalParent(parent);
                external_parents.clear();
                bind(forward_buffers.xyz_ws, views[0]);
                bind(forward_buffers.sh0, views[1]);
                bind(forward_buffers.shN, views[2]);
                bind(forward_buffers.rotations, views[3]);
                bind(forward_buffers.scaling_raw, views[4]);
                bind(forward_buffers.opacity_raw, views[5]);
                bind(forward_buffers.pixel_state, pixel_view);
                _VulkanBuffer visible;
                visible.buffer = reinterpret_cast<VkBuffer>(visible_view.buffer);
                visible.offset = visible_view.offset;
                visible.allocSize = visible_view.offset + visible_view.bytes;
                visible.size = visible.capacity = visible_view.bytes;
                if (external_parents.insert(visible.buffer).second)
                    renderer->trackExternalParent(visible.buffer);
                try {
                    renderer->beginCommandBatch();
                    gate = renderer->executeTrainingForward(u, forward_buffers, visible);
                    renderer->endCommandBatch();
                } catch (...) {
                    renderer->cancelCommandBatch();
                    throw;
                }
            });
            last_gate = gate;
            return gate;
        }
    };

    VulkanTrainingRasterizer::VulkanTrainingRasterizer(const std::filesystem::path& directory, uint32_t max_tile_instances)
        : impl_(std::make_unique<Impl>(rendering::vulkan::resolveVkSplatSpirvRoot(directory), max_tile_instances)) {}
    VulkanTrainingRasterizer::~VulkanTrainingRasterizer() = default;

    RenderOutput VulkanTrainingRasterizer::forward(const Camera& camera, SplatData& model,
                                                   std::array<float, 3> background, bool mip_filter) {
        GpuBackendScope backend(GpuBackend::Vulkan);
        if (!impl_->renderer || impl_->context_id != vulkan_backend_context_id())
            impl_ = std::make_unique<Impl>(impl_->shader_directory, impl_->max_tile_instances);
        auto& impl = *impl_;
        impl.pending_backward = false;
        if (gpu_backend_of(model.means()) != GpuBackend::Vulkan || model.size() <= 0 ||
            camera.image_width() <= 0 || camera.image_height() <= 0 ||
            (camera.has_distortion() && !camera.is_undistort_precomputed()))
            throw std::invalid_argument("Vulkan training requires nonempty Vulkan splats and an undistorted pinhole image");
        for (float channel : background)
            if (!std::isfinite(channel))
                throw std::invalid_argument("Training background must be finite");
        impl.model = &model;
        impl.model_generation = model.param_layout_generation();
        impl.model_storage = {model.means().data_ptr(), model.sh0().data_ptr(), (model.shN().is_valid() ? model.shN().data_ptr() : nullptr),
                              model.rotation_raw().data_ptr(), model.scaling_raw().data_ptr(), model.opacity_raw().data_ptr()};
        const size_t n = model.size(), width = camera.image_width(), height = camera.image_height();
        impl.uniforms = {};
        auto& u = impl.uniforms;
        u.image_width = u.camera_width = width;
        u.image_height = u.camera_height = height;
        u.num_splats = u.model_num_splats = n;
        u.active_sh = model.get_active_sh_degree();
        u.shN_layout_slots = sh_float4_slots_for_rest(model.max_sh_coeffs_rest());
        u.mip_filter = mip_filter;
        std::tie(u.fx, u.fy, u.cx, u.cy) = camera.get_intrinsics();
        const auto world_view = camera.world_view_transform().cpu().to_vector();
        for (size_t row = 0; row < 4; ++row)
            for (size_t col = 0; col < 4; ++col)
                u.world_view_transform[row * 4 + col] = world_view[col * 4 + row];
        Tensor sh = model.max_sh_coeffs_rest() ? (model.shN().dtype() == DataType::Float32 ? model.shN() : reorder_sh_to_swizzled(model.shN_canonical(), n, model.max_sh_coeffs_rest(), model.max_sh_coeffs_rest()))
                                               : Tensor::zeros({1}, Device::GPU);
        // Tensor assignment into an existing view copies into its old storage.
        // Rebinding must release those handles, including after model replacement.
        impl.parameters = {};
        impl.parameters = {model.means(), model.sh0(), std::move(sh), model.rotation_raw(), model.scaling_raw(), model.opacity_raw()};
        for (const auto& parameter : impl.parameters)
            if (parameter.dtype() != DataType::Float32 || !parameter.is_contiguous() || gpu_backend_of(parameter) != GpuBackend::Vulkan)
                throw std::invalid_argument("Vulkan rasterization parameters must be contiguous Float32 on Vulkan");
        impl.background = Tensor::from_vector(std::vector<float>(background.begin(), background.end()), {1, 1, 3}, Device::GPU);
        RenderOutput output;
        output.width = width;
        output.height = height;
        output.image = Tensor::zeros(TensorShape{3, height, width}, Device::GPU);
        output.alpha = Tensor::zeros(TensorShape{1, height, width}, Device::GPU);
        Tensor any_visible = Tensor::zeros_bool({n}, Device::GPU);
        std::vector<Impl::Region> pending{{0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height)}};
        impl.regions.clear();
        while (!pending.empty()) {
            const auto region = pending.back();
            pending.pop_back();
            const auto gate = impl.render(region);
            if (gate.count_overflow || gate.raw_count > impl.max_tile_instances) {
                if (region.width == 1 && region.height == 1)
                    throw std::runtime_error("One training pixel exceeds the raster instance budget");
                if (region.width >= region.height && region.width > 1) {
                    const uint32_t half = region.width / 2;
                    pending.push_back({region.x, region.y, half, region.height});
                    pending.push_back({region.x + half, region.y, region.width - half, region.height});
                } else {
                    const uint32_t half = region.height / 2;
                    pending.push_back({region.x, region.y, region.width, half});
                    pending.push_back({region.x, region.y + half, region.width, region.height - half});
                }
                continue;
            }
            impl.regions.push_back(region);
            auto rgba = impl.pixels.slice(0, 0, size_t{region.width} * region.height * 4).reshape(TensorShape{region.height, region.width, 4});
            auto rgb = rgba.slice(2, 0, 3) + rgba.slice(2, 3, 4) * impl.background;
            output.image.slice(1, region.y, region.y + region.height).slice(2, region.x, region.x + region.width).copy_from(rgb.permute({2, 0, 1}));
            output.alpha.slice(1, region.y, region.y + region.height).slice(2, region.x, region.x + region.width).copy_from((rgba.slice(2, 3, 4).neg() + 1.f).permute({2, 0, 1}));
            any_visible = any_visible.logical_or(impl.visibility.gt(0));
        }
        // Match the existing screen-share cap's world-space radius definition.
        impl.screen_share = {};
        if (model._max_screen_share.is_valid() && model._max_screen_share.numel() == n) {
            auto radius = model.scaling_raw().max(1).exp() * (model.opacity_raw().reshape(TensorShape{n}).sigmoid() * 255.f).clamp_min(1.f).log().mul(2.f).sqrt();
            auto distance = (model.means() - camera.cam_position().reshape({1, 3})).square().sum(1).sqrt();
            auto denominator = distance.maximum(radius) + (distance.square() - radius.square()).clamp_min(0.f).sqrt();
            impl.screen_share = Tensor::where(any_visible, (radius / denominator.clamp_min(1e-12f)).clamp(0.f, 1.f), Tensor::zeros_like(radius));
        }
        impl.pending_backward = true;
        return output;
    }

    void VulkanTrainingRasterizer::backward(const Tensor& image_gradient, AdamOptimizer& optimizer, const Tensor& alpha_gradient, const Tensor& error_map) {
        GpuBackendScope backend(GpuBackend::Vulkan);
        auto& impl = *impl_;
        const size_t width = impl.uniforms.camera_width, height = impl.uniforms.camera_height, n = impl.uniforms.num_splats;
        if (!impl.pending_backward || !impl.model || static_cast<size_t>(impl.model->size()) != n)
            throw std::logic_error("Vulkan backward requires an unchanged model and a preceding forward");
        const auto& model = *impl.model;
        const std::array<const void*, 6> storage{model.means().data_ptr(), model.sh0().data_ptr(), (model.shN().is_valid() ? model.shN().data_ptr() : nullptr),
                                               model.rotation_raw().data_ptr(), model.scaling_raw().data_ptr(), model.opacity_raw().data_ptr()};
        if (impl.model_generation != model.param_layout_generation() || storage != impl.model_storage ||
            model.get_active_sh_degree() != impl.uniforms.active_sh)
            throw std::logic_error("Vulkan model storage or SH state changed after forward");
        if (image_gradient.shape() != TensorShape{3, height, width} || image_gradient.dtype() != DataType::Float32 || gpu_backend_of(image_gradient) != GpuBackend::Vulkan)
            throw std::invalid_argument("Vulkan image gradient must be Float32 CHW on Vulkan");
        if (alpha_gradient.is_valid() && (alpha_gradient.shape() != TensorShape{1, height, width} || alpha_gradient.dtype() != DataType::Float32 || gpu_backend_of(alpha_gradient) != GpuBackend::Vulkan))
            throw std::invalid_argument("Vulkan alpha gradient must be Float32 1HW on Vulkan");
        if (error_map.is_valid() && (error_map.shape() != TensorShape{height, width} || error_map.dtype() != DataType::Float32 || gpu_backend_of(error_map) != GpuBackend::Vulkan ||
                                     impl.model->_densification_info.shape() != TensorShape{2, n}))
            throw std::invalid_argument("Vulkan densification needs a Float32 HW error map and 2N model statistics");
        impl.pending_backward = false;
        // The last forward region is still resident in the renderer.
        for (auto it = impl.regions.rbegin(); it != impl.regions.rend(); ++it) {
            const auto& region = *it;
            auto gate = impl.last_gate;
            if (it != impl.regions.rbegin()) {
                gate = impl.render(region);
                if (gate.count_overflow || gate.raw_count > impl.max_tile_instances)
                    throw std::runtime_error("Training region changed between forward and backward");
            }
            auto rgb = image_gradient.slice(1, region.y, region.y + region.height).slice(2, region.x, region.x + region.width).permute({1, 2, 0});
            auto transmittance = (rgb * impl.background).sum(2, true);
            if (alpha_gradient.is_valid())
                transmittance = transmittance - alpha_gradient.slice(1, region.y, region.y + region.height).slice(2, region.x, region.x + region.width).permute({1, 2, 0});
            auto pixel_gradient = Tensor::cat({rgb, transmittance}, 2).contiguous().reshape({-1});
            const auto pixel_view = tensor_vulkan_buffer(pixel_gradient).value();
            const auto visible_view = tensor_vulkan_buffer(impl.visibility).value();
            Tensor region_errors;
            std::optional<TensorVulkanBuffer> error_view, density_view;
            if (error_map.is_valid()) {
                region_errors = error_map.slice(0, region.y, region.y + region.height).slice(1, region.x, region.x + region.width).contiguous();
                error_view = tensor_vulkan_buffer(region_errors);
                density_view = tensor_vulkan_write_buffer(impl.model->_densification_info);
            }
            auto& raw_gradients = impl.raw_gradients;
            std::array<TensorVulkanBuffer, 6> views;
            for (size_t i = 0; i < views.size(); ++i) {
                // Projection writes live SH coefficients; padding and inactive
                // lanes must contribute zero when regions are accumulated.
                if (!raw_gradients[i].is_valid() || raw_gradients[i].shape() != impl.parameters[i].shape())
                    raw_gradients[i] = Tensor::zeros(impl.parameters[i].shape(), Device::GPU);
                else
                    raw_gradients[i].zero_();
                views[i] = tensor_vulkan_write_buffer(raw_gradients[i]).value();
            }
            with_idle_vulkan_device([&](const VulkanExternalDevice& device) {
                if (device.context_id != impl.context_id || !impl.renderer)
                    throw std::runtime_error("Training renderer outlived its Vulkan backend");
                auto& b = impl.gradient_buffers;
                impl.bind(b.xyz_ws, views[0]);
                impl.bind(b.sh0, views[1]);
                impl.bind(b.shN, views[2]);
                impl.bind(b.rotations, views[3]);
                impl.bind(b.scaling_raw, views[4]);
                impl.bind(b.opacity_raw, views[5]);
                impl.bind(b.pixel_state, pixel_view);
                _VulkanBuffer visible;
                visible.buffer = reinterpret_cast<VkBuffer>(visible_view.buffer);
                visible.offset = visible_view.offset;
                visible.allocSize = visible_view.offset + visible_view.bytes;
                visible.size = visible.capacity = visible_view.bytes;
                Buffer<float> error_buffer, density_buffer;
                if (error_view) {
                    impl.bind(error_buffer, *error_view);
                    impl.bind(density_buffer, *density_view);
                }
                auto u = impl.region_uniforms(region);
                try {
                    impl.renderer->beginCommandBatch();
                    impl.renderer->executeRasterizationBackward(u, impl.forward_buffers, gate.raw_count, b, error_buffer.deviceBuffer, density_buffer.deviceBuffer);
                    impl.renderer->executeProjectionBackward(u, impl.forward_buffers, visible, b);
                    impl.renderer->endCommandBatch();
                } catch (...) {
                    impl.renderer->cancelCommandBatch();
                    throw;
                }
            });
            const ParamType types[] = {ParamType::Means, ParamType::Sh0, ParamType::ShN, ParamType::Rotation, ParamType::Scaling, ParamType::Opacity};
            for (size_t i = 0; i < raw_gradients.size(); ++i) {
                if (types[i] == ParamType::ShN && impl.model->max_sh_coeffs_rest() == 0)
                    continue;
                auto& gradient = optimizer.get_grad(types[i]);
                gradient.add_(raw_gradients[i].reshape(gradient.shape()));
            }
        }
        // Forward-only evaluation must not influence subsequent scale clipping.
        if (impl.screen_share.is_valid())
            impl.model->_max_screen_share.copy_from(impl.model->_max_screen_share.maximum(impl.screen_share));
    }
} // namespace lfs::training
