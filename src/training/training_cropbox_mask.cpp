/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_cropbox_mask.hpp"

#include "core/assert.hpp"
#include "core/camera.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lfs::training {

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_geom(
        const core::Scene& scene) {
        const auto* training_node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
        if (!training_node) {
            return std::nullopt;
        }

        const core::NodeId cropbox_id = scene.getCropBoxForSplat(training_node->id);
        if (cropbox_id == core::NULL_NODE) {
            return std::nullopt;
        }

        const auto* cropbox = scene.getCropBoxData(cropbox_id);
        if (!cropbox || !cropbox->enabled) {
            return std::nullopt;
        }

        const glm::mat4 world_to_cropbox =
            glm::inverse(scene.getWorldTransform(cropbox_id));
        return TrainingCropBoxGeometry{
            .min = cropbox->min,
            .max = cropbox->max,
            .world_to_cropbox = world_to_cropbox,
            .model_to_cropbox =
                world_to_cropbox * scene.getWorldTransform(training_node->id),
            .inverse = cropbox->inverse};
    }

    std::optional<TrainingCropBoxGeometry> resolve_training_cropbox_loss_geom(
        const core::Scene& scene,
        const float outside_weight) {
        LFS_ASSERT_MSG(
            std::isfinite(outside_weight) &&
                outside_weight >= 0.0f &&
                outside_weight <= 1.0f,
            "crop box loss weight must be finite and within [0, 1]");
        if (outside_weight == 1.0f) {
            return std::nullopt;
        }
        return resolve_training_cropbox_geom(scene);
    }

    core::Tensor compute_cropbox_loss_weight(
        const core::Camera& camera, const TrainingCropBoxGeometry& geometry, const float outside_weight) {
        using core::Tensor;
        const int width = camera.image_width(), height = camera.image_height();
        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        if (width <= 0 || height <= 0 || !std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.f || fy <= 0.f ||
            !std::isfinite(outside_weight) || outside_weight < 0.f || outside_weight > 1.f)
            throw std::invalid_argument("ROI loss requires valid image dimensions, intrinsics and weight in [0,1]");
        // Only camera metadata crosses to the host; all pixel rays and slab
        // intersections remain on the image's GPU backend.
        const auto rotation = camera.world_view_transform().cpu().to_vector();
        const auto position = camera.cam_position().cpu().to_vector();
        const glm::vec3 origin(geometry.world_to_cropbox * glm::vec4(position[0], position[1], position[2], 1.f));
        const glm::vec3 basis_x(geometry.world_to_cropbox * glm::vec4(rotation[0], rotation[1], rotation[2], 0.f));
        const glm::vec3 basis_y(geometry.world_to_cropbox * glm::vec4(rotation[4], rotation[5], rotation[6], 0.f));
        const glm::vec3 basis_z(geometry.world_to_cropbox * glm::vec4(rotation[8], rotation[9], rotation[10], 0.f));
        const auto x = (Tensor::arange(static_cast<float>(width)).unsqueeze(0) + (.5f - cx)) / fx;
        const auto y = (Tensor::arange(static_cast<float>(height)).unsqueeze(1) + (.5f - cy)) / fy;
        const core::TensorShape shape{static_cast<size_t>(height), static_cast<size_t>(width)};
        auto enter = Tensor::full(shape, -std::numeric_limits<float>::max(), core::Device::GPU);
        auto leave = enter.neg();
        auto hit = Tensor::ones(shape, core::Device::GPU, core::DataType::Bool);
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(origin[axis]) || !std::isfinite(basis_x[axis]) || !std::isfinite(basis_y[axis]) ||
                !std::isfinite(basis_z[axis]) || !std::isfinite(geometry.min[axis]) ||
                !std::isfinite(geometry.max[axis]) || geometry.min[axis] > geometry.max[axis])
                throw std::invalid_argument("ROI crop box must have finite ordered bounds and transform");
            const auto direction = x * basis_x[axis] + y * basis_y[axis] + basis_z[axis];
            const auto parallel = direction.abs().lt(1e-8f);
            const auto divisor = Tensor::where(parallel, Tensor::ones_like(direction), direction);
            const auto a = (divisor.reciprocal()) * (geometry.min[axis] - origin[axis]);
            const auto b = (divisor.reciprocal()) * (geometry.max[axis] - origin[axis]);
            const auto near = Tensor::where(a.lt(b), a, b);
            const auto far = Tensor::where(a.gt(b), a, b);
            enter = Tensor::where(parallel, enter, Tensor::where(enter.gt(near), enter, near));
            leave = Tensor::where(parallel, leave, Tensor::where(leave.lt(far), leave, far));
            if (origin[axis] < geometry.min[axis] || origin[axis] > geometry.max[axis])
                hit = hit.logical_and(parallel.logical_not());
        }
        hit = hit.logical_and(leave.ge(enter)).logical_and(leave.ge(0.f));
        if (geometry.inverse)
            hit = hit.logical_not();
        return hit.to(core::DataType::Float32) * (1.f - outside_weight) + outside_weight;
    }

    std::optional<core::Tensor> compute_cropbox_remove_mask(
        const core::Tensor& means,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const glm::mat4& points_to_cropbox,
        const bool inverse,
        const core::Tensor* const deleted_mask) {
        if (!means.is_valid()) {
            return std::nullopt;
        }
        LFS_ASSERT_MSG(means.dtype() == core::DataType::Float32, "crop box means must be Float32");
        LFS_ASSERT_MSG(means.ndim() == 2, "crop box means must be a 2D tensor");
        LFS_ASSERT_MSG(means.size(1) >= 3, "crop box means must have at least three columns");
        if (means.size(0) == 0) {
            return std::nullopt;
        }

        auto inside_mask =
            core::compute_cropbox_mask(means, crop_min, crop_max, points_to_cropbox);
        if (!inside_mask.is_valid() || inside_mask.numel() == 0) {
            return std::nullopt;
        }

        auto remove_mask = inverse ? inside_mask : inside_mask.logical_not();
        if (deleted_mask) {
            LFS_ASSERT_MSG(deleted_mask->is_valid(), "crop box deleted mask must be valid");
            LFS_ASSERT_MSG(
                deleted_mask->numel() == remove_mask.numel(),
                "crop box deleted mask must match the means row count");
            remove_mask = remove_mask.logical_and(deleted_mask->logical_not());
        }

        return remove_mask;
    }

    std::optional<core::Tensor> compute_training_cropbox_remove_mask(
        const core::Scene& scene,
        const core::SplatData& model) {
        const auto geometry = resolve_training_cropbox_geom(scene);
        if (!geometry) {
            return std::nullopt;
        }

        const auto& deleted = model.deleted();
        const core::Tensor* const deleted_mask =
            model.has_deleted_mask() && deleted.is_valid() ? &deleted : nullptr;

        return compute_cropbox_remove_mask(
            model.means(),
            geometry->min,
            geometry->max,
            geometry->model_to_cropbox,
            geometry->inverse,
            deleted_mask);
    }

} // namespace lfs::training
