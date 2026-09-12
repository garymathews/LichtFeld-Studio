/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "selection_ops.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace lfs::rendering {
    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode,
        Tensor* const group_counts_scratch) {
        apply_selection_group_tensor_mask_program(cumulative_selection, existing_mask, output_mask, group_id, locked_groups, add_mode, transform_indices, valid_nodes, replace_mode, group_counts_scratch);
    }
    void apply_selection_group_indexed_tensor_mask(
        const Tensor& visible_selection,
        const Tensor& visible_indices,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode) {
        apply_selection_group_indexed_tensor_mask_program(visible_selection, visible_indices, existing_mask, output_mask, group_id, locked_groups, add_mode, transform_indices, valid_nodes, replace_mode);
    }
    void merge_selection_mask_or(Tensor& accumulated_mask, const Tensor& delta_mask) {
        merge_selection_mask_or_program(accumulated_mask, delta_mask);
    }
    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes) {
        filter_selection_by_node_mask_program(selection, transform_indices, valid_nodes);
    }
    void filter_selection_by_crop(
        Tensor& selection,
        const Tensor& means,
        const Tensor* const crop_box_transform,
        const Tensor* const crop_box_min,
        const Tensor* const crop_box_max,
        const bool crop_inverse,
        const Tensor* const ellipsoid_transform,
        const Tensor* const ellipsoid_radii,
        const bool ellipsoid_inverse,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices) {
        filter_selection_by_crop_program(selection, means, crop_box_transform, crop_box_min, crop_box_max, crop_inverse, ellipsoid_transform, ellipsoid_radii, ellipsoid_inverse, model_transforms, transform_indices);
    }
    void count_selection_groups_async(const Tensor& mask, Tensor& scratch) {
        count_selection_groups_tensor_program(mask, scratch);
    }
    void enqueue_selection_group_count_read(const Tensor& scratch, int* host_counts, cudaEvent_t event) {
        if (!host_counts || event || !scratch.is_valid() || scratch.numel() != 257 || scratch.dtype() != core::DataType::Int32)
            throw std::invalid_argument("Invalid Vulkan selection count destination");
        const auto host = scratch.cpu().contiguous();
        std::memcpy(host_counts, host.ptr<int>(), 257 * sizeof(int));
    }
    SelectionGroupDeltaResult read_selection_group_delta_result(const Tensor& scratch) {
        int values[257]{};
        enqueue_selection_group_count_read(scratch, values, nullptr);
        SelectionGroupDeltaResult result;
        std::copy_n(values, 256, result.group_deltas.begin());
        result.changed_count = static_cast<size_t>(std::max(values[256], 0));
        return result;
    }
    SelectionGroupCountResult read_selection_group_count_result(const Tensor& scratch) {
        const auto delta = read_selection_group_delta_result(scratch);
        SelectionGroupCountResult result;
        for (size_t i = 0; i < 256; ++i) result.group_counts[i] = static_cast<size_t>(std::max(delta.group_deltas[i], 0));
        result.changed_count = delta.changed_count;
        return result;
    }
    std::array<size_t, 256> read_selection_group_counts(const Tensor& scratch) {
        return read_selection_group_count_result(scratch).group_counts;
    }
    std::array<size_t, 256> count_selection_groups(const Tensor& mask, Tensor& scratch) {
        count_selection_groups_async(mask, scratch);
        return read_selection_group_counts(scratch);
    }
}
namespace lfs::rendering {
    void brush_select_tensor(const Tensor& positions, float x, float y, float radius, Tensor& output) {
        if (!positions.is_valid() || !positions.numel()) return;
        const auto px = positions.slice(1, 0, 1).squeeze(1);
        const auto py = positions.slice(1, 1, 2).squeeze(1);
        const auto hit = ((px - x).square() + (py - y).square() <= radius * radius)
            .logical_and(px >= kInvalidScreenPositionThreshold).logical_and(py >= kInvalidScreenPositionThreshold);
        output.copy_(output.logical_or(hit));
    }
    void polygon_select_tensor(const Tensor& positions, const Tensor& vertices, Tensor& output) {
        if (!positions.is_valid() || !positions.numel() || !vertices.is_valid() || vertices.shape()[0] < 3) return;
        const auto polygon = vertices.cpu().to(core::DataType::Float32).to_vector();
        const auto x = positions.slice(1, 0, 1).squeeze(1);
        const auto y = positions.slice(1, 1, 2).squeeze(1);
        auto inside = Tensor::zeros_like(output);
        const size_t n = polygon.size() / 2;
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            const float xi = polygon[2*i], yi = polygon[2*i+1], xj = polygon[2*j], yj = polygon[2*j+1];
            if (yi == yj) continue;
            const auto crossing = (y < yi).logical_xor(y < yj)
                .logical_and(x < (y - yi) * ((xj - xi) / (yj - yi)) + xi);
            inside = inside.logical_xor(crossing);
        }
        inside = inside.logical_and(x >= kInvalidScreenPositionThreshold).logical_and(y >= kInvalidScreenPositionThreshold);
        output.copy_(output.logical_or(inside));
    }
}
