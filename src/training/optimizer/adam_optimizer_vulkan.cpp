/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "adam_optimizer.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/tensor/internal/joint_moments.hpp"
#include "core/tensor/internal/tensor_index_validation.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/sh_value_storage.hpp"
#if LFS_TENSOR_CUDA
#include "core/cuda_error.hpp"
#endif
#include "core/logger.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace lfs::training {
    using core::DataType;
    using core::Device;
    using core::Tensor;
    using core::TensorShape;

    void AdamOptimizer::relocate_params_at_indices(ParamType type, const Tensor& indices) {
        if (indices.dtype() != DataType::Int64 && indices.dtype() != DataType::Int32)
            throw std::invalid_argument("Adam relocation indices must be integers");
#if LFS_TENSOR_CUDA
        const auto gpu_indices = indices.to(DataType::Int64).gpu().contiguous();
        if (core::gpu_backend_of(gpu_indices) != core::GpuBackend::CUDA)
            throw std::runtime_error("This CUDA optimizer requires CUDA relocation indices");
        LFS_CUDA_CHECK(cudaStreamSynchronize(gpu_indices.stream()));
        relocate_params_at_indices_gpu(type, gpu_indices.ptr<int64_t>(), gpu_indices.numel());
#else
        if (indices.numel() == 0 || !states_.contains(param_name(type)))
            return;
        auto& state = states_.at(param_name(type));
        if (!state.exp_avg.is_valid() || !state.exp_avg.numel())
            return;
        // Relocation rebuilds the packed state from scratch, so bring it up to date and
        // let the next update re-derive the fp32 moments.
        flush_moments_to_packed();
        state.moments_first = {};
        state.moments_second = {};
        state.packed_current = true;
        const size_t n = splat_data_.size();
        core::internal::assert_index_tensor(indices, n, "Adam reset", true);
        const auto selected = (indices.device() == Device::GPU ? indices : indices.to(Device::GPU)).to(DataType::Int32);
        const auto divisor = Tensor::full({1}, 256.f, Device::GPU, DataType::Int32);
        const auto blocks = ((selected - selected.mod(divisor)) / divisor).to(DataType::Int32);
        auto touched = Tensor::zeros({(n + 255) / 256}, Device::GPU, DataType::Int32);
        touched.index_fill_(0, blocks, 1);
        auto selected_rows = Tensor::zeros_bool({n}, Device::GPU);
        selected_rows.index_fill_(0, selected, 1.f);
        auto prepared = state;
        prepared.exp_avg = {};
        prepared.joint_bounds = {};
        prepared.exp_avg = Tensor::zeros_direct(state.exp_avg.shape(), std::max(state.exp_avg.capacity(), state.exp_avg.shape()[0]), Device::GPU, DataType::UInt8);
        prepared.exp_avg.copy_from(state.exp_avg);
        prepared.joint_bounds = state.joint_bounds.clone();
        if (state.joint_bounds.capacity() > prepared.joint_bounds.shape()[0])
            prepared.joint_bounds.reserve(state.joint_bounds.capacity());
        const auto slots = type == ParamType::ShN ? core::sh_float4_slots_for_rest(splat_data_.max_sh_coeffs_rest()) : 0;
        // Cap each contiguous batch; drain sparse batches after at least 4096 queued rows.
        constexpr size_t max_batch_rows = 131072;
        constexpr size_t sync_interval_rows = 4096;
        size_t pending_rows = 0;
        const auto touched_blocks = touched.nonzero().reshape({-1}).cpu().to_vector_int64();
        for (size_t i = 0; i < touched_blocks.size();) {
            const size_t start = size_t(touched_blocks[i++]) * 256;
            size_t end = start + 256;
            // Batch adjacent blocks without changing untouched quantization bounds.
            while (i < touched_blocks.size() && size_t(touched_blocks[i]) * 256 == end && end - start < max_batch_rows) {
                end += 256;
                ++i;
            }
            const size_t count = std::min(end, n) - start;
            auto rows = read_moment_rows(type, n, start, count);
            const auto mask = selected_rows.slice(0, start, start + count).unsqueeze(1).broadcast_to(rows.first.shape());
            rows.first.masked_fill_(mask, 0.f);
            rows.second.masked_fill_(mask, 0.f);
            write_moment_range(prepared, slots, rows, start);
            if ((pending_rows += count) >= sync_interval_rows) {
                core::with_idle_vulkan_device([](const auto&) {});
                pending_rows = 0;
            }
        }
        core::with_idle_vulkan_device([](const auto&) {});
        state.exp_avg = {};
        state.joint_bounds = {};
        state.exp_avg = std::move(prepared.exp_avg);
        state.joint_bounds = std::move(prepared.joint_bounds);
#endif
    }

#if !LFS_TENSOR_CUDA
    std::unique_ptr<AdamOptimizer> AdamOptimizer::clone_for_model(core::SplatData& model) const {
        auto copy = std::make_unique<AdamOptimizer>(model, config_);
        const auto clone = [](const Tensor& source) {
            if (!source.is_valid())
                return Tensor{};
            const size_t rows = source.ndim() ? source.shape()[0] : 1;
            auto result = Tensor::zeros_direct(source.shape(), std::max(source.capacity(), rows), source.device(), source.dtype());
            result.copy_from(source);
            return result;
        };
        copy->states_ = states_;
        for (auto& [name, state] : copy->states_) {
            const auto& source = states_.at(name);
            state.grad = {};
            state.grad = clone(source.grad);
            state.exp_avg = {};
            state.exp_avg = clone(source.exp_avg);
            state.joint_bounds = {};
            state.joint_bounds = clone(source.joint_bounds);
        }
        copy->frozen_mask_ = clone(frozen_mask_);
        copy->crop_damping_mask_ = clone(crop_damping_mask_);
        copy->mean_step_far_mask_storage_ = clone(mean_step_far_mask_storage_);
        copy->frozen_lr_scale_ = frozen_lr_scale_;
        copy->cropbox_lr_scale_ = cropbox_lr_scale_;
        copy->per_splat_mean_step_ = per_splat_mean_step_;
        copy->mean_step_median_extent_ = mean_step_median_extent_;
        copy->mean_step_r_min_ = mean_step_r_min_;
        copy->mean_step_r_max_ = mean_step_r_max_;
        copy->mean_step_far_mask_n_ = mean_step_far_mask_n_;
        copy->mean_step_far_mask_ = copy->mean_step_far_mask_storage_.is_valid() ? copy->mean_step_far_mask_storage_.ptr<bool>() : nullptr;
        copy->screen_share_limit_ = screen_share_limit_;
        copy->screen_share_penalty_ = screen_share_penalty_;
        copy->refresh_screen_share_buffer();
        copy->last_step_zeroed_gradients_ = last_step_zeroed_gradients_;
        return copy;
    }

    void AdamOptimizer::adopt_training_update(AdamOptimizer& prepared) noexcept {
        adopt_checkpoint_state(prepared);
        frozen_mask_ = {};
        frozen_mask_ = std::move(prepared.frozen_mask_);
        crop_damping_mask_ = {};
        crop_damping_mask_ = std::move(prepared.crop_damping_mask_);
        mean_step_far_mask_storage_ = {};
        mean_step_far_mask_storage_ = std::move(prepared.mean_step_far_mask_storage_);
        cropbox_lr_scale_ = prepared.cropbox_lr_scale_;
        per_splat_mean_step_ = prepared.per_splat_mean_step_;
        mean_step_median_extent_ = prepared.mean_step_median_extent_;
        mean_step_r_min_ = prepared.mean_step_r_min_;
        mean_step_r_max_ = prepared.mean_step_r_max_;
        mean_step_far_mask_ = prepared.mean_step_far_mask_;
        mean_step_far_mask_n_ = prepared.mean_step_far_mask_n_;
        screen_share_max_ = prepared.screen_share_max_;
        screen_share_n_ = prepared.screen_share_n_;
        screen_share_limit_ = prepared.screen_share_limit_;
        screen_share_penalty_ = prepared.screen_share_penalty_;
        last_step_zeroed_gradients_ = prepared.last_step_zeroed_gradients_;
    }

    namespace {
        Tensor rows_from_layout(const Tensor& flat, size_t n, size_t slots) {
            if (!slots)
                return flat.reshape(TensorShape{n, flat.numel() / n});
            const size_t blocks = (n + 31) / 32;
            return flat.reshape(TensorShape{blocks, slots, 32, 4}).permute({0, 2, 1, 3}).contiguous().reshape(TensorShape{blocks * 32, slots * 4}).slice(0, 0, n);
        }
        Tensor layout_from_rows(const Tensor& rows, size_t slots) {
            if (!slots)
                return rows.contiguous().reshape({-1});
            const size_t n = rows.shape()[0], blocks = (n + 31) / 32;
            Tensor padded = rows;
            if (n % 32) {
                padded = Tensor::zeros(TensorShape{blocks * 32, slots * 4}, rows.device(), rows.dtype());
                padded.slice(0, 0, n).copy_from(rows);
            }
            return padded.reshape(TensorShape{blocks, 32, slots, 4}).permute({0, 2, 1, 3}).contiguous().reshape({-1});
        }
        Tensor row_cell_indices(const Tensor& rows, size_t attributes, size_t slots) {
            const auto integer = [&](size_t value) { return Tensor::full({1}, float(value), rows.device(), DataType::Int32); };
            const auto indices = rows.to(DataType::Int32).reshape(TensorShape{rows.numel(), 1});
            const auto columns = Tensor::arange(float(attributes)).to(DataType::Int32).reshape(TensorShape{1, attributes});
            if (!slots)
                return (indices * integer(attributes) + columns).reshape({-1});
            const auto lane = indices.mod(integer(32)), component = columns.mod(integer(4));
            return ((indices - lane) * integer(attributes) + lane * integer(4) + (columns - component) * integer(32) + component).reshape({-1});
        }
        size_t slots_for(ParamType type, const core::SplatData& splat) {
            return type == ParamType::ShN ? core::sh_float4_slots_for_rest(static_cast<uint32_t>(splat.max_sh_coeffs_rest())) : 0;
        }
    } // namespace

    joint_adam::TensorMoments AdamOptimizer::read_moment_rows(ParamType type, size_t n, size_t offset, size_t count) {
        auto& state = states_.at(param_name(type));
        const auto slots = slots_for(type, splat_data_);
        const size_t attributes = slots ? slots * 4 : state.exp_avg.numel() / (n * joint_adam::bytes_per_cell(state.joint_bits));
        if (!count)
            count = n - offset;
        if (offset % 256 || offset > n || count > n - offset || (offset + count < n && count % 256))
            throw std::invalid_argument("Adam range must cover complete quantization blocks");
        const size_t cells = (slots ? ((count + 31) / 32) * 32 : count) * attributes;
        const size_t bpc = joint_adam::bytes_per_cell(state.joint_bits);
        const auto packed = state.exp_avg.reshape({-1}).slice(0, offset * attributes * bpc, offset * attributes * bpc + cells * bpc);
        const auto bounds = state.joint_bounds.slice(0, offset / 256, (offset + count + 255) / 256);
        auto flat = joint_adam::decode_tensor(packed, bounds, state.joint_bits, 256 * attributes);
        return {rows_from_layout(flat.first, count, slots), rows_from_layout(flat.second, count, slots)};
    }

    void AdamOptimizer::write_moment_range(AdamParamState& state, size_t slots, const joint_adam::TensorMoments& rows, size_t offset) {
        const size_t count = rows.first.shape()[0], attributes = rows.first.shape()[1];
        auto first = layout_from_rows(rows.first, slots), second = layout_from_rows(rows.second, slots);
        const size_t bpc = joint_adam::bytes_per_cell(state.joint_bits);
        auto packed = state.exp_avg.reshape({-1}).slice(0, offset * attributes * bpc, offset * attributes * bpc + first.numel() * bpc);
        auto bounds = state.joint_bounds.slice(0, offset / 256, (offset + count + 255) / 256);
        Tensor valid;
        if (slots && count % 32)
            valid = layout_from_rows(Tensor::full(rows.first.shape(), 1.0f, Device::GPU, DataType::Bool), slots);
        joint_adam::encode_tensor({first, second}, packed, bounds, state.joint_bits, 256 * attributes, valid.is_valid() ? &valid : nullptr);
    }

    void AdamOptimizer::ensure_fp32_moments(ParamType type) {
        auto* state = get_state_mutable(type);
        if (state == nullptr || !state->is_joint() || !state->exp_avg.is_valid())
            return;
        const size_t n = splat_data_.size();
        const size_t slots = slots_for(type, splat_data_);
        const size_t attributes = slots
                                      ? slots * 4
                                      : state->exp_avg.numel() / (n * joint_adam::bytes_per_cell(state->joint_bits));
        // Geometry check instead of an invalidation call at every grow site: any
        // reallocation changes the row count, which forces a fresh decode.
        if (state->moments_first.is_valid() && state->moments_first.shape() == TensorShape{n, attributes} &&
            state->moments_second.is_valid())
            return;
        auto decoded = read_moment_rows(type, n, 0, n);
        state->moments_first = decoded.first.contiguous();
        state->moments_second = decoded.second.contiguous();
    }

    void AdamOptimizer::invalidate_fp32_moments() {
        for (const auto type : all_param_types()) {
            auto* state = get_state_mutable(type);
            if (state == nullptr)
                continue;
            state->moments_first = {};
            state->moments_second = {};
            state->packed_current = true;
        }
    }

    void AdamOptimizer::sync_moments_for_external_access() {
        flush_moments_to_packed();
    }

    void AdamOptimizer::flush_moments_to_packed() {
        for (const auto type : all_param_types()) {
            auto* state = get_state_mutable(type);
            if (state == nullptr || !state->is_joint() || !state->exp_avg.is_valid() ||
                !state->moments_first.is_valid() || !state->moments_second.is_valid() ||
                state->packed_current)
                continue;
            const size_t n = splat_data_.size();
            if (state->moments_first.shape() != TensorShape{n, state->moments_first.shape()[1]}) {
                // Moments do not cover the current model; let the next update rebuild them.
                continue;
            }
            write_moment_range(*state, slots_for(type, splat_data_),
                               {state->moments_first, state->moments_second}, 0);
            state->packed_current = true;
        }
    }

    void AdamOptimizer::remap_moment_rows(ParamType type, size_t old_n, size_t n, const Tensor& mapping, bool append) {
        flush_moments_to_packed();
        auto& state = states_.at(param_name(type));
        const auto slots = slots_for(type, splat_data_);
        const size_t bpc = joint_adam::bytes_per_cell(state.joint_bits);
        const size_t attributes = slots ? slots * 4 : state.exp_avg.numel() / (old_n * bpc);
        const size_t logical = slots ? core::sh_swizzled_float_count(n, splat_data_.max_sh_coeffs_rest()) : n;
        const size_t capacity = logical <= state.capacity ? state.capacity : compute_new_capacity(state.capacity, logical);
        if (mapping.is_valid()) {
            if (mapping.numel() != (append ? n - old_n : n))
                throw std::invalid_argument("Adam mapping size disagrees with destination rows");
            core::internal::assert_index_tensor(mapping, old_n, "Adam remap", true);
        } else if (!append)
            throw std::invalid_argument("Adam reorder requires source indices");
        const TensorShape packed_shape = slots ? TensorShape{logical * bpc} : TensorShape{n, attributes * bpc};
        auto prepared = state;
        prepared.exp_avg = {};
        prepared.joint_bounds = {};
        prepared.grad = {};
        prepared.exp_avg = Tensor::zeros_direct(packed_shape, slots ? capacity * bpc : capacity, Device::GPU, DataType::UInt8);
        ensure_joint_bounds_capacity(prepared.joint_bounds, n, slots ? capacity / attributes : capacity, Device::GPU);
        if (state.grad.is_valid()) {
            auto shape = state.grad.shape().dims();
            shape[0] = logical;
            prepared.grad = Tensor::zeros_direct(TensorShape(shape), capacity, Device::GPU);
        }
        const size_t first = append ? (old_n / 256) * 256 : 0;
        if (first) {
            prepared.exp_avg.reshape({-1}).slice(0, 0, first * attributes * bpc).copy_from(state.exp_avg.reshape({-1}).slice(0, 0, first * attributes * bpc));
            prepared.joint_bounds.slice(0, 0, first / 256).copy_from(state.joint_bounds.slice(0, 0, first / 256));
            if (prepared.grad.is_valid())
                prepared.grad.reshape({-1}).slice(0, 0, first * attributes).copy_from(state.grad.reshape({-1}).slice(0, 0, first * attributes));
        }
        for (size_t offset = first; offset < n; offset += 4096) {
            const size_t end = std::min(offset + 4096, n), count = end - offset;
            joint_adam::TensorMoments rows{Tensor::zeros({count, attributes}, Device::GPU), Tensor::zeros({count, attributes}, Device::GPU)};
            Tensor gradients = prepared.grad.is_valid() ? Tensor::zeros({count, attributes}, Device::GPU) : Tensor{};
            const size_t keep = append && offset < old_n ? std::min(end, old_n) - offset : 0;
            if (keep) {
                const auto old = read_moment_rows(type, old_n, offset, keep);
                rows.first.slice(0, 0, keep).copy_from(old.first);
                rows.second.slice(0, 0, keep).copy_from(old.second);
                if (gradients.is_valid()) {
                    const size_t cells = (slots ? (keep + 31) / 32 * 32 : keep) * attributes;
                    gradients.slice(0, 0, keep).copy_from(rows_from_layout(state.grad.reshape({-1}).slice(0, offset * attributes, offset * attributes + cells), keep, slots));
                }
            }
            if (mapping.is_valid() && count > keep) {
                const size_t begin = append ? offset + keep - old_n : offset;
                const auto cells = row_cell_indices(mapping.slice(0, begin, begin + count - keep), attributes, slots);
                const auto gathered = joint_adam::decode_tensor(state.exp_avg, state.joint_bounds, state.joint_bits, 256 * attributes, &cells);
                rows.first.slice(0, keep, count).copy_from(gathered.first.reshape(TensorShape{count - keep, attributes}));
                rows.second.slice(0, keep, count).copy_from(gathered.second.reshape(TensorShape{count - keep, attributes}));
                if (gradients.is_valid())
                    gradients.slice(0, keep, count).copy_from(state.grad.reshape({-1}).index_select(0, cells).reshape(TensorShape{count - keep, attributes}));
            }
            write_moment_range(prepared, slots, rows, offset);
            if (gradients.is_valid()) {
                const auto flat = layout_from_rows(gradients, slots);
                prepared.grad.reshape({-1}).slice(0, offset * attributes, offset * attributes + flat.numel()).copy_from(flat);
            }
            core::with_idle_vulkan_device([](const auto&) {});
        }
        prepared.size = logical;
        prepared.capacity = capacity;
        prepared.moments_first = {};
        prepared.moments_second = {};
        prepared.packed_current = true;
        state = AdamParamState{};
        state = std::move(prepared);
    }

    void AdamOptimizer::permute_rows(const Tensor& permutation) {
        const size_t n = splat_data_.size();
        if (permutation.shape() != TensorShape{n})
            throw std::invalid_argument("Adam permutation must cover all model rows");
        if (!n)
            return;
        for (const auto type : all_param_types()) {
            auto* state = get_state_mutable(type);
            if (state && state->exp_avg.is_valid() && state->exp_avg.numel())
                remap_moment_rows(type, n, n, permutation, false);
        }
        if (frozen_mask_.is_valid())
            set_frozen_mask(frozen_mask_.index_select(0, permutation));
        if (crop_damping_mask_.is_valid())
            set_crop_damping_mask(crop_damping_mask_.index_select(0, permutation));
        if (mean_step_far_mask_storage_.is_valid())
            set_mean_step_far_mask(mean_step_far_mask_storage_.index_select(0, permutation));
    }

    void AdamOptimizer::step(int iteration) {
        core::GpuBackendScope backend(core::GpuBackend::Vulkan);
        validate_mean_step_far_mask();
        refresh_screen_share_buffer();
        last_step_zeroed_gradients_ = false;
        for (auto type : all_param_types())
            step_param(type, iteration);
    }

    void AdamOptimizer::step_param(ParamType type, int) {
        // Opt-in per-stage attribution (--log-level perf). The optimizer step is
        // the largest single phase of a mature Vulkan iteration, and the Adam
        // arithmetic is a minority of it, so the surrounding stages need names.
        LOG_TIMER("adam_step_param");
        auto& param = get_param(type);
        if (!param.is_valid() || !param.numel() ||
            (type == ParamType::ShN && !splat_data_.active_sh_coeffs_rest()))
            return;
        const auto name = param_name(type);
        if (!states_.contains(name))
            init_state(type, false);
        auto& state = states_.at(name);
        if (!state.grad.is_valid() || !state.grad.numel())
            return;
        const size_t n = splat_data_.size(), slots = slots_for(type, splat_data_);
        const size_t expected_size = slots ? core::sh_swizzled_float_count(n, splat_data_.max_sh_coeffs_rest()) : n;
        if (state.size != expected_size)
            throw std::runtime_error("Optimizer state desync: " + name);
        const int64_t next_step = state.step_count + 1;
        const float bc1 = static_cast<float>(1.0 / (1.0 - std::pow(config_.beta1, next_step)));
        const float bc2 = static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(config_.beta2, next_step)));
        const float beta1 = static_cast<float>(config_.beta1), beta2 = static_cast<float>(config_.beta2);
        const float learning_rate = static_cast<float>(get_param_lr(type)) * bc1;
        const size_t active_slots = core::sh_float4_slots_for_rest(splat_data_.active_sh_coeffs_rest());
        // Bound scratch by parameter width: SH needs fewer rows per chunk than
        // scalar/vector parameters. Preserve the single-chunk path for small models
        // and keep complete 256-row quantization blocks.
        const size_t attributes = slots ? slots * 4 : param.numel() / n;
        constexpr size_t chunk_cells = 3 * 1024 * 1024;
        const size_t chunk_rows = n <= 131072 ? 131072 : std::min(size_t{1048576}, chunk_cells / attributes / 256 * 256);
        ensure_fp32_moments(type);
        for (size_t offset = 0; offset < n; offset += chunk_rows) {
            const size_t count = std::min(chunk_rows, n - offset);
            // Fast path: the moments live in fp32 in memory, so the chunk needs no decode
            // and no encode at all. Whole 32-row blocks only so each slice lines up with
            // the stored rows. Everything else falls back to the fused decode, which reads
            // the packed form directly in the Adam shader.
            const bool fp32_moments = slots != 0 && count % 32 == 0 && state.moments_first.is_valid() &&
                                      state.moments_first.shape() == TensorShape{n, attributes} &&
                                      state.moments_second.is_valid();
            const bool fused = !fp32_moments && slots != 0 && count % 32 == 0;
            const auto old = [&] {
                LOG_TIMER("adam_read_moments");
                if (fp32_moments) {
                    return joint_adam::TensorMoments{state.moments_first.slice(0, offset, offset + count),
                                                     state.moments_second.slice(0, offset, offset + count)};
                }
                if (fused) {
                    // Placeholders: only the shape is used, the shader decodes the
                    // moments from the packed storage itself.
                    return joint_adam::TensorMoments{Tensor::empty(TensorShape{count, attributes}, Device::GPU),
                                                     Tensor::empty(TensorShape{count, attributes}, Device::GPU)};
                }
                return read_moment_rows(type, n, offset, count);
            }();
            const auto shape = old.first.shape();
            const size_t cells = (slots ? ((count + 31) / 32) * 32 : count) * attributes;
            // When the chunk fills whole 32-row blocks the packed layout needs no padding, so the
            // gradient can be read straight from the swizzled storage with the kernel's own packed
            // mapping and the rows_from_layout permute+materialise disappears. Padded chunks and
            // non-SH parameters keep the old path. The delta write is not swizzled: that half
            // diverged in testing (ledger section 105).
            const bool swizzled_grad = slots != 0 && count % 32 == 0;
            Tensor gradient = swizzled_grad
                ? state.grad.reshape({-1}).slice(0, offset * attributes, offset * attributes + cells).reshape(TensorShape{count, attributes})
                : rows_from_layout(state.grad.reshape({-1}).slice(0, offset * attributes, offset * attributes + cells), count, slots);
            Tensor lr_scale, enabled;
            const auto apply_mask = [&](const Tensor& mask, float scale) {
                if (!mask.is_valid() || !mask.numel() || scale == 1.0f)
                    return;
                const size_t length = std::min(count, mask.numel() > offset ? mask.numel() - offset : 0);
                auto effective = Tensor::zeros({count, 1}, Device::GPU, DataType::Bool);
                if (length)
                    effective.slice(0, 0, length).copy_from(mask.slice(0, offset, offset + length).reshape(TensorShape{length, 1}));
                const Tensor factor = effective.to(DataType::Float32) * (scale - 1.0f) + 1.0f;
                lr_scale = lr_scale.is_valid() ? lr_scale * factor : factor;
                if (scale == 0.0f)
                    enabled = enabled.is_valid() ? enabled.logical_and(effective.logical_not()) : effective.logical_not();
            };
            apply_mask(frozen_mask_, frozen_lr_scale_);
            apply_mask(crop_damping_mask_, cropbox_lr_scale_);
            if (type == ParamType::Means && per_splat_mean_step_ && mean_step_median_extent_ > 0 && mean_step_far_mask_storage_.is_valid()) {
                const Tensor ratio = (splat_data_.scaling_raw().slice(0, offset, offset + count).mean(1, true).exp() / mean_step_median_extent_).clamp(mean_step_r_min_, mean_step_r_max_);
                const Tensor factor = Tensor::where(mean_step_far_mask_storage_.slice(0, offset, offset + count).reshape(TensorShape{count, 1}), ratio, Tensor::ones_like(ratio));
                lr_scale = lr_scale.is_valid() ? lr_scale * factor : factor;
            }
            if (type == ParamType::Scaling && screen_share_n_ > 0 && screen_share_penalty_ > 0) {
                const Tensor share = splat_data_._max_screen_share.slice(0, offset, offset + count).reshape(TensorShape{count, 1});
                const Tensor hinge = (share.clamp_min(screen_share_limit_) / screen_share_limit_).log() * (screen_share_penalty_ / std::log(2.0f));
                gradient = gradient + hinge * (old.second.sqrt() * bc2 + static_cast<float>(config_.eps));
            }
            if (slots && active_slots < slots) {
                auto active = Tensor::zeros(shape, Device::GPU, DataType::Bool);
                active.slice(1, 0, active_slots * 4).fill_(1.0f);
                enabled = enabled.is_valid() ? enabled.logical_and(active) : active;
            }
            joint_adam::TensorMoments packed_source;
            core::internal::AdamMomentSource source;
            if (fused) {
                const size_t bpc = joint_adam::bytes_per_cell(state.joint_bits);
                const size_t block_cells = 256 * attributes;
                packed_source = {state.exp_avg.reshape({-1}).slice(0, offset * attributes * bpc,
                                                                   offset * attributes * bpc + cells * bpc),
                                 state.joint_bounds.slice(0, offset / 256, (offset + count + 255) / 256)};
                source = {&packed_source.first, &packed_source.second, static_cast<uint32_t>(slots),
                          static_cast<uint32_t>(block_cells), state.joint_bits, joint_adam::kEps};
            }
            if (swizzled_grad)
                source.slots = static_cast<uint32_t>(slots);
            Tensor first = old.first, second = old.second, delta;
            {
                LOG_TIMER("adam_kernel");
                core::internal::vulkan_adam_update(first, second, delta, gradient, enabled, lr_scale,
                                                   {beta1, beta2, bc2, static_cast<float>(config_.eps), learning_rate},
                                                   source, swizzled_grad);
            }
            if (slots) {
                LOG_TIMER("adam_write_params");
                const bool q16 = splat_data_.shN_value_quantized();
                const size_t rest = splat_data_.max_sh_coeffs_rest();
                if (q16) {
                    const size_t code_start = offset * rest * 3;
                    const size_t code_count = core::sh_value_quant::sh_value_u16_count(count, rest);
                    const auto codes = param.reshape({-1}).slice(0, code_start, code_start + code_count);
                    const auto bounds = splat_data_.shN_value_bounds().reshape({-1, 2}).slice(0, offset / 256, (offset + count + 255) / 256);
                    Tensor values = Tensor::zeros(shape, Device::GPU);
                    values.slice(1, 0, rest * 3).copy_from(core::sh_value_quant::decode_shN_u16_to_canonical_tensor(codes, bounds, count, rest).reshape(TensorShape{count, rest * 3}));
                    const auto swizzled = layout_from_rows(values - delta, slots);
                    Tensor encoded, encoded_bounds;
                    core::sh_value_quant::encode_shN_float4_to_u16_tensor(swizzled, count, slots, rest * 3, encoded, encoded_bounds);
                    param.reshape({-1}).slice(0, offset * rest * 3, offset * rest * 3 + encoded.numel()).copy_from(encoded.reshape({-1}));
                    splat_data_.shN_value_bounds().reshape({-1, 2}).slice(0, offset / 256, (offset + count + 255) / 256).copy_from(encoded_bounds);
                } else {
                    // The parameter storage is already in the swizzled order, and a
                    // permutation commutes with elementwise subtraction, so the
                    // un-swizzle/re-swizzle pair that used to bracket the update
                    // collapses exactly:
                    //     P(P^-1(values) - delta) == values - P(delta)
                    // That removes one full read/write pass over the chunk plus the
                    // materializing copy behind it, without changing any arithmetic.
                    auto range = param.reshape({-1}).slice(0, offset * attributes, offset * attributes + cells);
                    const Tensor flat_delta = layout_from_rows(delta, slots);
                    if (range.dtype() == DataType::Float32) {
                        range.sub_(flat_delta);
                    } else {
                        range.copy_from((range.to(DataType::Float32) - flat_delta).to(range.dtype()));
                    }
                }
            } else {
                LOG_TIMER("adam_write_params");
                auto range = param.slice(0, offset, offset + count);
                // Same collapse as the swizzled branch: update the parameter in place
                // instead of materializing the difference and copying it back.
                range.sub_(delta.reshape(range.shape()));
            }
            if (fp32_moments) {
                // first/second are this chunk's views into the state moments and the kernel now
                // updates moments where they already are, so there is nothing to fold back. The
                // packed form is stale until it is refreshed.
                state.packed_current = false;
            } else {
                LOG_TIMER("adam_write_moments");
                write_moment_range(state, slots, {first, second}, offset);
            }
            // Release this chunk's queued temporaries before producing the next.
            if (n > chunk_rows) {
                LOG_TIMER("adam_drain");
                core::with_idle_vulkan_device([](const auto&) {});
            }
        }
        state.step_count = next_step;
    }

    FastGSFusedAdamState AdamOptimizer::prepare_fastgs_fused_adam(int, cudaStream_t) {
        throw std::runtime_error("FastGS fused Adam requires CUDA; Vulkan uses the explicit optimizer");
    }
    void AdamOptimizer::commit_fastgs_fused_adam(int) {
        throw std::runtime_error("Cannot commit a CUDA fused optimizer step on Vulkan");
    }

    void AdamOptimizer::reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty() || !states_.contains(param_name(type)))
            return;
        const size_t n = splat_data_.size();
        std::vector<int> checked;
        checked.reserve(indices.size());
        for (auto index : indices) {
            if (index < 0 || static_cast<size_t>(index) >= n)
                throw std::out_of_range("Adam reset index is outside the model");
            checked.push_back(static_cast<int>(index));
        }
        relocate_params_at_indices(type, Tensor::from_vector(checked, {checked.size()}, Device::GPU));
    }

    void AdamOptimizer::extend_state_for_new_params(ParamType type, size_t count) {
        if (!count || !states_.contains(param_name(type)))
            return;
        const auto& state = states_.at(param_name(type));
        if (!state.exp_avg.is_valid() || !state.exp_avg.numel())
            return;
        const size_t old_n = type == ParamType::ShN ? splat_data_.size() - count : state.size;
        remap_moment_rows(type, old_n, old_n + count, {}, true);
    }

    void AdamOptimizer::extend_state_by_gather(ParamType type, const Tensor& indices) {
        if (!indices.numel() || !states_.contains(param_name(type)) || type == ParamType::ShN)
            return;
        const size_t old_n = states_.at(param_name(type)).size;
        remap_moment_rows(type, old_n, old_n + indices.numel(), indices, true);
    }

    void AdamOptimizer::add_new_params_gather(ParamType type, const Tensor& indices) {
        if (!indices.numel())
            return;
        auto& param = get_param(type);
        if (!param.is_valid() || !param.numel())
            return;
        if (type != ParamType::ShN) {
            const size_t new_n = param.shape()[0] + indices.numel();
            if (param.capacity() < new_n)
                param.reserve(compute_new_capacity(param.capacity(), new_n));
            param.append_gather(indices);
            extend_state_by_gather(type, indices);
            return;
        }
        const size_t new_n = splat_data_.size(), old_n = new_n - indices.numel();
        // Means are appended before SH in the existing strategy contract.
        Tensor source;
        sh_value::gather_shN_to_canonical(splat_data_, indices, source, old_n);
        sh_value::append_canonical_to_shN(splat_data_, source, old_n);
        if (states_.contains(param_name(type)))
            remap_moment_rows(type, old_n, new_n, indices, true);
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType, const int64_t*, size_t) {
        throw std::runtime_error("Raw CUDA relocation pointers are unsupported; use Tensor indices");
    }
#endif
} // namespace lfs::training
