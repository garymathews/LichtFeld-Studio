/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <array>
#include <tuple>

#include "mcmc.hpp"
#include "core/cuda/sh_layout.cuh"
#if LFS_TENSOR_CUDA
#include "core/cuda_error.hpp"
#endif
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor_backend.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "kernels/mcmc_tensor.hpp"
#if LFS_TENSOR_CUDA
#include "kernels/densification_kernels.hpp"
#endif
#include "kernels/mcmc_kernels.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "strategy_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        [[nodiscard]] std::uint64_t deterministic_mcmc_seed(const int iteration,
                                                            const std::uint64_t stream) {
            // Derive each stochastic operation from the absolute training
            // iteration.  A resumed trainer therefore emits the same MCMC
            // mutations as an uninterrupted run without serializing a CUDA
            // generator's opaque state.
            std::uint64_t value = static_cast<std::uint64_t>(std::max(iteration, 0));
            value ^= 0x9e3779b97f4a7c15ULL + stream + (value << 6) + (value >> 2);
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            return value ^ (value >> 31);
        }

        [[nodiscard]] inline bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] size_t deleted_mask_capacity(const lfs::core::SplatData& splat_data) {
            const size_t means_capacity = splat_data.means().capacity();
            return means_capacity > 0 ? means_capacity : static_cast<size_t>(splat_data.size());
        }

        void ensure_deleted_mask_size(lfs::core::SplatData& splat_data) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data);
            auto& deleted = splat_data.deleted();

            if (!deleted.is_valid() || deleted.ndim() != 1 || deleted.numel() != current_size) {
                deleted = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
            }

            deleted.reserve(desired_capacity);
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool({static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
            splat_data.notify_deleted_mask_changed();
        }

        void append_live_deleted_rows(lfs::core::SplatData& splat_data) {
            // keep deleted.numel() == size() after densify grow.
            if (!splat_data.has_deleted_mask()) {
                return;
            }
            const size_t target_size = static_cast<size_t>(splat_data.size());
            auto& deleted = splat_data.deleted();
            const size_t cur = static_cast<size_t>(deleted.numel());
            if (cur == target_size) {
                return;
            }
            if (cur < target_size) {
                const size_t pad = target_size - cur;
                const size_t desired_capacity = std::max(
                    deleted_mask_capacity(splat_data),
                    target_size);
                deleted.reserve(desired_capacity);
                deleted.append_zeros(pad);
                splat_data.notify_deleted_mask_changed();
                return;
            }
            splat_data.reconcile_deleted_mask();
        }

        void zero_optimizer_state(
            lfs::training::AdamOptimizer& optimizer,
            const ParamType param_type,
            const lfs::core::Tensor& indices) {
            if (indices.numel() == 0) {
                return;
            }

            auto* state = optimizer.get_state_mutable(param_type);
            if (!state) {
                return;
            }

            if (state->is_joint()) {
                // Joint codec: encode true zeros under current bounds via optimizer API.
                auto idx_cpu = indices.cpu();
                std::vector<int64_t> host_idx;
                host_idx.reserve(indices.numel());
                if (idx_cpu.dtype() == lfs::core::DataType::Int64) {
                    const auto* p = idx_cpu.ptr<int64_t>();
                    host_idx.assign(p, p + indices.numel());
                } else if (idx_cpu.dtype() == lfs::core::DataType::Int32) {
                    const auto* p = idx_cpu.ptr<int32_t>();
                    for (size_t i = 0; i < indices.numel(); ++i)
                        host_idx.push_back(static_cast<int64_t>(p[i]));
                }
                if (!host_idx.empty())
                    optimizer.reset_state_at_indices(param_type, host_idx);
            }

            // grad is transient (re-zeroed each step); only the contiguous case is handled here.
            if (param_type != ParamType::ShN && state->grad.is_valid() && state->grad.numel() > 0) {
                const auto& shape = state->grad.shape();
                if (has_zero_dimension(shape)) {
                    return;
                }
                std::vector<size_t> dims = {static_cast<size_t>(indices.numel())};
                for (size_t i = 1; i < shape.rank(); ++i) {
                    dims.push_back(shape[i]);
                }
                auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->grad.device());
                state->grad.index_put_(indices, zeros);
            }
        }
    } // anonymous namespace

    MCMC::MCMC(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    lfs::core::Tensor MCMC::multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement) {
        // Use the tensor library's built-in multinomial sampling
        return lfs::core::Tensor::multinomial(weights, n, replacement);
    }

    void MCMC::update_optimizer_for_relocate(
        const lfs::core::Tensor& sampled_indices,
        const lfs::core::Tensor& dead_indices,
        ParamType param_type) {

        // Reset optimizer moment state for rows whose params changed.
        // Source rows get adjusted opacity/scaling; destination rows receive fresh params.
#if LFS_TENSOR_CUDA
        _optimizer->relocate_params_at_indices(param_type, sampled_indices);
        _optimizer->relocate_params_at_indices(param_type, dead_indices);
#else
        // Shared quantization blocks must be decoded and committed just once.
        _optimizer->relocate_params_at_indices(param_type,
            lfs::core::Tensor::cat({sampled_indices, dead_indices}, 0));
#endif
    }

    void MCMC::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        // reuse densification_info / score buffers in place when possible.
        const size_t reserve =
            (_params && _params->max_cap > 0) ? static_cast<size_t>(_params->max_cap) : 0;
        ensure_densification_info_shape_inplace(
            _splat_data->_densification_info, n, _splat_data->means().device());
        if (_params) {
            ensure_max_screen_share_shape(*_splat_data, n, reserve);
            publish_screen_share_cap(_optimizer.get(), *_splat_data, *_params);
        }

        const size_t prev_n = _error_score_max.is_valid() ? _error_score_max.numel() : 0;
        ensure_score_buffer_inplace(
            _error_score_max, n, _splat_data->means().device(), reserve);
        if (prev_n != n) {
            _error_score_max.set_name("mcmc.error_score_max");
            _error_score_windows = 0;
        }
    }

    lfs::core::Tensor MCMC::get_sampling_weights() const {
        using namespace lfs::core;

        const size_t n = static_cast<size_t>(_splat_data->size());
        Tensor weights;
        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            weights = zero_frozen_scores(
                *_splat_data,
                Tensor::ones({n}, _splat_data->means().device()));
        } else {
            weights = zero_frozen_scores(*_splat_data, _error_score_max.clamp_min(1e-12f));
        }
        return apply_crop_damping_to_scores(*_optimizer, weights);
    }

    void MCMC::ensure_ratio_workspace_size(const size_t required) {
        if (!_ones_int32.is_valid() || _ones_int32.numel() < required) {
            _ones_int32 = lfs::core::Tensor::ones(
                {required}, _splat_data->means().device(), lfs::core::DataType::Int32);
        }
    }

#if !LFS_TENSOR_CUDA
    int MCMC::prepare_topology_update(const std::function<int()>& update) {
        if (preparing_topology_)
            return update();
        auto prepared_model = _splat_data->clone_async(nullptr, true);
        auto prepared_optimizer = _optimizer->clone_for_model(prepared_model);
        auto prepared_scores = _error_score_max.is_valid() ? _error_score_max.clone() : core::Tensor{};
        auto live_scores = std::move(_error_score_max);
        const int live_windows = _error_score_windows, live_iteration = _current_iteration;
        auto* live_model = _splat_data;
        auto live_optimizer = std::move(_optimizer);
        _error_score_max = std::move(prepared_scores);
        _splat_data = &prepared_model;
        _optimizer = std::move(prepared_optimizer);
        preparing_topology_ = true;
        try {
            const int result = update();
            // Materialize all deferred writes before the non-allocating commit.
            core::with_idle_vulkan_device([](const auto&) {});
            if (result) {
                live_model->adopt_training_update(std::move(prepared_model));
                live_optimizer->adopt_training_update(*_optimizer);
            } else {
                _error_score_max = {};
                _error_score_max = std::move(live_scores);
                _error_score_windows = live_windows;
                _current_iteration = live_iteration;
            }
            _optimizer = std::move(live_optimizer);
            _splat_data = live_model;
            preparing_topology_ = false;
            return result;
        } catch (...) {
            _optimizer = std::move(live_optimizer);
            _splat_data = live_model;
            _error_score_max = {};
            _error_score_max = std::move(live_scores);
            _error_score_windows = live_windows;
            _current_iteration = live_iteration;
            preparing_topology_ = false;
            throw;
        }
    }
#endif

    int MCMC::relocate_gs() {
        LOG_TIMER("MCMC::relocate_gs");
        LFS_TRACE("kernel.mcmc.relocate");
        const bool shN_expanded =
            !_splat_data->shN_value_quantized() &&
            lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);
        lfs::training::sh_value::ShNCommitGuard shn_guard(
            *_splat_data, shN_expanded, "MCMC::relocate_gs");
        using namespace lfs::core;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("relocate_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        // Find dead Gaussians: opacity <= min_opacity OR rotation magnitude near zero
        Tensor dead_mask, dead_indices;
        size_t n_dead;
        {
            LOG_TIMER("relocate_find_dead");
            dead_mask = compute_dead_mask_from_opacity_and_rotation(
                opacities,
                _splat_data->rotation_raw(),
                _params->min_opacity);
            dead_mask = exclude_frozen_from_mask(*_splat_data, dead_mask);
            dead_indices = dead_mask.nonzero().squeeze(-1);
            n_dead = dead_indices.numel();
        }

        if (n_dead == 0)
            return 0;

        Tensor alive_indices;
        {
            LOG_TIMER("relocate_find_alive");
            Tensor alive_mask = dead_mask.logical_not();
            alive_indices = alive_mask.nonzero().squeeze(-1);
        }

        if (alive_indices.numel() == 0)
            return 0;

        Tensor sampled_idxs, sampled_opacities, sampled_scales;
        {
            LOG_TIMER("relocate_multinomial_sample_and_gather_FUSED");
            const size_t N = opacities.numel();

            // Get source tensors (contiguous)
            Tensor opacities_contig = opacities.contiguous();
            const Tensor sampling_weights = get_sampling_weights();
            const auto alive_weights = sampling_weights.index_select(0, alive_indices);
            if (alive_weights.count_nonzero() == 0) {
                return 0;
            }
            Tensor scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()

            const uint64_t seed = deterministic_mcmc_seed(_current_iteration, 0x52454c4f43415445ULL);

            // does multinomial sampling + gathering in one pass
#if LFS_TENSOR_CUDA
            // Allocate outputs
            sampled_idxs = Tensor::empty({n_dead}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_dead}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_dead, 3}, Device::CUDA, DataType::Float32);

            mcmc::launch_multinomial_sample_and_gather(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                alive_indices.ptr<int64_t>(),
                alive_indices.numel(),
                n_dead,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                N);
#else
            const auto local = Tensor::multinomial(alive_weights, static_cast<int>(n_dead), true, seed);
            sampled_idxs = alive_indices.index_select(0, local);
            sampled_opacities = opacities_contig.index_select(0, sampled_idxs);
            sampled_scales = scaling_raw_contig.index_select(0, sampled_idxs).exp();
#endif
        }

        // Count occurrences of each sampled index (how many times each was sampled)
        Tensor ratios;
        {
            LOG_TIMER("relocate_count_occurrences");
            ensure_ratio_workspace_size(opacities.numel());
            auto ones_N = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ones_N.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs).contiguous();

            // Clamp ratios to [1, n_max]
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("relocate_cuda_kernel");
#if LFS_TENSOR_CUDA
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);
            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
#else
            std::tie(new_opacities, new_scales) = mcmc::relocate(
                sampled_opacities, sampled_scales, ratios, _params->min_opacity);
#endif
        }

        // Clamp new opacities and compute raw values
        Tensor new_opacity_raw;
        {
            LOG_TIMER("relocate_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

#if !LFS_TENSOR_CUDA
        return prepare_topology_update([&]() -> int {
#endif
        // Update parameters
        {
            LOG_TIMER("relocate_update_params");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0]; // Total number of Gaussians

            // Compute log(scales) for the new scales
            Tensor new_scales_log = new_scales.log();

            // Update sampled indices with new opacity/scaling using direct CUDA kernel
            // This preserves tensor capacity (unlike index_put_ which creates new tensors)
#if LFS_TENSOR_CUDA
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scales_log.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);
#else
            _splat_data->scaling_raw().index_put_(sampled_idxs, new_scales_log);
            _splat_data->opacity_raw().index_put_(sampled_idxs, new_opacity_raw);
#endif

            // Copy sampled params to dead slots. shN is stored swizzled, so the legacy
            // kernel skips it and the selected rows are copied below.
#if LFS_TENSOR_CUDA
            mcmc::launch_copy_gaussian_params(
                sampled_idxs.ptr<int64_t>(),
                dead_indices.ptr<int64_t>(),
                _splat_data->means().ptr<float>(),
                _splat_data->sh0().ptr<float>(),
                /*shN=*/nullptr,
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                dead_indices.numel(),
                /*sh_coeffs=*/0,
                opacity_dim,
                N);
#else
            for (Tensor* parameter : {&_splat_data->means(), &_splat_data->sh0(),
                                      &_splat_data->scaling_raw(), &_splat_data->rotation_raw(),
                                      &_splat_data->opacity_raw()}) {
                const auto source = parameter->index_select(0, sampled_idxs);
                parameter->index_put_(dead_indices, source);
            }
#endif

            // Copy sampled shN onto dead slots. q16 stays packed: gather-decode
            // the source rows and re-encode only the dest 256-splat blocks.
            if (_splat_data->shN().is_valid() && _splat_data->shN().numel() > 0 &&
                _splat_data->max_sh_coeffs_rest() > 0 && dead_indices.numel() > 0) {
                using namespace lfs::core;
                Tensor staged;
                lfs::training::sh_value::gather_shN_to_canonical(
                    *_splat_data, sampled_idxs, staged);
                lfs::training::sh_value::scatter_canonical_into_shN(
                    *_splat_data, dead_indices, staged);
            }
        }

        // Update optimizer states for all parameters
        {
            LOG_TIMER("relocate_update_optimizer");
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Means);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Sh0);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::ShN);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Scaling);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Rotation);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Opacity);
        }

        if (_splat_data->has_deleted_mask()) {
            set_deleted_mask_rows(*_splat_data, dead_indices, false);
        }

        return n_dead;
#if !LFS_TENSOR_CUDA
        });
#endif

    }

    int MCMC::add_new_gs() {
        LOG_TIMER("MCMC::add_new_gs");
        LFS_TRACE("kernel.densify.duplicate");
        using namespace lfs::core;
        const bool shN_expanded =
            !_splat_data->shN_value_quantized() &&
            lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);
        lfs::training::sh_value::ShNCommitGuard shn_guard(
            *_splat_data, shN_expanded, "MCMC::add_new_gs");

        if (!_optimizer) {
            LOG_ERROR("MCMC::add_new_gs: optimizer not initialized");
            return 0;
        }

        const int current_n = _splat_data->size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const size_t n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("add_new_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        Tensor sampled_idxs;
        Tensor sampled_opacities;
        Tensor sampled_scales;
        {
            LOG_TIMER("add_new_multinomial_sample_and_gather");

            const size_t N = opacities.numel();

            // Get raw scaling and ensure contiguity
            auto scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()
            auto opacities_contig = opacities.contiguous();
            const auto sampling_weights = get_sampling_weights();
            if (sampling_weights.count_nonzero() == 0) {
                return 0;
            }

            // Generate random seed
            const auto seed = deterministic_mcmc_seed(_current_iteration, 0x4144445f4e4557ULL);

            // Call fused CUDA kernel
#if LFS_TENSOR_CUDA
            // Allocate output tensors
            sampled_idxs = Tensor::empty({n_new}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_new}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_new, 3}, Device::CUDA, DataType::Float32);

            mcmc::launch_multinomial_sample_all(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                N,
                n_new,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>());
#else
            sampled_idxs = Tensor::multinomial(sampling_weights, static_cast<int>(n_new), true, seed);
            sampled_opacities = opacities_contig.index_select(0, sampled_idxs);
            sampled_scales = scaling_raw_contig.index_select(0, sampled_idxs).exp();
#endif
        }

        // Count occurrences as int32 to avoid float->int conversions in the hot path.
        Tensor ratios;
        {
            LOG_TIMER("add_new_count_occurrences");
            ensure_ratio_workspace_size(opacities.numel());
            ratios = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ratios.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs);

            // Clamp in int32 domain
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
            ratios = ratios.contiguous();
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation_kernel");
#if LFS_TENSOR_CUDA
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);
            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
#else
            std::tie(new_opacities, new_scales) = mcmc::relocate(
                sampled_opacities, sampled_scales, ratios, _params->min_opacity);
#endif
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

#if !LFS_TENSOR_CUDA
        return prepare_topology_update([&]() -> int {
#endif
        // preflight exportable capacity BEFORE parent-row
        // relocate writes or multi-param gather grow (no torn model on failure).
        if (!_optimizer->preflight_grow_capacity(static_cast<size_t>(n_new))) {
            LOG_ERROR(
                "MCMC densify aborted: capacity-ensure failed for +{} rows "
                "(no params mutated)",
                n_new);
            return 0;
        }

        // Update existing Gaussians first (before concatenation)
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
#if LFS_TENSOR_CUDA
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);
#else
            _splat_data->scaling_raw().index_put_(sampled_idxs, new_scaling_raw);
            _splat_data->opacity_raw().index_put_(sampled_idxs, new_opacity_raw);
#endif
        }

        // Use add_new_params_gather() to leverage reserved capacity
        {
            LOG_TIMER("add_new_append_gather");
            // Gather and append parameters for new Gaussians (done after updating opacity/scaling)
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs);
        }

        append_live_deleted_rows(*_splat_data);
        if (_splat_data->has_frozen_ranges()) {
            apply_frozen_ranges_to_optimizer(*_splat_data, *_optimizer);
        }

        return n_new;
#if !LFS_TENSOR_CUDA
        });
#endif

    }

    // Test helper: add_new_gs with explicitly specified indices (no multinomial sampling)
    int MCMC::add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs) {
        LOG_TIMER("MCMC::add_new_gs_with_indices_test");
        using namespace lfs::core;
        const bool shN_expanded =
            !_splat_data->shN_value_quantized() &&
            lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);
        lfs::training::sh_value::ShNCommitGuard shn_guard(
            *_splat_data, shN_expanded, "MCMC::add_new_gs_with_indices_test");

        if (!_optimizer) {
            LOG_ERROR("add_new_gs_with_indices_test called but optimizer not initialized");
            return 0;
        }

        // Ensure indices are Int64 (test may pass Int32)
        Tensor sampled_idxs_i64 = (sampled_idxs.dtype() == DataType::Int64) ? sampled_idxs : sampled_idxs.to(DataType::Int64);

        const size_t required = _splat_data->size();
        if (auto frozen_mask = make_frozen_mask(*_splat_data, required, sampled_idxs_i64.device());
            frozen_mask.is_valid()) {
            auto trainable = frozen_mask.index_select(0, sampled_idxs_i64).logical_not();
            sampled_idxs_i64 = sampled_idxs_i64.index_select(0, trainable.nonzero().squeeze(-1));
        }

        const int n_new = sampled_idxs_i64.numel();
        if (n_new == 0)
            return 0;

        // Get opacities
        auto opacities = _splat_data->get_opacity();

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs_i64);
        auto sampled_scales = _splat_data->get_scaling().index_select(0, sampled_idxs_i64);

        ensure_ratio_workspace_size(required);

        // Count occurrences in int32 and keep +1 baseline.
        auto ratios = _ones_int32.slice(0, 0, required).clone();
        ratios.index_add_(0, sampled_idxs_i64, _ones_int32.slice(0, 0, sampled_idxs_i64.numel()));
        ratios = ratios.index_select(0, sampled_idxs_i64);

        // Clamp in int32 domain
        const int n_max = _n_max;
        ratios = ratios.clamp(1, n_max);
        ratios = ratios.contiguous();

        // Call the CUDA relocation function
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation");
#if LFS_TENSOR_CUDA
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);
            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
#else
            std::tie(new_opacities, new_scales) = mcmc::relocate(
                sampled_opacities, sampled_scales, ratios, _params->min_opacity);
#endif
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        if (!_optimizer->preflight_grow_capacity(static_cast<size_t>(n_new))) {
            LOG_ERROR(
                "MCMC densify (test path) aborted: capacity-ensure failed for +{} rows",
                n_new);
            return 0;
        }

#if !LFS_TENSOR_CUDA
        return prepare_topology_update([&]() -> int {
#endif
        // Update existing Gaussians first
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
#if LFS_TENSOR_CUDA
            mcmc::launch_update_scaling_opacity(
                sampled_idxs_i64.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs_i64.numel(),
                opacity_dim,
                N);
#else
            _splat_data->scaling_raw().index_put_(sampled_idxs_i64, new_scaling_raw);
            _splat_data->opacity_raw().index_put_(sampled_idxs_i64, new_opacity_raw);
#endif
        }

        // Use fused append_gather() operation
        {
            LOG_TIMER("add_new_params_gather");
            // Gather opacity/scaling after updating them
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs_i64);
        }

        append_live_deleted_rows(*_splat_data);

        return n_new;
#if !LFS_TENSOR_CUDA
        });
#endif

    }

    void MCMC::inject_noise() {
        LOG_TIMER("MCMC::inject_noise");
        LFS_TRACE("kernel.mcmc.add_noise");
        using namespace lfs::core;

        // Get current learning rate from optimizer (after scheduler has updated it)
        const float current_lr = _optimizer->get_lr() * NOISE_LR;
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0) {
            return;
        }

        // one fused kernel (curand + cov transform + add); no noise buffer.
        const auto frozen_mask = make_frozen_mask(*_splat_data, n, Device::CUDA);
        const auto seed = deterministic_mcmc_seed(_current_iteration, 0x494e4a454354ULL);
#if LFS_TENSOR_CUDA
        mcmc::launch_inject_noise_kernel(
            _splat_data->opacity_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->means().ptr<float>(),
            frozen_mask.is_valid() ? frozen_mask.ptr<bool>() : nullptr,
            frozen_mask.is_valid() ? frozen_mask.numel() : 0,
            current_lr,
            n,
            seed);
#else
        mcmc::inject_noise(_splat_data->opacity_raw(), _splat_data->scaling_raw(),
                           _splat_data->rotation_raw(), _splat_data->means(), frozen_mask, current_lr, seed);
#endif
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
#if !LFS_TENSOR_CUDA
        if (is_refining(iter) && !preparing_topology_) {
            prepare_topology_update([&] { post_backward(iter, render_output); return 1; });
            return;
        }
#endif
        LOG_TIMER("MCMC::post_backward");
        _current_iteration = iter;

        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
            _splat_data->_max_screen_share = lfs::core::Tensor::empty({0});
            if (_params) {
                publish_screen_share_cap(_optimizer.get(), *_splat_data, *_params);
            }
            _error_score_max = lfs::core::Tensor::empty({0});
            _error_score_windows = 0;
        }

        if (iter < _params->stop_refine) {
            ensure_densification_info_shape();

            // One training iteration corresponds to one camera view, so info[1] is E_k^pi.
            // Keep the max over views as the densification priority.
            const auto& info = _splat_data->_densification_info;
            if (info.is_valid() &&
                info.ndim() == 2 &&
                info.shape()[0] >= 2 &&
                info.shape()[1] == _error_score_max.numel()) {
#if LFS_TENSOR_CUDA
                lfs::training::mcmc::launch_max_error_and_zero_densification(
                    _error_score_max.ptr<float>(),
                    _splat_data->_densification_info.ptr<float>(),
                    _error_score_max.numel());
#else
                _error_score_max.copy_from(_error_score_max.maximum(info.slice(0, 1, 2).squeeze(0)));
                _splat_data->_densification_info.zero_();
#endif
            } else if (info.is_valid() && info.numel() > 0) {
                _splat_data->_densification_info.zero_();
            }
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            if (_splat_data->_max_screen_share.is_valid() &&
                _splat_data->_max_screen_share.numel() > 0) {
#if LFS_TENSOR_CUDA
                LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(), "wait fused adam before screen-share mutate");
#endif
                // Vulkan Tensor operations retain their input dependencies on the shared queue.
            }
            const size_t n_clip = static_cast<size_t>(_splat_data->size());
            if (_params && screen_share_cap_active(_params->max_screen_share) &&
                _splat_data->_max_screen_share.is_valid() &&
                _splat_data->_max_screen_share.numel() == n_clip) {
                auto& log_scales = _splat_data->scaling_raw();
                assert(log_scales.shape()[0] == n_clip && log_scales.shape()[1] == 3);
                const bool* frozen = nullptr;
                size_t frozen_n = 0;
                if (_optimizer) {
                    const auto& mask = _optimizer->frozen_mask();
                    if (mask.is_valid()) {
                        frozen = mask.ptr<bool>();
                        frozen_n = mask.numel();
                    }
                }
#if LFS_TENSOR_CUDA
                kernels::launch_clip_log_scale_by_screen_share(
                    log_scales.ptr<float>(),
                    _splat_data->_max_screen_share.ptr<float>(),
                    frozen,
                    frozen_n,
                    _params->max_screen_share,
                    n_clip);
#else
                const auto& share = _splat_data->_max_screen_share;
                auto delta = (share / _params->max_screen_share).clamp_min(1.f).log().clamp_max(std::log(1.5f));
                auto axis = log_scales.max_with_indices(1, true).second;
                auto axes = lfs::core::Tensor::from_vector({0, 1, 2}, {1, 3}, log_scales.device());
                auto active = axis.eq(axes).logical_and(share.gt(_params->max_screen_share).unsqueeze(1));
                if (_optimizer && _optimizer->frozen_mask().is_valid())
                    active = active.logical_and(_optimizer->frozen_mask().logical_not().unsqueeze(1));
                log_scales.copy_from(lfs::core::Tensor::where(active, log_scales - delta.unsqueeze(1), log_scales));
#endif
            }

            const int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("MCMC: Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            const int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("MCMC: Added {} new Gaussians at iteration {} (total: {})",
                          n_added, iter, _splat_data->size());
                LFS_COUNTER_ADD("strategy.mcmc.added", n_added);
            }
            // Release cached pool memory to avoid bloat (important after add_new_gs)
            lfs::core::Tensor::trim_memory_pool();

            const size_t n = static_cast<size_t>(_splat_data->size());
            LFS_GAUGE("model.gaussians.live", n);
            LFS_GAUGE("model.gaussians.capacity", deleted_mask_capacity(*_splat_data));

            ensure_score_buffer_inplace(
                _error_score_max, n, _splat_data->means().device(),
                _params && _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);

            ++_error_score_windows;
            if (_error_score_windows >= 2) {
                ensure_score_buffer_inplace(
                    _error_score_max, n, _splat_data->means().device(),
                    _params && _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
                _error_score_max.zero_();
                _error_score_max.set_name("mcmc.error_score_max");
                _error_score_windows = 0;
            }

            ensure_densification_info_shape_inplace(
                _splat_data->_densification_info, n, _splat_data->means().device());
            _splat_data->_densification_info.zero_();
            if (_params) {
                const size_t cap = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
                ensure_max_screen_share_shape(*_splat_data, n, cap);
                _splat_data->_max_screen_share.zero_();
                publish_screen_share_cap(_optimizer.get(), *_splat_data, *_params);
            }
        }

        // Inject noise to positions every iteration
        inject_noise();
    }

    void MCMC::step(int iter) {
        LOG_TIMER("MCMC::step");
        if (iter < _params->iterations) {
            {
                LOG_TIMER("step_optimizer_step");
                _optimizer->step(iter);
            }
            {
                LOG_TIMER("step_zero_grad");
                _optimizer->zero_grad(iter);
            }
            {
                LOG_TIMER("step_scheduler");
                _scheduler->step();
            }
        }
    }

    void MCMC::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        if (!mask.is_valid() || mask.numel() == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        const auto prune_mask = exclude_frozen_from_mask(*_splat_data, mask);
        const int n_remove = prune_mask.to(DataType::Int32).sum().template item<int>();

        LOG_INFO("MCMC::remove_gaussians called: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", n_remove);
        LFS_COUNTER_ADD("strategy.mcmc.pruned", n_remove);

        const Tensor prune_indices = prune_mask.nonzero().squeeze(-1);

        set_deleted_mask_rows(*_splat_data, prune_indices, true);

        auto zero_rotation = Tensor::zeros(
            {static_cast<size_t>(n_remove), 4},
            _splat_data->rotation_raw().device());
        _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

        zero_optimizer_state(*_optimizer, ParamType::Means, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Sh0, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::ShN, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Scaling, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Rotation, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Opacity, prune_indices);

        if (_error_score_max.is_valid() &&
            _error_score_max.ndim() == 1 &&
            _error_score_max.numel() >= _splat_data->size()) {
            auto zeros = Tensor::zeros({static_cast<size_t>(n_remove)}, _error_score_max.device());
            _error_score_max.index_put_(prune_indices, zeros);
        }

        LOG_DEBUG("MCMC: soft-deleted {} Gaussians (rotation and optimizer state zeroed)", n_remove);
    }

    void MCMC::set_optimization_params(const lfs::core::param::OptimizationParameters& params) {
        auto next = std::make_unique<const lfs::core::param::OptimizationParameters>(params);
        if (_params && _optimizer && _scheduler) {
            const auto& old = *_params;
            const double scene_scale = _splat_data->get_scene_scale();
            const std::array<std::tuple<ParamType, double, double>, 6> rates{{
                {ParamType::Means, old.means_lr * scene_scale, params.means_lr * scene_scale},
                {ParamType::Sh0, old.shs_lr, params.shs_lr},
                {ParamType::ShN, old.shs_lr / 20.f, params.shs_lr / 20.f},
                {ParamType::Scaling, old.scaling_lr, params.scaling_lr},
                {ParamType::Rotation, old.rotation_lr, params.rotation_lr},
                {ParamType::Opacity, old.opacity_lr, params.opacity_lr},
            }};
            for (const auto& [type, previous, rate] : rates) {
                if (rate == previous) continue;
                _optimizer->set_param_lr(type, rate);
                if (type == ParamType::Means) _optimizer->set_lr(rate);
            }
            if (params.iterations != old.iterations || params.means_lr != old.means_lr) {
                const double current = _optimizer->get_param_lr(ParamType::Means);
                const double target = params.means_lr * _splat_data->get_scene_scale() * .01;
                const int remaining = std::max(1, int(params.iterations) - _current_iteration);
                _scheduler->set_gamma(current > 0 && target > 0 ? std::pow(target / current, 1.0 / remaining) : 1.0);
            }
        }
        _params = std::move(next);
        if (_splat_data) {
            publish_screen_share_cap(_optimizer.get(), *_splat_data, *_params);
        }
    }

    void MCMC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Pre-allocate tensor capacity if max_cap is specified
        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data->size();
            LOG_INFO("Pre-allocating capacity for {} Gaussians (current size: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            try {
                // ELIMINATE ALL POOL ALLOCATIONS: Replace pool-allocated parameters with direct cudaMalloc versions
                LOG_DEBUG("  Replacing pool-allocated parameters with direct cudaMalloc versions:");

                // When init_model_from_pointcloud was called with capacity = max_cap, every
                // param is already direct-allocated at that capacity. Re-allocating would briefly
                // hold both old and new buffers (≈2× peak) before the cuda caching allocator
                // releases the freed chunk — so only replace if the param's capacity is actually
                // below the target.
                auto ensure_capacity_direct = [capacity](Tensor& param) {
                    LFS_ASSERT_MSG(param.dtype() == DataType::Float32,
                                   "MCMC training parameter must be Float32");
                    if (param.capacity() >= capacity)
                        return;
                    // GUI exportable tensors grow with live N.
                    if (param.is_external_storage())
                        return;
                    auto new_param = Tensor::zeros_direct(param.shape(), capacity);
                    new_param.copy_from(param);
                    param = std::move(new_param);
                };

                const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
                const bool shN_quantized = _splat_data->shN_value_quantized();
                auto ensure_shN_capacity_direct = [capacity, layout_rest, shN_quantized](Tensor& param) {
                    const auto expected_dtype = shN_quantized ? DataType::Float16 : DataType::Float32;
                    LFS_ASSERT_MSG(param.dtype() == expected_dtype,
                                   "MCMC shN dtype does not match its storage representation");
                    const size_t required_capacity =
                        shN_quantized
                            ? lfs::core::sh_value_quant::sh_value_u16_count(capacity, layout_rest)
                            : lfs::core::sh_swizzled_float_count(capacity, layout_rest);
                    if (param.capacity() >= required_capacity)
                        return;
                    if (param.is_external_storage())
                        return;
                    auto new_param = Tensor::zeros_direct(
                        param.shape(), required_capacity, Device::CUDA, param.dtype());
                    new_param.copy_from(param);
                    param = std::move(new_param);
                };

                ensure_capacity_direct(_splat_data->means());
                ensure_capacity_direct(_splat_data->sh0());
                if (layout_rest > 0 && _splat_data->shN().is_valid() && _splat_data->shN().numel() > 0) {
                    ensure_shN_capacity_direct(_splat_data->shN());
                }
                ensure_capacity_direct(_splat_data->scaling_raw());
                ensure_capacity_direct(_splat_data->rotation_raw());
                ensure_capacity_direct(_splat_data->opacity_raw());

                // noise is generated inside inject_noise_kernel (no buffer).

                LOG_INFO("Pre-allocated capacity: {}/{} Gaussians ({:.1f}%)",
                         current_size, capacity, 100.0f * current_size / capacity);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to pre-allocate capacity: {}. Continuing without pre-allocation.", e.what());
            }
        }

        // Convert shN to pad-dropped u16 after reserving float capacity.
        lfs::training::sh_value::apply_shN_value_quant(*_splat_data);

        _n_max = 51;
#if LFS_TENSOR_CUDA
        mcmc::init_relocation_coefficients(_n_max);
#endif

        if (_params->max_cap > 0) {
            _ones_int32 = Tensor::ones({static_cast<size_t>(_params->max_cap)}, Device::CUDA, DataType::Int32);
        }

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        ensure_densification_info_shape();
        if (_splat_data->has_deleted_mask()) {
            ensure_deleted_mask_size(*_splat_data);
        }
        _error_score_windows = 0;

        LOG_INFO("MCMC strategy initialized with {} Gaussians", _splat_data->size());
    }

    void MCMC::permute_gaussian_rows(const lfs::core::Tensor& perm) {
#if !LFS_TENSOR_CUDA
        if (!_error_score_max.is_valid() || !_error_score_max.numel())
            return;
        auto prepared = _error_score_max.index_select(0, perm);
        (void)prepared.data_ptr();
        core::with_idle_vulkan_device([](const auto&) {});
        _error_score_max = {};
        _error_score_max = std::move(prepared);
#else
        morton::permute_row_tensor(_error_score_max, perm);
#endif
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t MCMC_MAGIC = 0x4C464D43; // "LFMC"
        constexpr uint32_t MCMC_VERSION = 2;
    } // namespace

    void MCMC::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&MCMC_MAGIC), sizeof(MCMC_MAGIC));
        os.write(reinterpret_cast<const char*>(&MCMC_VERSION), sizeof(MCMC_VERSION));

        // Serialize optimizer state
        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        // Serialize scheduler state
        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        os.write(reinterpret_cast<const char*>(&_current_iteration), sizeof(_current_iteration));
        os.write(reinterpret_cast<const char*>(&_error_score_windows), sizeof(_error_score_windows));
        os << _error_score_max;
        LOG_DEBUG("Serialized MCMC strategy");
    }

    void MCMC::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "MCMC magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "MCMC version");

        if (magic != MCMC_MAGIC) {
            throw std::runtime_error("Invalid MCMC checkpoint: wrong magic");
        }
        if (version < 1 || version > MCMC_VERSION) {
            throw std::runtime_error("Unsupported MCMC checkpoint version: " + std::to_string(version));
        }

        // Deserialize optimizer state
        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_optimizer, sizeof(has_optimizer), "MCMC optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer))
            throw std::runtime_error("Invalid MCMC checkpoint: optimizer flag/state mismatch");
        if (has_optimizer) {
            _optimizer->deserialize(is);
        }

        // Deserialize scheduler state
        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_scheduler, sizeof(has_scheduler), "MCMC scheduler flag");
        if (has_scheduler > 1 || (has_scheduler && !_scheduler))
            throw std::runtime_error("Invalid MCMC checkpoint: scheduler flag/state mismatch");
        if (has_scheduler) {
            _scheduler->deserialize(is);
        }

        if (version >= 2) {
            int iteration = 0, windows = 0;
            lfs::core::serialization_detail::read_exact(is, &iteration, sizeof(iteration), "MCMC iteration");
            lfs::core::serialization_detail::read_exact(is, &windows, sizeof(windows), "MCMC refinement windows");
            lfs::core::Tensor priorities;
            is >> priorities;
            if (iteration < 0 || windows < 0 || windows > 1 ||
                !priorities.is_valid() || priorities.ndim() != 1 ||
                priorities.dtype() != lfs::core::DataType::Float32 ||
                (priorities.numel() != 0 && priorities.numel() != static_cast<size_t>(_splat_data->size())) ||
                (priorities.numel() && (!priorities.isfinite().all().cpu().item<bool>() || priorities.min().cpu().item<float>() < 0)))
                throw std::runtime_error("Invalid MCMC checkpoint refinement state");
            _error_score_max = priorities.to(_splat_data->means().device());
            _current_iteration = iteration;
            _error_score_windows = windows;
        } else {
            // v1 omitted refinement priorities; retain compatibility, but cannot
            // reproduce its unsaved partial refinement window.
            _error_score_max.zero_();
            _error_score_windows = 0;
            _current_iteration = 0;
        }
        LOG_DEBUG("Deserialized MCMC strategy");
    }

    bool MCMC::can_adopt_checkpoint_state(const IStrategy& loaded) const noexcept {
        const auto* source = dynamic_cast<const MCMC*>(&loaded);
        return source && static_cast<bool>(_optimizer) == static_cast<bool>(source->_optimizer) &&
               static_cast<bool>(_scheduler) == static_cast<bool>(source->_scheduler);
    }

    void MCMC::adopt_checkpoint_state(IStrategy& loaded) noexcept {
        auto& source = checked_checkpoint_source<MCMC>(loaded);
        if (_optimizer)
            _optimizer->adopt_checkpoint_state(*source._optimizer);
        if (_scheduler)
            _scheduler->adopt_checkpoint_state(*source._scheduler);
        _params.swap(source._params);
        std::swap(_n_max, source._n_max);

        std::swap(_ones_int32, source._ones_int32);
        std::swap(_error_score_max, source._error_score_max);
        std::swap(_error_score_windows, source._error_score_windows);
        std::swap(_current_iteration, source._current_iteration);
    }

    void MCMC::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

} // namespace lfs::training
