/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lfs::training::morton {
    using namespace core;
    void permute_row_tensor(Tensor& tensor, const Tensor& perm, const int axis) {
        if (!tensor.is_valid() || !tensor.numel() || !perm.is_valid() || !perm.numel())
            return;
        const size_t n = perm.numel();
        if (axis != 0) {
            tensor.copy_from(tensor.index_select(axis, perm));
        } else if (tensor.ndim() && tensor.shape()[0] >= n) {
            auto prefix = tensor.slice(0, 0, n);
            prefix.copy_from(prefix.index_select(0, perm));
        }
    }
    void permute_shN(SplatData& splat, const Tensor& perm, cudaStream_t stream) {
        if (stream)
            throw std::invalid_argument("Vulkan SH reorder does not accept a CUDA stream");
        sh_value::compact_shN_gather(splat, perm, splat.size(), splat.means().capacity());
    }
    ReorderResult apply_morton_reorder(SplatData& splat, AdamOptimizer* optimizer, cudaStream_t stream,
                                      const std::function<void(const Tensor&)>& before_commit) {
        if (stream)
            throw std::invalid_argument("Vulkan Morton reorder does not accept a CUDA stream");
        if (splat.has_frozen_ranges() || splat.size() == 0)
            return {};
        LiveModelMutationGuard guard("morton_reorder");
        const size_t n = splat.size();
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Morton row count exceeds Int32");
        // Periodic host key sorting preserves all 30 key bits. The Tensor sort
        // currently accepts Float32 only, which would merge distinct Morton cells.
        const auto means = splat.means().cpu().to_vector();
        std::array<float, 3> low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(means[i * 3 + c]))
                    throw std::invalid_argument("Morton reorder requires finite means");
                low[c] = std::min(low[c], means[i * 3 + c]);
                high[c] = std::max(high[c], means[i * 3 + c]);
            }
        std::vector<uint32_t> codes(n, 0);
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                const float multiplier = high[c] == low[c] ? 0.f : 1024.f / (high[c] - low[c]);
                const auto coordinate = static_cast<uint32_t>(std::clamp((means[i * 3 + c] - low[c]) * multiplier, 0.f, 1023.f));
                for (int bit = 0; bit < 10; ++bit)
                    codes[i] |= ((coordinate >> bit) & 1u) << (3 * bit + c);
            }
        std::stable_sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) { return codes[a] < codes[b]; });
        GpuBackendScope backend(gpu_backend_of(splat.means()).value_or(default_gpu_backend()));
        auto perm = Tensor::from_vector(indices, TensorShape{n}, splat.means().device()).to(DataType::Int64);
        auto prepared = splat.clone_async(nullptr, true);
        auto prepared_optimizer = optimizer ? optimizer->clone_for_model(prepared) : nullptr;
        for (Tensor* parameter : {&prepared.means(), &prepared.sh0(), &prepared.scaling_raw(), &prepared.rotation_raw(), &prepared.opacity_raw()})
            parameter->copy_from(parameter->index_select(0, perm));
        permute_shN(prepared, perm);
        permute_row_tensor(prepared._densification_info, perm, 1);
        permute_row_tensor(prepared._max_screen_share, perm);
        if (prepared.has_deleted_mask()) {
            permute_row_tensor(prepared.deleted(), perm);
            prepared.notify_deleted_mask_changed();
        }
        if (prepared_optimizer) {
            prepared_optimizer->permute_rows(perm);
            prepared_optimizer->refresh_screen_share_buffer();
        }
        with_idle_vulkan_device([](const auto&) {});
        if (before_commit)
            before_commit(perm);
        splat.adopt_training_update(std::move(prepared));
        if (optimizer)
            optimizer->adopt_training_update(*prepared_optimizer);
        return {true, std::move(perm)};
    }
} // namespace lfs::training::morton
