/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "joint_adam_tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor/internal/tensor_index_validation.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include <limits>
#include <optional>
#include <stdexcept>

namespace lfs::training::joint_adam {
    using core::DataType;
    using core::Tensor;
    using core::TensorShape;

    namespace {
        size_t validate_storage(const Tensor& packed, const Tensor& bounds,
                                const int bits, const size_t cells_per_block) {
            if ((bits != 8 && bits != 16) || cells_per_block == 0 ||
                !packed.is_valid() || !bounds.is_valid() || !packed.is_contiguous() ||
                !bounds.is_contiguous() || packed.dtype() != DataType::UInt8 ||
                bounds.dtype() != DataType::Float32 || bounds.ndim() != 2 || bounds.shape()[1] != 4 ||
                packed.device() != bounds.device() || core::gpu_backend_of(packed) != core::gpu_backend_of(bounds))
                throw std::invalid_argument("Invalid joint Adam tensor storage");
            const size_t bpc = bytes_per_cell(bits);
            const size_t cells = packed.numel() / bpc;
            if (packed.numel() % bpc || cells == 0 ||
                bounds.shape()[0] != (cells - 1) / cells_per_block + 1)
                throw std::invalid_argument("Joint Adam packed cells and bounds disagree");
            if (cells_per_block > std::numeric_limits<size_t>::max() / bounds.shape()[0])
                throw std::invalid_argument("Joint Adam padded layout overflows size_t");
            return cells;
        }
    } // namespace

    TensorMoments decode_tensor(const Tensor& packed, const Tensor& bounds,
                                const int bits, const size_t cells_per_block, const Tensor* cell_indices) {
        const size_t stored_cells = validate_storage(packed, bounds, bits, cells_per_block);
        const size_t cells = cell_indices ? cell_indices->numel() : stored_cells;
        const size_t blocks = bounds.shape()[0];
        const size_t bpc = bytes_per_cell(bits);
        Tensor selected_bytes, selected_bounds;
        if (cell_indices) {
            core::internal::assert_index_tensor(*cell_indices, stored_cells, "Adam cell gather", true);
            const auto indices = cell_indices->to(DataType::Int32);
            const auto divisor = Tensor::full({1}, float(cells_per_block), packed.device(), DataType::Int32);
            const Tensor block_ids = ((indices - indices.mod(divisor)) / divisor).to(DataType::Int32);
            selected_bytes = packed.reshape(TensorShape{stored_cells, bpc}).index_select(0, indices);
            selected_bounds = bounds.index_select(0, block_ids);
        }
        const Tensor bytes = (cell_indices ? selected_bytes : packed.reshape(TensorShape{cells, bpc})).to(DataType::Float32);
        const auto byte = [&](size_t column) { return bytes.slice(1, column, column + 1).reshape(TensorShape{cells}); };
        const Tensor qu = bits == 8 ? byte(0) : byte(0) + byte(1) * 256.0f;
        const Tensor qs = bits == 8 ? byte(1) : byte(2) + byte(3) * 256.0f;
        const auto bound = [&](size_t column) {
            if (cell_indices)
                return selected_bounds.slice(1, column, column + 1).contiguous().reshape(TensorShape{cells});
            return bounds.slice(1, column, column + 1).expand(TensorShape{blocks, cells_per_block}).contiguous().reshape(TensorShape{blocks * cells_per_block}).slice(0, 0, cells);
        };
        const Tensor umin = bound(0), umax = bound(1), smin = bound(2), smax = bound(3);
        const float inverse_max = 1.0f / static_cast<float>((1 << bits) - 1);
        const Tensor u = umin + (umax - umin) * (qu * inverse_max);
        const Tensor s = smin + (smax - smin) * (qs * inverse_max);
        // exp(s)-1 loses small moments to cancellation. This short Taylor series
        // covers the codec's expm1 branch (s <= .118); the remainder is < 7e-11.
        const Tensor small = s * (((((s / 720.0f + 1.0f / 120.0f) * s + 1.0f / 24.0f) * s + 1.0f / 6.0f) * s + .5f) * s + 1.0f);
        const Tensor root = Tensor::where(s.le(.118f), small, s.exp() - 1.0f) * kEps;
        Tensor second = root * root;
        Tensor first = Tensor::where(second.eq(0.0f), Tensor::zeros_like(second), u * (root + kEps));
        return {std::move(first), std::move(second)};
    }

    void encode_tensor(const TensorMoments& moments, Tensor& packed, Tensor& bounds,
                       const int bits, const size_t cells_per_block, const Tensor* valid_cells) {
        const size_t cells = validate_storage(packed, bounds, bits, cells_per_block);
        if (valid_cells && (!valid_cells->is_valid() || valid_cells->numel() != cells ||
                            valid_cells->dtype() != DataType::Bool || valid_cells->device() != packed.device() ||
                            core::gpu_backend_of(*valid_cells) != core::gpu_backend_of(packed)))
            throw std::invalid_argument("Invalid joint Adam cell validity mask");
        for (const auto* moment : {&moments.first, &moments.second}) {
            if (!moment->is_valid() || moment->numel() != cells || moment->dtype() != DataType::Float32 ||
                moment->device() != packed.device() || core::gpu_backend_of(*moment) != core::gpu_backend_of(packed))
                throw std::invalid_argument("Joint Adam moments do not match packed storage");
        }
        std::optional<core::GpuBackendScope> backend;
        if (const auto kind = core::gpu_backend_of(packed))
            backend.emplace(*kind);
        const size_t blocks = bounds.shape()[0], padded = blocks * cells_per_block;
        const Tensor root = moments.second.reshape(TensorShape{cells}).clamp_min(0.0f).sqrt();
        const Tensor u = moments.first.reshape(TensorShape{cells}) / (root + kEps);
        const Tensor s = (root / kEps).log1p();
        Tensor valid;
        if (valid_cells) {
            valid = Tensor::zeros(TensorShape{padded}, packed.device(), DataType::Bool);
            valid.slice(0, 0, cells).copy_from(valid_cells->reshape(TensorShape{cells}));
        }
        const auto quantize = [&](const Tensor& values, const size_t column) {
            auto padded_values = Tensor::full(TensorShape{padded}, std::numeric_limits<float>::infinity(), packed.device());
            padded_values.slice(0, 0, cells).copy_from(values);
            if (valid.is_valid())
                padded_values.masked_fill_(valid.logical_not(), std::numeric_limits<float>::infinity());
            Tensor low = padded_values.reshape(TensorShape{blocks, cells_per_block}).min(1, true);
            if (padded > cells)
                padded_values.slice(0, cells, padded).fill_(-std::numeric_limits<float>::infinity());
            if (valid.is_valid())
                padded_values.masked_fill_(valid.logical_not(), -std::numeric_limits<float>::infinity());
            Tensor high = padded_values.reshape(TensorShape{blocks, cells_per_block}).max(1, true);
            // A block with no live cells keeps the codec's exact zero state.
            low = Tensor::where(low.isfinite(), low, Tensor::zeros_like(low));
            high = Tensor::where(high.isfinite(), high, Tensor::zeros_like(high));
            bounds.slice(1, column, column + 1).copy_from(low);
            bounds.slice(1, column + 1, column + 2).copy_from(high);
            Tensor q = ((padded_values.reshape(TensorShape{blocks, cells_per_block}) - low) /
                            (high - low).clamp_min(kEps) * static_cast<float>((1 << bits) - 1) +
                        .5f)
                           .floor()
                           .clamp(0.0f, static_cast<float>((1 << bits) - 1));
            return q.reshape(TensorShape{padded}).slice(0, 0, cells);
        };
        const Tensor qu = quantize(u, 0), qs = quantize(s, 2);
        Tensor encoded;
        if (bits == 8) {
            encoded = Tensor::stack({qu, qs}, 1).to(DataType::UInt8);
        } else {
            const Tensor uh = (qu / 256.0f).floor(), sh = (qs / 256.0f).floor();
            encoded = Tensor::stack({qu - uh * 256.0f, uh, qs - sh * 256.0f, sh}, 1).to(DataType::UInt8);
        }
        packed.copy_from(encoded.reshape(packed.shape()));
    }
} // namespace lfs::training::joint_adam
