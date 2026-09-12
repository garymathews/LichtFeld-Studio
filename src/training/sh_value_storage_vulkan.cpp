/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor/internal/tensor_index_validation.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include <algorithm>
#include <stdexcept>

namespace lfs::training::sh_value {
    using namespace core;
    namespace {
        Tensor block_indices(const Tensor& indices, int block_size) {
            const auto divisor = Tensor::full({1}, float(block_size), indices.device(), DataType::Int32);
            const auto rows = indices.dtype() == DataType::Int32 ? indices : indices.to(DataType::Int32);
            // Subtract the remainder before conversion: multiples of 256 (or
            // 32) are exact FP32 throughout the supported SH index range.
            return ((rows - rows.mod(divisor)) / divisor).to(DataType::Int32);
        }
        Tensor decode_selected(const SplatData& splat, const Tensor& indices) {
            const size_t count = indices.numel(), rest = splat.max_sh_coeffs_rest(), cells = rest * 3;
            const auto integer = [&](int value) { return Tensor::full({1}, float(value), indices.device(), DataType::Int32); };
            const auto rows = indices.to(DataType::Int32).reshape(TensorShape{count, 1});
            const auto lane = rows.mod(integer(32));
            const auto columns = Tensor::arange(float(cells)).to(DataType::Int32).reshape(TensorShape{1, cells});
            Tensor offsets;
            if (splat.shN_value_quantized()) {
                offsets = (rows - lane) * integer(cells) + lane + columns * integer(32);
                const auto bytes = splat.shN().view_as(DataType::UInt8).reshape({-1, 2}).index_select(0, offsets.reshape({-1})).to(DataType::Float32);
                const auto codes = (bytes.slice(1, 0, 1) + bytes.slice(1, 1, 2) * 256.f).reshape(TensorShape{count, cells});
                const auto bounds = splat.shN_value_bounds().reshape({-1, 2}).index_select(0, block_indices(indices, 256));
                const auto lo = bounds.slice(1, 0, 1), hi = bounds.slice(1, 1, 2);
                return (lo + (hi - lo) * (codes * (1.f / 65535.f))).reshape(TensorShape{count, rest, 3});
            }
            const size_t attributes = sh_float4_slots_for_rest(rest) * 4;
            const auto component = columns.mod(integer(4));
            offsets = (rows - lane) * integer(attributes) + lane * integer(4) + (columns - component) * integer(32) + component;
            return splat.shN().reshape({-1}).index_select(0, offsets.reshape({-1})).to(DataType::Float32).reshape(TensorShape{count, rest, 3});
        }
        std::pair<Tensor, Tensor> prepare_storage(const SplatData& splat, size_t n, size_t capacity) {
            const auto rest = splat.max_sh_coeffs_rest();
            const bool quantized = splat.shN_value_quantized();
            const auto cells = [&](size_t rows) { return quantized ? sh_value_quant::sh_value_u16_count(rows, rest) : sh_swizzled_float_count(rows, rest); };
            auto values = Tensor::zeros_direct({cells(n)}, cells(std::max(n, capacity)), splat.shN().device(), splat.shN().dtype());
            Tensor bounds;
            if (quantized)
                bounds = Tensor::zeros_direct({sh_value_quant::n_bounds_for_prims(n) * 2}, sh_value_quant::n_bounds_for_prims(std::max(n, capacity)) * 2, values.device());
            return {std::move(values), std::move(bounds)};
        }
        void encode_block(const Tensor& canonical, size_t offset, uint32_t rest, std::pair<Tensor, Tensor>& storage) {
            const auto swizzled = reorder_sh_to_swizzled(canonical, canonical.shape()[0], rest, rest);
            if (storage.second.is_valid()) {
                Tensor codes, bounds;
                sh_value_quant::encode_shN_float4_to_u16_tensor(swizzled, canonical.shape()[0], sh_float4_slots_for_rest(rest), rest * 3, codes, bounds);
                storage.first.slice(0, offset * rest * 3, offset * rest * 3 + codes.numel()).copy_from(codes.reshape({-1}));
                storage.second.slice(0, (offset / 256) * 2, (offset / 256) * 2 + bounds.numel()).copy_from(bounds.reshape({-1}));
            } else {
                const size_t start = offset * sh_float4_slots_for_rest(rest) * 4;
                storage.first.slice(0, start, start + swizzled.numel()).copy_from(swizzled.to(storage.first.dtype()));
            }
        }
        void publish_storage(SplatData& splat, std::pair<Tensor, Tensor>&& storage) {
            with_idle_vulkan_device([](const auto&) {});
            splat.shN() = {};
            splat.shN_value_bounds() = {};
            splat.shN() = std::move(storage.first);
            splat.shN_value_bounds() = std::move(storage.second);
        }
    } // namespace
    bool apply_shN_value_quant(SplatData& splat) { return splat.apply_shN_value_quant(); }
    bool ensure_shN_fp32_for_mutation(SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        if (!splat.shN().is_valid() || splat.shN().dtype() != DataType::Float16 || splat.shN().numel() == 0)
            return false;
        auto canonical = splat.shN_canonical();
        // Cast IEEE half explicitly too: set_from_canonical can otherwise retain half storage.
        auto values = reorder_sh_to_swizzled(canonical, splat.size(), splat.max_sh_coeffs_rest(), splat.max_sh_coeffs_rest());
        const size_t capacity = sh_swizzled_float_count(std::max<size_t>(splat.size(), splat.means().capacity()), splat.max_sh_coeffs_rest());
        auto storage = Tensor::zeros_direct(values.shape(), capacity, values.device());
        storage.copy_from(values);
        splat.shN() = std::move(storage);
        splat.shN_value_bounds() = {};
        return true;
    }
    bool commit_shN_after_mutation(SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        return splat.apply_shN_value_quant();
    }
    void gather_shN_to_canonical(SplatData& splat, const Tensor& indices, Tensor& dest, size_t source_n) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        if (!indices.is_valid() || indices.numel() == 0 || splat.max_sh_coeffs_rest() == 0)
            return;
        const size_t n = source_n ? source_n : splat.size();
        internal::assert_index_tensor(indices, n, "SH gather", true);
        const auto selected = indices.device() == splat.shN().device() ? indices : indices.to(splat.shN().device());
        auto gathered = Tensor::empty({indices.numel(), splat.max_sh_coeffs_rest(), 3}, splat.shN().device());
        for (size_t offset = 0; offset < indices.numel(); offset += 4096) {
            const size_t end = std::min(offset + 4096, indices.numel());
            gathered.slice(0, offset, end).copy_from(decode_selected(splat, selected.slice(0, offset, end)));
            if (indices.numel() > 4096)
                with_idle_vulkan_device([](const auto&) {});
        }
        if (dest.is_valid() && dest.shape() == gathered.shape())
            dest.copy_from(gathered);
        else {
            dest = {};
            dest = std::move(gathered);
        }
    }
    void scatter_canonical_into_shN(SplatData& splat, const Tensor& indices, const Tensor& source) {
        ShNMutationBatch batch(splat);
        batch.scatter(indices, source);
        batch.flush();
    }
    void append_canonical_to_shN(SplatData& splat, const Tensor& source, size_t offset) {
        if (source.is_valid() && source.numel() &&
            (offset > size_t(splat.size()) || source.shape()[0] != size_t(splat.size()) - offset))
            throw std::invalid_argument("SH append must cover the newly grown model suffix");
        ShNMutationBatch batch(splat);
        batch.append(source, offset);
        batch.flush();
    }

    void zero_shN_at_indices(SplatData& splat, const Tensor& indices) {
        if (!indices.is_valid() || indices.numel() == 0 || splat.max_sh_coeffs_rest() == 0)
            return;
        ShNMutationBatch batch(splat);
        batch.zero(indices);
        batch.flush();
    }
    struct ShNMutationBatch::Impl {
        struct Op {
            enum class Kind { Scatter,
                              Zero,
                              Append } kind = Kind::Scatter;
            Tensor dest_indices;
            Tensor row_to_source;
            Tensor canonical;
            std::size_t dest_offset = 0;
        };

        core::SplatData* splat = nullptr;
        std::vector<Op> ops;
        bool flushed = true;
    };

    ShNMutationBatch::ShNMutationBatch(core::SplatData& splat)
        : impl_(std::make_unique<Impl>()) {
        impl_->splat = &splat;
    }

    ShNMutationBatch::ShNMutationBatch(ShNMutationBatch&&) noexcept = default;
    ShNMutationBatch& ShNMutationBatch::operator=(ShNMutationBatch&&) noexcept = default;

    ShNMutationBatch::~ShNMutationBatch() {
        if (!impl_ || impl_->flushed || impl_->ops.empty() || !impl_->splat) {
            return;
        }
        try {
            flush();
        } catch (const std::exception& e) {
            try {
                LOG_ERROR("ShNMutationBatch: flush failed during scope exit: {}", e.what());
            } catch (...) {
            }
        } catch (...) {
            try {
                LOG_ERROR("ShNMutationBatch: flush failed during scope exit with unknown exception");
            } catch (...) {
            }
        }
    }

    void ShNMutationBatch::scatter(const Tensor& dest_indices, const Tensor& src_canonical) {
        if (!impl_ || !dest_indices.is_valid() || !dest_indices.numel() ||
            !src_canonical.is_valid() || !src_canonical.numel() || !impl_->splat->max_sh_coeffs_rest())
            return;
        impl_->ops.push_back({.kind = Impl::Op::Kind::Scatter, .dest_indices = dest_indices, .canonical = src_canonical});
        impl_->flushed = false;
    }

    void ShNMutationBatch::zero(const Tensor& dest_indices) {
        if (!impl_ || !dest_indices.is_valid() || !dest_indices.numel() || !impl_->splat->max_sh_coeffs_rest())
            return;
        impl_->ops.push_back({.kind = Impl::Op::Kind::Zero, .dest_indices = dest_indices});
        impl_->flushed = false;
    }

    void ShNMutationBatch::append(const Tensor& src_canonical, std::size_t dest_offset) {
        if (!impl_ || !src_canonical.is_valid() || !src_canonical.numel() || !impl_->splat->max_sh_coeffs_rest())
            return;
        impl_->ops.push_back({.kind = Impl::Op::Kind::Append, .canonical = src_canonical, .dest_offset = dest_offset});
        impl_->flushed = false;
    }

    void ShNMutationBatch::flush() {
        if (!impl_ || impl_->flushed || impl_->ops.empty())
            return;
        // A failed explicit flush must never be retried by the destructor after
        // its one-shot allocation fault (or other transient error) has cleared.
        impl_->flushed = true;
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        auto& splat = *impl_->splat;
        const size_t n = splat.size(), rest = splat.max_sh_coeffs_rest();
        size_t old_n = n;
        auto touched = Tensor::zeros({sh_value_quant::n_bounds_for_prims(n)}, splat.shN().device(), DataType::Int32);
        for (auto& op : impl_->ops) {
            if (op.kind == Impl::Op::Kind::Append) {
                if (op.dest_offset > n || op.canonical.shape()[0] > n - op.dest_offset)
                    throw std::invalid_argument("SH append exceeds the newly grown model suffix");
                old_n = std::min(old_n, op.dest_offset);
                touched.slice(0, op.dest_offset / 256, (op.dest_offset + op.canonical.shape()[0] + 255) / 256).fill_(1);
            } else {
                internal::assert_index_tensor(op.dest_indices, n, "SH scatter", true);
                auto selected = (op.dest_indices.device() == splat.shN().device() ? op.dest_indices : op.dest_indices.to(splat.shN().device())).to(DataType::Int32);
                op.dest_indices = {};
                op.dest_indices = std::move(selected);
                touched.index_fill_(0, block_indices(op.dest_indices, 256), 1);
                // One GPU lookup per operation replaces a full selection scan
                // for every quantization block. Only block metadata reaches CPU.
                op.row_to_source = Tensor::full({n}, -1.f, splat.shN().device(), DataType::Int32);
                if (op.kind == Impl::Op::Kind::Zero)
                    op.row_to_source.index_fill_(0, op.dest_indices, 0.f);
                else
                    op.row_to_source.index_put_(op.dest_indices, (Tensor::full({op.dest_indices.numel()}, 1.f, splat.shN().device(), DataType::Int32).cumsum(0) - Tensor::full({1}, 1.f, splat.shN().device(), DataType::Int32)));
            }
            if (op.kind != Impl::Op::Kind::Zero &&
                (op.canonical.ndim() != 3 || op.canonical.shape()[1] != rest || op.canonical.shape()[2] != 3 ||
                 op.canonical.dtype() != DataType::Float32 || op.canonical.device() != splat.shN().device() ||
                 (op.kind == Impl::Op::Kind::Scatter && op.canonical.shape()[0] != op.dest_indices.numel())))
                throw std::invalid_argument("SH mutation requires matching canonical Float32 rows");
        }
        auto storage = prepare_storage(splat, n, std::max(splat.means().capacity(), n));
        storage.first.slice(0, 0, splat.shN().numel()).copy_from(splat.shN());
        if (storage.second.is_valid())
            storage.second.slice(0, 0, splat.shN_value_bounds().numel()).copy_from(splat.shN_value_bounds());
        // Read back block IDs, never the primitive index array. Untouched codes
        // and bounds are copied verbatim and incur no requantization drift.
        const auto blocks = touched.nonzero().reshape({-1}).cpu().to_vector_int64();
        size_t pending_blocks = 0;
        for (const auto block : blocks) {
            const size_t start = size_t(block) * 256, end = std::min(start + 256, n);
            auto canonical = Tensor::zeros({end - start, rest, 3}, splat.shN().device());
            if (start < old_n) {
                const size_t keep = std::min(end, old_n) - start;
                const auto rows = Tensor::arange(float(keep)).to(DataType::Int32) + Tensor::full({1}, float(start), splat.shN().device(), DataType::Int32);
                canonical.slice(0, 0, keep).copy_from(decode_selected(splat, rows));
            }
            for (const auto& op : impl_->ops) {
                if (op.kind == Impl::Op::Kind::Append) {
                    const size_t lo = std::max(start, op.dest_offset), hi = std::min(end, op.dest_offset + op.canonical.shape()[0]);
                    if (lo < hi)
                        canonical.slice(0, lo - start, hi - start).copy_from(op.canonical.slice(0, lo - op.dest_offset, hi - op.dest_offset));
                } else {
                    const auto sources = op.row_to_source.slice(0, start, end);
                    const auto mask = sources.ge(0.f).reshape(core::TensorShape{end - start, 1, 1}).broadcast_to(canonical.shape());
                    if (op.kind == Impl::Op::Kind::Zero)
                        canonical.masked_fill_(mask, 0.f);
                    else {
                        const auto values = op.canonical.index_select(0, sources.clamp_min(0.f).to(DataType::Int32));
                        canonical.copy_from(Tensor::where(mask, values, canonical));
                    }
                }
            }
            encode_block(canonical, start, rest, storage);
            if (++pending_blocks == 16) {
                with_idle_vulkan_device([](const auto&) {});
                pending_blocks = 0;
            }
        }
        publish_storage(splat, std::move(storage));
        impl_->ops.clear();
        impl_->flushed = true;
    }
    void compact_shN_gather(SplatData& splat, const Tensor& indices, size_t source_n, size_t capacity) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        if (splat.max_sh_coeffs_rest() == 0)
            return;
        internal::assert_index_tensor(indices, source_n, "SH compact", true);
        const auto selected = indices.device() == splat.shN().device() ? indices : indices.to(splat.shN().device());
        auto storage = prepare_storage(splat, indices.numel(), std::max(capacity, indices.numel()));
        for (size_t offset = 0; offset < indices.numel(); offset += 4096) {
            const size_t end = std::min(offset + 4096, indices.numel());
            encode_block(decode_selected(splat, selected.slice(0, offset, end)), offset, splat.max_sh_coeffs_rest(), storage);
            with_idle_vulkan_device([](const auto&) {});
        }
        publish_storage(splat, std::move(storage));
    }
    ShNCommitGuard::~ShNCommitGuard() noexcept {
        if (!expanded_ || !splat_) {
            return;
        }
        try {
            (void)commit_shN_after_mutation(*splat_);
        } catch (const std::exception& e) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit: {}", site_, e.what());
            } catch (...) {
            }
        } catch (...) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit with unknown exception", site_);
            } catch (...) {
            }
        }
    }

} // namespace lfs::training::sh_value
