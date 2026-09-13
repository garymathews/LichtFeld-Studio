/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/sh_layout.cuh"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <climits>

namespace lfs::core {
    Tensor reorder_sh_to_swizzled(const Tensor& canonical, const size_t n,
                                  const uint32_t source_rest, const uint32_t layout_rest) {
        if (n > (static_cast<size_t>(INT_MAX) / 48 / 32) * 32 || !canonical.is_valid() || source_rest > layout_rest || layout_rest > kShMaxCoeffsRest ||
            (canonical.dtype() != DataType::Float32 && canonical.dtype() != DataType::Float16) ||
            canonical.numel() != n * source_rest * kShChannels) {
            throw TensorError("SH packing requires canonical floating-point [N,K,3] data and a sufficient layout");
        }
        const size_t slots = sh_float4_slots_for_rest(layout_rest);
        auto output = internal::allocate_like(canonical, {sh_swizzled_float_count(n, layout_rest)}, canonical.dtype());
        output.zero_();
        if (n == 0 || slots == 0 || source_rest == 0)
            return output;
#if LFS_TENSOR_CUDA
        if (gpu_backend_of(canonical) == GpuBackend::CUDA && canonical.dtype() == DataType::Float32) {
            auto source = canonical.contiguous();
            pin_operands({&source, &output});
            const auto stream = prepare_inputs_for_stream({&source, &output}, output.stream());
            reorder_sh_to_swizzled(source.ptr<float>(), output.ptr<float>(), n, source_rest, layout_rest, stream);
            source.record_stream(stream);
            output.record_stream(stream);
            output.set_stream(stream);
            return output;
        }
#endif
        const size_t blocks = sh_swizzled_block_count(n);
        auto rows = internal::allocate_like(canonical, {blocks * kShReorderSize, slots * 4}, canonical.dtype());
        rows.zero_();
        rows.slice(0, 0, n).slice(1, 0, source_rest * 3).copy_from(canonical.contiguous().reshape(TensorShape{n, source_rest * 3}));
        output.copy_from(rows.reshape(TensorShape{blocks, kShReorderSize, slots, 4})
                             .permute({0, 2, 1, 3})
                             .contiguous()
                             .reshape(output.shape()));
        return output;
    }

    Tensor undo_reorder_sh_from_swizzled(const Tensor& swizzled, const size_t n,
                                         const uint32_t destination_rest, const uint32_t layout_rest) {
        const size_t required = sh_swizzled_float_count(n, layout_rest);
        if (n > (static_cast<size_t>(INT_MAX) / 48 / 32) * 32 || !swizzled.is_valid() || destination_rest > layout_rest || layout_rest > kShMaxCoeffsRest ||
            (swizzled.dtype() != DataType::Float32 && swizzled.dtype() != DataType::Float16) ||
            swizzled.numel() < required) {
            throw TensorError("SH unpacking requires floating-point swizzled data and a valid layout");
        }
        auto output = internal::allocate_like(swizzled, {n, destination_rest, kShChannels}, swizzled.dtype());
        if (n == 0 || destination_rest == 0)
            return output;
#if LFS_TENSOR_CUDA
        if (gpu_backend_of(swizzled) == GpuBackend::CUDA && swizzled.dtype() == DataType::Float32) {
            auto source = swizzled.contiguous();
            pin_operands({&source, &output});
            const auto stream = prepare_inputs_for_stream({&source, &output}, output.stream());
            undo_reorder_sh_from_swizzled(source.ptr<float>(), output.ptr<float>(), n, destination_rest, layout_rest, stream);
            source.record_stream(stream);
            output.record_stream(stream);
            output.set_stream(stream);
            return output;
        }
#endif
        const size_t blocks = sh_swizzled_block_count(n);
        const size_t slots = sh_float4_slots_for_rest(layout_rest);
        auto rows = swizzled.contiguous().reshape(TensorShape{swizzled.numel()}).slice(0, 0, required).reshape(TensorShape{blocks, slots, kShReorderSize, 4}).permute({0, 2, 1, 3}).contiguous().reshape(TensorShape{blocks * kShReorderSize, slots * 4});
        output.copy_from(rows.slice(0, 0, n).slice(1, 0, destination_rest * 3).contiguous().reshape(output.shape()));
        return output;
    }
    void shN_swizzled_copy_range(const Tensor& source, Tensor& destination,
                                 const size_t source_offset, const size_t count,
                                 const size_t destination_offset,
                                 const uint32_t source_rest, const uint32_t destination_rest) {
        if (!source.is_valid() || !destination.is_valid() || !source.is_contiguous() ||
            !destination.is_contiguous() || source_rest == 0 || source_rest > destination_rest ||
            destination_rest > kShMaxCoeffsRest || source.dtype() != destination.dtype() ||
            (source.dtype() != DataType::Float32 && source.dtype() != DataType::Float16) ||
            source.device() != destination.device() || gpu_backend_of(source) != gpu_backend_of(destination)) {
            throw TensorError("SH range copy requires compatible contiguous floating-point layouts/backends");
        }
        const size_t source_slots = sh_float4_slots_for_rest(source_rest);
        const size_t destination_slots = sh_float4_slots_for_rest(destination_rest);
        const size_t source_blocks = source.numel() / (source_slots * 128);
        const size_t destination_blocks = destination.numel() / (destination_slots * 128);
        if (source.numel() % (source_slots * 128) || destination.numel() % (destination_slots * 128) ||
            source_offset > source_blocks * 32 || count > source_blocks * 32 - source_offset ||
            destination_offset > destination_blocks * 32 || count > destination_blocks * 32 - destination_offset) {
            throw TensorError("SH range copy exceeds source or destination storage");
        }
        if (count == 0)
            return;
#if LFS_TENSOR_CUDA
        if (gpu_backend_of(source) == GpuBackend::CUDA && source.dtype() == DataType::Float32) {
            pin_operands({&source, &destination});
            const auto stream = prepare_inputs_for_stream({&source, &destination}, destination.stream());
            shN_swizzled_copy_range(source.ptr<float>(), destination.ptr<float>(), source_offset,
                                    count, destination_offset, source_rest, destination_rest, stream);
            source.record_stream(stream);
            destination.record_stream(stream);
            destination.set_stream(stream);
            return;
        }
#endif
        // Work directly on strided float4 lanes: at most 32 copies regardless of
        // model size, with no full-model canonical temporary. Clone when aliased
        // to preserve memmove semantics for overlapping ranges.
        auto input = source.data_ptr() == destination.data_ptr() ? source.clone() : source;
        auto src = input.reshape(TensorShape{source_blocks, source_slots, 32, 4});
        auto dst = destination.reshape(TensorShape{destination_blocks, destination_slots, 32, 4});
        for (size_t i = 0; i < std::min(count, size_t{32}); ++i) {
            const size_t rows = (count - i + 31) / 32;
            const size_t si = source_offset + i, di = destination_offset + i;
            auto target = dst.slice(0, di / 32, di / 32 + rows).slice(2, di % 32, di % 32 + 1);
            target.zero_();
            target.slice(1, 0, source_slots).copy_from(src.slice(0, si / 32, si / 32 + rows).slice(2, si % 32, si % 32 + 1));
            // A partial final float4 must not leak source padding into additional
            // destination coefficients.
            const size_t tail = source_rest * 3 % 4;
            if (tail)
                target.slice(1, source_slots - 1, source_slots).slice(3, tail, 4).zero_();
        }
    }
} // namespace lfs::core
