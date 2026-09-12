/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "internal/tensor_impl.hpp"
#include "internal/lazy_config.hpp"
#include "core/tensor/backend/kernel_contracts.hpp"
#if LFS_TENSOR_CUDA
#include "internal/stream_lifetime.hpp"
#endif
namespace lfs::core {
    static thread_local cudaStream_t tl_current_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return tl_current_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
#if LFS_TENSOR_CUDA
        if (stream) unretire_stream(stream);
#else
        LFS_ASSERT_MSG(stream == nullptr, "CUDA streams are unavailable in this build");
#endif
        tl_current_stream = stream;
    }

    cudaStream_t prepare_inputs_for_stream(
        const std::initializer_list<const Tensor*> inputs,
        const std::optional<cudaStream_t> requested_stream) {
        const Tensor* backend_reference = nullptr;
        for (const Tensor* input : inputs) {
            if (input == nullptr || input->device() != Device::CUDA) {
                continue;
            }
            if (backend_reference == nullptr) {
                backend_reference = input;
            } else {
                internal::require_same_gpu_backend(
                    *backend_reference, *input, "prepare_inputs_for_stream");
            }
        }

        cudaStream_t execution_stream = requested_stream.has_value()
                                            ? *requested_stream
                                            : getCurrentCUDAStream();
        if (!requested_stream.has_value() && execution_stream == nullptr) {
            for (const Tensor* input : inputs) {
                LFS_ASSERT_MSG(input != nullptr && input->is_valid(),
                               "stream preparation requires valid tensor inputs");
                if (input->device() == Device::CUDA) {
                    execution_stream = input->stream();
                    break;
                }
            }
        }

        for (const Tensor* input : inputs) {
            LFS_ASSERT_MSG(input != nullptr && input->is_valid(),
                           "stream preparation requires valid tensor inputs");
            if (input->device() == Device::CUDA) {
                input->sync_to_stream(execution_stream);
            }
        }
        return execution_stream;
    }

}

namespace lfs::core::tensor_ops {
    namespace {
        thread_local ReducePathForTesting g_reduce_path_override =
            ReducePathForTesting::None;
        thread_local ReducePathForTesting g_reduce_last_path =
            ReducePathForTesting::Default;
    } // namespace

    void set_reduce_path_override_for_testing(ReducePathForTesting path) noexcept {
        g_reduce_path_override = path;
    }
    ReducePathForTesting reduce_path_override_for_testing() noexcept {
        return g_reduce_path_override;
    }
    ReducePathForTesting reduce_last_path_for_testing() noexcept {
        return g_reduce_last_path;
    }
    void set_reduce_last_path_for_testing(ReducePathForTesting path) noexcept {
        g_reduce_last_path = path;
    }

    bool should_prefer_strided_over_transpose(
        size_t outer_size, size_t reduce_size, size_t inner_size) noexcept {
        // Measured argmin on RTX 4080 (microbench, µs):
        //   [64,512,512] dim0:  strided ~109  vs transpose ~570  → strided
        //   [32,128,512] dim1:  strided ~17   vs transpose ~23   → strided
        //   [4,2048,256] dim1:  strided ~18   vs transpose ~38   → strided
        //   [1,4096,512] dim1:  strided ~18   vs transpose ~52   → strided
        //   [8,64,1024]  dim1:  strided ~9.5  vs transpose ~8.0  → transpose (edge)
        //   [16,16,256]  dim0:  strided ~3.9  vs transpose ~6.0  → strided
        //
        // New strided_fast kernel (coalesced inner, unrolled) beats the old
        // ~74µs strided path; full-tensor transpose copy only edges out when
        // the reduce axis is short (copy is cheap) and output is wide.
        const size_t output_elems = outer_size * inner_size;
        if (output_elems == 0 || reduce_size == 0 || inner_size < 256) {
            return false; // small-inner uses legacy warp_strided / other paths
        }
        const size_t numel = outer_size * reduce_size * inner_size;
        // Cheap-copy edge: short reduce + wide output + modest total size.
        if (reduce_size <= 64 && output_elems >= 8192 && numel <= (1u << 20)) {
            return false; // transpose class (measured)
        }
        return true; // strided_fast default for large-inner zone
    }

    namespace {


        std::atomic<uint64_t> g_tensor_kernel_launch_count{0};

    } // namespace

    void reset_tensor_kernel_launch_count() noexcept {
        g_tensor_kernel_launch_count.store(0, std::memory_order_relaxed);
    }

    uint64_t tensor_kernel_launch_count() noexcept {
        return g_tensor_kernel_launch_count.load(std::memory_order_relaxed);
    }

    void record_tensor_kernel_launch(uint64_t n) noexcept {
        g_tensor_kernel_launch_count.fetch_add(n, std::memory_order_relaxed);
        internal::telemetry_record_kernel_launch(n);
    }

}
