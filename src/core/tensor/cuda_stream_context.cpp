/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/stream_lifetime.hpp"
#include "internal/tensor_impl.hpp"

#include <format>

namespace lfs::core {

    void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream) {
        unretire_stream(execution_stream);
        unretire_stream(dependency_stream);
        if (dependency_stream == nullptr || dependency_stream == execution_stream) {
            return;
        }

        cudaError_t status = cudaErrorUnknown;
        if (cudaEvent_t ready = CudaEventPool::instance().acquire()) {
            const cudaError_t record_status = cudaEventRecord(ready, dependency_stream);
            status = record_status;
            if (record_status == cudaSuccess) {
                const cudaError_t wait_status =
                    cudaStreamWaitEvent(execution_stream, ready, 0);
                status = wait_status;
                if (wait_status != cudaSuccess) {
                    ensure_cuda_success(
                        wait_status, "cudaStreamWaitEvent(tensor dependency)",
                        std::format("dependency_stream={}, execution_stream={}; fallback=stream sync",
                                    static_cast<void*>(dependency_stream),
                                    static_cast<void*>(execution_stream)),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
            } else {
                ensure_cuda_success(
                    record_status, "cudaEventRecord(tensor dependency)",
                    std::format("dependency_stream={}, execution_stream={}; fallback=stream sync",
                                static_cast<void*>(dependency_stream),
                                static_cast<void*>(execution_stream)),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }
            CudaEventPool::instance().release(ready);
        }

        if (status != cudaSuccess) {
            const cudaError_t sync_status = cudaStreamSynchronize(dependency_stream);
            if (sync_status != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    sync_status, "cudaStreamSynchronize(tensor dependency fallback)",
                    std::format("dependency_stream={}, execution_stream={}",
                                static_cast<void*>(dependency_stream),
                                static_cast<void*>(execution_stream)));
            }
        }
    }

    cudaError_t memcpy_ordered(void* const dst, const void* const src, const size_t bytes,
                               const cudaMemcpyKind kind, const cudaStream_t stream) {
        if (stream != nullptr && kind == cudaMemcpyDeviceToHost) {
            bridgeStreams(stream, nullptr);
        }
        const cudaError_t status = cudaMemcpy(dst, src, bytes, kind);
        if (status == cudaSuccess && stream != nullptr && kind == cudaMemcpyHostToDevice) {
            bridgeStreams(nullptr, stream);
        }
        return status;
    }

} // namespace lfs::core
