/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include "core/cuda_stream_fwd.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#if LFS_TENSOR_CUDA
#include "core/cuda/memory_arena.hpp"
#include "core/device_fault.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"
#endif
#include <array>
#include <atomic>
namespace lfs::core {
    namespace {
        constexpr int kMaxGpuPreShutdownHooks = 32;
        std::array<GpuPreShutdownHook, kMaxGpuPreShutdownHooks> g_gpu_pre_shutdown_hooks{};
        std::atomic<int> g_gpu_pre_shutdown_hook_count{0};
        std::atomic<bool> g_gpu_pre_shutdown_hooks_ran{false};
        std::atomic<bool> g_gpu_process_teardown_started{false};
        std::atomic<bool> g_gpu_pre_shutdown_overflow_logged{false};

        void run_gpu_pre_shutdown_hooks_once() noexcept {
            bool expected = false;
            if (!g_gpu_pre_shutdown_hooks_ran.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                return;
            }
            const int n = g_gpu_pre_shutdown_hook_count.load(std::memory_order_acquire);
            for (int i = 0; i < n && i < kMaxGpuPreShutdownHooks; ++i) {
                if (GpuPreShutdownHook hook = g_gpu_pre_shutdown_hooks[static_cast<size_t>(i)]) {
                    try {
                        hook();
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): hooks must not escape;
                        // continue remaining holders so pool shutdown still runs.
                    }
                }
            }
        }
    } // namespace

    void register_gpu_pre_shutdown_hook(const GpuPreShutdownHook hook) noexcept {
        if (!hook) {
            return;
        }
        const int index = g_gpu_pre_shutdown_hook_count.fetch_add(1, std::memory_order_acq_rel);
        if (index < 0 || index >= kMaxGpuPreShutdownHooks) {
            if (!g_gpu_pre_shutdown_overflow_logged.exchange(true, std::memory_order_relaxed)) {
                try {
                    LOG_ERROR("register_gpu_pre_shutdown_hook: capacity {} exceeded; "
                              "hook dropped during static/TLS release",
                              kMaxGpuPreShutdownHooks);
                } catch (...) {
                }
            }
            return;
        }
        g_gpu_pre_shutdown_hooks[static_cast<size_t>(index)] = hook;
    }

    bool gpu_process_teardown_started() noexcept {
        return g_gpu_process_teardown_started.load(std::memory_order_acquire);
    }

    void teardown_gpu_before_exit() noexcept {
        try {
            static_cast<void>(shutdown_gpu_backend(GpuBackend::Vulkan));
            // release every registered long-lived CUDA holder
            // (TLS FastGS sort workspaces, rasterizer image caches, PPISP shared
            // statics, mirror mult cache, nan-check scratch, …) while the pool
            // and CUDA context are still usable. After this returns, static/TLS
            // dtors must find empty holders — otherwise they free after the
            // Meyers-singleton pool is destroyed → SIGSEGV (exit 139).
            const bool cuda_usable = gpu_backend_available(GpuBackend::CUDA);
            if (cuda_usable) run_gpu_pre_shutdown_hooks_once();
            g_gpu_process_teardown_started.store(true, std::memory_order_release);

            // Drain dedicated DeviceFaultRecord slots (cudaMalloc-owned, never
            // pool memory) before the tensor memory
            // pool shuts down. device_fault_registry_teardown is no-throw and
            // idempotent (LFS_CUDA_LOG_TEARDOWN on every free).
#if LFS_TENSOR_CUDA
            if (!cuda_usable) return;
            device_fault_registry_teardown();
            GlobalArenaManager::instance().shutdown();
            Tensor::shutdown_memory_pool();
            PinnedMemoryAllocator::instance().shutdown();
#endif
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): subsystem teardown reports CUDA
            // failures internally; none may escape this sanctioned pre-exit step.
        }
    }

}
