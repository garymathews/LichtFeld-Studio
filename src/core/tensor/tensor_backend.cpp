/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "core/cuda_safe_format.hpp"
#include "core/cuda_stream_fwd.hpp"
#include "core/float16.hpp"

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#if LFS_TENSOR_CUDA
#include <cuda_runtime.h>
#endif
#include <format>
#include <string>
#ifdef LFS_TENSOR_VULKAN
#include "backend/gpu_backend_ops.hpp"
#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#endif

namespace lfs::core {
    namespace {

        constexpr GpuBackend kDefaultBackend = LFS_TENSOR_CUDA ? GpuBackend::CUDA : GpuBackend::Vulkan;

        constexpr int kUnconfigured = -1;
        constexpr int kConfiguredCuda = -2;
        constexpr int kConfiguredVulkan = -3;

        std::atomic<int> process_backend_state{kUnconfigured};
        thread_local std::optional<GpuBackend> scoped_backend;

        bool is_resolved(const int state) {
            return state == static_cast<int>(GpuBackend::CUDA) ||
                   state == static_cast<int>(GpuBackend::Vulkan);
        }

        GpuBackend configured_backend(const int state) {
            return state == kConfiguredVulkan ? GpuBackend::Vulkan : GpuBackend::CUDA;
        }

        int configured_state(const GpuBackend backend) {
            return backend == GpuBackend::Vulkan ? kConfiguredVulkan : kConfiguredCuda;
        }

        std::optional<GpuBackend> backend_from_environment() {
            const char* value = std::getenv("LFS_TENSOR_BACKEND");
            if (value == nullptr) {
                return std::nullopt;
            }

            std::string normalized(value);
            for (char& character : normalized) {
                character = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(character)));
            }
            if (normalized == "cuda") {
                return GpuBackend::CUDA;
            }
            if (normalized == "vulkan") {
                return GpuBackend::Vulkan;
            }

            LOG_WARN("Ignoring invalid LFS_TENSOR_BACKEND='{}'; using {}", value, gpu_backend_name(kDefaultBackend));
            return kDefaultBackend;
        }

        [[noreturn]] void throw_backend_unavailable(const GpuBackend backend) {
#ifdef LFS_TENSOR_VULKAN
            if (backend == GpuBackend::Vulkan && internal::vulkan_backend_lost()) {
                throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::DeviceLost,
                    .domain = lfs::ErrorDomain::Vulkan,
                    .user_message = "Vulkan tensor backend device lost; shut the backend down "
                                    "to create a new context",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
#endif
            throw TensorError(std::format(
                "GPU backend '{}' is unavailable", gpu_backend_name(backend)));
        }

    } // namespace

    const char* gpu_backend_name(const GpuBackend backend) {
        switch (backend) {
        case GpuBackend::CUDA: return "CUDA";
        case GpuBackend::Vulkan: return "Vulkan";
        }
        return "Unknown";
    }

    GpuBackend default_gpu_backend() {
        for (;;) {
            int state = process_backend_state.load(std::memory_order_acquire);
            if (is_resolved(state)) {
                return static_cast<GpuBackend>(state);
            }

            const GpuBackend selected = state == kUnconfigured
                                            ? backend_from_environment().value_or(kDefaultBackend)
                                            : configured_backend(state);
            if (process_backend_state.compare_exchange_weak(
                    state, static_cast<int>(selected),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return selected;
            }
        }
    }

    lfs::Status set_default_gpu_backend(const GpuBackend backend) {
        int state = process_backend_state.load(std::memory_order_acquire);
        for (;;) {
            if (is_resolved(state)) {
                const auto frozen = static_cast<GpuBackend>(state);
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::FailedPrecondition,
                    .domain = lfs::ErrorDomain::Core,
                    .user_message = std::format(
                        "GPU backend default is frozen as {}; cannot change it to {}",
                        gpu_backend_name(frozen), gpu_backend_name(backend)),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            if (process_backend_state.compare_exchange_weak(
                    state, configured_state(backend),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return {};
            }
        }
    }

    bool gpu_backend_available(const GpuBackend backend) {
        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            return internal::vulkan_backend_probe_available();
#else
            return false;
#endif
        }

#if LFS_TENSOR_CUDA
        static const bool cuda_available = [] {
            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return device_count > 0;
        }();
        return cuda_available;
#else
        return false;
#endif
    }

    std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.device_ != Device::CUDA) {
            return std::nullopt;
        }
        return internal::gpu_backend_tag(tensor);
    }

    GpuBackendScope::GpuBackendScope(const GpuBackend backend)
        : previous_(scoped_backend) {
        scoped_backend = backend;
    }

    GpuBackendScope::~GpuBackendScope() {
        scoped_backend = previous_;
    }

    MemoryInfo gpu_backend_memory_info(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA) {
            return MemoryInfo::cuda();
        }
#ifdef LFS_TENSOR_VULKAN
        return internal::backend_ops(GpuBackend::Vulkan).stats();
#else
        return {};
#endif
    }

    lfs::Status shutdown_gpu_backend(const GpuBackend backend) {
#ifdef LFS_TENSOR_VULKAN
        if (backend == GpuBackend::Vulkan) {
            try {
                internal::backend_ops(backend).shutdown();
            } catch (...) {
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Vulkan,
                    .user_message = "Vulkan tensor backend shutdown failed",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
        }
#else
        (void)backend;
#endif
        return {};
    }

    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles) {
#ifdef LFS_TENSOR_VULKAN
        return internal::adopt_vulkan_context(internal::AdoptedDevice{
            .instance = static_cast<VkInstance>(handles.instance),
            .physical_device = static_cast<VkPhysicalDevice>(handles.physical_device),
            .device = static_cast<VkDevice>(handles.device),
            .queue = static_cast<VkQueue>(handles.queue),
            .queue_family = handles.queue_family,
            .shader_atomic_float = handles.shader_atomic_float,
            .memory_budget = handles.memory_budget,
            .shader_float16 = handles.shader_float16,
            .shared_buffer_queue_families = handles.shared_buffer_queue_families,
            .with_other_queues_idle = handles.with_other_queues_idle,
        });
#else
        (void)handles;
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
#endif
    }

    bool vulkan_backend_adopted() {
#ifdef LFS_TENSOR_VULKAN
        return internal::vulkan_context_adopted();
#else
        return false;
#endif
    }

    uint64_t vulkan_backend_context_id() {
#ifdef LFS_TENSOR_VULKAN
        return internal::acquire_vulkan_context()->context_id();
#else
        throw std::runtime_error("This build has no Vulkan tensor backend");
#endif
    }

    void* vulkan_backend_timeline() {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::try_live_vulkan_context();
        if (!context) {
            return nullptr;
        }
        const VkSemaphore timeline = context->timeline();
        if (timeline == VK_NULL_HANDLE) {
            return nullptr;
        }
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(timeline));
#else
        return nullptr;
#endif
    }

    std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor) {
#ifdef LFS_TENSOR_VULKAN
        if (!tensor.is_valid()) {
            return std::nullopt;
        }
        tensor.materialize_if_deferred();
        if (gpu_backend_of(tensor) != GpuBackend::Vulkan) {
            return std::nullopt;
        }
        const auto context = internal::try_live_vulkan_context();
        if (!context) {
            return std::nullopt;
        }
        const internal::StorageRef storage = internal::storage_ref(tensor);
        if (storage.meta == nullptr ||
            storage.backend != GpuBackend::Vulkan ||
            storage.meta->gpu_descriptor.native_buffer == 0) {
            return std::nullopt;
        }
        context->recorders().flush_storage(storage);
        TensorVulkanBuffer result;
        result.buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(
            storage.meta->gpu_descriptor.native_buffer));
        result.offset = storage.byte_offset;
        result.device_address =
            storage.meta->gpu_descriptor.base_address + storage.byte_offset;
        result.bytes = tensor.bytes();
        result.pending_timeline_value =
            storage.meta->pending_value.load(std::memory_order_acquire);
        result.keep_alive = tensor.data_owner_;
        return result;
#else
        (void)tensor;
        return std::nullopt;
#endif
    }

    std::optional<TensorVulkanBuffer> tensor_vulkan_write_buffer(Tensor& tensor) {
        if (!tensor.is_valid() || gpu_backend_of(tensor) != GpuBackend::Vulkan)
            return std::nullopt;
        if (!tensor.is_contiguous())
            throw std::invalid_argument("External Vulkan writes require contiguous tensors");
        // The mutable pointer escape materializes deferred values and preserves
        // dependent lazy snapshots. The address itself is never dereferenced here.
        (void)tensor.data_ptr();
        return tensor_vulkan_buffer(tensor);
    }

    void retire_vulkan_resources(void* device, const std::function<void()>& work) {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::try_live_vulkan_context();
        if (context && context->device() == device && !context->dead()) {
            context->recorders().retire_resources(work);
            return;
        }
#endif
        work();
    }

    void invalidate_vulkan_backend(const lfs::Error& error) {
#ifdef LFS_TENSOR_VULKAN
        if (const auto context = internal::try_live_vulkan_context()) {
            if (error.code() == lfs::ErrorCode::DeviceLost)
                context->mark_device_lost_once();
            else if (error.code() == lfs::ErrorCode::DeadlineExceeded)
                context->quarantine();
        }
#endif
    }

    lfs::Status vulkan_backend_status() {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::try_live_vulkan_context();
        if (context && !context->dead() && context->accepting_work())
            return {};
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = !context          ? lfs::ErrorCode::Unavailable
                    : context->dead() ? lfs::ErrorCode::DeviceLost
                                      : lfs::ErrorCode::DeadlineExceeded,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "Vulkan backend is unavailable; restart and restore the last checkpoint.",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
#else
        return {};
#endif
    }

    void with_idle_vulkan_device(const std::function<void(const VulkanExternalDevice&)>& work) {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::acquire_vulkan_context();
        const VulkanExternalDevice device{
            .handles = {.instance = context->instance(), .physical_device = context->physical_device(), .device = context->device(), .queue = context->queue(), .queue_family = context->queue_family(), .shader_atomic_float = context->caps().shader_atomic_float, .memory_budget = context->caps().memory_budget, .shader_float16 = context->caps().shader_float16},
            .allocator = context->allocator(),
            .context_id = context->context_id()};
        context->recorders().run_external([&] {
            context->require_work();
            work(device);
        });
#else
        (void)work;
        throw std::runtime_error("This build has no Vulkan tensor backend");
#endif
    }

    std::shared_ptr<void> register_vulkan_external_resource(
        const std::function<void(const VulkanExternalDevice&)>& initialize,
        std::function<void()> cleanup) {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::acquire_vulkan_context();
        auto token = std::shared_ptr<uint64_t>(new uint64_t{0},
            [weak = std::weak_ptr(context)](uint64_t* id) {
                if (const auto owner = weak.lock())
                    owner->release_external_resource(*id);
                delete id;
            });
        *token = context->register_external_resource([&] {
            const VulkanExternalDevice device{
                .handles = {.instance = context->instance(), .physical_device = context->physical_device(),
                            .device = context->device(), .queue = context->queue(), .queue_family = context->queue_family(),
                            .shader_atomic_float = context->caps().shader_atomic_float, .memory_budget = context->caps().memory_budget,
                            .shader_float16 = context->caps().shader_float16},
                .allocator = context->allocator(), .context_id = context->context_id()};
            initialize(device);
        }, std::move(cleanup));
        return token;
#else
        (void)initialize;
        (void)cleanup;
        throw std::runtime_error("This build has no Vulkan tensor backend");
#endif
    }

    namespace internal {

        void order_legacy_after_home(const Tensor& tensor) {
            if (tensor.device() != Device::CUDA || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{tensor.stream()}, ExecContext{nullptr});
        }

        void order_home_after_legacy(const Tensor& tensor) {
            if (tensor.device() != Device::CUDA || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{nullptr}, ExecContext{tensor.stream()});
        }

        void trim_live_gpu_backends() {
#if LFS_TENSOR_CUDA
            backend_ops(GpuBackend::CUDA).trim();
#endif
#ifdef LFS_TENSOR_VULKAN
            if (vulkan_backend_live()) {
                backend_ops(GpuBackend::Vulkan).trim();
            }
#endif
        }

        void trim_live_gpu_backends_if_reserved_unused_exceeds(const size_t threshold_bytes) {
#if LFS_TENSOR_CUDA
            backend_ops(GpuBackend::CUDA).trim_if_reserved_unused_exceeds(threshold_bytes);
#endif
#ifdef LFS_TENSOR_VULKAN
            if (vulkan_backend_live()) {
                backend_ops(GpuBackend::Vulkan).trim_if_reserved_unused_exceeds(threshold_bytes);
            }
#endif
        }

        GpuBackend resolve_new_gpu_storage_backend() {
            const GpuBackend backend = scoped_backend ? *scoped_backend : default_gpu_backend();
            if (!gpu_backend_available(backend)) {
                throw_backend_unavailable(backend);
            }
            return backend;
        }

        Tensor allocate_like(const Tensor& input,
                             const TensorShape& shape,
                             const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::empty(shape, Device::CPU, dtype);
            }

            const GpuBackend backend = gpu_backend_of(input).value();
            GpuBackendScope scope(backend);
            return Tensor::empty(shape, Device::CUDA, dtype);
        }

        Tensor allocate_like(const Tensor& input,
                             const TensorShape& shape,
                             const DataType dtype,
                             const float value) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::full(shape, value, Device::CPU, dtype);
            }

            const GpuBackend backend = gpu_backend_of(input).value();
            GpuBackendScope scope(backend);
            return Tensor::full(shape, value, Device::CUDA, dtype);
        }

        Tensor allocate_zeros_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_zeros_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::zeros(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::zeros(shape, Device::CUDA, dtype);
        }

        Tensor allocate_ones_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_ones_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::ones(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::ones(shape, Device::CUDA, dtype);
        }

        Tensor allocate_rand_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_rand_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::rand(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::rand(shape, Device::CUDA, dtype);
        }

        Tensor allocate_randn_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_randn_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::randn(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::randn(shape, Device::CUDA, dtype);
        }

        Tensor copy_to_backend(const Tensor& source, const GpuBackend target) {
            LFS_ASSERT_MSG(source.is_valid(), "copy_to_backend requires a valid source tensor");
            if (gpu_backend_of(source) == target) {
                return source.clone();
            }

            GpuBackendScope scope(target);
            if (source.device() == Device::CPU) {
                return source.to(Device::CUDA);
            }
            return source.to(Device::CPU).to(Device::CUDA);
        }

        void throw_gpu_backend_mismatch(const Tensor& reference,
                                        const Tensor& other,
                                        const std::string_view operation) {
            const auto reference_backend = gpu_backend_of(reference);
            const auto other_backend = gpu_backend_of(other);
            throw TensorError(std::format(
                "{} requires matching GPU backends, got {} and {}",
                operation,
                reference_backend ? gpu_backend_name(*reference_backend) : "CPU",
                other_backend ? gpu_backend_name(*other_backend) : "CPU"));
        }

    } // namespace internal

} // namespace lfs::core
