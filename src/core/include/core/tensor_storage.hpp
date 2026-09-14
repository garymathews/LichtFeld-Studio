/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace lfs::core {

    enum class StorageAccountingKind : uint8_t {
        CudaDirect,
        CudaExternal,
        VulkanOwned,
        VulkanExternal,
    };

    struct GpuStorageDescriptor {
        uint64_t native_buffer;
        uint64_t native_allocation;
        uint64_t native_context;
        uint64_t base_address;
        uint64_t byte_size;
        StorageAccountingKind accounting_kind;
    };
    static_assert(std::is_trivial_v<GpuStorageDescriptor>);
    static_assert(std::is_standard_layout_v<GpuStorageDescriptor>);

    struct StorageMeta {
        GpuBackend backend = GpuBackend::CUDA;
        GpuStorageDescriptor gpu_descriptor{};
        std::atomic<uint64_t> pending_recorder{0};
        std::atomic<uint64_t> pending_value{0};
        std::atomic<uint64_t> generation{0};
        std::atomic<uint32_t> pending_lazy_snapshots{0};
        std::mutex lazy_snapshot_mutex;
        std::vector<std::weak_ptr<Tensor>> lazy_snapshots;
        std::string external_kind;
        std::shared_ptr<void> external_owner;
        // Exportable packed-SoA provenance. When set, bind sites
        // re-resolve the device pointer through the live control block instead of
        // trusting the baked data_ pointer across a capacity grow.
        // exportable_control holds shared_ptr<SplatExportableStorage::Control>.
        std::shared_ptr<void> exportable_control;
        std::uint32_t exportable_region = 0;
        std::uint64_t exportable_bound_generation = 0;
    };

    // Memory info
    class LFS_CORE_API MemoryInfo {
    public:
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        size_t allocated_bytes = 0;
        int device_id = -1;

        static MemoryInfo cuda();
        static MemoryInfo cpu();

        void log() const;
    };

} // namespace lfs::core
