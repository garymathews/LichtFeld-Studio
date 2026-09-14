/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/memory_pressure.hpp"
#include <format>

namespace lfs::core {

    MemoryAllocationError::MemoryAllocationError(const AllocationFailure& failure)
        : std::runtime_error(std::format(
              "{} out of memory: failed to allocate {} for '{}' (op '{}', device {}, native {})",
              to_string(failure.domain),
              std::format("{:.1f} MiB", static_cast<double>(failure.requested_bytes) / (1024.0 * 1024.0)),
              failure.label ? failure.label : "",
              failure.operation ? failure.operation : "",
              failure.device,
              failure.native_error)),
          failure_(failure) {}



    const char* to_string(const MemoryDomain domain) noexcept {
        switch (domain) {
        case MemoryDomain::CudaDevice: return "cuda-device";
        case MemoryDomain::CudaVmm: return "cuda-vmm";
        case MemoryDomain::VulkanDevice: return "vulkan-device";
        case MemoryDomain::PinnedHost: return "pinned-host";
        case MemoryDomain::PageableHost: return "pageable-host";
        }
        return "unknown";
    }

    bool is_device_heap(const MemoryDomain domain) noexcept {
        return domain == MemoryDomain::CudaDevice ||
               domain == MemoryDomain::CudaVmm ||
               domain == MemoryDomain::VulkanDevice;
    }

} // namespace lfs::core
