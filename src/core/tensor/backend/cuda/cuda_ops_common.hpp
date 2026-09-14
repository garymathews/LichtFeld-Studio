/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../descriptors.hpp"
#include "core/assert.hpp"

namespace lfs::core::internal {
    template <class T>
    T* cuda_pointer(const StorageRef storage) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                       "CUDA adapter received non-CUDA storage");
        return reinterpret_cast<T*>(
            static_cast<unsigned char*>(storage.data) + storage.byte_offset);
    }
} // namespace lfs::core::internal
