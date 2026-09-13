/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/cuda_stream_fwd.hpp"
#if LFS_TENSOR_CUDA
#include <cuda_fp16.h>
#endif
namespace lfs::core {
#if !LFS_TENSOR_CUDA
    using Float16 = _Float16;
#else
    using Float16 = __half;
#endif
    static_assert(sizeof(Float16) == 2);
}
