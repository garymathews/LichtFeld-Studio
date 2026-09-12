/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_fwd.hpp"
#include "core/export.hpp"
#include <string_view>

namespace lfs::core::internal {
    // Validate before narrowing indices or issuing writes. Vulkan reduces an
    // exact Int64 range on-device and reads back only the two bounds.
    LFS_CORE_API void assert_index_tensor(const Tensor& indices, size_t upper_bound,
                                          std::string_view operation, bool check_bounds,
                                          bool allow_negative = false);
}
