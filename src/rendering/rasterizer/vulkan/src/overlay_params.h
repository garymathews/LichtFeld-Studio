/* SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
namespace lfs::rendering::vulkan {
    enum OverlayParamIndex : std::size_t {
#define LFS_OVERLAY_PARAM(host, shader, value) shader = value, host = shader,
#include "../shader/src/slang/overlay_layout.inc"
#undef LFS_OVERLAY_PARAM
    };
    static_assert(EllipsoidFlags + EllipsoidParamStride <= ViewFlags);
}
