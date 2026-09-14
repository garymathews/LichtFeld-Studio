/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/video/video_export_options.hpp"

#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace lfs::io::video {

    class IVideoEncoder {
    public:
        virtual ~IVideoEncoder() = default;

        [[nodiscard]] virtual std::expected<void, std::string> open(
            const std::filesystem::path& output_path,
            const VideoExportOptions& options) = 0;

        // Packed RGB24 in host memory. Vulkan exporters must read back explicitly.
        [[nodiscard]] virtual std::expected<void, std::string> writeFrame(
            std::span<const uint8_t> rgb, int width, int height) = 0;

        [[nodiscard]] virtual std::expected<void, std::string> writeFrameGpu(
            const void* rgba_gpu_ptr, int width, int height, void* cuda_stream = nullptr) = 0;

        [[nodiscard]] virtual std::expected<void, std::string> close() = 0;
    };

} // namespace lfs::io::video
