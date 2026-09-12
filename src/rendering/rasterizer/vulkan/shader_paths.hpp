/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/executable_path.hpp"
#include "core/path_utils.hpp"
#include <vector>
#include "vulkan_shader_manifest.hpp"
namespace lfs::rendering::vulkan {
    [[nodiscard]] inline std::filesystem::path resolveVkSplatSpirvRoot(const std::filesystem::path& requested = {}) {
        std::vector<std::filesystem::path> search_paths;

        search_paths.push_back(lfs::core::getResourceBaseDir() / "shaders" / "vulkan_rasterizer");

#ifdef LFS_VULKAN_RASTERIZER_DEV_SPV_DIR
        search_paths.push_back(lfs::core::utf8_to_path(LFS_VULKAN_RASTERIZER_DEV_SPV_DIR));
#endif

#ifdef PROJECT_ROOT_PATH
        search_paths.push_back(lfs::core::utf8_to_path(PROJECT_ROOT_PATH) /
                               "src/rendering/rasterizer/vulkan/shader");
#endif

        if (!requested.empty()) search_paths = {requested};
        for (const auto& path : search_paths) {
            if (!std::filesystem::exists(path)) continue;
            std::string missing;
            for (const auto shader : required_shaders) {
                std::error_code ec;
                if (!std::filesystem::is_regular_file(path / shader, ec) || std::filesystem::file_size(path / shader, ec) == 0 || ec)
                    missing += "\n  - " + std::string(shader);
            }
            if (!missing.empty())
                throw std::runtime_error("Incomplete VkSplat shader bundle at " + lfs::core::path_to_utf8(path) + ":" + missing);
            return path;
        }

        std::string error = "Cannot find VkSplat SPIR-V shaders. Searched in:";
        for (const auto& path : search_paths) {
            error += "\n  - " + lfs::core::path_to_utf8(path);
        }
        error += "\nExecutable directory: " + lfs::core::path_to_utf8(lfs::core::getExecutableDir());
        throw std::runtime_error(error);
    }

}
