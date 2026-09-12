/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <optional>
#include <string>
#include <string_view>
namespace lfs::core {
    // The single runtime diagnostics control, generalized from the boolean
    // LFS_CUDA_SYNC_DEBUG into a comma-separated mode list. See
    // parse_diagnostic_modes() for the parsing contract.
    enum class DiagnosticMode : unsigned {
        CudaSync = 1u << 0,
        DeviceTrap = 1u << 1,
        VkFatal = 1u << 2,
    };

    struct LFS_CORE_API ParsedDiagnosticModes {
        unsigned modes = 0;
        bool unknown_tokens_present = false;
        std::string unknown_tokens;
        bool legacy_alias_present = false;
    };

    // Pure string -> bitmask parser: no getenv, no caching, no logging.
    // sync_debug_value/vk_validation_fatal_value are the raw LFS_CUDA_SYNC_DEBUG
    // and deprecated LFS_VK_VALIDATION_FATAL values (nullopt when unset).
    [[nodiscard]] LFS_CORE_API ParsedDiagnosticModes parse_diagnostic_modes(
        std::optional<std::string_view> sync_debug_value,
        std::optional<std::string_view> vk_validation_fatal_value) noexcept;

    // Reads and parses the environment exactly once into an immutable bitmask,
    // logging any deprecation/unknown-token warnings on first use.
    [[nodiscard]] LFS_CORE_API unsigned diagnostic_modes() noexcept;
    [[nodiscard]] LFS_CORE_API bool diagnostic_mode_enabled(DiagnosticMode mode) noexcept;

}
