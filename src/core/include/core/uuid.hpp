/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfs::core {

    struct LFS_CORE_API Uuid {
        std::array<std::uint8_t, 16> bytes{};

        [[nodiscard]] bool is_nil() const noexcept;
        [[nodiscard]] std::string to_string() const;
        [[nodiscard]] static std::optional<Uuid> from_string(std::string_view value);

        friend bool operator==(const Uuid&, const Uuid&) = default;
    };

    class LFS_CORE_API UuidGenerationError final : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    // UUIDv4 backed by the operating system CSPRNG. This generator is independent
    // of all training and tensor pseudo-random number generators.
    [[nodiscard]] LFS_CORE_API Uuid generate_uuid_v4();

    /// Reproducible-artifact mode, opt-in through SOURCE_DATE_EPOCH - the convention reproducible
    /// builds use everywhere (Debian, gcc, Python).
    ///
    /// Enabled, identities are derived from what an artifact is and timestamps come from the pinned
    /// epoch, so the same build and invocation produce identical artifacts. Absent - the default -
    /// every value is drawn at random and stamped with the wall clock exactly as before, which
    /// matters because crash recovery orders candidates by save time (project_recovery.cpp:1585)
    /// and must see real times.
    [[nodiscard]] inline std::optional<std::uint64_t> reproducible_epoch_seconds() {
        const char* const raw = std::getenv("SOURCE_DATE_EPOCH");
        if (raw == nullptr || *raw == '\0')
            return std::nullopt;
        try {
            return std::stoull(raw);
        } catch (...) {
            return std::nullopt;
        }
    }

    [[nodiscard]] inline bool reproducible_artifacts() {
        return reproducible_epoch_seconds().has_value();
    }

    [[nodiscard]] inline std::uint64_t reproducible_epoch_ns() {
        return *reproducible_epoch_seconds() * 1'000'000'000ull;
    }

    /// Identity derived from what an artifact is rather than drawn at random: FNV-1a twice with
    /// different offsets to fill sixteen bytes, shaped like a v4 UUID so anything inspecting the
    /// version sees a valid one.
    [[nodiscard]] inline Uuid derive_uuid_from_seed(const std::string_view seed) {
        std::array<std::uint8_t, 16> bytes{};
        const auto fill = [&seed](const std::uint64_t offset, std::uint8_t* const out) {
            std::uint64_t h = offset;
            for (const char c : seed) {
                h ^= static_cast<std::uint8_t>(c);
                h *= 1099511628211ull;
            }
            std::uint64_t h2 = h ^ 0x9e3779b97f4a7c15ull;
            for (const char c : seed) {
                h2 ^= static_cast<std::uint8_t>(c);
                h2 *= 1099511628211ull;
            }
            std::memcpy(out, &h, sizeof(h));
            std::memcpy(out + sizeof(h), &h2, sizeof(h2));
        };
        fill(14695981039346656037ull, bytes.data());
        bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0f) | 0x40);
        bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3f) | 0x80);
        Uuid uuid;
        uuid.bytes = bytes;
        return uuid;
    }

    /// Identity that is derived in reproducible mode and generated otherwise.
    [[nodiscard]] inline Uuid reproducible_artifact_uuid(const std::string_view seed) {
        return reproducible_artifacts() ? derive_uuid_from_seed(seed) : generate_uuid_v4();
    }


} // namespace lfs::core

template <>
struct std::hash<lfs::core::Uuid> {
    [[nodiscard]] std::size_t operator()(const lfs::core::Uuid& uuid) const noexcept {
        std::size_t hash = sizeof(std::size_t) == 8
                               ? static_cast<std::size_t>(14695981039346656037ull)
                               : static_cast<std::size_t>(2166136261u);
        constexpr std::size_t PRIME = sizeof(std::size_t) == 8
                                          ? static_cast<std::size_t>(1099511628211ull)
                                          : static_cast<std::size_t>(16777619u);
        for (const std::uint8_t byte : uuid.bytes) {
            hash ^= byte;
            hash *= PRIME;
        }
        return hash;
    }
};
