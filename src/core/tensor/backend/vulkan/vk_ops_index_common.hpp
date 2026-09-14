/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_storage.hpp"
#include "core/assert.hpp"
#include "core/tensor/backend/descriptors.hpp"

#include "core/float16.hpp"

#include <cstdint>
#include <cstring>
#include <utility>

// Helpers shared by the index and mask adapters.
namespace lfs::core::internal::vk_index {

    // Element codes of index.slang and mask.slang follow DataType.
    inline uint32_t shader_dtype(const DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 0;
        case DataType::Float16: return 1;
        case DataType::Int32: return 2;
        case DataType::Int64: return 3;
        case DataType::UInt8: return 4;
        case DataType::Bool: return 5;
        default: break;
        }
        LFS_ASSERT_MSG(false, "Vulkan index operation received an unsupported dtype");
        return 0;
    }

    // Bit pattern of a scalar in the element type, split into two 32-bit words.
    inline std::pair<uint32_t, uint32_t> fill_bits(const DataType dtype, const ScalarOperand value) {
        float as_float = 0.0f;
        int64_t as_integer = 0;
        switch (value.kind) {
        case ScalarKind::Float:
            as_float = value.value.float_value;
            if (dtype != DataType::Float32 && dtype != DataType::Float16 && dtype != DataType::Bool)
                as_integer = static_cast<int64_t>(value.value.float_value);
            break;
        case ScalarKind::Int32:
            as_float = static_cast<float>(value.value.int32_value);
            as_integer = value.value.int32_value;
            break;
        case ScalarKind::Int64:
            as_float = static_cast<float>(value.value.int64_value);
            as_integer = value.value.int64_value;
            break;
        case ScalarKind::Bool:
            as_float = value.value.bool_value ? 1.0f : 0.0f;
            as_integer = value.value.bool_value ? 1 : 0;
            break;
        }
        switch (dtype) {
        case DataType::Float32: {
            uint32_t bits = 0;
            std::memcpy(&bits, &as_float, sizeof(bits));
            return {bits, 0};
        }
        case DataType::Float16: {
            const Float16 converted = static_cast<Float16>(as_float);
            uint16_t bits = 0;
            std::memcpy(&bits, &converted, sizeof(bits));
            return {bits, 0};
        }
        case DataType::Int32:
        case DataType::Int64:
        case DataType::UInt8: {
            const auto bits = static_cast<uint64_t>(as_integer);
            return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
        }
        case DataType::Bool: return {as_float != 0.0f ? 1u : 0u, 0};
        default: break;
        }
        LFS_ASSERT_MSG(false, "unsupported Vulkan fill dtype");
        return {0, 0};
    }

} // namespace lfs::core::internal::vk_index
