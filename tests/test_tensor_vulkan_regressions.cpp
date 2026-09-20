/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/nn.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor/internal/joint_moments.hpp"
#include "core/tensor/internal/mcmc_noise.hpp"
#include "core/vulkan_phase_timing.hpp"
#include "training/kernels/mcmc_tensor.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <future>
#include <thread>
#ifdef LFS_TENSOR_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <optional>
#include <numbers>

using namespace lfs::core;

namespace {
    class VulkanTensorRegression : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GpuBackend::Vulkan))
                GTEST_SKIP() << "Vulkan backend unavailable";
            backend_.emplace(GpuBackend::Vulkan);
        }

        std::optional<GpuBackendScope> backend_;
    };
} // namespace

TEST_F(VulkanTensorRegression, FusedMomentDecodeMatchesSeparateDecode) {
    // The optimizer can let the Adam shader decode the joint-moment codec instead of
    // running the decode kernel first. That is only worth having if it reproduces the
    // separate decode bit for bit, including the swizzled packed layout.
    constexpr size_t rows = 256, slots = 12, lanes = 32, attributes = slots * 4, blocks = rows / lanes;
    constexpr size_t cells = rows * attributes;
    constexpr int bits = 8;
    constexpr size_t block_cells = 256 * attributes;
    const auto to_rows = [&](const Tensor& flat) {
        return flat.reshape(TensorShape{blocks, slots, lanes, 4}).permute({0, 2, 1, 3}).contiguous()
            .reshape(TensorShape{rows, attributes});
    };
    const auto to_flat = [&](const Tensor& row_major) {
        return row_major.reshape(TensorShape{blocks, lanes, slots, 4}).permute({0, 2, 1, 3}).contiguous()
            .reshape(TensorShape{rows, attributes});
    };
    const auto index = Tensor::arange(0.f, static_cast<float>(cells));
    const auto second_1d = ((index.mod(Tensor::full({1}, 11.f, Device::GPU, DataType::Float32)) + 1.f) * 0.25f).to(DataType::Float32);
    const auto first_1d = (((index - static_cast<float>(cells) * 0.5f) * 0.01f) * (second_1d + 1.f)).to(DataType::Float32);
    const auto gradient_1d = ((index - static_cast<float>(cells) * 0.25f) * 0.003f).to(DataType::Float32);
    const auto second_flat = second_1d.reshape(TensorShape{rows, attributes});
    const auto first_flat = first_1d.reshape(TensorShape{rows, attributes});
    const auto gradient = gradient_1d.reshape(TensorShape{rows, attributes});
    auto packed = Tensor::zeros({cells * 2}, Device::GPU, DataType::UInt8);
    auto bounds = Tensor::zeros({rows / 256, 4}, Device::GPU);
    ASSERT_TRUE(internal::try_encode_joint_moments(first_flat, second_flat, packed, bounds, bits, block_cells, 1e-15f, nullptr));
    // The reference decodes through the codec first and permutes into the row-major
    // order the Adam step uses; the fused run is handed only the packed storage and is
    // expected to land on exactly the same values.
    Tensor decoded_first, decoded_second;
    ASSERT_TRUE(internal::try_decode_joint_moments(packed, bounds, bits, block_cells, 1e-15f,
                                                   decoded_first, decoded_second));
    const auto first_rows = to_rows(decoded_first);
    const auto second_rows = to_rows(decoded_second);
    const auto gradient_rows = to_rows(gradient);
    const internal::AdamUpdateConfig config{0.9f, 0.999f, 1.0f, 1e-8f, 1e-3f};

    auto ref_first = first_rows.clone(), ref_second = second_rows.clone(), ref_delta = Tensor{};
    internal::vulkan_adam_update(ref_first, ref_second, ref_delta, gradient_rows, Tensor{}, Tensor{},
                                 config);

    auto fused_first = Tensor::empty(TensorShape{rows, attributes}, Device::GPU);
    auto fused_second = Tensor::empty(TensorShape{rows, attributes}, Device::GPU);
    auto fused_delta = Tensor{};
    const internal::AdamMomentSource source{&packed, &bounds, static_cast<uint32_t>(slots),
                                           static_cast<uint32_t>(block_cells), bits, 1e-15f};
    internal::vulkan_adam_update(fused_first, fused_second, fused_delta, gradient_rows, Tensor{}, Tensor{},
                                 config, source);

    const auto compare = [&](const Tensor& fused, const Tensor& reference, const char* what) {
        const auto a = fused.contiguous().cpu().to_vector();
        const auto b = reference.contiguous().cpu().to_vector();
        ASSERT_EQ(a.size(), b.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < a.size(); ++i)
            mismatches += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        EXPECT_EQ(mismatches, 0u) << what << " differs in " << mismatches << " of " << a.size()
                                  << " values between fused and separate moment decode";
    };
    compare(fused_delta, ref_delta, "delta");
    compare(fused_first, ref_first, "first moment");
    compare(fused_second, ref_second, "second moment");
}

TEST_F(VulkanTensorRegression, SwizzlePermutationCommutesWithSliceUpdate) {
    // The Vulkan Adam parameter write relies on two properties:
    //   P(P^-1(values) - delta) == values - P(delta)
    // so the un-swizzle/re-swizzle pair around the update can be dropped, and
    // in-place subtraction through a view writes into the parent storage.
    constexpr size_t rows = 256, slots = 12, lanes = 32, attributes = slots * 4, blocks = rows / lanes;
    const size_t cells = rows * attributes;
    const auto to_rows = [&](const Tensor& flat) {
        return flat.reshape(TensorShape{blocks, slots, lanes, 4}).permute({0, 2, 1, 3}).contiguous()
            .reshape(TensorShape{rows, attributes});
    };
    const auto to_flat = [&](const Tensor& row_major) {
        return row_major.reshape(TensorShape{blocks, lanes, slots, 4}).permute({0, 2, 1, 3}).contiguous().reshape({-1});
    };
    const auto values = Tensor::arange(0.f, static_cast<float>(cells)).to(DataType::Float32);
    const auto delta = to_rows(Tensor::arange(0.f, static_cast<float>(cells)).to(DataType::Float32)) * 0.25f;

    const auto expected = values - to_flat(delta);
    const auto composed = to_flat(to_rows(values) - delta);
    ASSERT_EQ(composed.numel(), expected.numel());
    const auto composed_host = composed.cpu().to_vector();
    const auto expected_host = expected.cpu().to_vector();
    EXPECT_EQ(std::memcmp(composed_host.data(), expected_host.data(), cells * sizeof(float)), 0)
        << "swizzle round trip is not exact"
        << " max |diff| = " << (composed - expected).abs().max().item<float>();

    auto in_place = values.clone();
    auto view = in_place.slice(0, 0, cells);
    ASSERT_TRUE(view.is_contiguous());
    view.sub_(to_flat(delta));
    const auto in_place_host = in_place.cpu().to_vector();
    EXPECT_EQ(std::memcmp(in_place_host.data(), expected_host.data(), cells * sizeof(float)), 0)
        << "in-place view update did not match the materialized form";
}

TEST_F(VulkanTensorRegression, FusedMcmcNoiseMatchesSeededCovarianceReference) {
    float max_reference_difference = 0.f;
    for (const size_t count : {1, 31, 32, 33, 255, 256, 257}) {
        for (const uint64_t seed : {0ULL, 1ULL, 0xfedcba9876543210ULL}) {
            SCOPED_TRACE(::testing::Message() << count << "/" << seed);
            std::vector<float> opacity(count, -5.f), scales(count * 3), quats(count * 4), initial(count * 3), frozen(count);
            for (size_t i = 0; i < count; ++i) {
                frozen[i] = i % 7 == 1;
                for (size_t j = 0; j < 3; ++j) {
                    scales[i * 3 + j] = -float(j + 1) * .5f;
                    initial[i * 3 + j] = float(j) * .25f;
                }
                for (size_t j = 0; j < 4; ++j)
                    quats[i * 4 + j] = i % 5 == 0 ? 0.f : float(int((i + j) % 9) - 4) * .3f;
            }
            auto draws = Tensor::empty({2, count * 3}, Device::GPU);
            draws.uniform_(0.f, 1.f, seed);
            const auto uniforms = draws.cpu().to_vector();
            auto means = Tensor::from_vector(initial, {count, 3}, Device::GPU);
            internal::vulkan_mcmc_noise(Tensor::from_vector(opacity, {count}, Device::GPU),
                                        Tensor::from_vector(scales, {count, 3}, Device::GPU),
                                        Tensor::from_vector(quats, {count, 4}, Device::GPU), means,
                                        Tensor::from_vector(frozen, {count}, Device::GPU).to(DataType::Bool), .1f, seed);
            const auto actual = means.cpu().to_vector();
            auto reference = Tensor::from_vector(initial, {count, 3}, Device::GPU);
            const auto noise = ((draws.slice(0, 0, 1).squeeze(0).clamp_min(1e-7f).log() * -2.f).sqrt() *
                                (draws.slice(0, 1, 2).squeeze(0) * (2.f * std::numbers::pi_v<float>)).cos()).reshape(reference.shape());
            lfs::training::mcmc::add_noise(Tensor::from_vector(opacity, {count}, Device::GPU),
                                          Tensor::from_vector(scales, {count, 3}, Device::GPU),
                                          Tensor::from_vector(quats, {count, 4}, Device::GPU), noise, reference,
                                          Tensor::from_vector(frozen, {count}, Device::GPU).to(DataType::Bool), .1f);
            const auto previous = reference.cpu().to_vector();
            for (size_t i = 0; i < actual.size(); ++i)
                max_reference_difference = std::max(max_reference_difference, std::abs(actual[i] - previous[i]));
            for (size_t i = 0; i < count; ++i) {
                double q[4], squared_norm = 0;
                for (size_t j = 0; j < 4; ++j) squared_norm += double(quats[i * 4 + j]) * quats[i * 4 + j];
                for (size_t j = 0; j < 4; ++j) q[j] = quats[i * 4 + j] * std::min(1.0 / std::sqrt(squared_norm), 1e12);
                const double w = q[0], x = q[1], y = q[2], z = q[3];
                const double rotation[3][3] = {{1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)},
                                                {2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)},
                                                {2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)}};
                double noise[3];
                for (size_t j = 0; j < 3; ++j)
                    noise[j] = std::sqrt(-2.0 * std::log(std::max(double(uniforms[i * 3 + j]), 1e-7))) *
                               std::cos(2.0 * std::numbers::pi * uniforms[count * 3 + i * 3 + j]);
                const double factor = .1 / (1 + std::exp(100 / (1 + std::exp(-double(opacity[i]))) - .5));
                // Form the covariance independently, in double precision.
                for (size_t row = 0; row < 3; ++row) {
                    double delta = 0;
                    for (size_t col = 0; col < 3; ++col) {
                        double covariance = 0;
                        for (size_t axis = 0; axis < 3; ++axis)
                            covariance += rotation[row][axis] * std::exp(2.0 * scales[i * 3 + axis]) * rotation[col][axis];
                        delta += covariance * noise[col];
                    }
                    if (frozen[i]) EXPECT_EQ(actual[i * 3 + row], initial[i * 3 + row]);
                    else EXPECT_NEAR(actual[i * 3 + row], initial[i * 3 + row] + factor * delta, 2e-6);
                }
            }
        }
    }
    EXPECT_LE(max_reference_difference, 2e-6f);
    std::cout << "Maximum fused/unfused noise update difference: " << max_reference_difference << '\n';
}

TEST_F(VulkanTensorRegression, PhaseTimestampsFollowSubmittedWork) {
    // The GPU phase timing used to attribute the training loop must produce device
    // timestamps that advance with submitted work, not host call time. Without this the
    // per-phase numbers cannot be trusted at all, which is what made the backward look
    // expensive while its body and cooperative loads turned out to be free.
    lfs::core::vulkan_phase_timing_begin(64);
    if (!lfs::core::vulkan_phase_timing_supported()) {
        lfs::core::vulkan_phase_timing_end();
        GTEST_SKIP() << "device cannot write timestamps";
    }

    std::vector<Tensor> results;
    constexpr int kBoundaries = 3;
    for (int step = 0; step < kBoundaries; ++step) {
        lfs::core::vulkan_phase_timing_announce(static_cast<std::uint32_t>(step));
        // Real work between boundaries so the stamps must advance.
        auto value = Tensor::full({1u << 20}, 1.0f, Device::GPU);
        for (int repeat = 0; repeat < 4; ++repeat)
            value = value + 1.0f;
        results.push_back(value);
    }
    // Drain before reading: the stamps are only valid once the submissions completed.
    for (const auto& result : results)
        (void)result.slice(0, 0, 1).cpu().to_vector();

    const auto stamps = lfs::core::vulkan_phase_timestamps();
    lfs::core::vulkan_phase_timing_end();
    ASSERT_EQ(stamps.size(), size_t{kBoundaries});
    for (size_t i = 0; i < stamps.size(); ++i)
        EXPECT_EQ(stamps[i].tag, i);
    for (size_t i = 1; i < stamps.size(); ++i)
        EXPECT_GE(stamps[i].gpu_ms, stamps[i - 1].gpu_ms)
            << "device timestamps went backwards between boundaries " << i - 1 << " and " << i;
    EXPECT_GT(stamps.back().gpu_ms - stamps.front().gpu_ms, 0.0)
        << "the device clock did not advance across submitted work";
}

TEST_F(VulkanTensorRegression, FloatingMaskedFillPreservesInfinity) {
    auto values = Tensor::zeros({4}, Device::GPU);
    values.masked_fill_(Tensor::ones_bool({4}, Device::GPU), std::numeric_limits<float>::infinity());
    for (float value : values.cpu().to_vector())
        EXPECT_EQ(value, std::numeric_limits<float>::infinity());
}

TEST_F(VulkanTensorRegression, SubnormalTruthAgreesAcrossOperations) {
    for (uint32_t bits : {1u, 0x80000001u, 0u, 0x80000000u}) {
        const auto value = Tensor::from_vector({std::bit_cast<float>(bits)}, {1}, Device::GPU);
        const bool nonzero = (bits & 0x7fffffffu) != 0;
        EXPECT_EQ(value.count_nonzero(), size_t(nonzero));
        EXPECT_EQ(value.nonzero().numel(), size_t(nonzero));
        EXPECT_EQ(value.logical_not().cpu().ptr<bool>()[0], !nonzero);
        EXPECT_EQ(value.to(DataType::Bool).cpu().ptr<bool>()[0], nonzero);
    }
}

TEST_F(VulkanTensorRegression, BroadcastIntegerPowerUsesIntegerArithmetic) {
    for (const auto dtype : {DataType::Int32, DataType::Int64}) {
        const auto base = Tensor::from_vector({-2, 3}, {2}, Device::GPU).to(dtype);
        const auto exponent = Tensor::from_vector({3}, {1}, Device::GPU).to(dtype);
        EXPECT_EQ(base.pow(exponent).cpu().to(DataType::Int64).to_vector_int64(),
                  (std::vector<int64_t>{-8, 27}));
    }
}

TEST_F(VulkanTensorRegression, AffineParametersKeepTheirFeatureAxis) {
    const auto input = Tensor::from_vector({1.f, 2.f, 3.f, 4.f}, {2, 2}, Device::GPU);
    const auto identity = Tensor::from_vector({1.f, 0.f, 0.f, 1.f}, {2, 2}, Device::GPU);
    const auto parameter = Tensor::from_vector({10.f, 20.f}, {2, 1}, Device::GPU);
    EXPECT_EQ(nn::gemm(input, identity, false, true, &parameter).cpu().to_vector(),
              (std::vector<float>{11, 22, 13, 24}));
    EXPECT_EQ(nn::gemm(input, identity, false, true, nullptr, nn::Activation::None, nullptr, &parameter).cpu().to_vector(),
              (std::vector<float>{10, 40, 30, 80}));
    const auto bias = Tensor::zeros({2, 1}, Device::GPU);
    const auto normalized = nn::layer_norm(input, parameter, bias).cpu().to_vector();
    for (size_t i = 0; i < normalized.size(); ++i)
        EXPECT_NEAR(normalized[i], i % 2 ? 20.f : -10.f, 0.001f);
}

TEST_F(VulkanTensorRegression, MultinomialNeverSelectsZeroWeight) {
    std::vector<float> values(257, 1.f);
    values.front() = 100000000.f;
    values.back() = 0.f;
    const auto weights = Tensor::from_vector(values, {values.size()}, Device::GPU);
    // This seed previously selected the trailing zero after inconsistent sums.
    const auto samples = Tensor::multinomial(weights, 1024, true, 8958351).cpu().to_vector_int64();
    for (const auto index : samples) {
        ASSERT_GE(index, 0);
        ASSERT_LT(size_t(index), values.size());
        EXPECT_GT(values[index], 0.f);
    }
}

TEST_F(VulkanTensorRegression, MultinomialPreservesFiniteWeightScale) {
    for (float scale : {1.f, std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()}) {
        const auto weights = Tensor::from_vector({scale, 0.f, scale}, {3}, Device::GPU);
        const auto reference = Tensor::from_vector({1.f, 0.f, 1.f}, {3}, Device::GPU);
        EXPECT_EQ(Tensor::multinomial(weights, 4096, true, 19).cpu().to_vector_int64(),
                  Tensor::multinomial(reference, 4096, true, 19).cpu().to_vector_int64());
    }
}

TEST_F(VulkanTensorRegression, PackedShRetainsPaddingAndOwnedOutput) {
    for (size_t n : {1, 31, 32, 33}) {
        for (const auto dtype : {DataType::Float16, DataType::Float32}) {
            const auto canonical = Tensor::full({n, 1, 3}, 7.f, Device::GPU, dtype);
            auto packed = reorder_sh_to_swizzled(canonical, n, 1, 15);
            EXPECT_FALSE(packed.is_view());
            const auto unpacked = undo_reorder_sh_from_swizzled(packed, n, 15, 15).cpu().to(DataType::Float32);
            for (size_t i = 0; i < unpacked.numel(); ++i)
                EXPECT_EQ(unpacked.ptr<float>()[i], i % 45 < 3 ? 7.f : 0.f);
            const auto retained = packed;
            const auto before = retained.cpu().clone();
            packed = Tensor::zeros(packed.shape(), Device::GPU, dtype);
            const auto after = retained.cpu();
            EXPECT_EQ(std::memcmp(before.data_ptr(), after.data_ptr(), before.bytes()), 0);
        }
    }
}

TEST_F(VulkanTensorRegression, TypedConstantsMatchHostConversion) {
    for (auto dtype : {DataType::Float16, DataType::Int32, DataType::Int64}) {
        for (float value : {-7.75f, 0.f, -0.f, 7.f}) {
            // The GPU factory canonicalizes floating zeros through memset.
            const auto expected = Tensor::full({257}, value == 0.f ? 0.f : value, Device::CPU, dtype);
            const auto actual = Tensor::full({257}, value, Device::GPU, dtype).cpu();
            EXPECT_EQ(std::memcmp(expected.data_ptr(), actual.data_ptr(), actual.bytes()), 0);
        }
    }
}

#ifdef LFS_TENSOR_VULKAN
TEST_F(VulkanTensorRegression, WaitingExternalWorkIsRejectedAfterQuarantine) {
    std::promise<void> entered, release, caller_started;
    auto released = release.get_future();
    auto first = std::async(std::launch::async, [&] {
        with_idle_vulkan_device([&](const auto&) {
            entered.set_value();
            released.wait();
        });
    });
    entered.get_future().wait();
    std::atomic<bool> called{false};
    auto second = std::async(std::launch::async, [&] {
        caller_started.set_value();
        with_idle_vulkan_device([&](const auto&) { called = true; });
    });
    caller_started.get_future().wait();
    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    invalidate_vulkan_backend(lfs::make_error(lfs::ErrorInit{
        .code = lfs::ErrorCode::DeadlineExceeded,
        .domain = lfs::ErrorDomain::Vulkan,
        .user_message = "Quarantine while an external caller waits",
        .detection = LFS_SOURCE_SITE_CURRENT(),
    }));
    release.set_value();
    EXPECT_NO_THROW(first.get());
    EXPECT_THROW(second.get(), lfs::Exception);
    EXPECT_FALSE(called);
    EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
}

TEST_F(VulkanTensorRegression, NestedRetirementPreservesCleanupAndExceptionBoundaries) {
    void* device = nullptr;
    with_idle_vulkan_device([&](const auto& external) { device = external.handles.device; });
    int retired = 0;
    EXPECT_THROW(retire_vulkan_resources(device, [&] {
                     retire_vulkan_resources(device, [&] { ++retired; });
                     throw std::runtime_error("cleanup failed");
                 }),
                 std::runtime_error);
    EXPECT_EQ(retired, 1);
    EXPECT_NO_THROW(retire_vulkan_resources(device, [&] {
        retire_vulkan_resources(device, [&] { ++retired; });
    }));
    EXPECT_EQ(retired, 2);
    EXPECT_NO_THROW(with_idle_vulkan_device([](const auto&) {}));
}

TEST_F(VulkanTensorRegression, BoundsFaultIsConsumedAndSubsequentReadSucceeds) {
    const auto input = Tensor::from_vector({1.f, 2.f, 3.f}, {3}, Device::GPU);
    const auto indices = Tensor::from_vector({99}, {1}, Device::GPU);
    try {
        (void)input.index_select(0, indices, BoundaryMode::Assert).cpu();
        FAIL() << "Expected an out-of-range device fault";
    } catch (const lfs::Exception& error) {
        EXPECT_EQ(error.error().code(), lfs::ErrorCode::BoundsViolation);
        EXPECT_EQ(error.error().domain(), lfs::ErrorDomain::Vulkan);
        EXPECT_NE(lfs::format_for_developer(error.error()).find("99"), std::string::npos);
    }
    EXPECT_EQ(input.cpu().to_vector(), (std::vector<float>{1.f, 2.f, 3.f}));
}

TEST_F(VulkanTensorRegression, ExternalCompletionWaitsForSubmittedWorkIncludingFailure) {
    for (bool fail_callback : {false, true}) {
        // Establish a tensor timeline value before the external completion marker.
        auto values = Tensor::zeros({1024}, Device::GPU);
        EXPECT_EQ(values.cpu().to_vector().front(), 0.f);
        VkDevice device = VK_NULL_HANDLE;
        VkSemaphore gate = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        with_idle_vulkan_device([&](const VulkanExternalDevice& external) {
            device = static_cast<VkDevice>(external.handles.device);
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            semaphore_info.pNext = &type;
            ASSERT_EQ(vkCreateSemaphore(device, &semaphore_info, nullptr, &gate), VK_SUCCESS);
            const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            ASSERT_EQ(vkCreateFence(device, &fence_info, nullptr, &fence), VK_SUCCESS);
        });
        ASSERT_NE(gate, VK_NULL_HANDLE);
        ASSERT_NE(fence, VK_NULL_HANDLE);
        // Put another tensor submission immediately before the tested marker.
        values.zero_();
        std::jthread release([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = gate;
            signal.value = 1;
            EXPECT_EQ(vkSignalSemaphore(device, &signal), VK_SUCCESS);
        });
        bool threw = false;
        try {
            with_idle_vulkan_device([&](const VulkanExternalDevice& external) {
                VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                wait.semaphore = gate;
                wait.value = 1;
                wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
                submit.waitSemaphoreInfoCount = 1;
                submit.pWaitSemaphoreInfos = &wait;
                ASSERT_EQ(vkQueueSubmit2(static_cast<VkQueue>(external.handles.queue), 1, &submit, fence), VK_SUCCESS);
                if (fail_callback)
                    throw std::runtime_error("callback failed after submission");
            });
        } catch (const std::runtime_error& error) {
            threw = true;
            EXPECT_STREQ(error.what(), "callback failed after submission");
        }
        EXPECT_EQ(threw, fail_callback);
        EXPECT_EQ(vkGetFenceStatus(device, fence), VK_SUCCESS);
        release.join();
        ASSERT_EQ(vkWaitForFences(device, 1, &fence, VK_TRUE, 5'000'000'000ull), VK_SUCCESS);
        vkDestroyFence(device, fence, nullptr);
        vkDestroySemaphore(device, gate, nullptr);
    }
}
TEST_F(VulkanTensorRegression, QuarantineRejectsNewWorkWithoutAborting) {
    {
        auto tensor = Tensor::zeros({16}, Device::GPU);
        EXPECT_EQ(tensor.cpu().to_vector().front(), 0.f);
        invalidate_vulkan_backend(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::DeadlineExceeded,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "Injected timeout after completed work",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
        EXPECT_FALSE(vulkan_backend_status());
        EXPECT_FALSE(gpu_backend_available(GpuBackend::Vulkan));
        EXPECT_THROW(with_idle_vulkan_device([](const auto&) {}), lfs::Exception);
        EXPECT_THROW(Tensor::zeros({16}, Device::GPU), std::exception);
    }
    // A healthy-but-quarantined test device can still drain through shutdown.
    EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
}

TEST_F(VulkanTensorRegression, ExternalCallbackFailurePreservesTerminalCompletionError) {
    VkDevice device = VK_NULL_HANDLE;
    try {
        with_idle_vulkan_device([&](const VulkanExternalDevice& external) {
            device = static_cast<VkDevice>(external.handles.device);
            invalidate_vulkan_backend(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DeviceLost,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "Injected device loss",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
            throw std::runtime_error("original callback failure");
        });
        ADD_FAILURE() << "Completion on a lost backend must fail";
    } catch (const lfs::Exception& error) {
        EXPECT_EQ(error.error().code(), lfs::ErrorCode::DeviceLost);
        EXPECT_FALSE(error.error().suppressed().empty());
        if (!error.error().suppressed().empty())
            EXPECT_NE(lfs::format_for_developer(error.error().suppressed().front()).find("original callback failure"), std::string::npos);
    }
    // The injected loss did not affect the real device: drain its marker before teardown.
    ASSERT_NE(device, VK_NULL_HANDLE);
    EXPECT_EQ(vkDeviceWaitIdle(device), VK_SUCCESS);
    EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
}
#endif
