/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/nn.hpp"
#include "core/tensor_backend.hpp"
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
#include <limits>
#include <optional>

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
