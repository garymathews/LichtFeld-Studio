/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Gradient checks for the Vulkan training rasterizer backward.
//
// Any backward experiment (analytic adjoint, Gaussian-owned backward, the per-splat
// variant) changes how splat derivatives are produced and cannot be judged without an
// independent reference. These tests render small controlled models, take one backward
// pass, and compare the accumulated parameter gradients against central finite
// differences of the same forward.
//
// The loss is a plain squared error on the rendered image, so the reference is smooth and
// no clipping rule inside the trainable path is being probed. The cases that *do* sit on a
// boundary (alpha-threshold rejection, saturated contributor tails) assert the boundary
// behaviour explicitly instead. Only a few entries per parameter are perturbed, because
// each one costs two forwards.

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/vulkan_rasterizer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using namespace lfs::core;
namespace training = lfs::training;

namespace {
    constexpr int kWidth = 32;
    constexpr int kHeight = 32;
    constexpr int kGaussians = 8;

    [[nodiscard]] Tensor gpu(const std::vector<float>& values, const TensorShape shape) {
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    // Central-difference step per parameter, chosen from measurement rather than taste.
    // With eps = 1e-3 the positional gradient agrees with the analytic one to 4-5 digits
    // and the numeric derivative is stable across 1e-4..1e-2; the finite-difference noise
    // floor is ~6e-8 absolute, which is what makes the 5e-2 relative tolerance below
    // meaningful. Scaling and opacity change a splat's coverage, so a step that large moves
    // pixels across the alpha threshold and the numeric derivative scatters by tens of
    // percent instead of converging; they need the smaller step.
    struct ParameterCase {
        const char* name;
        training::ParamType type;
        Tensor* tensor;
        float epsilon;
    };

    std::vector<ParameterCase> parameter_cases(SplatData& model) {
        return {{"means", training::ParamType::Means, &model.means(), 1e-3f},
                {"sh0", training::ParamType::Sh0, &model.sh0(), 1e-3f},
                {"scaling_raw", training::ParamType::Scaling, &model.scaling_raw(), 1e-4f},
                {"rotation_raw", training::ParamType::Rotation, &model.rotation_raw(), 1e-3f},
                {"opacity_raw", training::ParamType::Opacity, &model.opacity_raw(), 1e-4f}};
    }

    [[nodiscard]] Camera make_camera(const int width, const int height) {
        const auto identity = std::vector<float>{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        const auto origin = std::vector<float>{0.f, 0.f, 0.f};
        return Camera(gpu(identity, TensorShape{3, 3}), gpu(origin, TensorShape{3}),
                      /*focal_x=*/1.5f * float(width), /*focal_y=*/1.5f * float(height),
                      /*center_x=*/float(width) * 0.5f, /*center_y=*/float(height) * 0.5f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                      "gradient-check", std::filesystem::path{}, std::filesystem::path{},
                      width, height, /*uid=*/0);
    }

    // Eight small splats spread across the view frustum, no SH rest, so every trainable
    // parameter is exercised without needing a degree schedule.
    [[nodiscard]] SplatData make_spread_model() {
        std::vector<float> means, colors, scales, rotations, opacities;
        for (int i = 0; i < kGaussians; ++i) {
            const float u = float(i % 4) * 0.5f - 0.75f;
            const float v = float(i / 4) * 0.5f - 0.25f;
            means.insert(means.end(), {u, v, 4.0f + 0.1f * float(i)});
            const float tint = 0.2f + 0.1f * float(i);
            colors.insert(colors.end(), {tint, 1.0f - tint, 0.5f});
            scales.insert(scales.end(), {-1.2f, -1.4f, -1.6f});
            rotations.insert(rotations.end(), {1.0f, 0.0f, 0.0f, 0.0f});
            opacities.push_back(1.5f);
        }
        return SplatData(/*sh_degree=*/0,
                         gpu(means, TensorShape{kGaussians, 3}),
                         gpu(colors, TensorShape{kGaussians, 1, 3}),
                         Tensor{},
                         gpu(scales, TensorShape{kGaussians, 3}),
                         gpu(rotations, TensorShape{kGaussians, 4}),
                         gpu(opacities, TensorShape{kGaussians, 1}),
                         /*scene_scale=*/1.0f);
    }

    [[nodiscard]] Tensor render_now(training::VulkanTrainingRasterizer& rasterizer, SplatData& model,
                                    const Camera& camera) {
        return rasterizer.forward(camera, model, {0.f, 0.f, 0.f}).image.contiguous();
    }

    // A horizontally shifted copy of the render: a non-zero loss where every lit pixel
    // contributes a gradient.
    [[nodiscard]] Tensor shifted_target(const Tensor& rendered, const int width, const int height, const int shift) {
        const auto host = rendered.cpu().to_vector();
        std::vector<float> shifted(host.size(), 0.f);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t row = (size_t(c) * size_t(height) + size_t(y)) * size_t(width);
                    shifted[row + size_t(x)] = host[row + size_t((x + shift) % width)];
                }
            }
        }
        return gpu(shifted, TensorShape{size_t{3}, size_t(height), size_t(width)});
    }

    [[nodiscard]] double squared_error_loss(const Tensor& image, const Tensor& target) {
        const auto values = (image - target).cpu().to_vector();
        double total = 0.0;
        for (const float value : values)
            total += double(value) * double(value);
        return 0.5 * total / double(values.size());
    }

    [[nodiscard]] int lit_values(const Tensor& image, const float threshold = 1e-4f) {
        const auto values = image.cpu().to_vector();
        return static_cast<int>(std::count_if(values.begin(), values.end(),
                                              [threshold](const float value) { return value > threshold; }));
    }

    struct CheckOutcome {
        double worst_relative = 0.0;
        std::string worst_description;
        size_t checked = 0;
    };

    // Compares the analytic gradients of the *last* backward against central differences of
    // the forward, for a sample of entries in each parameter group.
    [[nodiscard]] CheckOutcome check_gradients_against_finite_differences(
        training::VulkanTrainingRasterizer& rasterizer, training::AdamOptimizer& optimizer, SplatData& model,
        const Camera& camera, const Tensor& target, const std::vector<ParameterCase>& cases,
        const size_t samples_per_parameter) {
        CheckOutcome outcome;
        for (const auto& entry : cases) {
            const auto analytic = optimizer.get_grad(entry.type).cpu().to_vector();
            const auto original = entry.tensor->cpu().to_vector();
            EXPECT_EQ(analytic.size(), original.size());
            const size_t stride = std::max<size_t>(1, original.size() / samples_per_parameter);
            for (size_t index = 0; index < original.size(); index += stride) {
                const auto loss_at = [&](const float delta) {
                    auto values = original;
                    values[index] += delta;
                    entry.tensor->copy_from(Tensor::from_vector(values, entry.tensor->shape(), Device::GPU));
                    return squared_error_loss(render_now(rasterizer, model, camera), target);
                };
                const double plus = loss_at(entry.epsilon);
                const double minus = loss_at(-entry.epsilon);
                entry.tensor->copy_from(Tensor::from_vector(original, entry.tensor->shape(), Device::GPU));

                const double numeric = (plus - minus) / (2.0 * double(entry.epsilon));
                const double analytic_value = double(analytic[index]);
                const double scale = std::max({std::abs(numeric), std::abs(analytic_value), 1e-8});
                const double relative = std::abs(numeric - analytic_value) / scale;
                ++outcome.checked;
                if (relative > outcome.worst_relative) {
                    outcome.worst_relative = relative;
                    outcome.worst_description = std::string(entry.name) + "[" + std::to_string(index) + "] numeric=" +
                                                std::to_string(numeric) + " analytic=" + std::to_string(analytic_value);
                }
                EXPECT_LE(relative, 5e-2) << entry.name << "[" << index << "]: numeric " << numeric
                                          << " vs analytic " << analytic_value;
            }
        }
        return outcome;
    }

    // Shared fixture: Vulkan backend plus the model/camera/rasterizer plumbing.
    class VulkanBackwardGradient : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GpuBackend::Vulkan))
                GTEST_SKIP() << "Vulkan backend unavailable";
            backend_.emplace(GpuBackend::Vulkan);
        }

        [[nodiscard]] static double gradient_magnitude(training::AdamOptimizer& optimizer,
                                                       const training::ParamType type, const size_t row,
                                                       const size_t width) {
            const auto values = optimizer.get_grad(type).cpu().to_vector();
            double total = 0.0;
            for (size_t c = 0; c < width; ++c)
                total += std::abs(double(values[row * width + c]));
            return total;
        }

        std::optional<GpuBackendScope> backend_;
    };
} // namespace

#if !LFS_TENSOR_CUDA
TEST_F(VulkanBackwardGradient, MatchesCentralFiniteDifferences) {
    auto model = make_spread_model();
    const auto camera = make_camera(kWidth, kHeight);
    training::VulkanTrainingRasterizer rasterizer;

    const auto rendered = render_now(rasterizer, model, camera);
    ASSERT_EQ(rendered.numel(), size_t{3} * kWidth * kHeight);
    // A vacuous check would pass on an empty image, so the reference must be a real one.
    ASSERT_GT(lit_values(rendered), 32) << "the test model does not render anything";

    const auto target = shifted_target(rendered, kWidth, kHeight, /*shift=*/2);
    ASSERT_GT(squared_error_loss(rendered, target), 0.0) << "the reference loss is degenerate";
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();

    training::AdamOptimizer optimizer(model, training::AdamConfig{});
    optimizer.zero_grad(0);
    rasterizer.backward(image_gradient, optimizer);

    // Every group must receive a gradient at all before any of them is trusted.
    const auto cases = parameter_cases(model);
    for (const auto& entry : cases) {
        const auto values = optimizer.get_grad(entry.type).cpu().to_vector();
        const double magnitude = std::accumulate(values.begin(), values.end(), 0.0,
                                                [](double sum, float value) { return sum + std::abs(double(value)); });
        EXPECT_GT(magnitude, 0.0) << "no gradient reached " << entry.name;
    }

    const auto outcome = check_gradients_against_finite_differences(rasterizer, optimizer, model, camera,
                                                                    target, cases, /*samples_per_parameter=*/3);
    std::cout << "worst relative gradient error: " << outcome.worst_relative << " (" << outcome.worst_description
              << ") over " << outcome.checked << " entries\n";
}

TEST_F(VulkanBackwardGradient, MatchesFiniteDifferencesAcrossSubdividedRegions) {
    // A deliberately small instance budget forces the forward to subdivide the image and
    // the backward to replay each region, which is the path large training images take.
    constexpr int width = 64, height = 64;
    auto model = make_spread_model();
    const auto camera = make_camera(width, height);
    training::VulkanTrainingRasterizer rasterizer({}, /*max_tile_instances=*/8);

    const auto rendered = render_now(rasterizer, model, camera);
    ASSERT_GT(lit_values(rendered), 32) << "the subdivided model does not render anything";

    const auto target = shifted_target(rendered, width, height, /*shift=*/3);
    ASSERT_GT(squared_error_loss(rendered, target), 0.0);
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();

    training::AdamOptimizer optimizer(model, training::AdamConfig{});
    optimizer.zero_grad(0);
    rasterizer.backward(image_gradient, optimizer);

    auto cases = parameter_cases(model);
    cases.resize(3); // means, sh0, scaling_raw: enough to cover positions and extents
    const auto outcome = check_gradients_against_finite_differences(rasterizer, optimizer, model, camera,
                                                                    target, cases, /*samples_per_parameter=*/2);
    std::cout << "subdivided regions: worst relative gradient error " << outcome.worst_relative << " over "
              << outcome.checked << " entries\n";
}

TEST_F(VulkanBackwardGradient, RejectsGaussiansBelowTheAlphaThreshold) {
    // A splat whose alpha never reaches the rasterizer's threshold contributes nothing, so
    // its gradients must be exactly zero rather than merely small. The accepted splat in the
    // same scene is the control: the scene must still render and differentiate normally.
    auto model = SplatData(/*sh_degree=*/0,
                           gpu({0.f, 0.f, 4.f, 0.2f, 0.2f, 4.2f}, TensorShape{2, 3}),
                           gpu({0.3f, 0.6f, 0.2f, 0.3f, 0.6f, 0.2f}, TensorShape{2, 1, 3}),
                           Tensor{},
                           gpu({-1.2f, -1.4f, -1.6f, -1.2f, -1.4f, -1.6f}, TensorShape{2, 3}),
                           gpu({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}, TensorShape{2, 4}),
                           /*opacity_raw: sigmoid(-12) is far below 0.5/255*/ gpu({1.5f, -12.f}, TensorShape{2, 1}),
                           /*scene_scale=*/1.0f);
    const auto camera = make_camera(kWidth, kHeight);
    training::VulkanTrainingRasterizer rasterizer;

    const auto rendered = render_now(rasterizer, model, camera);
    ASSERT_GT(lit_values(rendered), 32) << "the control splat did not render";

    const auto target = shifted_target(rendered, kWidth, kHeight, /*shift=*/2);
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();
    training::AdamOptimizer optimizer(model, training::AdamConfig{});
    optimizer.zero_grad(0);
    rasterizer.backward(image_gradient, optimizer);

    for (const auto& [type, width] : {std::pair{training::ParamType::Means, size_t{3}},
                                      std::pair{training::ParamType::Scaling, size_t{3}},
                                      std::pair{training::ParamType::Rotation, size_t{4}},
                                      std::pair{training::ParamType::Opacity, size_t{1}}}) {
        const double accepted = gradient_magnitude(optimizer, type, 0, width);
        const double rejected = gradient_magnitude(optimizer, type, 1, width);
        // Rotation is asserted only for rejection. Measured over 30 runs of this scene, the
        // accepted splat's rotation gradient sits at the fp32-atomic noise floor (every other
        // group is orders of magnitude above it: Means 6.9e-03, Scaling 2.7e-04, Opacity
        // 1.1e-04), so demanding it be non-zero would be a coin flip on rounding - and was,
        // at roughly a 15% failure rate - rather than a statement about the backward. The
        // rejected splat is exactly zero for it, which is the property this test is about.
        if (type == training::ParamType::Rotation) {
            EXPECT_EQ(rejected, 0.0) << "the rejected splat must contribute exactly nothing";
            continue;
        }
        EXPECT_GT(accepted, 0.0) << "the accepted splat received no gradient";
        // Relative, not absolute: the rejected splat must not contribute materially. An
        // absolute floor is flaky because the comparison is against accumulated fp32
        // atomics across the whole image.
        EXPECT_LT(rejected, 1e-6 * accepted)
            << "the rejected splat received a material gradient (" << rejected << " vs " << accepted << ")";
    }

    auto cases = parameter_cases(model);
    cases.resize(1); // means only; the rejected row is covered by the assertions above
    const auto outcome = check_gradients_against_finite_differences(rasterizer, optimizer, model, camera,
                                                                    target, cases, /*samples_per_parameter=*/2);
    std::cout << "threshold rejection: worst relative gradient error " << outcome.worst_relative << "\n";
}

TEST_F(VulkanBackwardGradient, SuppressesGradientsOfSaturatedContributorTails) {
    // Four splats stacked along the view ray: the nearest one alone drives the pixel close
    // to saturation, so the ones behind it must contribute almost nothing. The tail is the
    // path the contributor-tail experiments touch, and the front splat is the control that
    // the same scene still differentiates normally.
    constexpr size_t stack = 6;
    std::vector<float> means, scales, rotations, colors, opacities;
    for (size_t i = 0; i < stack; ++i) {
        means.insert(means.end(), {0.f, 0.f, 4.0f + 0.02f * float(i)});
        colors.insert(colors.end(), {0.4f, 0.4f, 0.4f});
        scales.insert(scales.end(), {-1.0f, -1.0f, -1.0f});
        rotations.insert(rotations.end(), {1.0f, 0.0f, 0.0f, 0.0f});
        opacities.push_back(10.0f); // sigmoid ~ 0.99995
    }
    auto model = SplatData(/*sh_degree=*/0,
                           gpu(means, TensorShape{stack, 3}),
                           gpu(colors, TensorShape{stack, 1, 3}),
                           Tensor{},
                           gpu(scales, TensorShape{stack, 3}),
                           gpu(rotations, TensorShape{stack, 4}),
                           gpu(opacities, TensorShape{stack, 1}),
                           /*scene_scale=*/1.0f);
    const auto camera = make_camera(kWidth, kHeight);
    training::VulkanTrainingRasterizer rasterizer;

    const auto rendered = render_now(rasterizer, model, camera);
    ASSERT_GT(lit_values(rendered), 32) << "the stacked splats did not render";

    // Target: the render everywhere except a black block at the centre. Concentrating the
    // loss on the saturated pixels is what makes this a test of tail suppression: away from
    // the centre every splat is only partially covered and all of them legitimately
    // contribute, which is why a whole-image loss shows a flat gradient across the stack.
    const auto centre = [](const int extent, const int limit) { return std::clamp(extent / 2 - 2, 0, limit - 4); };
    std::vector<float> target_host = rendered.cpu().to_vector();
    for (int c = 0; c < 3; ++c) {
        for (int y = centre(kHeight, kHeight); y < centre(kHeight, kHeight) + 4; ++y) {
            for (int x = centre(kWidth, kWidth); x < centre(kWidth, kWidth) + 4; ++x) {
                target_host[(size_t(c) * kHeight + size_t(y)) * kWidth + size_t(x)] = 0.f;
            }
        }
    }
    const auto target = gpu(target_host, TensorShape{size_t{3}, size_t(kHeight), size_t(kWidth)});
    const double loss = squared_error_loss(rendered, target);
    std::cout << "stacked splats: loss=" << loss << " lit=" << lit_values(rendered) << "\n";
    ASSERT_GT(loss, 1e-4) << "the reference loss is too small to differentiate";
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();
    training::AdamOptimizer optimizer(model, training::AdamConfig{});
    optimizer.zero_grad(0);
    rasterizer.backward(image_gradient, optimizer);

    std::vector<double> magnitudes;
    for (size_t row = 0; row < stack; ++row)
        magnitudes.push_back(gradient_magnitude(optimizer, training::ParamType::Opacity, row, 1));
    std::cout << "opacity gradient magnitudes (near-to-far):";
    for (const double value : magnitudes)
        std::cout << " " << value;
    std::cout << "\n";

    // Transmittance falls geometrically along the stack, so each successive splat must
    // contribute less than the one in front of it, and the splats behind the transmittance
    // break must receive exactly nothing rather than merely little. A rewrite that changes
    // which contributors are evaluated has to preserve both.
    EXPECT_GT(magnitudes.front(), 0.0) << "the nearest splat received no gradient";
    for (size_t row = 1; row < magnitudes.size(); ++row) {
        EXPECT_LE(magnitudes[row], magnitudes[row - 1])
            << "splat " << row << " received more gradient than the splat in front of it";
    }
    EXPECT_EQ(magnitudes.back(), 0.0) << "the splat behind the transmittance break kept a gradient";

    // No finite-difference check in this case on purpose: the scene sits exactly on the
    // transmittance break, so a perturbed splat crosses it and the numeric derivative is
    // meaningless. The suppression assertions above are the check, and they are the
    // behaviour a contributor-tail rewrite has to preserve.
}

TEST_F(VulkanBackwardGradient, ErrorMapFeedsDensificationOnly) {
    // The densification statistics are the blend weight and that weight scaled by the pixel
    // error, so an all-zero error map must leave the second row exactly zero, an all-ones
    // error map must make the two rows identical, and neither may change the parameter
    // gradients at all.
    auto model = make_spread_model();
    const auto camera = make_camera(kWidth, kHeight);
    training::VulkanTrainingRasterizer rasterizer;

    const auto rendered = render_now(rasterizer, model, camera);
    const auto target = shifted_target(rendered, kWidth, kHeight, /*shift=*/2);
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();

    training::AdamOptimizer optimizer(model, training::AdamConfig{});
    const size_t n = model.size();

    const auto statistics = [&] {
        const auto rows = model._densification_info.cpu().to_vector();
        double first = 0.0, second = 0.0;
        for (size_t i = 0; i < n; ++i) {
            first += std::abs(double(rows[i]));
            second += std::abs(double(rows[n + i]));
        }
        return std::pair{first, second};
    };
    const auto run_with_error_map = [&](const double error) {
        model._densification_info = Tensor::zeros({2, n}, Device::GPU);
        auto error_map = Tensor::full({kHeight, kWidth}, float(error), Device::GPU);
        rasterizer.forward(camera, model, {0.f, 0.f, 0.f});
        optimizer.zero_grad(0);
        rasterizer.backward(image_gradient, optimizer, {}, error_map);
    };

    run_with_error_map(0.0);
    const auto plain_gradients = optimizer.get_grad(training::ParamType::Means).cpu().to_vector();
    const auto [weight_zero_error, error_zero] = statistics();
    EXPECT_GT(weight_zero_error, 0.0) << "no blend weights were accumulated";
    EXPECT_EQ(error_zero, 0.0) << "a zero error map produced error statistics";

    run_with_error_map(1.0);
    const auto [weight, with_error] = statistics();
    const auto error_gradients = optimizer.get_grad(training::ParamType::Means).cpu().to_vector();

    EXPECT_NEAR(with_error, weight, 1e-4 * std::max(weight, 1.0)) << "error row is not weight times the error map";
    double worst_shift = 0.0;
    for (size_t i = 0; i < plain_gradients.size(); ++i) {
        const double scale = std::max(std::abs(double(plain_gradients[i])), 1e-12);
        worst_shift = std::max(worst_shift, std::abs(double(error_gradients[i]) - double(plain_gradients[i])) / scale);
        EXPECT_NEAR(double(error_gradients[i]), double(plain_gradients[i]), 1e-4 * scale)
            << "the error map changed parameter gradient " << i;
    }
    std::cout << "error map: weight=" << weight << " error_row=" << with_error
              << " worst relative gradient shift=" << worst_shift << "\n";
}
TEST_F(VulkanBackwardGradient, SecondBackwardReplacesEveryGradientElement) {
    // The backward replaces the gradient buffers it accumulates into rather than adding to what a
    // previous call left behind, which is why the host no longer clears them first. Calling it
    // again with a zero upstream gradient must therefore leave exactly zero in every group. A
    // buffer that were merely accumulated, or one the projection epilogue failed to overwrite on
    // some lane, would keep the first call's values.
    auto model = make_spread_model();
    const auto camera = make_camera(kWidth, kHeight);
    training::VulkanTrainingRasterizer rasterizer;

    const auto rendered = render_now(rasterizer, model, camera);
    ASSERT_GT(lit_values(rendered), 32) << "the test model does not render anything";
    const auto target = shifted_target(rendered, kWidth, kHeight, /*shift=*/2);
    const auto image_gradient = ((rendered - target) * (1.0 / double(rendered.numel()))).contiguous();

    training::AdamOptimizer optimizer(model, training::AdamConfig{});

    optimizer.zero_grad(0);
    rasterizer.backward(image_gradient, optimizer);
    double reference = 0.0;
    for (const auto& entry : parameter_cases(model)) {
        const auto values = optimizer.get_grad(entry.type).cpu().to_vector();
        for (const float value : values) reference += std::abs(double(value));
    }
    ASSERT_GT(reference, 0.0) << "the first backward wrote nothing to compare against";

    // The backward consumes the most recent forward, so replay one before the second call - which
    // is what a training loop does every iteration anyway.
    (void)render_now(rasterizer, model, camera);
    const auto zero_gradient = Tensor::zeros(image_gradient.shape(), Device::GPU);
    optimizer.zero_grad(0);
    rasterizer.backward(zero_gradient, optimizer);
    for (const auto& entry : parameter_cases(model)) {
        const auto values = optimizer.get_grad(entry.type).cpu().to_vector();
        double magnitude = 0.0;
        for (const float value : values) magnitude += std::abs(double(value));
        EXPECT_EQ(magnitude, 0.0) << entry.name << " kept a stale contribution from the previous backward";
    }
}

#endif
