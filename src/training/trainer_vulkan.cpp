/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/scene.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "rasterization/vulkan_rasterizer.hpp"
#include "trainer.hpp"
#include "training_cropbox_mask.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace lfs::training {
    void Trainer::drain_control_commands() {
        auto view = CommandCenter::instance().snapshot();
        std::unique_lock render_lock(render_mutex_);
        std::unique_lock model_lock(model_access_mutex_);
        LiveModelMutationGuard mutation("training commands");
        if (CommandCenter::instance().drain_enqueued(view, [&] {
                waitForModelReaders();
                ++mutation_epoch_;
            })) {
            recordParamsReady();
            completed_mutation_epoch_ = mutation_epoch_;
        }
    }

    lfs::Result<Trainer::StepDisposition> Trainer::train_step(
        int iter, core::Camera* camera, core::Tensor gt_image, std::stop_token stop_token) {
        active_step_ms_ = 0;
        const auto initial_epoch = mutation_epoch_;
        std::string_view phase = "AcquireData";
        const auto stamp = [&](lfs::Error error) {
            lfs::SmallFields fields;
            fields.add("iteration", static_cast<int64_t>(iter))
                .add("mutation_epoch", static_cast<int64_t>(mutation_epoch_))
                .add("step_phase", phase)
                .add("persistent_commit", mutation_epoch_ != initial_epoch);
            return std::move(error).with_context("train_step", LFS_SOURCE_SITE_CURRENT(), std::move(fields));
        };
        try {
            core::GpuBackendScope backend(core::GpuBackend::Vulkan);
            if (completed_mutation_epoch_ != mutation_epoch_)
                throw std::runtime_error("Cannot train after an incomplete persistent update; restore a checkpoint or reinitialize");
            current_iteration_ = iter;
            handle_control_requests(iter, stop_token);
            if (stop_requested_ || stop_token.stop_requested())
                return StepDisposition::Stop;
            const auto& opt = params_.optimization;
            if (!camera || !strategy_)
                throw std::runtime_error("Training camera or strategy is missing");
            if (auto error = opt.validate_training_backend(core::GpuBackend::Vulkan); !error.empty())
                throw std::runtime_error(error);
            if (camera->has_distortion() && !camera->is_undistort_prepared())
                throw std::runtime_error("Vulkan training requires pinhole images; enable undistortion for this dataset");
            HookContext context{.iteration = iter, .loss = current_loss_.load(), .num_gaussians = strategy_->get_model().size(), .is_refining = strategy_->is_refining(iter), .trainer = this};
            CommandCenter::instance().set_phase(TrainingPhase::SafeControl);
            CommandCenter::instance().update_snapshot(context, get_total_iterations(), is_paused_.load(),
                                                      is_running_.load(), stop_requested_.load(), TrainingPhase::SafeControl);
            ControlBoundary::instance().notify(ControlHook::IterationStart, context);
            drain_control_commands();
            // Keep the model and iteration fixed while paused, but allow saves
            // of the last completed step just as the CUDA trainer does.
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                drain_control_commands();
                consume_requested_project_snapshot(project_snapshot_iteration());
                CommandCenter::instance().update_snapshot(context, get_total_iterations(), is_paused_.load(),
                                                          is_running_.load(), stop_requested_.load(), TrainingPhase::SafeControl);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }
            if (stop_requested_.load() || stop_token.stop_requested())
                return StepDisposition::Stop;
            if (auto error = opt.validate_training_backend(core::GpuBackend::Vulkan); !error.empty())
                throw std::runtime_error(error);
            const auto active_begin = std::chrono::steady_clock::now();
            CommandCenter::instance().set_phase(TrainingPhase::Forward);
            {
                // Hold the existing live-model lock through forward and backward:
                // visibility and sorted raster scratch refer to these exact rows.
                std::unique_lock render_lock(render_mutex_);
                std::unique_lock model_lock(model_access_mutex_);
                if (stop_requested_ || stop_token.stop_requested())
                    return StepDisposition::Stop;
                LiveModelMutationGuard mutation("Vulkan training step");
                waitForModelReaders();
                if (!vulkan_rasterizer_)
                    vulkan_rasterizer_ = std::make_unique<VulkanTrainingRasterizer>();
                auto& model = strategy_->get_model();
                auto& optimizer = strategy_->get_optimizer();
                auto background = background_for_step(iter).cpu().to_vector();
                core::Tensor background_image;
                if (opt.bg_mode == core::param::BackgroundMode::Image)
                    background_image = get_background_image_for_camera(camera->image_width(), camera->image_height());
                else if (opt.bg_mode == core::param::BackgroundMode::Random)
                    background_image = get_random_background_for_camera(camera->image_width(), camera->image_height(), iter);
                if (background_image.is_valid())
                    background.assign(3, 0.f);
                phase = "Forward";
                auto rendered = vulkan_rasterizer_->forward(*camera, model,
                                                            {background.at(0), background.at(1), background.at(2)}, opt.mip_filter);
                if (background_image.is_valid())
                    rendered.composite_background(background_image);
                rendered.camera = camera;
                rendered.target_image = gt_image;
                phase = "Loss";
                core::Tensor roi_weight;
                if (scene_) {
                    if (const auto geometry = resolve_training_cropbox_loss_geom(*scene_, opt.cropbox_loss_weight))
                        roi_weight = compute_cropbox_loss_weight(*camera, *geometry, opt.cropbox_loss_weight);
                }
                auto loss = compute_photometric_loss_with_mask(rendered.image, gt_image,
                                                               pipelined_mask_, roi_weight, rendered.alpha, opt, {});
                if (!loss)
                    throw std::runtime_error(loss.error());
                auto image_gradient = loss->grad_corrected;
                if (loss->grad_raw.is_valid())
                    image_gradient = image_gradient + loss->grad_raw;
                auto alpha_gradient = loss->grad_alpha;
                if (alpha_gradient.is_valid())
                    alpha_gradient = alpha_gradient.reshape(rendered.alpha.shape());
                if (background_image.is_valid()) {
                    const auto background_gradient = (image_gradient * background_image).sum(0, true).neg();
                    alpha_gradient = alpha_gradient.is_valid() ? alpha_gradient + background_gradient : background_gradient;
                }
                core::Tensor error_map;
                if (iter < static_cast<int>(opt.stop_refine)) {
                    // Vulkan SSIM maps precede mask weighting; the current
                    // photometric pass already supplies the identical maps.
                    // Pure L1 does not compute them and needs the scoring pass.
                    if (opt.lambda_dssim == 0.f) {
                        const auto score = photometric_loss_.forward(rendered.image, gt_image, {.lambda_dssim = 1.f});
                        if (!score)
                            throw std::runtime_error(score.error());
                    }
                    const auto& maps = photometric_loss_.ssim_workspace();
                    const auto& map = opt.densify_error_map == core::param::DensifyErrorMap::SsimCs ? maps.cs_map : maps.ssim_map;
                    error_map = (map.mean(1).squeeze(0).neg() + 1.f).clamp_min(0.f);
                }
                phase = "Backward";
                // Backward commits screen-share and densification statistics,
                // so even a later loss/regularizer failure invalidates a save.
                ++mutation_epoch_;
                vulkan_rasterizer_->backward(image_gradient, optimizer, alpha_gradient, error_map);
                auto total_loss = loss->loss;
                if (opt.scale_reg > 0.f) {
                    auto regularization = compute_scale_reg_loss(model, optimizer, opt);
                    if (!regularization)
                        throw std::runtime_error(regularization.error());
                    total_loss = total_loss + *regularization;
                }
                if (opt.opacity_reg > 0.f) {
                    auto regularization = compute_opacity_reg_loss(model, optimizer, opt);
                    if (!regularization)
                        throw std::runtime_error(regularization.error());
                    total_loss = total_loss + *regularization;
                }
                const float scalar_loss = total_loss.cpu().item<float>();
                if (!std::isfinite(scalar_loss))
                    throw std::runtime_error("Non-finite training loss");
                update_camera_loss_heatmap(*camera, loss->loss);
                strategy_->pre_step(iter, rendered);
                install_cropbox_step_damping(model, optimizer);
                context.loss = scalar_loss;
                CommandCenter::instance().set_phase(TrainingPhase::OptimizerStep);
                CommandCenter::instance().update_snapshot(context, get_total_iterations(), is_paused_.load(),
                                                          is_running_.load(), stop_requested_.load(), TrainingPhase::OptimizerStep);
                ControlBoundary::instance().notify(ControlHook::PreOptimizerStep, context);
                const auto previous_size = model.size();
                phase = "RefinementCommit";
                strategy_->post_backward(iter, rendered);
                phase = "OptimizerCommit";
                strategy_->step(iter);
                maybe_morton_reorder(iter);
                if (scene_ && previous_size != model.size())
                    scene_->syncTrainingModelTopology(model.size());
                recordParamsReady();
                completed_iteration_ = iter;
                completed_mutation_epoch_ = mutation_epoch_;
                current_loss_ = scalar_loss;
                if (iter == 1 || iter % 10 == 0 || iter == get_total_iterations()) {
                    submitLossReadback(total_loss, iter);
                    const auto harvested = harvestLossReadbacks(true, false);
                    if (!harvested)
                        throw std::runtime_error(harvested.error());
                }
            }
            active_step_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - active_begin).count();
            phase = "Publish";
            maybe_publish_camera_loss_heatmap(iter);
            if (evaluator_ && evaluator_->should_evaluate(iter)) {
                const std::shared_lock render_lock(render_mutex_);
                const std::shared_lock model_lock(model_access_mutex_);
                evaluator_->evaluate(iter, strategy_->get_model(), val_dataset_, background_,
                    params_.optimization.bg_mode == core::param::BackgroundMode::Image ? bg_image_base_ : core::Tensor{});
            }
            context.loss = current_loss_.load();
            context.num_gaussians = strategy_->get_model().size();
            CommandCenter::instance().set_phase(TrainingPhase::SafeControl);
            CommandCenter::instance().update_snapshot(context, get_total_iterations(), is_paused_.load(),
                                                      is_running_.load(), stop_requested_.load(), TrainingPhase::SafeControl);
            ControlBoundary::instance().notify(ControlHook::PostStep, context);
            consume_requested_project_snapshot(iter);
            capture_scheduled_project_snapshot(iter);
            return iter < get_total_iterations() && !stop_requested_ && !stop_token.stop_requested()
                       ? StepDisposition::Continue
                       : StepDisposition::Stop;
        } catch (const lfs::Exception& error) {
            return stamp(error.error());
        } catch (const std::bad_alloc& error) {
            return stamp(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::ResourceExhausted,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Insufficient memory for Vulkan training",
                .detail = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT()}));
        } catch (const std::exception& error) {
            return stamp(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Training,
                .user_message = std::string("Vulkan training step failed: ") + error.what(),
                .detail = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    }
} // namespace lfs::training
