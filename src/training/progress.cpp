/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "progress.hpp"

#include "indicators.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::training {
    struct TrainingProgress::Impl {
        std::unique_ptr<indicators::ProgressBar> progress_bar;
        std::chrono::steady_clock::time_point start_time;
        std::optional<std::chrono::steady_clock::time_point> pause_started;
        std::chrono::steady_clock::duration paused_duration{};
        std::function<std::chrono::steady_clock::time_point()> clock;
        bool force_update = true;
        int starting_iteration = 0;
        int last_iteration = 0;
        int total_iterations;
        int update_frequency;
    };

    namespace {
        const char* phase_label(const TrainingProgress::Phase phase) {
            switch (phase) {
            case TrainingProgress::Phase::Train:
                return "";
            case TrainingProgress::Phase::Refine:
                return "(+)";
            case TrainingProgress::Phase::Controller:
                return "Ctrl";
            case TrainingProgress::Phase::Sparse:
                return "(-)";
            }
            return "";
        }
    } // namespace

    TrainingProgress::TrainingProgress(const int total_iterations, const int update_frequency,
                                       std::function<std::chrono::steady_clock::time_point()> clock)
        : impl_(std::make_unique<Impl>()) {
        impl_->clock = std::move(clock);
        impl_->total_iterations = std::max(1, total_iterations);
        impl_->update_frequency = std::max(1, update_frequency);
        impl_->progress_bar = std::make_unique<indicators::ProgressBar>();

        impl_->progress_bar->set_option(indicators::option::Start("["));

#ifdef _WIN32
        impl_->progress_bar->set_option(indicators::option::BarWidth(38));
        impl_->progress_bar->set_option(indicators::option::Fill("="));
        impl_->progress_bar->set_option(indicators::option::Lead(">"));
        impl_->progress_bar->set_option(indicators::option::Remainder(" "));
#else
        impl_->progress_bar->set_option(indicators::option::BarWidth(40));
        impl_->progress_bar->set_option(indicators::option::Fill("█"));
        impl_->progress_bar->set_option(indicators::option::Lead("▌"));
        impl_->progress_bar->set_option(indicators::option::Remainder("░"));
#endif
        impl_->progress_bar->set_option(indicators::option::End("]"));
        impl_->progress_bar->set_option(indicators::option::PrefixText("Training "));
        impl_->progress_bar->set_option(indicators::option::PostfixText("Initializing..."));
        impl_->progress_bar->set_option(indicators::option::ShowPercentage(true));
        impl_->progress_bar->set_option(indicators::option::ShowElapsedTime(false));
        impl_->progress_bar->set_option(indicators::option::ShowRemainingTime(false));
        impl_->progress_bar->set_option(indicators::option::ForegroundColor(indicators::Color::cyan));

        std::vector<indicators::FontStyle> styles;
        styles.push_back(indicators::FontStyle::bold);
        impl_->progress_bar->set_option(indicators::option::FontStyles(styles));
        impl_->start_time = impl_->clock();
    }

    void TrainingProgress::set_total_iterations(const int total) {
        impl_->total_iterations = std::max(1, total);
        impl_->force_update = true;
    }

    TrainingProgress::~TrainingProgress() {
        complete();
    }

    void TrainingProgress::start(const int completed_iterations) {
        impl_->force_update = true;
        impl_->starting_iteration = completed_iterations;
        impl_->last_iteration = completed_iterations;
        impl_->start_time = impl_->clock();
        impl_->paused_duration = {};
        impl_->pause_started.reset();
    }

    void TrainingProgress::update(
        const int current_iteration,
        const float loss,
        const int splat_count,
        const Phase phase) {
        impl_->last_iteration = current_iteration;
        if (!impl_->force_update && current_iteration != 1 && current_iteration != impl_->total_iterations &&
            current_iteration % impl_->update_frequency != 0) {
            return;
        }

        impl_->force_update = false;
        const auto now = impl_->pause_started.value_or(impl_->clock());
        const double elapsed = std::max(0.0, std::chrono::duration<double>(now - impl_->start_time - impl_->paused_duration).count());
        const int updates = std::max(0, current_iteration - impl_->starting_iteration);
        const double remaining = updates > 0 ? elapsed * std::max(0, impl_->total_iterations - current_iteration) / updates : 0.0;
        const float progress = static_cast<float>(current_iteration) / impl_->total_iterations * 100;

        std::ostringstream postfix;
        postfix << current_iteration << "/" << impl_->total_iterations
                << " | Loss: " << std::fixed << std::setprecision(4) << loss
                << " | Splats: " << splat_count
                << " " << phase_label(phase)
                << " | Time: " << std::setprecision(1) << elapsed << "s | ETA: ";
        if (updates > 0) postfix << remaining << "s";
        else postfix << "--";
        impl_->progress_bar->set_option(indicators::option::PostfixText(postfix.str()));
        impl_->progress_bar->set_progress(static_cast<size_t>(progress));
    }

    void TrainingProgress::pause() {
        if (!impl_->pause_started)
            impl_->pause_started = impl_->clock();
        if (!impl_->progress_bar->is_completed()) {
            impl_->progress_bar->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void TrainingProgress::resume(
        const int current_iteration,
        const float loss,
        const int splat_count,
        const Phase phase) {
        if (impl_->pause_started) {
            impl_->paused_duration += impl_->clock() - *impl_->pause_started;
            impl_->pause_started.reset();
        }
        impl_->force_update = true;
        update(current_iteration, loss, splat_count, phase);
    }

    void TrainingProgress::complete(const io::project::TrainingFinishReason reason, const int actual_iterations) {
        if (!impl_->progress_bar->is_completed()) {
            const int iterations = actual_iterations >= 0 ? actual_iterations : impl_->last_iteration;
            const float fraction = impl_->total_iterations > 0
                                       ? static_cast<float>(iterations) / impl_->total_iterations
                                       : 0.0f;
            impl_->progress_bar->set_progress(static_cast<size_t>(std::clamp(fraction, 0.0f, 1.0f) * 100.0f));
            if (reason == io::project::TrainingFinishReason::UserStopped || reason == io::project::TrainingFinishReason::Error) {
                impl_->progress_bar->set_option(indicators::option::PostfixText(
                    (reason == io::project::TrainingFinishReason::Error ? "Failed at " : "Stopped by user at ") + std::to_string(iterations) + "/" +
                    std::to_string(impl_->total_iterations)));
            }
            impl_->progress_bar->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void TrainingProgress::print_final_summary(const int final_splats, const int actual_iterations,
                                               const io::project::TrainingFinishReason reason) {
        complete(reason, actual_iterations);
        const auto end_time = impl_->clock();
        const auto paused = impl_->paused_duration + (impl_->pause_started ? end_time - *impl_->pause_started
                                                                         : std::chrono::steady_clock::duration{});
        const double elapsed = std::chrono::duration<double>(end_time - impl_->start_time - paused).count();
        const int iteration = actual_iterations >= 0 ? actual_iterations : impl_->last_iteration;
        const int updates = std::max(0, iteration - impl_->starting_iteration);
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3);
        switch (reason) {
        case io::project::TrainingFinishReason::Completed: summary << "Training completed"; break;
        case io::project::TrainingFinishReason::UserStopped: summary << "Training stopped by user"; break;
        case io::project::TrainingFinishReason::Error: summary << "Training failed"; break;
        case io::project::TrainingFinishReason::None: summary << "Training ended"; break;
        }
        summary << " at iteration " << iteration << " after " << elapsed << "s ("
                << updates << " updates, avg " << std::setprecision(1)
                << (elapsed > 0.0 ? updates / elapsed : 0.0) << " iter/s)";
        std::cout << '\n' << summary.str() << "\nFinal splats: " << final_splats << std::endl;
    }
} // namespace lfs::training
