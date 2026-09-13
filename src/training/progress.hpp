/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/session_chapters.hpp"
#include <memory>
#include <chrono>
#include <functional>

namespace lfs::training {
    class TrainingProgress {
    public:
        enum class Phase {
            Train,
            Refine,
            Controller,
            Sparse
        };

        TrainingProgress(int total_iterations, int update_frequency = 100,
                         std::function<std::chrono::steady_clock::time_point()> clock = std::chrono::steady_clock::now);
        ~TrainingProgress();

        TrainingProgress(const TrainingProgress&) = delete;
        TrainingProgress& operator=(const TrainingProgress&) = delete;
        TrainingProgress(TrainingProgress&&) = delete;
        TrainingProgress& operator=(TrainingProgress&&) = delete;

        void set_total_iterations(int total);
        void start(int completed_iterations);
        void update(int current_iteration, float loss, int splat_count, Phase phase = Phase::Train);
        void pause();
        void resume(int current_iteration, float loss, int splat_count, Phase phase = Phase::Train);
        void complete(io::project::TrainingFinishReason reason = io::project::TrainingFinishReason::None, int actual_iterations = -1);
        void print_final_summary(int final_splats, int actual_iterations = -1,
                                 io::project::TrainingFinishReason reason = io::project::TrainingFinishReason::Completed);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::training
