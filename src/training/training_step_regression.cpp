/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_snapshot_service.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace lfs::training {
    TrainingStepRegressionTracker::
        TrainingStepRegressionTracker(
            const std::size_t window_size)
        : window_size_(window_size) {
        if (window_size_ == 0) {
            throw std::invalid_argument(
                "Training step regression window must be non-zero");
        }
    }

    TrainingStepWindowMetrics
    TrainingStepRegressionTracker::summarize(
        const std::deque<Sample>& samples) const noexcept {
        TrainingStepWindowMetrics result;
        if (samples.empty()) {
            return result;
        }
        result.first_iteration =
            samples.front().iteration;
        result.last_iteration =
            samples.back().iteration;
        result.sample_count = samples.size();
        result.mean_ms =
            std::accumulate(
                samples.begin(), samples.end(), 0.0,
                [](const double sum,
                   const Sample& sample) {
                    return sum + sample.elapsed_ms;
                }) /
            static_cast<double>(samples.size());
        return result;
    }

    void TrainingStepRegressionTracker::observe(
        const int iteration,
        const double elapsed_ms,
        const bool topology_changed) {
        if (!(elapsed_ms >= 0.0) ||
            !std::isfinite(elapsed_ms)) {
            return;
        }
        if (topology_changed) {
            reset();
            return;
        }

        steady_run_.push_back({
            .iteration = iteration,
            .elapsed_ms = elapsed_ms,
        });
        if (steady_run_.size() > window_size_) {
            steady_run_.pop_front();
        }
        if (steady_run_.size() == window_size_) {
            latest_steady_window_ =
                summarize(steady_run_);
        }

        if (!armed_ ||
            iteration <= snapshot_iteration_ ||
            metrics_.gate_evaluated) {
            return;
        }
        post_resume_run_.push_back({
            .iteration = iteration,
            .elapsed_ms = elapsed_ms,
        });
        metrics_.post_resume =
            summarize(post_resume_run_);
        if (post_resume_run_.size() != window_size_) {
            return;
        }
        if (metrics_.pre_snapshot.sample_count !=
                window_size_ ||
            !(metrics_.pre_snapshot.mean_ms > 0.0)) {
            return;
        }
        metrics_.regression_percent =
            (metrics_.post_resume.mean_ms /
                 metrics_.pre_snapshot.mean_ms -
             1.0) *
            100.0;
        metrics_.gate_evaluated = true;
        metrics_.within_gate =
            metrics_.regression_percent <= 10.0;
    }

    void TrainingStepRegressionTracker::arm_after_snapshot(
        const int snapshot_iteration) {
        snapshot_iteration_ = snapshot_iteration;
        post_resume_run_.clear();
        metrics_ = {};
        if (latest_steady_window_) {
            metrics_.pre_snapshot =
                *latest_steady_window_;
        }
        // Without a baseline there is no comparison to make. A later snapshot
        // can arm again once observe() has collected a complete steady window.
        armed_ = metrics_.pre_snapshot.sample_count == window_size_ &&
                 metrics_.pre_snapshot.mean_ms > 0.0;
    }

    TrainingStepRegressionMetrics
    TrainingStepRegressionTracker::metrics() const noexcept {
        return metrics_;
    }

    void TrainingStepRegressionTracker::reset() noexcept {
        steady_run_.clear();
        latest_steady_window_.reset();
        post_resume_run_.clear();
        metrics_ = {};
        snapshot_iteration_ = 0;
        armed_ = false;
    }

}
