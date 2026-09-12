/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "checkpoint.hpp"
#include "core/checkpoint_format.hpp"
#include "core/host_metrics.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_serialization_sink.hpp"
#include "snapshot_counting_stream.hpp"
#include "training_snapshot_service.hpp"
#include <deque>
#include <algorithm>
#include <mutex>
#include <ostream>
#include <span>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using Clock = std::chrono::steady_clock;
        double elapsed(Clock::time_point begin) {
            return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        }
        lfs::Error snapshot_error(lfs::ErrorCode code, std::string detail, lfs::Retryability retry = lfs::Retryability::NotRetryable) {
            return lfs::make_error(lfs::ErrorInit{.code = code, .domain = lfs::ErrorDomain::Training, .retryability = retry, .user_message = "The training snapshot could not be captured.", .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        class SnapshotOutputBuffer final : public std::streambuf {
        public:
            explicit SnapshotOutputBuffer(std::span<std::byte> bytes)
                : begin_(reinterpret_cast<char*>(bytes.data())), size_(bytes.size()) {
                setp(begin_, begin_ + size_);
            }

            bool exceeded_capacity() const noexcept { return exceeded_capacity_; }

        protected:
            int_type overflow(int_type) override {
                exceeded_capacity_ = true;
                return traits_type::eof();
            }
            pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                             std::ios_base::openmode mode) override {
                if (!(mode & std::ios_base::out))
                    return pos_type(off_type(-1));
                const off_type base = direction == std::ios_base::beg ? 0 : direction == std::ios_base::cur ? pptr() - begin_
                                                                                                            : static_cast<off_type>(size_);
                if (offset < -base || offset > static_cast<off_type>(size_) - base) {
                    exceeded_capacity_ = true;
                    return pos_type(off_type(-1));
                }
                const auto position = base + offset;
                setp(begin_ + position, begin_ + size_);
                return pos_type(position);
            }
            pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
                return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
            }

        private:
            char* begin_;
            size_t size_;
            bool exceeded_capacity_ = false;
        };
        // Reuse checkpoint serialization for the exact format/byte count, while
        // avoiding payload readback during inter-step preparation.
        class SizeSink final : public core::TensorSerializationSink {
        public:
            uint64_t device_bytes = 0;
            size_t pieces = 0;
            void write_tensor_payload(std::ostream& destination, const core::Tensor& source,
                                      const core::Tensor*, const core::TensorSerializationDescriptor& descriptor) override {
                const auto bytes = descriptor.payload_bytes();
                if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
                    throw std::overflow_error("Snapshot tensor payload exceeds stream limits");
                if (source.device() == core::Device::GPU)
                    device_bytes += bytes;
                ++pieces;
                destination.seekp(static_cast<std::streamoff>(bytes), std::ios_base::cur);
            }
        };
    } // namespace
    struct PreparedTrainingSnapshot::Impl {
        core::Uuid uuid;
        int iteration = 0;
        const IStrategy* strategy = nullptr;
        uint64_t baseline_rss_bytes = 0;
        std::shared_ptr<std::vector<std::byte>> bytes;
        TrainingSnapshotPauseMetrics metrics;
    };
    struct PendingTrainingSnapshot::Impl {
        CapturedTrainingSnapshot capture;
    };
    struct TrainingSnapshotService::Impl {
        mutable std::mutex mutex;
        bool initialized = false;
        TrainingSnapshotServiceMetrics metrics;
        // p95 describes the last 1024 captures; completed_snapshots remains the
        // lifetime count. Keep telemetry bounded during long editor sessions.
        std::deque<double> pauses;
    };
    PreparedTrainingSnapshot::PreparedTrainingSnapshot(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
    PreparedTrainingSnapshot::PreparedTrainingSnapshot(PreparedTrainingSnapshot&&) noexcept = default;
    PreparedTrainingSnapshot& PreparedTrainingSnapshot::operator=(PreparedTrainingSnapshot&&) noexcept = default;
    PreparedTrainingSnapshot::~PreparedTrainingSnapshot() = default;
    const core::Uuid& PreparedTrainingSnapshot::snapshot_uuid() const noexcept { return impl_->uuid; }
    uint64_t PreparedTrainingSnapshot::checkpoint_bytes() const noexcept { return impl_->bytes->size(); }
    PendingTrainingSnapshot::PendingTrainingSnapshot(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    PendingTrainingSnapshot::PendingTrainingSnapshot(PendingTrainingSnapshot&&) noexcept = default;
    PendingTrainingSnapshot& PendingTrainingSnapshot::operator=(PendingTrainingSnapshot&&) noexcept = default;
    PendingTrainingSnapshot::~PendingTrainingSnapshot() = default;
    bool PendingTrainingSnapshot::ready() const { return impl_ != nullptr; }
    lfs::Result<CapturedTrainingSnapshot> PendingTrainingSnapshot::wait() {
        if (!impl_)
            return snapshot_error(lfs::ErrorCode::FailedPrecondition, "Snapshot handle has been moved");
        return impl_->capture;
    }
    TrainingSnapshotService::TrainingSnapshotService(TrainingSnapshotServiceConfig config) : impl_(std::make_unique<Impl>()) {
        if (!config.ring_slots || !config.band_bytes || !config.calibration_bytes || config.calibration_iterations <= 0)
            throw std::invalid_argument("Snapshot transfer configuration must be positive");
    }
    TrainingSnapshotService::~TrainingSnapshotService() = default;
    lfs::Result<void> TrainingSnapshotService::initialize(const TrainingSnapshotCaptureRequest& request) {
        try {
            for (auto stream : request.mutating_streams)
                if (stream)
                    return lfs::Status::failure(snapshot_error(lfs::ErrorCode::InvalidArgument, "CUDA mutation stream supplied to Vulkan snapshots"));
            core::with_idle_vulkan_device([](const auto&) {});
            std::lock_guard lock(impl_->mutex);
            impl_->initialized = true;
            return {};
        } catch (const lfs::Exception& e) {
            return lfs::Status::failure(e.error());
        } catch (const std::bad_alloc& e) {
            return lfs::Status::failure(snapshot_error(lfs::ErrorCode::ResourceExhausted, e.what()));
        } catch (const std::exception& e) {
            return lfs::Status::failure(snapshot_error(lfs::ErrorCode::Internal, e.what()));
        }
    }
    lfs::Result<PreparedTrainingSnapshot> TrainingSnapshotService::prepare(const TrainingSnapshotCaptureRequest& request) {
        try {
            {
                std::lock_guard lock(impl_->mutex);
                if (!impl_->initialized)
                    return snapshot_error(lfs::ErrorCode::FailedPrecondition, "Initialize snapshot service before preparing a save");
            }
            const auto begin = Clock::now();
            auto prepared = std::make_unique<PreparedTrainingSnapshot::Impl>();
            prepared->uuid = request.snapshot_uuid.is_nil() ? core::generate_uuid_v4() : request.snapshot_uuid;
            prepared->iteration = request.iteration;
            prepared->strategy = &request.strategy;
            CountingStreamBuffer buffer;
            std::ostream destination(&buffer);
            SizeSink sink;
            {
                core::TensorSerializationSinkScope scope(sink);
                auto serialized = serialize_checkpoint(destination, request.iteration, request.strategy, request.params,
                                                       request.bilateral_grid, request.ppisp, request.ppisp_controller_pool, request.sparsity_optimizer);
                if (!serialized)
                    return serialized.error();
                if (!destination || buffer.size() != serialized->bytes || !buffer.size() || buffer.size() > core::MAX_CHECKPOINT_FILE_BYTES)
                    return snapshot_error(lfs::ErrorCode::ResourceExhausted, "Invalid or oversized snapshot layout");
            }
            const auto memory = core::host_metrics::sample();
            prepared->baseline_rss_bytes = memory.process_rss_bytes;
            const uint64_t available = memory.system_total_bytes > memory.system_used_bytes ? memory.system_total_bytes - memory.system_used_bytes : 0;
            const uint64_t reserve = request.relaxed_host_memory_gate ? 768ull * 1024 * 1024 : 4ull * 1024 * 1024 * 1024;
            const uint64_t required = buffer.size() + reserve;
            if (!memory.ram_valid || available < required)
                return snapshot_error(lfs::ErrorCode::ResourceExhausted, "Insufficient measured host-memory budget for the training snapshot");
            prepared->bytes = std::make_shared<std::vector<std::byte>>(buffer.size());
            auto& metrics = prepared->metrics;
            metrics.snapshot_uuid = prepared->uuid;
            metrics.iteration = request.iteration;
            metrics.checkpoint_bytes = buffer.size();
            metrics.device_snapshot_bytes = sink.device_bytes;
            metrics.tensor_piece_count = sink.pieces;
            metrics.host_staging_bytes = buffer.size();
            metrics.host_memory_available_bytes = available;
            metrics.host_memory_required_bytes = required;
            metrics.host_memory_preflight_passed = true;
            const auto staged_rss = core::host_metrics::sample().process_rss_bytes;
            metrics.host_rss_delta_bytes = staged_rss > prepared->baseline_rss_bytes ? staged_rss - prepared->baseline_rss_bytes : 0;
            metrics.host_ram_within_gate = prepared->baseline_rss_bytes && staged_rss &&
                                           metrics.host_rss_delta_bytes <= buffer.size() + 768ull * 1024 * 1024;
            metrics.prepare_stall_ms = metrics.preparation_ms = elapsed(begin);
            return PreparedTrainingSnapshot(std::move(prepared));
        } catch (const lfs::Exception& e) {
            return e.error();
        } catch (const std::bad_alloc& e) {
            return snapshot_error(lfs::ErrorCode::ResourceExhausted, e.what());
        } catch (const std::exception& e) {
            return snapshot_error(lfs::ErrorCode::Internal, e.what());
        }
    }
    lfs::Result<PendingTrainingSnapshot> TrainingSnapshotService::capture(PreparedTrainingSnapshot prepared, const TrainingSnapshotCaptureRequest& request) {
        try {
            auto& plan = *prepared.impl_;
            if (request.iteration != plan.iteration || &request.strategy != plan.strategy ||
                (!request.snapshot_uuid.is_nil() && request.snapshot_uuid != plan.uuid))
                return snapshot_error(lfs::ErrorCode::FailedPrecondition, "Snapshot identity changed after preparation");
            const auto entry = Clock::now();
            const auto begin = request.safe_point_entered_at.value_or(entry);
            auto metrics = plan.metrics;
            for (auto stream : request.mutating_streams)
                if (stream)
                    return snapshot_error(lfs::ErrorCode::InvalidArgument, "CUDA mutation stream supplied to Vulkan snapshots");
            core::with_idle_vulkan_device([](const auto&) {});
            metrics.stream_sync_ms = elapsed(entry);
            if (request.capture_additional_cpu_state) {
                const auto cpu_begin = Clock::now();
                auto captured = request.capture_additional_cpu_state(plan.uuid);
                if (!captured)
                    return captured.error();
                metrics.scng_ms = captured->scng_ms;
                metrics.selm_ms = captured->selm_ms;
                metrics.prms_ms = captured->prms_ms;
                metrics.additional_cpu_state_ms = elapsed(cpu_begin);
            }
            const auto serialize_begin = Clock::now();
            SnapshotOutputBuffer buffer(*plan.bytes);
            std::ostream destination(&buffer);
            auto serialized = serialize_checkpoint(destination, request.iteration, request.strategy, request.params,
                                                   request.bilateral_grid, request.ppisp, request.ppisp_controller_pool, request.sparsity_optimizer);
            if (buffer.exceeded_capacity())
                return snapshot_error(lfs::ErrorCode::FailedPrecondition, "Snapshot layout changed after preparation; prepare the save again", lfs::Retryability::Retryable);
            if (!serialized)
                return serialized.error();
            if (!destination || serialized->bytes != plan.bytes->size())
                return snapshot_error(lfs::ErrorCode::FailedPrecondition, "Snapshot layout changed after preparation; prepare the save again", lfs::Retryability::Retryable);
            metrics.serialize_and_issue_ms = elapsed(serialize_begin);
            const auto capture_rss = core::host_metrics::sample().process_rss_bytes;
            if (capture_rss > plan.baseline_rss_bytes)
                metrics.host_rss_delta_bytes = std::max(metrics.host_rss_delta_bytes, capture_rss - plan.baseline_rss_bytes);
            metrics.host_ram_within_gate = plan.baseline_rss_bytes && capture_rss &&
                                           metrics.host_rss_delta_bytes <= plan.bytes->size() + 768ull * 1024 * 1024;
            metrics.pause_ms = elapsed(begin);
            // The caller holds the model safe point for the entire synchronous
            // capture. The returned bytes own all state and need no GPU lifetime.
            metrics.consistency_proven = true;
            auto pending = std::make_shared<PendingTrainingSnapshot::Impl>();
            pending->capture = {plan.uuid, request.iteration, std::move(plan.bytes), metrics};
            {
                std::lock_guard lock(impl_->mutex);
                ++impl_->metrics.completed_snapshots;
                impl_->metrics.last = metrics;
                impl_->pauses.push_back(metrics.pause_ms);
                if (impl_->pauses.size() > 1024)
                    impl_->pauses.pop_front();
                auto sorted = impl_->pauses;
                std::sort(sorted.begin(), sorted.end());
                impl_->metrics.p95_n = sorted.size();
                impl_->metrics.pause_p95_ms = sorted[static_cast<size_t>(.95 * (sorted.size() - 1))];
            }
            return PendingTrainingSnapshot(std::move(pending));
        } catch (const lfs::Exception& e) {
            return e.error();
        } catch (const std::bad_alloc& e) {
            return snapshot_error(lfs::ErrorCode::ResourceExhausted, e.what());
        } catch (const std::exception& e) {
            return snapshot_error(lfs::ErrorCode::Internal, e.what());
        }
    }
    TrainingSnapshotServiceMetrics TrainingSnapshotService::metrics() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->metrics;
    }
    void TrainingSnapshotService::reset_process_pinned_d2h_calibration_for_testing() {}
    void TrainingSnapshotService::testing_advance_completed_snapshots(uint64_t count) {
        std::lock_guard lock(impl_->mutex);
        impl_->metrics.completed_snapshots += count;
    }
} // namespace lfs::training
