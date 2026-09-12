#pragma once

#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

class VulkanGSPipeline;

namespace PerfTimer {

#define PERF_TIMER_TRAIN_STAGES   \
    _(ProjectionForward)          \
    _(RasterizeForward)           \
    _(_Cumsum)                    \
    _(CalculateIndexBufferOffset) \
    _(SortPrimitivesByDepth)      \
    _(BuildVisibleFlags)          \
    _(VisiblePrefix)              \
    _(PrepareVisibleSort)         \
    _(CompactVisiblePrimitives)   \
    _(SortVisiblePrimitives)      \
    _(CopyPrimitiveSortIndices)   \
    _(ApplyDepthOrdering)         \
    _(PrepareTileSort)            \
    _(CullSplats)                 \
    _(ProjectionSurvivors)

#define _(name) name,
    enum TrainStage {
        PERF_TIMER_TRAIN_STAGES
            END
    };
#undef _

    using Marker = std::pair<int, int>;
    struct State {
        std::vector<Marker> marks;
        std::vector<TrainStage> pushedMarks;
        bool hostHold = false;
        std::chrono::time_point<std::chrono::high_resolution_clock> hostStartTime;
        double hostTimeDelta = -1.0;
    };

    void hostTic(VulkanGSPipeline* module);
    void hostToc(VulkanGSPipeline* module);

    template <TrainStage stage>
    struct Timer {
        VulkanGSPipeline* module;

        Timer(VulkanGSPipeline* module);
        ~Timer();
    };

    void pushMarker(VulkanGSPipeline* module);
    void popMarkers(VulkanGSPipeline* module);

    std::vector<Marker> takeMarkers(VulkanGSPipeline* module);
    void discardMarkers(VulkanGSPipeline* module) noexcept;
    std::vector<std::pair<size_t, double>> update(std::vector<double> times,
                                                  const std::vector<Marker>& batch_marks);

    const char* stage_name(size_t stage);
    size_t stage_count();

} // namespace PerfTimer
