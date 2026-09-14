/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/selection_ops.hpp"
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace lfs::core::cuda {
    namespace {
        Tensor selection_morphology(const Tensor& mask, const Tensor& means, float radius, uint8_t group, bool grow) {
            if (!(radius > 0.f) || !std::isfinite(radius) || mask.dtype() != DataType::UInt8 ||
                means.ndim() != 2 || means.shape()[1] != 3 || means.shape()[0] != mask.numel())
                throw std::invalid_argument("Selection morphology requires UInt8 [N], positions [N,3], and a finite positive radius");
            auto host_mask = mask.cpu().contiguous();
            auto output = host_mask.clone();
            const auto points = means.cpu().to(DataType::Float32).to_vector();
            const auto* source = host_mask.ptr<uint8_t>();
            auto* destination = output.ptr<uint8_t>();
            using Cell = std::array<int64_t, 3>;
            std::vector<Cell> cells(mask.numel());
            std::map<Cell, std::vector<size_t>> neighbors;
            for (size_t i = 0; i < mask.numel(); ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    const double coordinate = std::floor(static_cast<double>(points[i*3+c]) / radius);
                    // Leave room for adjacent-cell offsets and avoid undefined float-to-int conversion.
                    if (!std::isfinite(coordinate) || std::abs(coordinate) >= 9e18)
                        throw std::invalid_argument("Selection position is nonfinite or exceeds the spatial grid range");
                    cells[i][c] = static_cast<int64_t>(coordinate);
                }
                if ((source[i] != 0) == grow) neighbors[cells[i]].push_back(i);
            }
            const double radius_squared = static_cast<double>(radius) * radius;
            for (size_t i = 0; i < mask.numel(); ++i) {
                if ((source[i] != 0) == grow) continue;
                bool found = false;
                for (int z = -1; z <= 1 && !found; ++z)
                    for (int y = -1; y <= 1 && !found; ++y)
                        for (int x = -1; x <= 1 && !found; ++x) {
                            const auto it = neighbors.find({cells[i][0]+x, cells[i][1]+y, cells[i][2]+z});
                            if (it == neighbors.end()) continue;
                            for (const size_t j : it->second) {
                                double distance = 0.;
                                for (size_t c = 0; c < 3; ++c) {
                                    const double delta = static_cast<double>(points[i*3+c]) - points[j*3+c];
                                    distance += delta * delta;
                                }
                                if (distance <= radius_squared) { found = true; break; }
                            }
                        }
                if (found) destination[i] = grow ? group : 0;
            }
            return output.to(mask.device());
        }
    }
    Tensor selection_grow(const Tensor& mask, const Tensor& means, float radius, uint8_t group) {
        return selection_morphology(mask, means, radius, group, true);
    }
    Tensor selection_shrink(const Tensor& mask, const Tensor& means, float radius) {
        return selection_morphology(mask, means, radius, 0, false);
    }
}
