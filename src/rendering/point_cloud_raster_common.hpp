/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_cloud_raster.cuh"
#include <algorithm>
#include <bit>
#include <cmath>
#if defined(__CUDACC__)
#define LFS_POINT_HD __device__ inline
#else
#define LFS_POINT_HD inline
#endif
namespace lfs::rendering::pcraster {
        LFS_POINT_HD float matMulRow(const float* M, int row, float x, float y, float z, float w) {
            return M[row + 0] * x + M[row + 4] * y + M[row + 8] * z + M[row + 12] * w;
        }

        LFS_POINT_HD std::uint64_t packDepthColor(float depth, float r, float g, float b) {
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);
            #if defined(__CUDA_ARCH__)
            const std::uint32_t depth_u = __float_as_uint(fmaxf(depth, 0.0f));
#else
            const std::uint32_t depth_u = std::bit_cast<std::uint32_t>(fmaxf(depth, 0.0f));
#endif
            const std::uint32_t r8 = static_cast<std::uint32_t>(r * 255.0f + 0.5f);
            const std::uint32_t g8 = static_cast<std::uint32_t>(g * 255.0f + 0.5f);
            const std::uint32_t b8 = static_cast<std::uint32_t>(b * 255.0f + 0.5f);
            const std::uint32_t color = r8 | (g8 << 8) | (b8 << 16);
            return (static_cast<std::uint64_t>(depth_u) << 32) | static_cast<std::uint64_t>(color);
        }

        template <typename Plot>
        LFS_POINT_HD void rasterizePoint(const LaunchParams& params, const int idx, Plot plot) {
            if (idx >= static_cast<int>(params.n_points)) {
                return;
            }
            if (params.deleted_mask && params.deleted_mask[idx]) {
                return;
            }

            int transform_index = 0;
            if (params.transform_indices) {
                int t = params.transform_indices[idx];
                if (t < 0) {
                    t = 0;
                }
                if (t >= params.n_transforms) {
                    t = params.n_transforms > 0 ? params.n_transforms - 1 : 0;
                }
                transform_index = t;
            }
            if (params.visibility_mask && transform_index < params.n_visibility &&
                !params.visibility_mask[transform_index]) {
                return;
            }

            float x = params.positions[idx * 3 + 0];
            float y = params.positions[idx * 3 + 1];
            float z = params.positions[idx * 3 + 2];

            if (params.transforms && params.n_transforms > 0) {
                const float* M = &params.transforms[transform_index * 16];
                const float nx = matMulRow(M, 0, x, y, z, 1.0f);
                const float ny = matMulRow(M, 1, x, y, z, 1.0f);
                const float nz = matMulRow(M, 2, x, y, z, 1.0f);
                const float nw = matMulRow(M, 3, x, y, z, 1.0f);
                if (fabsf(nw) > 1e-6f) {
                    x = nx / nw;
                    y = ny / nw;
                    z = nz / nw;
                } else {
                    x = nx;
                    y = ny;
                    z = nz;
                }
            }

            bool desaturate = false;
            if (params.has_crop) {
                const float* T = params.crop.to_local;
                const float lx = matMulRow(T, 0, x, y, z, 1.0f);
                const float ly = matMulRow(T, 1, x, y, z, 1.0f);
                const float lz = matMulRow(T, 2, x, y, z, 1.0f);
                const bool inside = (lx >= params.crop.min[0] && lx <= params.crop.max[0] &&
                                     ly >= params.crop.min[1] && ly <= params.crop.max[1] &&
                                     lz >= params.crop.min[2] && lz <= params.crop.max[2]);
                const bool visible = params.crop.inverse ? !inside : inside;
                if (!visible) {
                    if (!params.crop.desaturate) {
                        return;
                    }
                    desaturate = true;
                }
            } else if (params.has_crop_ellipsoid) {
                const float* T = params.crop_ellipsoid.to_local;
                const float lx = matMulRow(T, 0, x, y, z, 1.0f);
                const float ly = matMulRow(T, 1, x, y, z, 1.0f);
                const float lz = matMulRow(T, 2, x, y, z, 1.0f);
                const float rx = fmaxf(fabsf(params.crop_ellipsoid.radii[0]), 1e-8f);
                const float ry = fmaxf(fabsf(params.crop_ellipsoid.radii[1]), 1e-8f);
                const float rz = fmaxf(fabsf(params.crop_ellipsoid.radii[2]), 1e-8f);
                const float norm = (lx * lx) / (rx * rx) +
                                   (ly * ly) / (ry * ry) +
                                   (lz * lz) / (rz * rz);
                const bool inside = norm <= 1.0f;
                const bool visible = params.crop_ellipsoid.inverse ? !inside : inside;
                if (!visible) {
                    if (!params.crop_ellipsoid.desaturate) {
                        return;
                    }
                    desaturate = true;
                }
            }

            const float* V = params.view;
            const float view_x = matMulRow(V, 0, x, y, z, 1.0f);
            const float view_y = matMulRow(V, 1, x, y, z, 1.0f);
            const float view_z = matMulRow(V, 2, x, y, z, 1.0f);

            float pixel_x = 0.0f;
            float pixel_y = 0.0f;
            float depth = 0.0f;

            if (params.equirectangular) {
                const float len = sqrtf(view_x * view_x + view_y * view_y + view_z * view_z);
                if (len <= 1e-6f) {
                    return;
                }
                const float dx = view_x / len;
                const float dy = view_y / len;
                const float dz = view_z / len;
                const float pi = 3.14159265358979323846f;
                const float u = 0.5f + atan2f(dx, -dz) / (2.0f * pi);
                const float v = 0.5f - asinf(fminf(fmaxf(dy, -1.0f), 1.0f)) / pi;
                pixel_x = u * static_cast<float>(params.width - 1);
                pixel_y = v * static_cast<float>(params.height - 1);
                if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y) ||
                    pixel_x < 0.0f || pixel_x >= static_cast<float>(params.width) ||
                    pixel_y < 0.0f || pixel_y >= static_cast<float>(params.height)) {
                    return;
                }
                depth = len;
            } else {
                const float* P = params.view_proj;
                const float cx = matMulRow(P, 0, x, y, z, 1.0f);
                const float cy = matMulRow(P, 1, x, y, z, 1.0f);
                const float cz = matMulRow(P, 2, x, y, z, 1.0f);
                const float cw = matMulRow(P, 3, x, y, z, 1.0f);
                if (fabsf(cw) <= 1e-6f) {
                    return;
                }
                const float ndc_x = cx / cw;
                const float ndc_y = cy / cw;
                const float ndc_z = cz / cw;
                if (!std::isfinite(ndc_x) || !std::isfinite(ndc_y) || !std::isfinite(ndc_z) ||
                    ndc_x < -1.0f || ndc_x > 1.0f ||
                    ndc_y < -1.0f || ndc_y > 1.0f ||
                    ndc_z < 0.0f || ndc_z > 1.0f) {
                    return;
                }
                pixel_x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(params.width - 1);
                pixel_y = (0.5f - ndc_y * 0.5f) * static_cast<float>(params.height - 1);
                depth = params.orthographic ? -view_z : fmaxf(-view_z, 0.0f);
                if (depth <= 0.0f && !params.orthographic) {
                    return;
                }
            }

            const float voxel = fmaxf(params.voxel_size * params.scaling_modifier, 1e-5f);
            int radius = 1;
            if (params.orthographic) {
                const float pixels_per_world =
                    static_cast<float>(params.height) / fmaxf(params.ortho_scale, 1e-5f);
                radius = std::max(1, static_cast<int>(ceilf(voxel * pixels_per_world * 0.5f)));
            } else {
                radius = std::max(1, static_cast<int>(ceilf(voxel * params.focal_y / fmaxf(depth, 1e-4f))));
            }

            float r = params.colors[idx * 3 + 0];
            float g = params.colors[idx * 3 + 1];
            float b = params.colors[idx * 3 + 2];
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);
            if (desaturate) {
                const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                r = r + (gray - r) * 0.75f;
                g = g + (gray - g) * 0.75f;
                b = b + (gray - b) * 0.75f;
            }

            const std::uint64_t packed = packDepthColor(depth, r, g, b);
            const int center_x = static_cast<int>(rintf(pixel_x));
            const int center_y = static_cast<int>(rintf(pixel_y));
            const int r_sq = radius * radius;

            for (int dy = -radius; dy <= radius; ++dy) {
                const int yy = center_y + dy;
                if (yy < 0 || yy >= params.height) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = center_x + dx;
                    if (xx < 0 || xx >= params.width) {
                        continue;
                    }
                    if (dx * dx + dy * dy > r_sq) {
                        continue;
                    }
                    plot(yy * params.width + xx, packed);
                }
            }
        }

}
#undef LFS_POINT_HD
