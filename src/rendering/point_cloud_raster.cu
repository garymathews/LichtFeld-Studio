/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_raster_common.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::rendering::pcraster {

    namespace {

        constexpr std::uint64_t kEmptyPacked = ~0ULL;

        __global__ void clearPackedBuffer(std::uint64_t* buffer, int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            buffer[idx] = kEmptyPacked;
        }

        struct AtomicPixelWriter {
            std::uint64_t* buffer;
            __device__ void operator()(int pixel, std::uint64_t packed) const {
                atomicMin(reinterpret_cast<unsigned long long*>(&buffer[pixel]),
                          static_cast<unsigned long long>(packed));
            }
        };

        __global__ void rasterizePointsKernel(LaunchParams params, std::uint64_t* packed_buffer) {
            rasterizePoint(params, blockIdx.x * blockDim.x + threadIdx.x, AtomicPixelWriter{packed_buffer});
        }

        __global__ void unpackKernel(const std::uint64_t* packed_buffer,
                                     int width, int height, int channels,
                                     float bg_r, float bg_g, float bg_b, float bg_a,
                                     bool transparent, float far_plane,
                                     float* image, float* depth) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            const int idx = y * width + x;
            const int pixel_count = width * height;

            const std::uint64_t packed = packed_buffer[idx];
            if (packed == kEmptyPacked) {
                image[0 * pixel_count + idx] = bg_r;
                image[1 * pixel_count + idx] = bg_g;
                image[2 * pixel_count + idx] = bg_b;
                if (channels == 4) {
                    image[3 * pixel_count + idx] = transparent ? 0.0f : bg_a;
                }
                depth[idx] = far_plane;
                return;
            }

            const std::uint32_t color_u = static_cast<std::uint32_t>(packed & 0xFFFFFFFFu);
            const std::uint32_t depth_u = static_cast<std::uint32_t>(packed >> 32);
            const float depth_v = __uint_as_float(depth_u);

            image[0 * pixel_count + idx] = static_cast<float>(color_u & 0xFFu) / 255.0f;
            image[1 * pixel_count + idx] = static_cast<float>((color_u >> 8) & 0xFFu) / 255.0f;
            image[2 * pixel_count + idx] = static_cast<float>((color_u >> 16) & 0xFFu) / 255.0f;
            if (channels == 4) {
                image[3 * pixel_count + idx] = 1.0f;
            }
            depth[idx] = depth_v;
        }

    } // namespace

    cudaError_t launchPointCloudRaster(const LaunchParams& params) {
        if (params.width <= 0 || params.height <= 0 || params.n_points == 0 ||
            (params.channels != 3 && params.channels != 4) ||
            !params.image || !params.depth) {
            return cudaErrorInvalidValue;
        }

        const std::size_t pixels = static_cast<std::size_t>(params.width) *
                                   static_cast<std::size_t>(params.height);
        std::uint64_t* packed = nullptr;
        cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&packed),
                                             pixels * sizeof(std::uint64_t), params.stream);
        if (status != cudaSuccess) {
            return status;
        }

        const int clear_threads = 256;
        const int clear_blocks = static_cast<int>((pixels + clear_threads - 1) / clear_threads);
        clearPackedBuffer<<<clear_blocks, clear_threads, 0, params.stream>>>(
            packed, static_cast<int>(pixels));
        if ((status = cudaGetLastError()) != cudaSuccess) {
            cudaFreeAsync(packed, params.stream);
            return status;
        }

        const int raster_threads = 256;
        const int raster_blocks =
            static_cast<int>((params.n_points + raster_threads - 1) / raster_threads);
        rasterizePointsKernel<<<raster_blocks, raster_threads, 0, params.stream>>>(params, packed);
        if ((status = cudaGetLastError()) != cudaSuccess) {
            cudaFreeAsync(packed, params.stream);
            return status;
        }

        const dim3 unpack_threads(16, 16);
        const dim3 unpack_blocks((params.width + unpack_threads.x - 1) / unpack_threads.x,
                                 (params.height + unpack_threads.y - 1) / unpack_threads.y);
        unpackKernel<<<unpack_blocks, unpack_threads, 0, params.stream>>>(
            packed,
            params.width, params.height, params.channels,
            params.bg_r, params.bg_g, params.bg_b, params.bg_a,
            params.transparent_background, params.far_plane,
            params.image, params.depth);
        status = cudaGetLastError();
        cudaFreeAsync(packed, params.stream);
        return status;
    }

} // namespace lfs::rendering::pcraster
