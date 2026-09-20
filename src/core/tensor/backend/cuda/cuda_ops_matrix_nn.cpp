/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../gpu_backend_ops.hpp"
#include "cuda_ops_common.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/tensor/internal/tensor_nn_ops.hpp"
#include "core/tensor/internal/tensor_ops.hpp"

namespace lfs::core::internal {
    void CudaBackendOps::sgemm(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {

        tensor_ops::launch_sgemm(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<float>(output), program.m, program.n, program.k,
            context.cuda_stream);
    }

    void CudaBackendOps::sgemm_tn(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {

        tensor_ops::launch_sgemm_tn(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<float>(output), program.m, program.n, program.k,
            context.cuda_stream);
    }

    void CudaBackendOps::sgemm_batched(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const GemmProgram& program, const ExecContext context) {

        tensor_ops::launch_sgemm_batched(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<float>(output), program.batch, program.m, program.n,
            program.k, context.cuda_stream);
    }

    void CudaBackendOps::sgemm_bias_relu(
        const StorageRef lhs, const StorageRef rhs, const StorageRef bias,
        const StorageRef output, const GemmProgram& program,
        const ExecContext context) {

        tensor_ops::launch_sgemm_bias_relu(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<const float>(bias), cuda_pointer<float>(output),
            program.m, program.n, program.k, context.cuda_stream);
    }

    void CudaBackendOps::dot_product(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t count, const ExecContext context) {

        tensor_ops::launch_dot_product(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<float>(output), count, context.cuda_stream);
    }

    void CudaBackendOps::diag(
        const StorageRef diagonal, const StorageRef output, const size_t count,
        const ExecContext context) {

        tensor_ops::launch_diag(
            cuda_pointer<const float>(diagonal), cuda_pointer<float>(output),
            count, context.cuda_stream);
    }

    void CudaBackendOps::eye(
        const StorageRef output, const size_t rows, const size_t columns,
        const ExecContext context) {

        tensor_ops::launch_eye(
            cuda_pointer<float>(output), rows, columns, context.cuda_stream);
    }

    void CudaBackendOps::cdist(
        const StorageRef lhs, const StorageRef rhs, const StorageRef output,
        const size_t lhs_rows, const size_t rhs_rows, const size_t columns,
        const float p, const ExecContext context) {

        tensor_ops::launch_cdist(
            cuda_pointer<const float>(lhs), cuda_pointer<const float>(rhs),
            cuda_pointer<float>(output), lhs_rows, rhs_rows, columns, p,
            context.cuda_stream);
    }

    void CudaBackendOps::max_pool2d(
        const StorageRef input, const StorageRef output,
        const PoolProgram& program, const ExecContext context) {

        tensor_ops::launch_max_pool2d(
            cuda_pointer<const float>(input), cuda_pointer<float>(output),
            program.batch, program.channels, program.input_height, program.input_width,
            program.output_height, program.output_width, program.kernel_size,
            program.stride, program.padding, context.cuda_stream);
    }

    void CudaBackendOps::adaptive_avg_pool2d(
        const StorageRef input, const StorageRef output,
        const PoolProgram& program, const ExecContext context) {

        tensor_ops::launch_adaptive_avg_pool2d(
            cuda_pointer<const float>(input), cuda_pointer<float>(output),
            program.batch, program.channels, program.input_height, program.input_width,
            program.output_height, program.output_width, context.cuda_stream);
    }

    void CudaBackendOps::bias_add(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size,
        const ExecContext context) {

        tensor_ops::launch_bias_add(
            cuda_pointer<const float>(input), cuda_pointer<const float>(bias),
            cuda_pointer<float>(output), count, channels, spatial_size,
            context.cuda_stream);
    }

    void CudaBackendOps::bias_relu(
        const StorageRef input, const StorageRef bias, const StorageRef output,
        const int count, const int channels, const int spatial_size,
        const ExecContext context) {

        tensor_ops::launch_bias_relu(
            cuda_pointer<const float>(input), cuda_pointer<const float>(bias),
            cuda_pointer<float>(output), count, channels, spatial_size,
            context.cuda_stream);
    }

    void CudaBackendOps::relu(
        const StorageRef input, const StorageRef output, const int count,
        const ExecContext context) {

        tensor_ops::launch_relu(
            cuda_pointer<const float>(input), cuda_pointer<float>(output),
            count, context.cuda_stream);
    }

} // namespace lfs::core::internal
