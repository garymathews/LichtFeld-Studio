/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#ifndef LFS_TENSOR_CUDA
#ifdef __APPLE__
#define LFS_TENSOR_CUDA 0
#else
#define LFS_TENSOR_CUDA 1
#endif
#endif
// CUDA's opaque stream handle is retained at the compatibility boundary.
// Declaring its pointer type does not load or emulate the CUDA runtime.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

struct CUevent_st;
using cudaEvent_t = CUevent_st*;
