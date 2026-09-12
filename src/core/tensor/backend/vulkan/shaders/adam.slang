// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
#include "float_math.slangh"

struct Parameters
{
    uint64_t first, second, gradient, enabled, lrScale, nextFirst, nextSecond, delta;
    uint cells, maskStride, scaleStride;
    float beta1, beta2, biasCorrection2, eps, learningRate;
};

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    for (uint index = group.x * 256u + lane; index < p.cells; index += groups.x * 256u)
    {
        const float oldFirst = ((Ptr<float, Access.Read>)p.first)[index];
        const float oldSecond = ((Ptr<float, Access.Read>)p.second)[index];
        const float gradient = ((Ptr<float, Access.Read>)p.gradient)[index];
        bool enabled = true;
        if (p.enabled != 0)
        {
            const uint64_t address = p.enabled + uint64_t(index / p.maskStride);
            const uint word = ((Ptr<uint, Access.Read>)(address & ~uint64_t(3)))[0];
            enabled = ((word >> (uint(address & uint64_t(3)) * 8u)) & 255u) != 0;
        }
        // Match the separate Tensor operations, including rounding at each product.
        const float first = enabled ? roundedMultiply(oldFirst, p.beta1) + roundedMultiply(gradient, 1.0f - p.beta1) : oldFirst;
        const float second = enabled ? roundedMultiply(oldSecond, p.beta2) + roundedMultiply(roundedMultiply(gradient, gradient), 1.0f - p.beta2) : oldSecond;
        const float denominator = roundedMultiply(sqrt(second), p.biasCorrection2) + p.eps;
        float change = first / denominator;
        uint sign = asuint(first) ^ asuint(denominator) ^ asuint(p.learningRate);
        if (p.lrScale != 0)
        {
            const float scale = ((Ptr<float, Access.Read>)p.lrScale)[index / p.scaleStride];
            change = roundedMultiply(change, scale);
            sign ^= asuint(scale);
        }
        change = roundedMultiply(change, p.learningRate);
        ((Ptr<float, Access.ReadWrite>)p.nextFirst)[index] = first;
        ((Ptr<float, Access.ReadWrite>)p.nextSecond)[index] = second;
        // Preserve the signed zeros of the separate division/multiplication kernels.
        uint bits = asuint(change);
        if ((bits & 0x7fffffffu) == 0u) bits = sign & 0x80000000u;
        ((Ptr<uint, Access.ReadWrite>)p.delta)[index] = enabled ? bits : 0u;
    }
}
