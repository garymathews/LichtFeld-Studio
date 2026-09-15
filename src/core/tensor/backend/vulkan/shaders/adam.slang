// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
#include "float_math.slangh"

// Selects the moment source at pipeline level so the row-major path carries no branch
// or unused load. Both variants are cached separately by the pipeline table.
[SpecializationConstant]
const uint kFusedMoments = 0;

// When set, `gradient` is stored in the packed swizzled order instead of row-major, so the host
// skips the per-chunk permute+materialise of rows_from_layout and the kernel reads it through the
// same packed mapping it uses for the moments. (The delta write stays row-major: swizzling it
// diverged in testing, ledger section 105.)
[SpecializationConstant]
const uint kSwizzledGrad = 0;

struct Parameters
{
    uint64_t first, second, gradient, enabled, lrScale, nextFirst, nextSecond, delta;
    // When `packed` is set the moment inputs are not read: they are decoded here from
    // the joint (u, log_s) codec, which removes a whole dispatch and the fp32 moment
    // round trip through memory on every optimizer step.
    uint64_t packed, bounds;
    uint cells, maskStride, scaleStride, slots, blockCells, bits;
    // The codec's own epsilon, which is not the Adam epsilon. Keeping it in its own
    // field also makes the block 128 bytes so it matches the host struct exactly
    // (Slang packs tightly and would otherwise declare 124).
    float codecEps;
    float beta1, beta2, biasCorrection2, eps, learningRate;
};

// Joint-moment cell index for a row-major cell. The packed storage keeps the parameter
// swizzle, so a row-major index has to be permuted back into
// ((block * slots + slot) * 32 + lane) * 4 + component. With no slots the two orders
// are identical.
uint packedCellIndex(Parameters p, uint index)
{
    if (p.slots == 0u)
        return index;
    const uint component = index & 3u;
    const uint cell = index >> 2u;
    const uint slot = cell % p.slots;
    const uint row = cell / p.slots;
    return ((row >> 5u) * p.slots + slot) * 128u + (row & 31u) * 4u + component;
}

// Same arithmetic as the decode path of joint_moments.slang, cell for cell, so the
// fused step reproduces the separate decode exactly.
void decodeMoment(Parameters p, uint index, out float first, out float second)
{
    const uint cell = packedCellIndex(p, index);
    const uint word = p.bits == 8 ? uint(((Ptr<uint16_t, Access.Read>)p.packed)[cell])
                                 : ((Ptr<uint, Access.Read>)p.packed)[cell];
    const float qmax = float((1u << p.bits) - 1u);
    const float qu = float(word & uint(qmax));
    const float qs = float(word >> p.bits);
    const uint block = index / p.blockCells * 4u;
    const float umin = ((Ptr<float, Access.Read>)p.bounds)[block];
    const float umax = ((Ptr<float, Access.Read>)p.bounds)[block + 1u];
    const float smin = ((Ptr<float, Access.Read>)p.bounds)[block + 2u];
    const float smax = ((Ptr<float, Access.Read>)p.bounds)[block + 3u];
    const float u = umin + roundedMultiply(umax - umin, qu * (1.0f / qmax));
    const float s = smin + roundedMultiply(smax - smin, qs * (1.0f / qmax));
    float small = s / 720.0f + 1.0f / 120.0f;
    small = roundedMultiply(small, s) + 1.0f / 24.0f;
    small = roundedMultiply(small, s) + 1.0f / 6.0f;
    small = roundedMultiply(small, s) + .5f;
    small = roundedMultiply(small, s) + 1.0f;
    small = roundedMultiply(s, small);
    const float root = roundedMultiply(s <= .118f ? small : exp(s) - 1.0f, p.codecEps);
    second = roundedMultiply(root, root);
    first = second == 0.0f ? 0.0f : roundedMultiply(u, root + p.codecEps);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    for (uint index = group.x * 256u + lane; index < p.cells; index += groups.x * 256u)
    {
        float oldFirst, oldSecond;
        if (kFusedMoments != 0u)
            decodeMoment(p, index, oldFirst, oldSecond);
        else
        {
            oldFirst = ((Ptr<float, Access.Read>)p.first)[index];
            oldSecond = ((Ptr<float, Access.Read>)p.second)[index];
        }
        const uint grad_index = (kSwizzledGrad != 0u) ? packedCellIndex(p, index) : index;
        const float gradient = ((Ptr<float, Access.Read>)p.gradient)[grad_index];
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
