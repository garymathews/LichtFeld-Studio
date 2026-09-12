// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Quantized SH bounds cover 256 primitives; codes omit the float4 tail padding.
static const uint LFS_LOCAL_SIZE_X = 256;
#include "half_bits.slangh"
#include "float_math.slangh"

[SpecializationConstant]
const uint kEncode = 0;
[SpecializationConstant]
const uint kHalf = 0;

struct Parameters
{
    uint64_t input;
    uint64_t output;
    uint64_t bounds;
    uint n;
    uint slots;
    uint cells;
    uint padding;
};

groupshared float lows[256];
groupshared float highs[256];

float loadCell(Parameters p, uint row, uint cell)
{
    if (cell >= p.slots * 4u) return 0.0f;
    const uint index = ((row / 32u * p.slots + cell / 4u) * 32u + row % 32u) * 4u + cell % 4u;
    if (kHalf != 0)
        return halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)p.input)[index]));
    return ((Ptr<float, Access.Read>)p.input)[index];
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    if (kEncode == 0)
    {
        for (uint index = group.x * 256u + lane; index < p.n * p.cells; index += groups.x * 256u)
        {
            const uint row = index / p.cells;
            const uint cell = index % p.cells;
            const uint offset = (row / 32u * p.cells + cell) * 32u + row % 32u;
            const float code = float(((Ptr<uint16_t, Access.Read>)p.input)[offset]);
            const float lo = ((Ptr<float, Access.Read>)p.bounds)[row / 256u * 2u];
            const float hi = ((Ptr<float, Access.Read>)p.bounds)[row / 256u * 2u + 1u];
            const float value = lo + roundedMultiply(hi - lo, code * (1.0f / 65535.0f));
            ((Ptr<float, Access.ReadWrite>)p.output)[index] = value;
        }
        return;
    }
    for (uint block = group.x; block < (p.n + 255u) / 256u; block += groups.x)
    {
        const uint row = block * 256u + lane;
        float lo = asfloat(0x7f800000u);
        float hi = -asfloat(0x7f800000u);
        for (uint cell = 0; cell < p.cells; ++cell)
        {
            const float value = row < p.n ? loadCell(p, row, cell) : 0.0f;
            lo = ieeeMinimum(lo, row < p.n ? value : 1e30f);
            hi = ieeeMaximum(hi, row < p.n ? value : -1e30f);
        }
        lows[lane] = lo;
        highs[lane] = hi;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 128u; step > 0u; step /= 2u)
        {
            if (lane < step)
            {
                lows[lane] = ieeeMinimum(lows[lane], lows[lane + step]);
                highs[lane] = ieeeMaximum(highs[lane], highs[lane + step]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        lo = lows[0];
        hi = highs[0];
        if (lo > hi) { lo = 0.0f; hi = 0.0f; }
        if (lane == 0)
        {
            ((Ptr<float, Access.ReadWrite>)p.bounds)[block * 2u] = lo;
            ((Ptr<float, Access.ReadWrite>)p.bounds)[block * 2u + 1u] = hi;
        }
        if (row < (p.n + 31u) / 32u * 32u)
        {
            for (uint cell = 0; cell < p.cells; ++cell)
            {
                precise float scale = (loadCell(p, min(row, p.n - 1u), cell) - lo) * 65535.0f / tensorClampMin(hi - lo, 1e-20f);
                const float lower = floor(scale);
                precise float rounded = scale - lower < 0.5f ? lower : lower + 1.0f;
                rounded = ieeeMinimum(ieeeMaximum(rounded, 0.0f), 65535.0f);
                const uint code = row < p.n && isfinite(rounded) ? uint(rounded) : 0u;
                const uint offset = (row / 32u * p.cells + cell) * 32u + row % 32u;
                ((Ptr<uint16_t, Access.ReadWrite>)p.output)[offset] = uint16_t(code);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
