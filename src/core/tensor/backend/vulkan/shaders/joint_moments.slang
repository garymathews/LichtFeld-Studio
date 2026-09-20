// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
#include "float_math.slangh"

[SpecializationConstant]
const uint kEncode = 0;

struct Parameters
{
    uint64_t first;
    uint64_t second;
    uint64_t packed;
    uint64_t bounds;
    uint64_t valid;
    uint cells;
    uint blockCells;
    uint bits;
    float eps;
};

groupshared float4 extrema[256];

float2 transformed(Parameters p, uint index)
{
    const float root = sqrt(tensorClampMin(((Ptr<float, Access.Read>)p.second)[index], 0.0f));
    return float2(((Ptr<float, Access.Read>)p.first)[index] / (root + p.eps), accurateLog1p(root / p.eps));
}

bool liveCell(Parameters p, uint index)
{
    if (p.valid == 0) return true;
    const uint64_t address = p.valid + uint64_t(index);
    const uint bits = ((Ptr<uint, Access.Read>)(address & ~uint64_t(3)))[0];
    return ((bits >> (uint(address & uint64_t(3)) * 8u)) & 255u) != 0;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const float qmax = float((1u << p.bits) - 1u);
    if (kEncode == 0)
    {
        for (uint index = group.x * 256u + lane; index < p.cells; index += groups.x * 256u)
        {
            const uint word = p.bits == 8 ? uint(((Ptr<uint16_t, Access.Read>)p.packed)[index])
                                         : ((Ptr<uint, Access.Read>)p.packed)[index];
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
            const float root = roundedMultiply(s <= .118f ? small : exp(s) - 1.0f, p.eps);
            const float second = roundedMultiply(root, root);
            ((Ptr<float, Access.ReadWrite>)p.second)[index] = second;
            ((Ptr<float, Access.ReadWrite>)p.first)[index] = second == 0.0f ? 0.0f : roundedMultiply(u, root + p.eps);
        }
        return;
    }
    const uint blocks = (p.cells + p.blockCells - 1u) / p.blockCells;
    for (uint block = group.x; block < blocks; block += groups.x)
    {
        const uint start = block * p.blockCells;
        const uint end = min(start + p.blockCells, p.cells);
        float4 range = float4(asfloat(0x7f800000u), -asfloat(0x7f800000u), asfloat(0x7f800000u), -asfloat(0x7f800000u));
        for (uint index = start + lane; index < end; index += 256u)
        {
            if (!liveCell(p, index)) continue;
            const float2 v = transformed(p, index);
            range = float4(ieeeMinimum(range.x, v.x), ieeeMaximum(range.y, v.x),
                           ieeeMinimum(range.z, v.y), ieeeMaximum(range.w, v.y));
        }
        extrema[lane] = range;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 128u; step > 0u; step /= 2u)
        {
            if (lane < step)
            {
                const float4 a = extrema[lane], b = extrema[lane + step];
                extrema[lane] = float4(ieeeMinimum(a.x, b.x), ieeeMaximum(a.y, b.y),
                                      ieeeMinimum(a.z, b.z), ieeeMaximum(a.w, b.w));
            }
            GroupMemoryBarrierWithGroupSync();
        }
        range = extrema[0];
        for (uint c = 0; c < 4u; ++c)
        {
            if (!isfinite(range[c])) range[c] = 0.0f;
            if (lane == 0) ((Ptr<float, Access.ReadWrite>)p.bounds)[block * 4u + c] = range[c];
        }
        for (uint index = start + lane; index < end; index += 256u)
        {
            const bool live = liveCell(p, index);
            const float2 v = transformed(p, index);
            uint2 q;
            for (uint c = 0; c < 2u; ++c)
            {
                const float normalized = (v[c] - range[c * 2u]) / tensorClampMin(range[c * 2u + 1u] - range[c * 2u], p.eps);
                const float rounded = floor(roundedMultiply(normalized, qmax) + .5f);
                const float clamped = ieeeMinimum(ieeeMaximum(rounded, 0.0f), qmax);
                q[c] = live && isfinite(clamped) ? uint(clamped) : 0u;
            }
            const uint word = q.x | (q.y << p.bits);
            if (p.bits == 8) ((Ptr<uint16_t, Access.ReadWrite>)p.packed)[index] = uint16_t(word);
            else ((Ptr<uint, Access.ReadWrite>)p.packed)[index] = word;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
