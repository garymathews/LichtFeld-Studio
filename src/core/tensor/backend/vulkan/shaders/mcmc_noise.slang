// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "float_math.slangh"
#include "byte_access.slangh"
#include "philox.slangh"

struct Parameters
{
    uint64_t opacity;
    uint64_t scales;
    uint64_t quaternions;
    uint64_t means;
    uint64_t frozen;
    uint64_t output;
    uint64_t seed;
    uint count;
    float learningRate;
};

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    for (uint row = tid.x; row < p.count; row += groups.x * 256u)
    {
        const uint base = row * 3u;
        const float3 mean = float3(((Ptr<float, Access.Read>)p.means)[base],
                                  ((Ptr<float, Access.Read>)p.means)[base + 1u],
                                  ((Ptr<float, Access.Read>)p.means)[base + 2u]);
        float3 updated = mean;
        if (p.frozen == 0 || loadByte(p.frozen, row) == 0)
        {
            float3 noise;
            for (uint axis = 0; axis < 3u; ++axis)
            {
                const uint64_t index = uint64_t(base + axis);
                const float u = tensorClampMin(unitInterval(draw(index, p.seed).x), 1e-7f);
                const float v = unitInterval(draw(index + uint64_t(p.count) * 3u, p.seed).x);
                noise[axis] = sqrt(log(u) * -2.0f) * cos(v * kTwoPi);
            }
            float4 q;
            for (uint axis = 0; axis < 4u; ++axis)
                q[axis] = ((Ptr<float, Access.Read>)p.quaternions)[row * 4u + axis];
            const float4 sq = q * q;
            q *= ieeeMinimum(rsqrt(((sq.x + sq.y) + sq.z) + sq.w), 1e12f);
            const float w = q.x, x = q.y, y = q.z, z = q.w;
            const float3 r0 = float3((y*y + z*z) * -2.0f + 1.0f, (x*y - w*z) * 2.0f, (x*z + w*y) * 2.0f);
            const float3 r1 = float3((x*y + w*z) * 2.0f, (x*x + z*z) * -2.0f + 1.0f, (y*z - w*x) * 2.0f);
            const float3 r2 = float3((x*z - w*y) * 2.0f, (y*z + w*x) * 2.0f, (x*x + y*y) * -2.0f + 1.0f);
            float3 local = ((0.0f + r0 * noise.x) + r1 * noise.y) + r2 * noise.z;
            for (uint axis = 0; axis < 3u; ++axis)
                local[axis] *= exp(((Ptr<float, Access.Read>)p.scales)[base + axis] * 2.0f);
            float3 transformed;
            for (uint axis = 0; axis < 3u; ++axis)
            {
                const float3 r = axis == 0u ? r0 : axis == 1u ? r1 : r2;
                transformed[axis] = ((0.0f + r.x * local.x) + r.y * local.y) + r.z * local.z;
            }
            const float opacity = 1.0f / (1.0f + exp(-((Ptr<float, Access.Read>)p.opacity)[row]));
            const float factor = (1.0f / (1.0f + exp(-(opacity * -100.0f + 0.5f)))) * p.learningRate;
            updated += transformed * factor;
        }
        for (uint axis = 0; axis < 3u; ++axis)
            ((Ptr<float, Access.ReadWrite>)p.output)[base + axis] = updated[axis];
    }
}
