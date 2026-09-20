#define SUBGROUP_REDUCE_BEFORE_ATOMIC 1
#define SKIP_ZERO_IN_ATOMIC 1


#if SHARED_STRUCT
groupshared Splat_2D_AlphaBlend collected_splats[TILE_SIZE];
#else
groupshared float2 collected_xy_vs[TILE_SIZE];
groupshared float4 collected_inv_cov_vs_opacity[TILE_SIZE];
#endif

groupshared uint32_t collected_idx[TILE_SIZE];

groupshared float subgroup_shared[TILE_SIZE];


layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) StructuredBuffer<float4> final_pixel_state;
layout(binding=6) StructuredBuffer<int32_t> n_contributors;
layout(binding=7) StructuredBuffer<float4> v_current_pixel_state;

layout(binding=8) RWByteAddressBuffer v_xy_vs;
layout(binding=9) RWByteAddressBuffer v_inv_cov_vs_opacity;
layout(binding=10) RWByteAddressBuffer v_rgb;
#if DENSIFICATION_MCMC
layout(binding=11) StructuredBuffer<float> pixel_error;
layout(binding=12) RWByteAddressBuffer densification_info;
#endif

// Scale for the integer accumulation of densification statistics. Matched pair with
// kDensificationStatScale in the Vulkan training rasterizer's backward(); keep them equal.
static const int kDensificationStatScale = 65536;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint2 pix_coord = dispatchThreadID.xy;

    uint tile_idx = groupID.y * grid_width + groupID.x;
    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];

    uint32_t thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;

    float4 d_current_pixel_state;

    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = TILE_HEIGHT * TILE_WIDTH;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        uint32_t pix_id = pix_coord.y * W + pix_coord.x;
        current_pixel_state = final_pixel_state[pix_id];
        n_contrib_fwd = n_contributors[pix_id];
        d_current_pixel_state = v_current_pixel_state[pix_id];
    }

    float2 center_pix_coord = pix_coord;

    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xy_vs, inv_cov_vs_opacity, rgb);
        }
        GroupMemoryBarrierWithGroupSync();

        for (int j = 0; j < min(block_size, splats_left_to_process); j++)
        {
            uint32_t g_idx = collected_idx[j];
            DifferentialPair<Splat_2D_AlphaBlend> dp_g;
#if DENSIFICATION_MCMC
            float blend_weight = 0.0f;
            float weighted_error = 0.0f;
#endif
            #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
                float2 d_xy_vs = float2(0.0);
                float4 d_inv_cov_vs_opacity = float4(0.0);
                float3 d_rgb = float3(0.0);
            #endif

            bool contributes = false;
            if (is_inside) {
                current_splat_offset--;
                if (current_splat_offset < n_contrib_fwd) {

                Splat_2D_AlphaBlend g = collected_splats[j];

                float4 gauss_rgba = evaluate_splat(g, center_pix_coord);
                if (gauss_rgba.a >= ALPHA_THRESHOLD) {

                contributes = true;
                // Undo pixel state
                current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);
#if DENSIFICATION_MCMC
                blend_weight = current_pixel_state.w * gauss_rgba.a;
                weighted_error = blend_weight * pixel_error[pix_coord.y * W + pix_coord.x];
#endif

                // Back-prop automatically through blending and gaussian evaluation.
                dp_g = diffPair(g);
                DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_splat)(dp_g, center_pix_coord, dp_gauss_rgba.d);
                d_xy_vs = dp_g.d.xy_vs;
                d_inv_cov_vs_opacity = dp_g.d.inv_cov_vs_opacity;
                d_rgb = dp_g.d.rgb;

                }}
            }

            // A subgroup with no contributing pixels has only zero gradients.
            if (!WaveActiveAnyTrue(contributes))
                continue;

            // add back
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

          #if SKIP_ZERO_IN_ATOMIC
            #define ifNonZero(x) if (x != 0.0)
          #else
            #define ifNonZero(x)
          #endif

          #if SUBGROUP_REDUCE_BEFORE_ATOMIC
            d_xy_vs = WaveActiveSum(d_xy_vs);
            d_inv_cov_vs_opacity = WaveActiveSum(d_inv_cov_vs_opacity);
            d_rgb = WaveActiveSum(d_rgb);
#if DENSIFICATION_MCMC
            blend_weight = WaveActiveSum(blend_weight);
            weighted_error = WaveActiveSum(weighted_error);
#endif
          #endif
            // Each adjoint is accumulated as two scaled int32 words: a high part at 2^12 and a low
            // part at 2^30 carrying the residual. Both sums are exact integer additions, so neither
            // can depend on atomic scheduling order. A single int32 would need a scale coarse
            // enough to flush the smallest gradients to zero; the pair keeps a ~1e-9 floor and uses
            // the native integer atomic the count/index shaders already use, since no 64-bit
            // integer atomic is dependable on this backend.
            #define _ATOMIC_ADD_REFINED(buffer, address, value)                          \
                if (value != 0.0) {                                                       \
                    const int refined_hi = int(round(value * 4096.0));                    \
                    const int refined_lo = int(round((value - float(refined_hi) *         \
                                                       (1.0 / 4096.0)) * 1073741824.0));   \
                    buffer.InterlockedAdd(8u * (address), refined_hi);                    \
                    buffer.InterlockedAdd(8u * (address) + 4u, refined_lo);               \
                }
            // WaveActiveSum broadcasts each result; spread independent atomics across lanes.
          #if SUBGROUP_REDUCE_BEFORE_ATOMIC
            const uint lane = WaveGetLaneIndex(), lane_stride = WaveGetLaneCount();
          #else
            const uint lane = 0, lane_stride = 1;
          #endif
            for (uint c = lane; c < 2; c += lane_stride) {
                _ATOMIC_ADD_REFINED(v_xy_vs, 2*g_idx+c, d_xy_vs[c]);
            }
            for (uint c = lane; c < 4; c += lane_stride) {
                _ATOMIC_ADD_REFINED(v_inv_cov_vs_opacity, 4*g_idx+c, d_inv_cov_vs_opacity[c]);
            }
            for (uint c = lane; c < 3; c += lane_stride) {
                _ATOMIC_ADD_REFINED(v_rgb, 3*g_idx+c, d_rgb[c]);
            }
#if DENSIFICATION_MCMC
            for (uint c = lane; c < 2; c += lane_stride) {
                const float2 gradient = float2(blend_weight, weighted_error);
                // Integer accumulation so the statistics cannot depend on atomic ordering; the host
                // divides by kDensificationStatScale after the backward. Matched pair.
                const int scaled = int(round(gradient[c] * float(kDensificationStatScale)));
                densification_info.InterlockedAdd(4u * (c * uniforms.num_splats + g_idx), scaled);
            }
#endif

        #endif  // EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
        }
        splats_left_to_process -= block_size;
        // All subgroups must finish reading before the next round overwrites
        // the shared splat batch. Subgroup reduction is not a workgroup fence.
        GroupMemoryBarrierWithGroupSync();
    }
}
