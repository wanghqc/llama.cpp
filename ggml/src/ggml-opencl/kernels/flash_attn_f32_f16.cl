#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) ((float4)((float)(x).s0, (float)(x).s1, (float)(x).s2, (float)(x).s3))
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define Q1_WG_SIZE 64

// N_SPLIT: number of threads that collaborate on each query's dot product.
// When N_SPLIT > 1, each query is processed by N_SPLIT threads, each owning
// 1/N_SPLIT of the DK and DV dimensions.  This reduces register pressure for
// large head dimensions (e.g. DK=256 where N_SPLIT=1 causes ~512 float
// registers per thread → heavy spilling).
#ifndef N_SPLIT
#define N_SPLIT 1
#endif

#define SPLIT_DK_VEC (DK_VEC / N_SPLIT)
#define SPLIT_DV_VEC (DV_VEC / N_SPLIT)

#if N_SPLIT > 1
#define WG_SIZE (BLOCK_M * N_SPLIT)
#else
#define WG_SIZE (BLOCK_M)
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}
__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    const global void * k_pad_void,
    const global void * v_pad_void,
    const global void * mask_pad_void,
    const global char * blk,
    const int n_kv_blocks,
    const ulong mask_pad_nb1,
    const ulong mask_pad_nb2,
    const ulong mask_pad_nb3
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    // When N_SPLIT > 1: q_lane identifies which query row within the block,
    // split_idx identifies which DK/DV slice this thread owns.
    // When N_SPLIT == 1: q_lane == tid, split_idx == 0 (compile-time constant).
#if N_SPLIT > 1
    const int q_lane    = tid / N_SPLIT;
    const int split_idx = tid % N_SPLIT;
#else
    const int q_lane    = tid;
    const int split_idx = 0;
#endif

    const int my_query_row = block_q_idx * BLOCK_M + q_lane;
    const int query_valid = my_query_row < n_q;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;
    const int mask_head_idx = mask_void != NULL ? head_idx % mask_ne2 : 0;
    const int mask_batch_idx = mask_void != NULL ? batch_idx % mask_ne3 : 0;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }
    const global char* mask_pad_base = NULL;
    if (mask_pad_void != NULL) {
        mask_pad_base = (const global char*)mask_pad_void + mask_batch_idx * mask_pad_nb3 + mask_head_idx * mask_pad_nb2;
    }
    const global char* blk_base = NULL;
    if (blk != NULL) {
        const int n_q_blocks = (n_q + BLOCK_M - 1) / BLOCK_M;
        blk_base = blk + (((mask_batch_idx * mask_ne2) + mask_head_idx) * n_q_blocks + block_q_idx) * n_kv_blocks;
    }

    // Each thread owns SPLIT_DK_VEC float4 elements of the query row.
    // For N_SPLIT==1: SPLIT_DK_VEC == DK_VEC, identical to original.
    ACC_TYPE4 q_priv[SPLIT_DK_VEC];
    const int dk_off = split_idx * SPLIT_DK_VEC;
    if (query_valid) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[dk_off + i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < SPLIT_DK_VEC; ++i) {
            q_priv[i] = (ACC_TYPE4)(0.0f);
        }
    }

    ACC_TYPE4 o_acc[SPLIT_DV_VEC];
    #pragma unroll
    for (int i = 0; i < SPLIT_DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }

    // Softmax running state — maintained only by split_idx==0 threads when
    // N_SPLIT > 1; maintained by every thread when N_SPLIT==1 (original path).
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

#if N_SPLIT > 1
    // Local memory for the dot-product reduction across N_SPLIT threads and
    // for broadcasting the per-query softmax update to all split_idx threads.
    __local ACC_TYPE local_partial0[BLOCK_M][N_SPLIT];
    __local ACC_TYPE local_partial1[BLOCK_M][N_SPLIT];
    __local ACC_TYPE local_softmax_scale[BLOCK_M];
    __local ACC_TYPE local_softmax_p0[BLOCK_M];
    __local ACC_TYPE local_softmax_p1[BLOCK_M];
    __local ACC_TYPE local_l_inv[BLOCK_M];
#endif

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        const int use_kv_pad = k_pad_void != NULL && k_start + BLOCK_N > n_kv;
        const int k_tile_start = use_kv_pad ? 0 : k_start;
        const ulong k_tile_nb2 = use_kv_pad ? (ulong) BLOCK_N * k_nb1 : k_nb2;
        const ulong k_tile_nb3 = use_kv_pad ? (ulong) n_head_kv * k_tile_nb2 : k_nb3;
        const ulong v_tile_nb2 = use_kv_pad ? (ulong) BLOCK_N * v_nb1 : v_nb2;
        const ulong v_tile_nb3 = use_kv_pad ? (ulong) n_head_kv * v_tile_nb2 : v_nb3;
        const global char* k_tile_base = use_kv_pad ? (const global char*) k_pad_void : k_base;
        const global char* v_tile_base = use_kv_pad ? (const global char*) v_pad_void : v_base;

        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_tile_start + row;
            if (use_kv_pad || k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_tile_nb3 + head_kv_idx * k_tile_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_tile_base + k_row_offset))[col];
            } else {
                l_k[row][col] = (KV_DATA_TYPE4)(0.0h);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_tile_start + row;
            if (use_kv_pad || v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_tile_nb3 + head_kv_idx * v_tile_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_tile_base + v_row_offset))[col];
            } else {
                l_v[row][col] = (KV_DATA_TYPE4)(0.0h);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        char blk_cur = 1;
        if (blk_base != NULL) {
            blk_cur = blk_base[k_start / BLOCK_N];
        }
        if (blk_cur == 0) {
            continue;
        }

#if N_SPLIT > 1
        // ----------------------------------------------------------------
        // N_SPLIT > 1 path: split dot product across threads per query.
        // All threads execute the loop body to keep barriers uniform.
        // ----------------------------------------------------------------
        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            // Each thread computes a partial dot product over its DK slice.
            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < SPLIT_DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j  ][dk_off + k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][dk_off + k]), dot_acc1);
            }
            local_partial0[q_lane][split_idx] = dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3;
            local_partial1[q_lane][split_idx] = dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3;
            barrier(CLK_LOCAL_MEM_FENCE);

            // split_idx==0 reduces the partial sums and computes the softmax
            // update for this query.  For invalid query rows it writes neutral
            // values (scale=1, p=0) so all threads' o_acc update is a no-op.
            if (split_idx == 0) {
                ACC_TYPE score0 = 0.0f;
                ACC_TYPE score1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < N_SPLIT; s++) {
                    score0 += local_partial0[q_lane][s];
                    score1 += local_partial1[q_lane][s];
                }
                score0 *= scale;
                score1 *= scale;

                if (query_valid) {
                    if (is_causal) {
                        if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                        if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
                    }

                    if (k_row0 >= n_kv) score0 = -INFINITY;
                    if (k_row1 >= n_kv) score1 = -INFINITY;

                    if (mask_base != NULL && blk_cur != 2) {
                        if (use_kv_pad && mask_pad_base != NULL) {
                            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_pad_base + my_query_row * mask_pad_nb1);
                            score0 += slope * (ACC_TYPE)mask_ptr[j];
                            score1 += slope * (ACC_TYPE)mask_ptr[j + 1];
                        } else {
                            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                            if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                            if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                        }
                    }

                    if (logit_softcap > 0.0f) {
                        score0 = logit_softcap * tanh(score0 / logit_softcap);
                        score1 = logit_softcap * tanh(score1 / logit_softcap);
                    }

                    const ACC_TYPE m_new = max(m_i, max(score0, score1));
                    const ACC_TYPE p0 = exp(score0 - m_new);
                    const ACC_TYPE p1 = exp(score1 - m_new);
                    const ACC_TYPE sp = exp(m_i - m_new);

                    local_softmax_scale[q_lane] = sp;
                    local_softmax_p0[q_lane] = p0;
                    local_softmax_p1[q_lane] = p1;

                    l_i = l_i * sp + p0 + p1;
                    m_i = m_new;
                } else {
                    // invalid query row — neutral values
                    local_softmax_scale[q_lane] = 1.0f;
                    local_softmax_p0[q_lane] = 0.0f;
                    local_softmax_p1[q_lane] = 0.0f;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // All threads update their DV slice using the broadcast values.
            const ACC_TYPE sp_val = local_softmax_scale[q_lane];
            const ACC_TYPE p0_val = local_softmax_p0[q_lane];
            const ACC_TYPE p1_val = local_softmax_p1[q_lane];
            const int dv_off = split_idx * SPLIT_DV_VEC;
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * sp_val
                         + p0_val * CONVERT_KV_ACC4(l_v[j  ][dv_off + i])
                         + p1_val * CONVERT_KV_ACC4(l_v[j+1][dv_off + i]);
            }
        }
#else
        // ----------------------------------------------------------------
        // N_SPLIT == 1 path: original single-thread-per-query logic.
        // ----------------------------------------------------------------
        if (query_valid) {
            for (int j = 0; j < BLOCK_N; j += 2) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;

                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
                #pragma unroll
                for (int k = 0; k < DK_VEC; k++) {
                    dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]), dot_acc0);
                    dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]), dot_acc1);
                }
                ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
                ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

                if (is_causal) {
                    if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                    if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
                }

                if (k_row0 >= n_kv) score0 = -INFINITY;
                if (k_row1 >= n_kv) score1 = -INFINITY;

                if (mask_base != NULL && blk_cur != 2) {
                    if (use_kv_pad && mask_pad_base != NULL) {
                        const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_pad_base + my_query_row * mask_pad_nb1);
                        score0 += slope * (ACC_TYPE)mask_ptr[j];
                        score1 += slope * (ACC_TYPE)mask_ptr[j + 1];
                    } else {
                        const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                        if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                        if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                    }
                }

                if (logit_softcap > 0.0f) {
                    score0 = logit_softcap * tanh(score0 / logit_softcap);
                    score1 = logit_softcap * tanh(score1 / logit_softcap);
                }

                const ACC_TYPE m_new = max(m_i, max(score0, score1));
                const ACC_TYPE p0 = exp(score0 - m_new);
                const ACC_TYPE p1 = exp(score1 - m_new);
                const ACC_TYPE scale_prev = exp(m_i - m_new);

                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
                }
                l_i = l_i * scale_prev + p0 + p1;
                m_i = m_new;
            }
        }
#endif
    }

    // ----------------------------------------------------------------
    // Write output
    // ----------------------------------------------------------------
#if N_SPLIT > 1
    // split_idx==0 computes and stores l_inv (and handles sinks), then all
    // threads collectively write their DV slice.
    if (split_idx == 0) {
        ACC_TYPE sinks_sp = 1.0f;
        if (query_valid && sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            sinks_sp = exp(m_i - m_final);
            l_i = l_i * sinks_sp + exp(m_sink - m_final);
            m_i = m_final;
        }
        local_softmax_scale[q_lane] = sinks_sp;
        local_l_inv[q_lane] = (query_valid && l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (query_valid) {
        const ACC_TYPE sinks_sp = local_softmax_scale[q_lane];
        const ACC_TYPE l_inv    = local_l_inv[q_lane];
        const int dv_off = split_idx * SPLIT_DV_VEC;
        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_inv > 0.0f) {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = CONVERT_O_DATA4(o_acc[i] * sinks_sp * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < SPLIT_DV_VEC; ++i) {
                o_row[dv_off + i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
#else
    if (query_valid) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
#endif
}

__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}
