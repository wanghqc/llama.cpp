#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Subgroup extensions — same detection pattern as mul_mv_q6_k_f32_flat.cl.
// NVIDIA does not expose cl_khr_subgroups as a compile-time macro, so the
// pragma is skipped there; NVIDIA uses __local tree-reduction instead.
#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// SOA flat kernel for Q4_K x f32 matrix-vector multiply.
//
// GPU-specific parameters:
//   NVIDIA  : N_SIMDWIDTH=32,  BLOCK_STRIDE=2,  __local tree-reduction
//   Adreno  : N_SIMDWIDTH=64,  BLOCK_STRIDE=4,  sub_group_reduce_add
//
// Q4_K qs byte layout (SOA, 128 bytes per super-block = QK_K/2):
//   qs[ 0..31]: lo nibble = element   0..31  (sg=0),  hi nibble = element  32..63  (sg=1)
//   qs[32..63]: lo nibble = element  64..95  (sg=2),  hi nibble = element  96..127 (sg=3)
//   qs[64..95]: lo nibble = element 128..159 (sg=4),  hi nibble = element 160..191 (sg=5)
//   qs[96..127]:lo nibble = element 192..223 (sg=6),  hi nibble = element 224..255 (sg=7)
//
// Thread mapping (16 threads per super-block):
//   tid = lid / BLOCK_STRIDE  (0..15)   role within super-block
//   ix  = lid % BLOCK_STRIDE  (0..BLOCK_STRIDE-1)  which super-block this thread handles
//   ip  = tid / 8  (0..1)   first or second 64-byte qs section
//   il  = tid % 8  (0..7)   8-byte segment within section
//   sg_lo = 4*ip + 2*(il/4)   scale-group for lo nibbles (0,2,4,6)
//   sg_hi = sg_lo + 1          scale-group for hi nibbles (1,3,5,7)
//   q_off    = 64*ip + 8*il   byte offset into qs[0..127]
//   y_lo_off = 32*sg_lo + 8*(il&3)
//   y_hi_off = y_lo_off + 32
//
// Scale decode follows get_scale_min_k4 from ggml-quants.c.

#define QK_K         256
#define K_SCALE_SIZE 12
#define N_DST        4
#define N_SIMDGROUP  1

#ifdef INTEL_GPU
#define N_SIMDWIDTH 16
#elif defined(ADRENO_GPU)
#define N_SIMDWIDTH 64
#else
// NVIDIA: cl_khr_subgroups not exposed as compile-time macro; use __local tree-reduction
#define N_SIMDWIDTH 32
#endif

// 16 threads collaborate on one super-block, BLOCK_STRIDE super-blocks per iteration
#define BLOCK_STRIDE (N_SIMDWIDTH/16)

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q4_K_f32_flat(
    global uchar * src0_qs,
    global uchar * src0_scales,
    global half  * src0_d,
    global half  * src0_dmin,
    global char  * src1,
    int            offset1,
    global char  * dst,
    int            offsetd,
    int            ne00,
    int            ne01,
    int            ne02,
    int            ne12,
    ulong          nb11,
    ulong          nb12,
    ulong          nb13,
    int            ne0,
    int            ne1,
    int            r2,
    int            r3
) {
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int nb = ne00 / QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = r0 * N_DST;
    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong offset_src1 = (ulong)r1*nb11 + (ulong)i12*nb12 + (ulong)i13*nb13;
    global float * y = (global float *)(src1 + offset_src1);

    // Flat block index base for src0 (in units of super-blocks).
    // Layout: [ne02][ne01][nb] super-blocks, with GQA broadcasting via r2/r3.
    ulong offset_src0 = (ulong)first_row * nb
                      + (ulong)(i12/r2) * ((ulong)nb * ne01)
                      + (ulong)(i13/r3) * ((ulong)nb * ne01 * ne02);

    int lid = get_local_id(0);
    int tid = lid / BLOCK_STRIDE;  // 0..15: role within a super-block
    int ix  = lid % BLOCK_STRIDE;  // 0..BLOCK_STRIDE-1: which super-block per iter

    int ip  = tid / 8;             // 0 or 1 (which 64-byte qs section)
    int il  = tid % 8;             // 0..7  (8-byte segment within section)

    // Scale groups for lo and hi nibbles of this thread's 8 qs bytes.
    int sg_lo = 4*ip + 2*(il/4);  // in {0,2,4,6}
    int sg_hi = sg_lo + 1;        // in {1,3,5,7}
    // Clamped versions for safe speculative evaluation in ternary false-branch
    // (GPU evaluates both branches; negative index -> page fault).
    int sg_lo_m4 = max(sg_lo - 4, 0);  // safe substitute for sg_lo-4
    int sg_hi_m4 = max(sg_hi - 4, 0);  // safe substitute for sg_hi-4

    // Byte offset into qs[0..127] for this thread's 8 bytes.
    int q_off = 64*ip + 8*il;

    // Y element offsets.
    int y_lo_off = 32*sg_lo + 8*(il & 3);
    int y_hi_off = y_lo_off + 32;

    float4 sumf = (float4)(0.f);

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        global float * yb = y + (ulong)ib * QK_K;

        float4 ylo0 = vload4(0, yb + y_lo_off);      // y[y_lo_off..y_lo_off+3]
        float4 ylo1 = vload4(0, yb + y_lo_off + 4);  // y[y_lo_off+4..y_lo_off+7]
        float4 yhi0 = vload4(0, yb + y_hi_off);
        float4 yhi1 = vload4(0, yb + y_hi_off + 4);

        float sumy_lo = dot(ylo0 + ylo1, (float4)(1.f));
        float sumy_hi = dot(yhi0 + yhi1, (float4)(1.f));

        if (first_row + 0 < ne01) {
            ulong bi = offset_src0 + (ulong)0*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s0 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 1 < ne01) {
            ulong bi = offset_src0 + (ulong)1*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s1 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 2 < ne01) {
            ulong bi = offset_src0 + (ulong)2*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s2 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 3 < ne01) {
            ulong bi = offset_src0 + (ulong)3*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s3 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
    }

    global float * dst_f32 = (global float *)dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

#ifdef ADRENO_GPU
    // Adreno: sub_group_reduce_add across the 64-wide wavefront.
    float4 result = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );
    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = result.s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = result.s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = result.s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = result.s3;
    }
#else
    // NVIDIA: cl_khr_subgroups not exposed as compile-time macro,
    // so use __local tree-reduction instead of sub_group_reduce_add.
    __local float4 lm[N_SIMDWIDTH];
    lm[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = N_SIMDWIDTH/2; s > 0; s >>= 1) {
        if (lid < s) lm[lid] += lm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = lm[0].s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = lm[0].s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = lm[0].s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = lm[0].s3;
    }
#endif
}
