#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef NVIDIA_GPU

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

//------------------------------------------------------------------------------
// block_q4_K
//------------------------------------------------------------------------------
#define QK_K            256
#define BLOCK_Q4K_SIZE  144
#define K_SCALE_SIZE    12

// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
typedef struct {
    half d;    // super-block scale for quantized scales
    half dmin; // super-block scale for quantized mins

    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qs[QK_K/2];           // 4-bit quants
} block_q4_K;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 16
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// number of (super) blocks each subgroup processes
// each thread in a subgroup processes a block (32 weights)
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q4_K_f32_flat(
    global uchar * src0_q,
    global uchar * src0_s,
    global half  * src0_d,
    global half  * src0_dm,
    global char  * src1,
    int offset1,
    global char  * dst,
    int offsetd,
    int ne00,
    int ne01,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne12,
    ulong nb11,
    ulong nb12,
    ulong nb13,
    int ne0,
    int ne1,
    int r2,
    int r3
) {
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id()/8;
    int it = get_sub_group_local_id()%8;
    int iq = it/4;
    int ir = it%4;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = (first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03)/BLOCK_Q4K_SIZE;
    uint blk = nb01 / BLOCK_Q4K_SIZE;
    global uchar * blk_q     = (global uchar *)src0_q  + offset_src0*(QK_K/2);
    global uchar * blk_s     = (global uchar *)src0_s  + offset_src0*K_SCALE_SIZE;
    global half  * blk_d     = (global half  *)src0_d  + offset_src0;
    global half  * blk_dm    = (global half  *)src0_dm + offset_src0;

    int offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
    global float * y = (global float *)(src1 + offset_src1);

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort  sc16[4];
    uchar * sc8 = (uchar *)sc16;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0];
            sumy.s0 += yl[i+0];

            yl[i+8] = y4[i+32];
            sumy.s1 += yl[i+8];

            yh[i+0] = y4[i+128];
            sumy.s2 += yh[i+0];

            yh[i+8] = y4[i+160];
            sumy.s3 += yh[i+8];
        }

        global ushort * q1 = (global ushort *)(blk_q + ib * (QK_K/2)) + (16 * iq + 4 * ir);
        global ushort * sc = (global ushort *)(blk_s + ib * K_SCALE_SIZE) + iq;
        global half   * d  = blk_d + ib;
        global half   * dm = blk_dm + ib;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            global ushort * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = *d;
            float dmin = *dm;
            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += blk*64;
            sc += blk*6;
            d  += blk;
            dm += blk;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add(sumf[row]);
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}

#endif // !NVIDIA_GPU

// NVIDIA-only SOA flat kernel for Q4_K x f32 matrix-vector multiply.
// Uses separate qs / scales / d / dmin arrays (AOS -> SOA) for better
// memory coalescing vs the generic AOS kernel.
//
// Thread mapping (32 lanes = 1 warp):
//   ix  = lid/8   super-block group index (0..3)
//   it  = lid%8   thread within group
//   iq  = it/4    first (0) or second (1) half of super-block
//   ir  = it%4    element group within half (0..3)
//   BLOCK_STRIDE = N_SIMDWIDTH/8 = 4  (super-blocks per outer loop step)
//
// Each warp computes N_DST=4 output rows and reduces with __local tree.

#ifdef NVIDIA_GPU

#define QK_K         256
#define K_SCALE_SIZE 12
#define N_DST        4
#define N_SIMDWIDTH  32
#define BLOCK_STRIDE (N_SIMDWIDTH/8)   // = 4

kernel void kernel_mul_mv_q4_K_f32_flat(
    global uchar * src0_qs,      // SOA: quantized weights  [n_blocks * 128 bytes]
    global uchar * src0_scales,  // SOA: scales             [n_blocks * 12 bytes]
    global half  * src0_d,       // SOA: super-block scale  [n_blocks halves]
    global half  * src0_dmin,    // SOA: super-block min    [n_blocks halves]
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

    const ushort kmask1 = 0x3f3f;
    const ushort kmask2 = 0x0f0f;
    const ushort kmask3 = 0xc0c0;

    int nb = ne00 / QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = r0 * N_DST;   // N_SIMDGROUP=1 for NVIDIA

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong offset_src1 = (ulong)r1*nb11 + (ulong)i12*nb12 + (ulong)i13*nb13;
    global float * y = (global float *)(src1 + offset_src1);

    // Linear block index of the first block of first_row in this batch slice.
    ulong offset_src0 = (ulong)first_row * nb
                      + (ulong)(i12/r2) * ((ulong)nb * ne01)
                      + (ulong)(i13/r3) * ((ulong)nb * ne01 * ne02);

    int lid = get_local_id(0);
    int ix  = lid / 8;    // super-block group (0..3)
    int it  = lid % 8;    // thread within group
    int iq  = it / 4;     // half of super-block (0 or 1)
    int ir  = it % 4;     // element group within half (0..3)

    global float * y4 = y + (ulong)ix * QK_K + 64 * iq + 8 * ir;

    float sumf[N_DST] = {0.f, 0.f, 0.f, 0.f};
    float yl[16], yh[16];

    ushort sc16[4];
    uchar * sc8 = (uchar *)sc16;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        // Load y values once; reuse for all 4 output rows.
        float4 sumy = (float4)(0.f);
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0];   sumy.s0 += yl[i+0];
            yl[i+8] = y4[i+32];  sumy.s1 += yl[i+8];
            yh[i+0] = y4[i+128]; sumy.s2 += yh[i+0];
            yh[i+8] = y4[i+160]; sumy.s3 += yh[i+8];
        }

        // Row 0
        if (first_row + 0 < ne01) {
            ulong bi = offset_src0 + (ulong)0*nb + ib;
            global ushort * q1 = (global ushort *)(src0_qs    + bi*(ulong)128) + 16*iq + 4*ir;
            global ushort * sc = (global ushort *)(src0_scales + bi*(ulong)K_SCALE_SIZE) + iq;
            float dall = vload_half(0, src0_d    + bi);
            float dmin = vload_half(0, src0_dmin + bi);
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            global ushort * q2 = q1 + 32;
            float4 acc1 = (float4)(0.f), acc2 = (float4)(0.f);
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }
            sumf[0] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                               (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                               (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                               (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                       dmin * (sumy.s0*sc8[2] + sumy.s1*sc8[3] +
                               sumy.s2*sc8[6] + sumy.s3*sc8[7]);
        }

        // Row 1
        if (first_row + 1 < ne01) {
            ulong bi = offset_src0 + (ulong)1*nb + ib;
            global ushort * q1 = (global ushort *)(src0_qs    + bi*(ulong)128) + 16*iq + 4*ir;
            global ushort * sc = (global ushort *)(src0_scales + bi*(ulong)K_SCALE_SIZE) + iq;
            float dall = vload_half(0, src0_d    + bi);
            float dmin = vload_half(0, src0_dmin + bi);
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            global ushort * q2 = q1 + 32;
            float4 acc1 = (float4)(0.f), acc2 = (float4)(0.f);
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }
            sumf[1] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                               (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                               (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                               (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                       dmin * (sumy.s0*sc8[2] + sumy.s1*sc8[3] +
                               sumy.s2*sc8[6] + sumy.s3*sc8[7]);
        }

        // Row 2
        if (first_row + 2 < ne01) {
            ulong bi = offset_src0 + (ulong)2*nb + ib;
            global ushort * q1 = (global ushort *)(src0_qs    + bi*(ulong)128) + 16*iq + 4*ir;
            global ushort * sc = (global ushort *)(src0_scales + bi*(ulong)K_SCALE_SIZE) + iq;
            float dall = vload_half(0, src0_d    + bi);
            float dmin = vload_half(0, src0_dmin + bi);
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            global ushort * q2 = q1 + 32;
            float4 acc1 = (float4)(0.f), acc2 = (float4)(0.f);
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }
            sumf[2] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                               (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                               (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                               (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                       dmin * (sumy.s0*sc8[2] + sumy.s1*sc8[3] +
                               sumy.s2*sc8[6] + sumy.s3*sc8[7]);
        }

        // Row 3
        if (first_row + 3 < ne01) {
            ulong bi = offset_src0 + (ulong)3*nb + ib;
            global ushort * q1 = (global ushort *)(src0_qs    + bi*(ulong)128) + 16*iq + 4*ir;
            global ushort * sc = (global ushort *)(src0_scales + bi*(ulong)K_SCALE_SIZE) + iq;
            float dall = vload_half(0, src0_d    + bi);
            float dmin = vload_half(0, src0_dmin + bi);
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            global ushort * q2 = q1 + 32;
            float4 acc1 = (float4)(0.f), acc2 = (float4)(0.f);
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }
            sumf[3] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                               (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                               (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                               (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                       dmin * (sumy.s0*sc8[2] + sumy.s1*sc8[3] +
                               sumy.s2*sc8[6] + sumy.s3*sc8[7]);
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    // __local tree reduction — N_SIMDWIDTH=32 (warp), N_SIMDGROUP=1.
    __local float lm[N_DST * N_SIMDWIDTH];
    lm[0*N_SIMDWIDTH + lid] = sumf[0];
    lm[1*N_SIMDWIDTH + lid] = sumf[1];
    lm[2*N_SIMDWIDTH + lid] = sumf[2];
    lm[3*N_SIMDWIDTH + lid] = sumf[3];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = N_SIMDWIDTH/2; s > 0; s >>= 1) {
        if (lid < s) {
            lm[0*N_SIMDWIDTH + lid] += lm[0*N_SIMDWIDTH + lid + s];
            lm[1*N_SIMDWIDTH + lid] += lm[1*N_SIMDWIDTH + lid + s];
            lm[2*N_SIMDWIDTH + lid] += lm[2*N_SIMDWIDTH + lid + s];
            lm[3*N_SIMDWIDTH + lid] += lm[3*N_SIMDWIDTH + lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    global float * dst_f32 = (global float *)dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;
    if (lid == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = lm[0*N_SIMDWIDTH];
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = lm[1*N_SIMDWIDTH];
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = lm[2*N_SIMDWIDTH];
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = lm[3*N_SIMDWIDTH];
    }
}

#endif // NVIDIA_GPU
