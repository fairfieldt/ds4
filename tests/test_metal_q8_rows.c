#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The Q8_0 decode-rows matvec for 2..8 rows must equal, bit for bit, one
 * one-row call per row: verifier blocks rely on matching one-token decode. */
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;

static uint32_t rng = 12345u;
static uint32_t next_u32(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static float next_f32(void) { return ((int32_t)(next_u32() >> 8) - (1 << 23)) / (float)(1 << 22); }

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = (x >> 13) & 0x3ffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (uint32_t)exp << 10 | mant);
}

static int run(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t row_bytes = in_dim / 32u * sizeof(q8_block);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t model_size = (row_bytes * out_dim + page - 1u) / page * page;
    void *model = NULL;
    enum { MAX_ROWS = 8 };
    float *x = malloc((size_t)in_dim * MAX_ROWS * sizeof(float));
    float *got = malloc((size_t)out_dim * MAX_ROWS * sizeof(float));
    float *want = malloc((size_t)out_dim * MAX_ROWS * sizeof(float));
    if (posix_memalign(&model, page, model_size) || !x || !got || !want) return 0;
    memset(model, 0, model_size);
    for (uint32_t r = 0; r < out_dim; r++) {
        q8_block *b = (q8_block *)((char *)model + r * row_bytes);
        for (uint32_t k = 0; k < in_dim / 32u; k++) {
            b[k].d = f32_to_f16(0.001f + (next_u32() % 1000u) * 0.00003f);
            for (int i = 0; i < 32; i++) b[k].qs[i] = (int8_t)(next_u32() % 255u) - 127;
        }
    }
    for (uint32_t i = 0; i < in_dim * MAX_ROWS; i++) x[i] = next_f32();
    int ok = ds4_gpu_set_model_map(model, model_size);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)in_dim * MAX_ROWS * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)out_dim * MAX_ROWS * sizeof(float));
    ok = ok && xt && ot && ds4_gpu_tensor_write(xt, 0, x, (uint64_t)in_dim * MAX_ROWS * sizeof(float));
    for (uint32_t t = 0; ok && t < MAX_ROWS; t++) {
        ds4_gpu_tensor *xv = ds4_gpu_tensor_view(xt, (uint64_t)t * in_dim * 4u, (uint64_t)in_dim * 4u);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(ot, (uint64_t)t * out_dim * 4u, (uint64_t)out_dim * 4u);
        ok = xv && ov && ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            ov, model, model_size, 0, in_dim, out_dim, xv, 1);
        ds4_gpu_tensor_free(xv);
        ds4_gpu_tensor_free(ov);
    }
    ok = ok && ds4_gpu_tensor_read(ot, 0, want, (uint64_t)out_dim * MAX_ROWS * sizeof(float));
    for (uint32_t rows = 2; ok && rows <= MAX_ROWS; rows++) {
        ok = ds4_gpu_tensor_fill_f32(ot, NAN, (uint64_t)out_dim * MAX_ROWS) &&
            ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                ot, model, model_size, 0, in_dim, out_dim, xt, rows) &&
            ds4_gpu_tensor_read(ot, 0, got, (uint64_t)out_dim * MAX_ROWS * sizeof(float));
        for (uint32_t i = 0; ok && i < rows * out_dim; i++) {
            if (memcmp(&got[i], &want[i], sizeof(float)) != 0) {
                fprintf(stderr, "q8 rows in=%u out=%u rows=%u row=%u col=%u got=%.9g want=%.9g\n",
                        in_dim, out_dim, rows, i / out_dim, i % out_dim, got[i], want[i]);
                ok = 0;
            }
        }
        for (uint32_t i = rows * out_dim; ok && i < MAX_ROWS * out_dim; i++)
            if (!isnan(got[i])) ok = 0;
    }
    ds4_gpu_tensor_free(ot);
    ds4_gpu_tensor_free(xt);
    free(model); free(x); free(got); free(want);
    if (ok) fprintf(stderr, "q8 decode rows in=%u out=%u rows 2..8: bitwise equal to one-row calls\n",
                    in_dim, out_dim);
    return ok;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const int ok = run(5120, 1031) && run(1024, 4096) && run(256, 70001);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
