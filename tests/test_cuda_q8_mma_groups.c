/* The exact Q8 MMA kernel with several 16-token groups per warp must write
 * the bits of the one-group kernel (DS4_CUDA_Q8_MMA_G1) on the V4.1
 * attention output projections: Q8 rows and the grouped low projection. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ds4_test_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t random_state = 67;
static uint32_t random_u32(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static void fill_q8(unsigned char *w, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        unsigned char *blk = w + b * 34;
        blk[0] = (unsigned char)random_u32();
        blk[1] = (unsigned char)(0x14 + random_u32() % 8);
        for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(random_u32() >> 24);
    }
}

static void compare(const char *what, const float *a, const float *b, size_t n, uint32_t rows) {
    if (!memcmp(a, b, n * 4)) return;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], 4)) {
            fprintf(stderr, "%s rows=%u: [%zu] %.9g != %.9g\n", what, rows, i, a[i], b[i]);
            exit(1);
        }
    }
}

int main(void) {
    /* attn_output_b: 8192 -> 5120; attn_output_a: 8 groups of 4096 -> 1024 */
    const size_t b_blocks = (size_t)5120 * (8192 / 32), a_blocks = (size_t)8 * 1024 * (4096 / 32);
    const size_t model_bytes = (b_blocks + a_blocks) * 34;
    unsigned char *model = malloc(model_bytes);
    CHECK(model);
    fill_q8(model, b_blocks + a_blocks);
    const uint64_t a_offset = b_blocks * 34;
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    const uint32_t max_rows = 600;
    float *input = malloc((size_t)max_rows * 32768 * 4);
    float *ra = malloc((size_t)max_rows * 8192 * 4), *rb = malloc((size_t)max_rows * 8192 * 4);
    CHECK(input && ra && rb);
    for (size_t i = 0; i < (size_t)max_rows * 32768; i++) input[i] = ((int)(random_u32() >> 16) - 32768) / 8192.0f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((size_t)max_rows * 32768 * 4);
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((size_t)max_rows * 8192 * 4);
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, (size_t)max_rows * 32768 * 4));
    const uint32_t counts[] = {8, 31, 32, 47, 64, 100, 128, 513, 600};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        const uint32_t rows = counts[c];
        float *dst[2] = {ra, rb};
        for (unsigned v = 0; v < 2; v++) {
            if (v == 0) setenv("DS4_CUDA_Q8_MMA_G1", "1", 1);
            else unsetenv("DS4_CUDA_Q8_MMA_G1");
            CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(y, model, model_bytes, 0, 8192, 5120, x, rows));
            CHECK(ds4_gpu_tensor_read(y, 0, dst[v], (size_t)rows * 5120 * 4));
        }
        compare("Q8 rows 8192->5120", ra, rb, (size_t)rows * 5120, rows);
        for (unsigned v = 0; v < 2; v++) {
            if (v == 0) setenv("DS4_CUDA_Q8_MMA_G1", "1", 1);
            else unsetenv("DS4_CUDA_Q8_MMA_G1");
            CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(y, model, model_bytes, a_offset,
                4096, 1024, 8, 0, 8, x, rows));
            CHECK(ds4_gpu_tensor_read(y, 0, dst[v], (size_t)rows * 8192 * 4));
        }
        compare("grouped low 8x4096->1024", ra, rb, (size_t)rows * 8192, rows);
    }
    unsetenv("DS4_CUDA_Q8_MMA_G1");
    fprintf(stderr, "CUDA exact Q8 MMA token groups match one group: PASS\n");
    if (getenv("DS4_TEST_Q8_MMA_TIMING")) {
        /* Kernel time per call at prefill sizes: G caps 1, 2, 4. */
        static float big_in[8192 * 8192];
        for (size_t i = 0; i < 8192u * 8192u; i++) big_in[i] = ((int)(random_u32() >> 16) - 32768) / 8192.0f;
        ds4_gpu_tensor *bx = ds4_gpu_tensor_alloc(sizeof(big_in));
        ds4_gpu_tensor *by = ds4_gpu_tensor_alloc((size_t)8192 * 8192 * 4);
        CHECK(bx && by && ds4_gpu_tensor_write(bx, 0, big_in, sizeof(big_in)));
        const uint32_t sizes[] = {512, 2048, 4096, 8192};
        const char *caps[] = {"1", "2", "4"};
        for (unsigned si = 0; si < 4; si++) {
            for (unsigned ci = 0; ci < 3; ci++) {
                setenv("DS4_CUDA_Q8_MMA_G", caps[ci], 1);
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(by, model, model_bytes, 0, 8192, 5120, bx, sizes[si]));
                CHECK(ds4_gpu_synchronize());
                double t0 = ds4_test_now();
                for (int r = 0; r < 5; r++)
                    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(by, model, model_bytes, 0, 8192, 5120, bx, sizes[si]));
                CHECK(ds4_gpu_synchronize());
                const double rows_ms = (ds4_test_now() - t0) * 1e3 / 5;
                t0 = ds4_test_now();
                for (int r = 0; r < 5; r++)
                    CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(by, model, model_bytes, a_offset,
                        4096, 1024, 8, 0, 8, bx, sizes[si] < 2048 ? sizes[si] : 2048));
                CHECK(ds4_gpu_synchronize());
                const double low_ms = (ds4_test_now() - t0) * 1e3 / 5;
                fprintf(stderr, "rows=%u G<=%s: Q8 8192->5120 %.3f ms, grouped low (%u rows) %.3f ms\n",
                        sizes[si], caps[ci], rows_ms, sizes[si] < 2048 ? sizes[si] : 2048, low_ms);
            }
        }
        unsetenv("DS4_CUDA_Q8_MMA_G");
        ds4_gpu_tensor_free(bx);
        ds4_gpu_tensor_free(by);
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(y);
    ds4_gpu_cleanup();
    free(model); free(input); free(ra); free(rb);
    return 0;
}
