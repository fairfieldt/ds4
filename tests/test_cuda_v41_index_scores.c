/* The thread-per-score V4.1 indexer kernel must write the same bits as the
 * warp-per-score kernel (DS4_CUDA_V41_INDEX_SCORES_WARP), decode and prefill. */
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t random_state = 61;
static float random_value(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return ((int)(random_state >> 12) - (1 << 19)) / (float)(1 << 18);
}

int main(void) {
    /* rows, width (keys), start, ratio */
    const uint32_t cases[][4] = {
        {1, 16384, 32767, 2}, {1, 777, 776, 1}, {7, 1000, 1990, 2}, {32, 4096, 4000, 1},
        {32, 8192, 16300, 2}, {33, 333, 300, 1}, {2, 64, 126, 2},
    };
    const uint32_t max_rows = 33, max_width = 16384;
    float *q = malloc((size_t)max_rows * 4096 * 4), *w = malloc((size_t)max_rows * 32 * 4);
    float *keys = malloc((size_t)max_width * 128 * 4);
    float *a = malloc((size_t)max_rows * max_width * 4), *b = malloc((size_t)max_rows * max_width * 4);
    CHECK(q && w && keys && a && b);
    for (size_t i = 0; i < (size_t)max_rows * 4096; i++) q[i] = random_value();
    for (size_t i = 0; i < (size_t)max_rows * 32; i++) w[i] = random_value();
    for (size_t i = 0; i < (size_t)max_width * 128; i++) keys[i] = random_value();
    CHECK(ds4_gpu_init());
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc((size_t)max_rows * 4096 * 4);
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc((size_t)max_rows * 32 * 4);
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc((size_t)max_width * 128 * 4);
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc((size_t)max_rows * max_width * 4);
    CHECK(tq && tw && tk && ts);
    CHECK(ds4_gpu_tensor_write(tq, 0, q, (size_t)max_rows * 4096 * 4));
    CHECK(ds4_gpu_tensor_write(tw, 0, w, (size_t)max_rows * 32 * 4));
    CHECK(ds4_gpu_tensor_write(tk, 0, keys, (size_t)max_width * 128 * 4));
    const char *variants[] = {"fused", "unfused"};
    int same[2] = {1, 1};
    for (unsigned c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        const uint32_t rows = cases[c][0], width = cases[c][1], start = cases[c][2], ratio = cases[c][3];
        const size_t n = (size_t)rows * width;
        setenv("DS4_CUDA_V41_INDEX_SCORES_WARP", "1", 1);
        CHECK(ds4_gpu_dsv41_indexer_scores_batch(ts, tq, tw, tk, width, rows, start, ratio));
        CHECK(ds4_gpu_tensor_read(ts, 0, a, n * 4));
        unsetenv("DS4_CUDA_V41_INDEX_SCORES_WARP");
        setenv("DS4_CUDA_V41_INDEX_SCORES_THREAD", "1", 1);
        for (unsigned v = 0; v < 2; v++) {
            if (v) setenv("DS4_CUDA_V41_INDEX_SCORES_UNFUSED", "1", 1);
            else unsetenv("DS4_CUDA_V41_INDEX_SCORES_UNFUSED");
            CHECK(ds4_gpu_dsv41_indexer_scores_batch(ts, tq, tw, tk, width, rows, start, ratio));
            CHECK(ds4_gpu_tensor_read(ts, 0, b, n * 4));
            if (memcmp(a, b, n * 4)) {
                same[v] = 0;
                for (size_t i = 0; i < n; i++) {
                    if (memcmp(&a[i], &b[i], 4)) {
                        fprintf(stderr, "%s rows=%u width=%u start=%u ratio=%u: [%zu] %.9g != %.9g\n",
                                variants[v], rows, width, start, ratio, i, a[i], b[i]);
                        break;
                    }
                }
            }
        }
        unsetenv("DS4_CUDA_V41_INDEX_SCORES_UNFUSED");
        unsetenv("DS4_CUDA_V41_INDEX_SCORES_THREAD");
    }
    for (unsigned v = 0; v < 2; v++)
        fprintf(stderr, "V4.1 index scores, thread per score, %s heads: %s\n", variants[v],
                same[v] ? "bit-identical" : "differs");
    CHECK(same[0]);
    fprintf(stderr, "CUDA V4.1 thread-per-score index scores match the warp kernel: PASS\n");
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(ts);
    ds4_gpu_cleanup();
    free(q); free(w); free(keys); free(a); free(b);
    return 0;
}
