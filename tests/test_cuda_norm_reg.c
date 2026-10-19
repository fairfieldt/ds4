/* The register-resident RMS norm kernels must write the same bits as the
 * kernels they replace (DS4_CUDA_NORM_NO_REG), weighted, BF16 and plain. */
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t random_state = 53;
static float random_value(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return ((int)(random_state >> 12) - (1 << 19)) / (float)(1 << 17);
}

enum { MAX_N = 20480, MAX_ROWS = 64 };
static float input[MAX_N * MAX_ROWS], a[MAX_N * MAX_ROWS], b[MAX_N * MAX_ROWS];

static void compare(const char *what, uint32_t n, uint32_t rows) {
    if (!memcmp(a, b, (size_t)n * rows * 4)) return;
    for (size_t i = 0; i < (size_t)n * rows; i++) {
        if (memcmp(&a[i], &b[i], 4)) {
            fprintf(stderr, "%s n=%u rows=%u: [%zu] %.9g != %.9g\n", what, n, rows, i, a[i], b[i]);
            exit(1);
        }
    }
}

int main(void) {
    static float weights[8192];
    for (unsigned i = 0; i < 8192; i++) weights[i] = 0.5f + random_value() / 8.0f;
    for (unsigned i = 0; i < MAX_N * MAX_ROWS; i++) input[i] = random_value();
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(weights, sizeof(weights)));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input)), *y = ds4_gpu_tensor_alloc(sizeof(input));
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
    const uint32_t widths[] = {128, 512, 1280, 5120, 8192}, rows_list[] = {1, 3, 64};
    for (unsigned wi = 0; wi < 5; wi++) {
        for (unsigned ri = 0; ri < 3; ri++) {
            const uint32_t n = widths[wi], rows = rows_list[ri];
            for (unsigned kind = 0; kind < 3; kind++) {
                if (kind == 0 && rows != 1) continue;
                float *dst[2] = {a, b};
                for (unsigned v = 0; v < 2; v++) {
                    if (v == 0) setenv("DS4_CUDA_NORM_NO_REG", "1", 1);
                    else unsetenv("DS4_CUDA_NORM_NO_REG");
                    if (kind == 0) CHECK(ds4_gpu_rms_norm_weight_tensor(y, x, weights, sizeof(weights), 0, n, 1e-6f));
                    else if (kind == 1) CHECK(ds4_gpu_rms_norm_weight_rows_tensor(y, x, weights, sizeof(weights), 0, n, rows, 1e-6f));
                    else CHECK(ds4_gpu_dsv41_norm(y, x, weights, sizeof(weights), 0, n, rows, 1e-6f));
                    CHECK(ds4_gpu_tensor_read(y, 0, dst[v], (size_t)n * rows * 4));
                }
                compare(kind == 2 ? "V4.1 BF16 norm" : "weighted norm", n, rows);
            }
        }
    }
    const uint32_t plain[] = {2048, 8192, 20480};
    for (unsigned pi = 0; pi < 3; pi++) {
        for (unsigned ri = 0; ri < 2; ri++) {
            const uint32_t n = plain[pi], rows = ri ? 3 : 1;
            float *dst[2] = {a, b};
            for (unsigned v = 0; v < 2; v++) {
                if (v == 0) setenv("DS4_CUDA_NORM_NO_REG", "1", 1);
                else unsetenv("DS4_CUDA_NORM_NO_REG");
                if (rows == 1) CHECK(ds4_gpu_rms_norm_plain_tensor(y, x, n, 1e-6f));
                else CHECK(ds4_gpu_rms_norm_plain_rows_tensor(y, x, n, rows, 1e-6f));
                CHECK(ds4_gpu_tensor_read(y, 0, dst[v], (size_t)n * rows * 4));
            }
            compare("plain norm", n, rows);
        }
    }
    fprintf(stderr, "CUDA register-resident RMS norms match: PASS\n");
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(y);
    ds4_gpu_cleanup();
    return 0;
}
