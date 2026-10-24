/* Batched V4.1 F16 projection rows must write the bits of the per-row
 * one-row cuBLAS calls they replace, for every projection shape V4.1 uses. */
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t random_state = 41;
static uint32_t random_u32(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static uint16_t f16_bits(float f) {
    /* Small normal values only: sign, a few exponents, 10 mantissa bits. */
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int exp = (int)((u >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((u >> 13) & 0x3ffu));
}

int main(void) {
    /* compressor kv/gate, indexer q_b, indexer projection, indexer key */
    const uint32_t shapes[][2] = {{5120, 512}, {1280, 4096}, {5120, 32}, {512, 128}};
    const uint32_t counts[] = {2, 7, 513, 2048, 2049, 4096, 8192};
    CHECK(ds4_gpu_init());
    size_t model_bytes = 0;
    for (unsigned s = 0; s < 4; s++) model_bytes += (size_t)shapes[s][0] * shapes[s][1] * 2;
    uint16_t *model = malloc(model_bytes);
    CHECK(model);
    for (size_t i = 0; i < model_bytes / 2; i++)
        model[i] = f16_bits(((int)(random_u32() >> 20) - 2048) / 40960.0f);
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    const uint32_t max_rows = 8192, max_width = 5120, max_out = 4096;
    float *input = malloc((size_t)max_rows * max_width * 4);
    float *a = malloc((size_t)max_rows * max_out * 4), *b = malloc((size_t)max_rows * max_out * 4);
    CHECK(input && a && b);
    for (size_t i = 0; i < (size_t)max_rows * max_width; i++) input[i] = ((int)(random_u32() >> 16) - 32768) / 16384.0f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((size_t)max_rows * max_width * 4);
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((size_t)max_rows * max_out * 4);
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, (size_t)max_rows * max_width * 4));
    uint64_t offset = 0;
    for (unsigned s = 0; s < 4; s++) {
        const uint32_t width = shapes[s][0], outputs = shapes[s][1];
        for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
            const uint32_t rows = counts[c];
            if ((uint64_t)rows * outputs > (uint64_t)max_rows * max_out) continue;
            setenv("DS4_CUDA_V41_SERIAL_PROJECTION_ROWS", "1", 1);
            CHECK(ds4_gpu_dsv41_projection_rows(y, model, model_bytes, offset, width, outputs, rows, x));
            CHECK(ds4_gpu_tensor_read(y, 0, a, (size_t)rows * outputs * 4));
            unsetenv("DS4_CUDA_V41_SERIAL_PROJECTION_ROWS");
            CHECK(ds4_gpu_dsv41_projection_rows(y, model, model_bytes, offset, width, outputs, rows, x));
            CHECK(ds4_gpu_tensor_read(y, 0, b, (size_t)rows * outputs * 4));
            if (memcmp(a, b, (size_t)rows * outputs * 4)) {
                for (size_t i = 0; i < (size_t)rows * outputs; i++) {
                    if (memcmp(&a[i], &b[i], 4)) {
                        fprintf(stderr, "width=%u outputs=%u rows=%u: [%zu] %.9g != %.9g\n",
                                width, outputs, rows, i, a[i], b[i]);
                        break;
                    }
                }
                exit(1);
            }
        }
        offset += (uint64_t)width * outputs * 2;
        fprintf(stderr, "V4.1 F16 projection rows width=%u outputs=%u: batched == per-row\n", width, outputs);
    }
    fprintf(stderr, "CUDA V4.1 batched projection rows match one-row cuBLAS: PASS\n");
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(y);
    ds4_gpu_cleanup();
    free(model); free(input); free(a); free(b);
    return 0;
}
