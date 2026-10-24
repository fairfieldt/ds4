/* The warp-per-row IQ2_XXS decode gate/up kernel must write the same bits as
 * the qwarp32 kernel it replaces, resident and from the SSD expert cache. */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { EXPERTS = 8, SELECTED = 6, IN = 5120, MID = 2304, OUT = 5120, ROWS = 8 };

static uint32_t random_state = 29;
static uint32_t random_u32(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static ds4_gpu_tensor *tensor(size_t bytes, const void *data) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && (!data || ds4_gpu_tensor_write(t, 0, data, bytes)));
    return t;
}

int main(void) {
    const size_t gate_row = IN / 256 * 66, down_row = MID / 256 * 84;
    const size_t gate_bytes = (size_t)MID * gate_row, down_bytes = (size_t)OUT * down_row;
    const size_t model_bytes = EXPERTS * (2 * gate_bytes + down_bytes);
    unsigned char *model = malloc(model_bytes);
    CHECK(model);
    for (size_t i = 0; i < model_bytes; i++) model[i] = (unsigned char)(random_u32() >> 24);
    /* Finite FP16 superblock scales of mixed magnitude. */
    for (size_t i = 0; i < 2 * EXPERTS * (size_t)MID * (IN / 256); i++) {
        unsigned char *d = model + i * 66;
        d[0] = (unsigned char)random_u32();
        d[1] = (unsigned char)(0x20 + random_u32() % 16);
    }
    for (size_t i = 0; i < EXPERTS * (size_t)OUT * (MID / 256); i++) {
        unsigned char *b = model + 2 * EXPERTS * gate_bytes + i * 84;
        b[80] = (unsigned char)random_u32(); b[81] = 0x24;
        b[82] = (unsigned char)random_u32(); b[83] = 0x20;
    }
    char path[] = "ds4-moe-decode-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0 && write(fd, model, model_bytes) == (ssize_t)model_bytes);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model));
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    static float input[ROWS * IN], weights[ROWS * SELECTED];
    int32_t ids[ROWS * SELECTED];
    for (unsigned i = 0; i < ROWS * IN; i++) input[i] = ((int)(random_u32() >> 20) - 2048) / 1024.0f;
    for (unsigned i = 0; i < ROWS * SELECTED; i++) weights[i] = (1 + random_u32() % 7) / 16.0f;
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned j = 0; j < SELECTED; j++) ids[r * SELECTED + j] = (r * 3 + j * 5) % EXPERTS;
    ds4_gpu_tensor *x = tensor(sizeof(input), input), *sw = tensor(sizeof(weights), weights);
    ds4_gpu_tensor *si = tensor(sizeof(ids), ids);
    ds4_gpu_tensor *out = tensor((size_t)ROWS * OUT * 4, NULL);
    const size_t scratch = (size_t)ROWS * SELECTED * (MID > OUT ? MID : OUT) * 4;
    ds4_gpu_tensor *gate = tensor(scratch, NULL), *up = tensor(scratch, NULL);
    ds4_gpu_tensor *mid = tensor(scratch, NULL), *down = tensor(scratch, NULL);
    const uint64_t g = 0, u = EXPERTS * gate_bytes, d = 2 * EXPERTS * gate_bytes;
    static float ref_out[ROWS * OUT], ref_mid[ROWS * SELECTED * MID];
    static float new_out[ROWS * OUT], new_mid[ROWS * SELECTED * MID];
    const char *variants[] = {"fused", "unfused"};
    int matched[2] = {1, 1};
    for (unsigned streaming = 0; streaming < 2; streaming++) {
        ds4_gpu_set_ssd_streaming(streaming != 0);
        if (streaming) {
            ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
            ds4_gpu_set_streaming_expert_cache_budget(2 * EXPERTS);
        }
        const unsigned counts[] = {1, 3, 8};
        for (unsigned ci = 0; ci < (streaming ? 3u : 1u); ci++) {
            const unsigned rows = counts[ci];
            bool half = false;
            setenv("DS4_CUDA_MOE_DECODE_QWARP_GATE", "1", 1);
            CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes, g, u, d,
                16, 10, gate_bytes, gate_row, down_bytes, down_row, IN, MID, OUT, si, sw,
                EXPERTS, SELECTED, 10, x, 0, rows, &half, !streaming));
            CHECK(!half && ds4_gpu_tensor_read(out, 0, ref_out, rows * OUT * 4) &&
                  ds4_gpu_tensor_read(mid, 0, ref_mid, (size_t)rows * SELECTED * MID * 4));
            unsetenv("DS4_CUDA_MOE_DECODE_QWARP_GATE");
            for (unsigned v = 0; v < 2; v++) {
                if (v) setenv("DS4_CUDA_MOE_DECODE_WARP_UNFUSED", "1", 1);
                else unsetenv("DS4_CUDA_MOE_DECODE_WARP_UNFUSED");
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes, g, u, d,
                    16, 10, gate_bytes, gate_row, down_bytes, down_row, IN, MID, OUT, si, sw,
                    EXPERTS, SELECTED, 10, x, 0, rows, &half, !streaming));
                CHECK(!half && ds4_gpu_tensor_read(out, 0, new_out, rows * OUT * 4) &&
                      ds4_gpu_tensor_read(mid, 0, new_mid, (size_t)rows * SELECTED * MID * 4));
                const int same = !memcmp(ref_mid, new_mid, (size_t)rows * SELECTED * MID * 4) &&
                                 !memcmp(ref_out, new_out, (size_t)rows * OUT * 4);
                if (!same) {
                    matched[v] = 0;
                    for (size_t i = 0; i < (size_t)rows * SELECTED * MID; i++) {
                        if (memcmp(&ref_mid[i], &new_mid[i], 4)) {
                            fprintf(stderr, "%s %s rows=%u: mid[%zu] %.9g != %.9g\n", variants[v],
                                    streaming ? "streaming" : "resident", rows, i, ref_mid[i], new_mid[i]);
                            break;
                        }
                    }
                }
            }
            unsetenv("DS4_CUDA_MOE_DECODE_WARP_UNFUSED");
        }
    }
    for (unsigned v = 0; v < 2; v++)
        fprintf(stderr, "IQ2_XXS decode gate/up warp kernel, %s accumulation: %s\n",
                variants[v], matched[v] ? "bit-identical" : "differs");
    CHECK(matched[0]);
    fprintf(stderr, "CUDA MoE decode gate/up warp kernel matches qwarp32: PASS\n");
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_cleanup();
    close(fd);
    free(model);
    return 0;
}
