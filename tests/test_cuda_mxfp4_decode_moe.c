/* MXFP4 routed-MoE decode at V4.1 shapes (5120 -> 2304 -> 5120, six of N
 * experts): every kernel variant must write the bits of the reference
 * qwarp32 kernels (DS4_CUDA_MXFP4_DECODE_REF), resident and from the SSD
 * expert cache. DS4_TEST_MXFP4_TIMING=1 also times one decode step with the
 * experts resident in VRAM (DS4_CUDA_COPY_MODEL=1 copies them there). */
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { EXPERTS = 64, SELECTED = 6, IN = 5120, MID = 2304, OUT = 5120, ITERS = 64 };

static uint32_t random_state = 71;
static uint32_t random_u32(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static ds4_gpu_tensor *tensor(size_t bytes, const void *data) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && (!data || ds4_gpu_tensor_write(t, 0, data, bytes)));
    return t;
}

/* MXFP4 blocks: an E8M0 scale near 2^-6 and random nibbles. */
static void fill_mxfp4(unsigned char *w, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        unsigned char *blk = w + b * 17;
        blk[0] = (unsigned char)(119 + random_u32() % 5);
        for (int i = 1; i < 17; i++) blk[i] = (unsigned char)(random_u32() >> 24);
    }
}

int main(void) {
    const size_t gate_row = IN / 32 * 17, down_row = MID / 32 * 17;
    const size_t gate_bytes = (size_t)MID * gate_row, down_bytes = (size_t)OUT * down_row;
    const size_t model_bytes = EXPERTS * (2 * gate_bytes + down_bytes);
    unsigned char *model = malloc(model_bytes);
    CHECK(model);
    fill_mxfp4(model, model_bytes / 17);
    char path[] = "ds4-mxfp4-decode-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0);
    for (size_t done = 0; done < model_bytes;) {
        const ssize_t n = write(fd, model + done, model_bytes - done);
        CHECK(n > 0);
        done += (size_t)n;
    }
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model));
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    static float input[IN], weights[ITERS * SELECTED];
    static int32_t ids[ITERS * SELECTED];
    for (unsigned i = 0; i < IN; i++) input[i] = ((int)(random_u32() >> 20) - 2048) / 1024.0f;
    for (unsigned i = 0; i < ITERS * SELECTED; i++) {
        weights[i] = (1 + random_u32() % 7) / 16.0f;
        ids[i] = (int32_t)((i / SELECTED * 7 + i % SELECTED * 11) % EXPERTS);
    }
    ds4_gpu_tensor *x = tensor(sizeof(input), input), *sw = tensor(sizeof(weights), weights);
    ds4_gpu_tensor *si = tensor(sizeof(ids), ids);
    ds4_gpu_tensor *out = tensor((size_t)OUT * 4, NULL);
    const size_t scratch = (size_t)SELECTED * (MID > OUT ? MID : OUT) * 4;
    ds4_gpu_tensor *gate = tensor(scratch, NULL), *up = tensor(scratch, NULL);
    ds4_gpu_tensor *mid = tensor(scratch, NULL), *down = tensor(scratch, NULL);
    const uint64_t g = 0, u = EXPERTS * gate_bytes, d = 2 * EXPERTS * gate_bytes;
    static float ref_out[OUT], new_out[OUT], ref_mid[SELECTED * MID], new_mid[SELECTED * MID];
#define MOE(sel, wts, resident) ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, down, model, model_bytes, \
        g, u, d, 39, 39, gate_bytes, gate_row, down_bytes, down_row, IN, MID, OUT, sel, wts, \
        EXPERTS, SELECTED, 10.0f, x, NULL, 0, resident)
    int same = 1;
    for (unsigned streaming = 0; streaming < 2; streaming++) {
        ds4_gpu_set_ssd_streaming(streaming != 0);
        if (streaming) {
            ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
            ds4_gpu_set_streaming_expert_cache_budget(2 * SELECTED);
        }
        for (unsigned it = 0; it < 4; it++) {
            ds4_gpu_tensor *sel = ds4_gpu_tensor_view(si, (uint64_t)it * SELECTED * 4, SELECTED * 4);
            ds4_gpu_tensor *wts = ds4_gpu_tensor_view(sw, (uint64_t)it * SELECTED * 4, SELECTED * 4);
            CHECK(sel && wts);
            setenv("DS4_CUDA_MXFP4_DECODE_REF", "1", 1);
            CHECK(MOE(sel, wts, !streaming));
            CHECK(ds4_gpu_tensor_read(out, 0, ref_out, sizeof(ref_out)) &&
                  ds4_gpu_tensor_read(mid, 0, ref_mid, sizeof(ref_mid)));
            unsetenv("DS4_CUDA_MXFP4_DECODE_REF");
            CHECK(MOE(sel, wts, !streaming));
            CHECK(ds4_gpu_tensor_read(out, 0, new_out, sizeof(new_out)) &&
                  ds4_gpu_tensor_read(mid, 0, new_mid, sizeof(new_mid)));
            if (memcmp(ref_mid, new_mid, sizeof(ref_mid)) || memcmp(ref_out, new_out, sizeof(ref_out))) {
                same = 0;
                for (unsigned i = 0; i < SELECTED * MID; i++)
                    if (memcmp(&ref_mid[i], &new_mid[i], 4)) {
                        fprintf(stderr, "%s step %u: mid[%u] %.9g != %.9g\n",
                                streaming ? "streaming" : "resident", it, i, ref_mid[i], new_mid[i]);
                        break;
                    }
                for (unsigned i = 0; i < OUT; i++)
                    if (memcmp(&ref_out[i], &new_out[i], 4)) {
                        fprintf(stderr, "%s step %u: out[%u] %.9g != %.9g\n",
                                streaming ? "streaming" : "resident", it, i, ref_out[i], new_out[i]);
                        break;
                    }
            }
            ds4_gpu_tensor_free(sel);
            ds4_gpu_tensor_free(wts);
        }
    }
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    CHECK(same);
    fprintf(stderr, "CUDA MXFP4 decode MoE matches the reference kernels: PASS\n");
    if (getenv("DS4_TEST_MXFP4_TIMING")) {
        ds4_gpu_tensor *sel[ITERS], *wts[ITERS];
        for (unsigned it = 0; it < ITERS; it++) {
            sel[it] = ds4_gpu_tensor_view(si, (uint64_t)it * SELECTED * 4, SELECTED * 4);
            wts[it] = ds4_gpu_tensor_view(sw, (uint64_t)it * SELECTED * 4, SELECTED * 4);
            CHECK(sel[it] && wts[it]);
        }
        for (unsigned ref = 0; ref < 2; ref++) {
            if (ref) setenv("DS4_CUDA_MXFP4_DECODE_REF", "1", 1);
            else unsetenv("DS4_CUDA_MXFP4_DECODE_REF");
            for (unsigned it = 0; it < 8; it++) CHECK(MOE(sel[it], wts[it], true));
            CHECK(ds4_gpu_synchronize());
            const double t0 = now_sec();
            for (unsigned rep = 0; rep < 4; rep++)
                for (unsigned it = 0; it < ITERS; it++) CHECK(MOE(sel[it], wts[it], true));
            CHECK(ds4_gpu_synchronize());
            const double us = (now_sec() - t0) * 1e6 / (4 * ITERS);
            fprintf(stderr, "%s: %.1f us per decode MoE, %.0f GB/s of expert weights\n",
                    ref ? "reference" : "default", us, SELECTED * (2.0 * gate_bytes + down_bytes) / us / 1e3);
        }
        unsetenv("DS4_CUDA_MXFP4_DECODE_REF");
        for (unsigned it = 0; it < ITERS; it++) {
            ds4_gpu_tensor_free(sel[it]);
            ds4_gpu_tensor_free(wts[it]);
        }
    }
#undef MOE
    ds4_gpu_cleanup();
    close(fd);
    free(model);
    return 0;
}
