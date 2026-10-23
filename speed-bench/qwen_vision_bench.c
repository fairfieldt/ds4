/* Qwen3.8 vision latency: image encode and image-prompt TTFT / decode speed.
 *
 * Loads the model and the vision encoder once.  For every --image it first
 * times the encoder alone (the first call separately, then --encode-repeat
 * warm calls), then runs --repeat end-to-end blocks after one discarded
 * warm-up block.  A block is a fresh session: encode the image from memory,
 * prefill [system, user: image + prompt, assistant prefix] without thinking,
 * sample the first token (TTFT = encode + prefill + first sample), then
 * decode -n tokens greedily the way ds4_cli.c does (MTP when --mtp).  Token
 * hashes are compared across blocks.
 *
 * qwen_vision_bench -m MODEL --vision MMPROJ --image PATH[@MAX_TOKENS]...
 *                   [--mtp] [--ctx 8192] [-n 128] [--repeat 3] [--encode-repeat 8]
 *                   [--encode-only] [--prompt TEXT]
 *
 * @MAX_TOKENS sets DS4_QWEN4_IMAGE_MAX_TOKENS for that image; an image
 * already a multiple of 32 pixels within the cap keeps its size, so a
 * 512x512 PNG gives 256 image tokens (1,024 patches). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec;
}

static uint8_t *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void encode(ds4_engine *e, const uint8_t *img, size_t len, ds4_vision_embedding *out) {
    char err[256] = {0};
    if (!ds4_engine_vision_encode_memory(e, img, len, out, err, sizeof(err))) {
        fprintf(stderr, "vision encode failed: %s\n", err);
        exit(1);
    }
}

typedef struct { double encode_ms, prefill_ms, ttft_ms, decode_tps; int prompt_tokens, n; uint64_t hash; } block_result;

static block_result run_block(ds4_engine *e, const uint8_t *img, size_t len, const char *prompt,
                              int ctx, int n_predict, bool spec) {
    const ds4_think_mode tm = DS4_THINK_NONE;
    block_result r = {0};
    ds4_session *s = NULL;
    char err[256] = {0};
    if (ds4_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); exit(1); }
    ds4_session_gpu_warmup(s);
    const double t0 = now_ms();
    ds4_vision_embedding emb = {0};
    encode(e, img, len, &emb);
    const double t1 = now_ms();
    ds4_tokens tokens = {0};
    ds4_vision_span span = {0};
    const char *parts[2] = {"", prompt};
    ds4_chat_begin(e, &tokens);
    ds4_chat_append_think_prefix(e, &tokens, tm);
    ds4_chat_append_message(e, &tokens, "system", "You are a helpful assistant");
    if (!ds4_chat_append_multimodal_message(e, &tokens, "user", parts, &emb, 1, &span, err, sizeof(err))) {
        fprintf(stderr, "image prompt failed: %s\n", err);
        exit(1);
    }
    ds4_chat_append_assistant_prefix(e, &tokens, tm);
    if (ds4_session_sync_multimodal(s, &tokens, &span, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "prefill failed: %s\n", err);
        exit(1);
    }
    uint64_t rng = 1, h = 1469598103934665603ull;
    int token = ds4_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &rng);
    const double t2 = now_ms();
    r.encode_ms = t1 - t0;
    r.prefill_ms = t2 - t1;
    r.ttft_ms = t2 - t0;
    r.prompt_tokens = tokens.len;
    int max_tokens = n_predict;
    const int room = ds4_session_ctx(s) - ds4_session_pos(s);
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    int generated = 0;
    bool first = true;
    const double t3 = now_ms();
    while (generated < max_tokens) {
        if (!first) token = ds4_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &rng);
        first = false;
        if (ds4_token_is_stop_for_think_mode(e, token, tm)) break;
        if (spec) {
            int toks[17];
            const int ntok = ds4_session_eval_speculative(s, token, max_tokens - generated, ds4_token_eos(e),
                                                          0.0f, 0, 1.0f, 0.0f, &rng, toks, 17, err, sizeof(err));
            if (ntok < 0) { fprintf(stderr, "speculative decode failed: %s\n", err); exit(1); }
            bool stop = false;
            for (int j = 0; j < ntok; j++) {
                if (ds4_token_is_stop_for_think_mode(e, toks[j], tm)) { stop = true; break; }
                h = (h ^ (uint64_t)(uint32_t)toks[j]) * 1099511628211ull;
                if (++generated >= max_tokens) break;
            }
            if (stop) break;
        } else {
            h = (h ^ (uint64_t)(uint32_t)token) * 1099511628211ull;
            if (++generated >= max_tokens) break;
            if (ds4_session_eval(s, token, err, sizeof(err)) != 0) { fprintf(stderr, "decode failed: %s\n", err); exit(1); }
        }
    }
    const double t4 = now_ms();
    r.decode_tps = t4 > t3 ? 1e3 * generated / (t4 - t3) : 0.0;
    r.n = generated;
    r.hash = h;
    ds4_vision_embedding_free(&span.embedding);
    ds4_tokens_free(&tokens);
    ds4_session_free(s);
    return r;
}

int main(int argc, char **argv) {
    const char *model = NULL, *vision = NULL, *prompt = "Describe this image in detail.";
    const char *images[16];
    int n_images = 0, ctx = 8192, n_predict = 128, repeat = 3, encode_repeat = 8;
    bool mtp = false, encode_only = false;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(a, "--vision") && i + 1 < argc) vision = argv[++i];
        else if (!strcmp(a, "--image") && i + 1 < argc && n_images < 16) images[n_images++] = argv[++i];
        else if (!strcmp(a, "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(a, "--mtp")) mtp = true;
        else if (!strcmp(a, "--encode-only")) encode_only = true;
        else if (!strcmp(a, "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(a, "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(a, "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--encode-repeat") && i + 1 < argc) encode_repeat = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s -m MODEL --vision MMPROJ --image PATH[@MAX_TOKENS]... [--mtp] [--ctx N] "
                    "[-n N] [--repeat N] [--encode-repeat N] [--encode-only] [--prompt TEXT]\n", argv[0]);
            return 2;
        }
    }
    if (!model || !vision || !n_images || repeat < 1 || encode_repeat < 1 || encode_repeat > 64) {
        fprintf(stderr, "need -m, --vision and --image\n");
        return 2;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .vision_path = vision,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .context_size = ctx,
        .placement_ctx_hint = ctx,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .glm_mtp = mtp,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "engine open failed\n"); return 1; }
    const bool spec = mtp && ds4_engine_mtp_draft_tokens(e) > 1;
    int bad = 0;
    for (int i = 0; i < n_images; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s", images[i]);
        char *at = strrchr(path, '@');
        if (at) { *at = 0; setenv("DS4_QWEN4_IMAGE_MAX_TOKENS", at + 1, 1); }
        else unsetenv("DS4_QWEN4_IMAGE_MAX_TOKENS");
        const char *name = strrchr(path, '/');
        name = name ? name + 1 : path;
        size_t len = 0;
        uint8_t *img = slurp(path, &len);
        double ms[65];
        uint32_t tokens = 0;
        for (int r = 0; r <= encode_repeat; r++) {
            ds4_vision_embedding emb = {0};
            const double t0 = now_ms();
            encode(e, img, len, &emb);
            ms[r] = now_ms() - t0;
            tokens = emb.token_count;
            ds4_vision_embedding_free(&emb);
        }
        const double first = ms[0];
        double mn = 1e30, mx = 0;
        for (int r = 1; r <= encode_repeat; r++) { if (ms[r] < mn) mn = ms[r]; if (ms[r] > mx) mx = ms[r]; }
        printf("encode image=%s tokens=%u first_ms=%.2f warm_median_ms=%.2f min_ms=%.2f max_ms=%.2f repeats=%d\n",
               name, tokens, first, median(ms + 1, encode_repeat), mn, mx, encode_repeat);
        fflush(stdout);
        if (encode_only) { free(img); continue; }
        block_result b[33];
        const int n = repeat > 32 ? 32 : repeat;
        for (int r = 0; r <= n; r++) {
            b[r] = run_block(e, img, len, prompt, ctx, n_predict, spec);
            printf("block image=%s mode=%s %s encode_ms=%.2f prefill_ms=%.2f ttft_ms=%.2f decode_tps=%.2f "
                   "prompt_tokens=%d n=%d hash=%016llx\n", name, mtp ? "mtp" : "plain", r ? "run" : "warmup",
                   b[r].encode_ms, b[r].prefill_ms, b[r].ttft_ms, b[r].decode_tps, b[r].prompt_tokens, b[r].n,
                   (unsigned long long)b[r].hash);
            fflush(stdout);
        }
        double enc[32], pre[32], ttft[32], tps[32];
        bool identical = true;
        for (int r = 1; r <= n; r++) {
            enc[r - 1] = b[r].encode_ms; pre[r - 1] = b[r].prefill_ms;
            ttft[r - 1] = b[r].ttft_ms; tps[r - 1] = b[r].decode_tps;
            if (b[r].hash != b[0].hash || b[r].n != b[0].n) identical = false;
        }
        printf("E2E image=%s mode=%s tokens=%u prompt_tokens=%d encode_ms=%.2f prefill_ms=%.2f ttft_ms=%.2f "
               "decode_tps=%.2f n=%d hash=%016llx outputs_identical=%s\n", name, mtp ? "mtp" : "plain", tokens,
               b[0].prompt_tokens, median(enc, n), median(pre, n), median(ttft, n), median(tps, n), b[0].n,
               (unsigned long long)b[0].hash, identical ? "yes" : "NO");
        fflush(stdout);
        if (!identical) bad = 1;
        free(img);
    }
    ds4_engine_close(e);
    return bad ? 3 : 0;
}
