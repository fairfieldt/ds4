/* Model-backed CUDA oracle for the Qwen3.8 decode host path.
 *
 * Two sessions decode the same prompt in lockstep on the same kernels.  The
 * reference session copies its logits to the host with
 * ds4_session_copy_logits right after every prefill, eval, rewind and
 * speculative step, so each of its readers runs on the host row: CPU argmax,
 * host logprobs and host sampling.  The candidate leaves its frontier on the
 * device, where the greedy id comes from the GPU argmax and the row is
 * materialized only by the reader that needs it.  Every lazy-logits consumer
 * is compared bit for bit at varying points (argmax, the excluding and
 * EOS-ignoring forms, top logprobs, token logprob, copy_logits, sampling at a
 * positive temperature, payload save/load, rewind and replay), and with --mtp
 * the speculative path must commit the same tokens and frontiers.
 *
 * Not part of `make test`: it needs the model and a CUDA device.
 *   ./tests/test_qwen4_host_path [-m MODEL] [--mtp] [--steps N]
 */
#include "ds4.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(bool ok, const char *what, int step) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s step=%d\n", what, step);
        failures++;
    }
}

static int vocab;
static float *la, *lb;

/* The reference reads every frontier on the host: copying the logits
 * materializes its row, so the reference's argmax, logprobs and sampling
 * run on the host copy. */
static void settle(ds4_session *a, int step) {
    check(ds4_session_copy_logits(a, la, vocab) == vocab, "reference logits", step);
}

static void compare_logits(ds4_session *a, ds4_session *b, int step, const char *what) {
    const int na = ds4_session_copy_logits(a, la, vocab);
    const int nb = ds4_session_copy_logits(b, lb, vocab);
    check(na == vocab && nb == vocab && memcmp(la, lb, (size_t)vocab * sizeof(float)) == 0, what, step);
}

static void eval_both(ds4_session *a, ds4_session *b, int token, int step) {
    char err[160];
    const int ra = ds4_session_eval(a, token, err, sizeof(err));
    settle(a, step);
    const int rb = ds4_session_eval(b, token, err, sizeof(err));
    check(ra == 0 && rb == 0, "eval", step);
}

static ds4_session *reload(ds4_engine *e, ds4_session *src, int ctx, int step) {
    char err[160];
    ds4_session *c = NULL;
    FILE *fp = tmpfile();
    if (!fp || ds4_session_create(&c, e, ctx) != 0) {
        check(false, "payload setup", step);
        if (fp) fclose(fp);
        return c;
    }
    const bool saved = ds4_session_save_payload(src, fp, err, sizeof(err)) == 0;
    const long bytes = ftell(fp);
    rewind(fp);
    check(saved && bytes > 0 &&
          ds4_session_load_payload(c, fp, (uint64_t)bytes, err, sizeof(err)) == 0, "payload round trip", step);
    fclose(fp);
    return c;
}

static void consumers(ds4_session *a, ds4_session *b, int eos, int kind, int step) {
    switch (kind) {
    case 0: break;   /* the frontier is never materialized before the next eval */
    case 1: {
        const int a0 = ds4_session_argmax_excluding(a, eos);
        const int a1 = ds4_session_argmax_excluding(a, ds4_session_argmax(a));
        const int b0 = ds4_session_argmax_excluding(b, eos);
        const int b1 = ds4_session_argmax_excluding(b, ds4_session_argmax(b));
        check(a0 == b0 && a1 == b1, "argmax_excluding", step);
        break;
    }
    case 2: {
        ds4_token_score sa[5], sb[5], ta, tb;
        const int na = ds4_session_top_logprobs(a, sa, 5);
        const int ka = ds4_session_token_logprob(a, sa[0].id, &ta);
        const int nb = ds4_session_top_logprobs(b, sb, 5);
        const int kb = ds4_session_token_logprob(b, sb[0].id, &tb);
        check(na == nb && ka == kb && memcmp(sa, sb, sizeof(sa)) == 0 && memcmp(&ta, &tb, sizeof(ta)) == 0,
              "top_logprobs", step);
        break;
    }
    case 3:
        compare_logits(a, b, step, "copy_logits");
        break;
    case 4: {
        uint64_t ra = 0x1234u + (uint64_t)step, rb = ra;
        const int ta = ds4_session_sample(a, 0.8f, 40, 0.95f, 0.0f, &ra);
        const int tb = ds4_session_sample(b, 0.8f, 40, 0.95f, 0.0f, &rb);
        check(ta == tb && ra == rb, "sample temperature 0.8", step);
        break;
    }
    case 5: {
        const int ta = ds4_session_argmax_ignoring_eos(a, DS4_THINK_NONE);
        const int tb = ds4_session_argmax_ignoring_eos(b, DS4_THINK_NONE);
        check(ta == tb, "argmax_ignoring_eos", step);
        break;
    }
    }
}

static void run_plain(ds4_engine *e, const ds4_tokens *prompt, int ctx, int steps) {
    char err[160];
    ds4_session *a = NULL, *b = NULL;
    if (ds4_session_create(&a, e, ctx) != 0 || ds4_session_create(&b, e, ctx) != 0) {
        check(false, "session create", -1);
        return;
    }
    check(ds4_session_sync(a, prompt, err, sizeof(err)) == 0, "prefill reference", -1);
    settle(a, -1);
    check(ds4_session_sync(b, prompt, err, sizeof(err)) == 0, "prefill candidate", -1);
    const int eos = ds4_token_eos(e);
    for (int step = 0; step < steps && !failures; step++) {
        const int ta = ds4_session_argmax(a);
        const int tb = ds4_session_argmax(b);
        check(ta == tb, "argmax", step);
        consumers(a, b, eos, step % 6, step);
        if (step % 16 == 7) {
            /* the payload carries the frontier logits and the recurrent state */
            ds4_session *c = reload(e, b, ctx, step);
            if (c) {
                compare_logits(a, c, step, "payload logits");
                ds4_session_free(c);
            }
        }
        if (step == 21) {
            /* rewind two tokens and replay them from the kept prefix */
            const int pos = ds4_session_pos(a);
            const ds4_tokens *toks = ds4_session_tokens(a);
            const int t1 = toks->v[pos - 2], t2 = toks->v[pos - 1];
            ds4_session_rewind(a, pos - 2);
            settle(a, step);
            ds4_session_rewind(b, pos - 2);
            eval_both(a, b, t1, step);
            eval_both(a, b, t2, step);
            compare_logits(a, b, step, "rewind replay logits");
        }
        eval_both(a, b, ta, step);
    }
    compare_logits(a, b, steps, "final logits");
    ds4_session_free(a);
    ds4_session_free(b);
}

static void run_mtp(ds4_engine *e, const ds4_tokens *prompt, int ctx, int steps, float temperature) {
    char err[160];
    ds4_session *a = NULL, *b = NULL;
    if (ds4_session_create(&a, e, ctx) != 0 || ds4_session_create(&b, e, ctx) != 0) {
        check(false, "session create", -1);
        return;
    }
    check(ds4_session_sync(a, prompt, err, sizeof(err)) == 0, "prefill reference", -1);
    settle(a, -1);
    check(ds4_session_sync(b, prompt, err, sizeof(err)) == 0, "prefill candidate", -1);
    const int eos = ds4_token_eos(e);
    uint64_t ra = 99, rb = 99;
    int generated = 0, cycles = 0, multi = 0;
    while (generated < steps && !failures) {
        const int ta = ds4_session_sample(a, temperature, 40, 0.95f, 0.0f, &ra);
        const int tb = ds4_session_sample(b, temperature, 40, 0.95f, 0.0f, &rb);
        check(ta == tb, "mtp sample", cycles);
        int acc_a[17], acc_b[17];
        const int na = ds4_session_eval_speculative(a, ta, steps - generated, eos, temperature, 40, 0.95f, 0.0f,
                                                    &ra, acc_a, 17, err, sizeof(err));
        settle(a, cycles);
        const int nb = ds4_session_eval_speculative(b, tb, steps - generated, eos, temperature, 40, 0.95f, 0.0f,
                                                    &rb, acc_b, 17, err, sizeof(err));
        check(na > 0 && na == nb && memcmp(acc_a, acc_b, (size_t)na * sizeof(int)) == 0, "mtp accepted tokens",
              cycles);
        if (na <= 0) break;
        multi += na > 1;
        generated += na;
        if (cycles % 5 == 3) compare_logits(a, b, cycles, "mtp frontier logits");
        cycles++;
    }
    compare_logits(a, b, cycles, "mtp final logits");
    printf("mtp temperature=%.1f: %d tokens in %d cycles (%d multi-token)\n", temperature, generated, cycles, multi);
    ds4_session_free(a);
    ds4_session_free(b);
}

int main(int argc, char **argv) {
    const char *model = "/srv/models/ds4/Qwen3.8-Flash-Next-Q4.gguf";
    bool mtp = false;
    int steps = 48;
    const int ctx = 1024;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--mtp")) mtp = true;
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-m MODEL] [--mtp] [--steps N]\n", argv[0]);
            return 2;
        }
    }
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_CUDA,
        .context_size = ctx,
        .placement_ctx_hint = ctx,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .glm_mtp = mtp,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "engine open failed\n");
        return 1;
    }
    vocab = 248320;
    la = malloc((size_t)vocab * sizeof(float));
    lb = malloc((size_t)vocab * sizeof(float));
    static const char *texts[] = {
        "Write a C function that reverses a singly linked list, then explain it line by line.",
        "Describe the water cycle to a curious ten year old, with three vivid examples.",
    };
    enum { NP = sizeof(texts) / sizeof(texts[0]) };
    ds4_tokens tokens[NP] = {{0}};
    for (int p = 0; p < NP; p++)
        ds4_encode_chat_prompt(e, "You are a helpful assistant", texts[p], DS4_THINK_NONE, &tokens[p]);
    for (int p = 0; p < NP && !failures; p++) {
        run_plain(e, &tokens[p], ctx, steps);
        if (mtp && !failures) {
            run_mtp(e, &tokens[p], ctx, 4 * steps, 0.0f);
            if (!failures) run_mtp(e, &tokens[p], ctx, steps, 0.8f);
        }
    }
    for (int p = 0; p < NP; p++) ds4_tokens_free(&tokens[p]);
    free(la);
    free(lb);
    ds4_engine_close(e);
    if (failures) {
        fprintf(stderr, "qwen4 host path: %d failure(s)\n", failures);
        return 1;
    }
    printf("qwen4 host path tests passed (%s)\n", mtp ? "plain + mtp" : "plain");
    return 0;
}
