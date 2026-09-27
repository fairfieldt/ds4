# Qwen CUDA fused decode on RTX PRO 6000 Blackwell

Results for the Qwen3.8-Flash-Next CUDA decode changes of this series, on
one RTX PRO 6000 Blackwell Workstation Edition (188 SMs, 96 GiB, 450 W
limit), CUDA 13.4, driver 615.71, Ryzen 9 9950X host, with the
Qwen3.8-Flash-Next-Q4 GGUF (Q4_K routed gate/up, MXFP4 routed down, Q8_0
dense projections, F16 HC mixers, BF16 disk-resident n-gram table). No
weights were requantized.

## What was measured

The series was developed as one combined build and split into these
commits afterwards; reordering the commits does not change the code that
runs. The numbers below compare two executables:

- **Baseline**: the code of the previous part of this stack (the
  batching series) before it was split into pull requests. It differs from
  that series' tip in one dispatch threshold that matters here: its expert
  GEMM takes decode batches from 25 rows, where the tip keeps the
  per-token decode expert kernels through 32 rows.
- **Candidate**: the combined build. Its default path is the one these
  commits produce, except for the MTP verify-row attention: "Split each
  Qwen CUDA decode attention row like its one-token step" and the verify
  rows of "Split the grouped Qwen attention across keys for decode rows"
  were measured separately afterwards (see *MTP verify rows*). The combined build
  also contained code that is not part of this series: an opt-in CUDA-graph
  replay of the MoE block (off by default, 4.0% slower single-stream), a
  batched reduced draft head for session batches (within 1% at 8 sessions:
  483.0 against 485.0 t/s with the full head), a native FP4 expert
  prototype (slower and less accurate; never used for inference), a
  frequency-selected draft vocabulary experiment, and on/off switches for
  each change. Those were left out; the switches were only used for the
  paired component measurements below.

All throughput runs are greedy, and the generated text is byte-identical
between baseline and candidate on every fixed prompt.

## Single stream

Fixed prompts ([`prompts/`](prompts/): a code and a prose request), 256
output tokens, 4096-token context capacity, CLI generation rate excluding
model load. MTP uses the served configuration's draft prefix
(`DS4_QWEN4_MTP_DRAFT_ROWS=65536`) on both executables. Each comparison
ran baseline, candidate, candidate, baseline; the first pair starts with
the model's n-gram table evicted from the page cache (cold start), the
second reuses it (warm).

| Prompt / mode | Baseline warm t/s | Candidate warm t/s | Change | Baseline cold t/s | Candidate cold t/s | Change |
|---|---:|---:|---:|---:|---:|---:|
| Code, plain | 135.79 | 140.15 | +3.2% | 124.67 | 133.21 | +6.9% |
| Prose, plain | 135.91 | 140.47 | +3.4% | 126.27 | 131.34 | +4.0% |
| Code, MTP | 199.60 | 207.36 | +3.9% | 186.96 | 196.18 | +4.9% |
| Prose, MTP | 171.39 | 177.93 | +3.8% | 158.18 | 166.53 | +5.3% |

MTP first-draft acceptance is the same on both: 90.2% code, 61.9% prose
([`single.csv`](single.csv)).

At 4096 tokens of context, `ds4-bench` (one full prefill chunk, 256
generated tokens) gives 128.85 -> 133.60 t/s warm (+3.7%), prefill 3,095 ->
3,111 t/s ([`context.csv`](context.csv)). The session benchmark at the same
context (512-token prefill chunks) gives 129.78 -> 134.49 t/s plain and
141.45 -> 146.19 t/s with MTP; p95 step latency 7.737 -> 7.468 ms and 9.902
-> 9.586 ms ([`sessions.csv`](sessions.csv)).

## Sessions

`speed-bench/session_concurrency_bench`, warm, staggered slices of an
Italian corpus, context 256, 128 measured steps after 16 warmup steps, full
draft head on both executables. Wall throughput includes token selection.

| Mode | Sessions | Aggregate t/s | Wall t/s | p95 step ms |
|---|---:|---:|---:|---:|
| Plain | 1 | 135.6 -> 140.9 | 134.6 -> 139.9 | 7.41 -> 7.13 |
| Plain | 8 | 361.6 -> 430.8 | 354.7 -> 421.0 | 22.38 -> 18.98 |
| Plain | 16 | 414.0 -> 576.3 | 405.1 -> 558.9 | 38.75 -> 28.25 |
| MTP | 1 | 142.0 -> 146.4 | 141.3 -> 146.4 | 9.82 -> 9.53 |
| MTP | 8 | 344.4 -> 434.1 | 338.7 -> 434.1 | 37.83 -> 26.56 |
| MTP | 16 | 393.0 -> 576.0 | 385.5 -> 576.0 | 71.94 -> 42.87 |

The candidate's MTP commits 1.383 / 1.033 / 1.030 tokens per stream per
step at 1 / 8 / 16 sessions: on this corpus the batch scheduler often
chooses plain decode. Cold-start aggregate rates were 127.1 -> 131.3, 352.5
-> 413.7 and 405.8 -> 549.5 t/s plain, and 133.4 -> 137.5, 345.2 -> 419.9
and 390.1 -> 554.5 t/s with MTP. Warm prefill stayed within about 1%.

## Per-change measurements

Paired in-process ABBA runs of the session benchmark on the candidate
build, each with one change turned off in one arm. ABBA means both arms
run in one process in the order A, B, B, A, so that drift over the run
affects both arms equally. These continuations advance further than the
sweep above, so compare within a row only; the effects do not add up. In
the `ab-*` rows of [`sessions.csv`](sessions.csv), `control_tps` is the
change on and `candidate_tps` the change off.

| Commit | Sessions | With | Without | Change |
|---|---:|---:|---:|---:|
| Run the Qwen CUDA session-row APIs as batched kernels | 8 | 414.0 | 381.5 | +8.5% |
| | 16 | 554.9 | 493.7 | +12.4% |
| Run the Qwen shared expert densely for CUDA batches of 8+ rows (routed grouping off in both arms) | 8 | 415.3 | 412.5 | within 1% |
| | 16 | 522.2 | 509.8 | +2.4% |
| Reuse unpacked Qwen expert weights across batch rows on CUDA | 8 | 423.0 | 425.6 | within 1% |
| | 16 | 565.3 | 537.2 | +5.2% |
| Both expert changes together | 8 | 423.3 | 421.9 | within 1% |
| | 16 | 563.3 | 520.0 | +8.3% |

Single-stream pilots ([`component-pilots.csv`](component-pilots.csv)) ran
while the surrounding code was still being tuned, so they are indicative:
shared-input projections in one launch 135.6 -> 138.6 t/s (context 256);
one-token decode on the grouped attention 138.5 -> 140.6 (context 256) and
127.3 -> 128.6 (context 4096); the HC combine folded into the HC norm 124.4
-> 124.6 and the overlapped n-gram read 124.5 -> 124.5 (both within noise).

GPU verification of greedy MTP drafts, CLI with fixed prompts and identical
256-token outputs, two runs each ([`verifier.csv`](verifier.csv)): code
207.27 / 207.34 t/s against 202.36 / 202.42 with CPU verification (+2.4%),
prose 178.06 / 177.44 against 173.86 / 173.76 (+2.3%), same acceptance. An
earlier advancing-session A/B (`ab-single-gpu-verify`) favoured the CPU
variant, but its arms consumed different continuations with different
acceptance, so it does not isolate the verifier.

## MTP verify rows

Greedy MTP must commit exactly the tokens plain decode commits, so each
verify row must round like the one-token decode at its position. The first
commit gives every row of a multi-row decode its own key split, and the
grouped-attention commit sends each row to the kernel its one-token step
uses. Measured on the candidate build with and without that fix (CLI-style
greedy decode, 256 tokens, automatic MTP depth, separate processes in the
order without, with, with, without, two pairs):

| Prompt | Plain t/s | MTP t/s |
|---|---:|---:|
| Code | 140.64 -> 140.63 | 227.0 -> 228.3 (+0.6%) |
| Prose | 140.15 -> 140.38 | 175.4 -> 176.9 (+0.9%) |

Tokens and verify outcomes are the same in both builds (code: 105 cycles,
93 first and 57 second drafts accepted; prose: 156, 95, 3). These MTP rates
are higher than the single-stream table above because automatic depth
includes the depth-3 predictor; the two tables use different setups and
should not be compared with each other.

Without the fix, on an op-ed prompt, depth-2 MTP changed the greedy text:
the first argmax flip is at generated token 111, where plain decode has
token 74102 at 19.8210 and 1149 at 19.8026 and the verify row has them the
other way round. With it, every compared verify row is bit-identical to
plain decode (depth 2: 254 of 254 rows, depth 3: 254 of 254), plain decode
logits are unchanged (255 of 255 rows), and `tests/test_qwen4_mtp_exact.py`
reproduces the plain greedy continuation at depths 2 and 3 on ten benchmark
prompts, two prompts of about 3K tokens and its built-in prompts. An earlier
pilot that put the verify rows on the grouped kernel while they still shared
one key split measured 159.8 -> 158.0 t/s (session benchmark, one stream,
MTP, context 4096; [`component-pilots.csv`](component-pilots.csv)); the
per-row split is what makes the grouped verify rows both exact and faster.

## Kernel microbenchmarks

Median of seven samples after warmup, including activation packing and
launches, weights in VRAM.

Decode-batch projections ([`projections.csv`](projections.csv); "GEMV" is
the row GEMV in the same executable):

| Matrix | Rows | GEMV us | Selected us |
|---|---:|---:|---:|
| Q8 K=2560, M=512 | 4 | 2.878 | 2.875 (GEMV) |
| Q8 K=2560, M=512 | 8 | 5.252 | 4.146 |
| Q8 K=2560, M=512 | 16 | 10.482 | 4.716 |
| Q8 K=2560, M=512 | 25 | 17.466 | 5.405 |
| F16 K=10240, M=320 | 4 | 7.974 | 5.103 |
| F16 K=10240, M=320 | 8 | 14.959 | 4.704 |
| F16 K=10240, M=320 | 16 | 31.671 | 6.078 |
| F16 K=10240, M=320 | 25 | 52.745 | 8.709 |

Maximum error relative to the reference maximum is below 2.7e-7. Two
choices of the selection have no recorded tensor-core timing, because both
arms of those rows in `projections.csv` ran the GEMV: short F16 inputs
(K = 320) keep the GEMV, since a tensor-core trial there was not faster but
was not recorded, and four Q8 rows of a K >= 1024 input keep the GEMV
without the tensor-core kernel having been timed at that width.

One-token decode attention, scalar -> split grouped kernel
([`attention.csv`](attention.csv)): 128 keys 11.08 -> 8.23 us, 256 keys
11.29 -> 8.58, 2048 keys 16.65 -> 13.02, 4096 keys 26.00 -> 14.66; 32 and
64 keys stay scalar. Maximum relative error below 5.8e-7. A synthetic
32,768-key case measured 156.13 -> 38.53 us, but the sparse path normally
selects about 2,048 keys, so that is not an end-to-end claim.

Routed experts, synthetic 25 rows ([`experts.csv`](experts.csv)): with 8
routing patterns 411 us per-row, 336 us grouped, 639 us expert GEMM; with 25
distinct patterns 491 / 683 / 1,477 us, so grouping pays only with real
overlap. Measured overlap over 384 layer samples
([`expert-overlap.csv`](expert-overlap.csv)): 8 rows choose 80 (row, expert)
pairs from 52.4 distinct experts, 16 rows 160 from 78.6, peak reuse all 16.
Nsight Compute on a real eight-session layer (dependent launch off for
these counter captures only; [`memory-counters.csv`](memory-counters.csv)):
L2 traffic gate/up 169.4 -> 152.8 MB and down 113.6 -> 100.2 MB, DRAM reads
unchanged; a synthetic 16-row batch over eight patterns drops gate/up DRAM
reads 252.0 -> 147.5 MB. Some derived L2 hit rates above 100% in that file
are invalid and were ignored.

N-gram reader, 40 trials, wall median / p95 ms, threads per read -> pool
([`ngram-reader.csv`](ngram-reader.csv)): 256 warm rows 0.425 / 0.466 ->
0.118 / 0.125; 512 warm rows 0.438 / 0.475 -> 0.169 / 0.174; 16 rows
0.0049 either way; cold reads are disk-bound (256 rows 3.59 -> 3.41 ms, 512
rows 5.58 -> 5.60). Process CPU time can be higher with the pool (512 warm
rows 0.99 -> 1.42 ms).

## Launch and transfer profiles

Nsight Systems, 32 measured steps of the session benchmark at context 4096
after prefill and warmup ([`launch-profile.csv`](launch-profile.csv),
[`kernel-profile.csv`](kernel-profile.csv)):

| Mode / sessions | Kernels | Launch API ms | Device-to-host |
|---|---:|---:|---:|
| Plain / 1 | 34,144 -> 29,920 | 66.90 -> 58.78 | 31.785 MB -> 31.785 MB |
| MTP / 1 | 35,423 -> 32,671 | 69.39 -> 64.05 | 63.570 MB -> 384 B |
| Plain / 8 | 75,808 -> 54,848 | 382.51 -> 208.40 | 254.280 MB -> 254.280 MB |
| MTP / 8 | 79,121 -> 55,057 | 413.47 -> 218.19 | 270.172 MB -> 1,152 B |

Single-stream plain decode goes from 1,067 to 935 kernels per token; the
384 indexer tile-maxima launches of the plain capture disappear. Plain
decode still downloads each token's logits; greedy MTP reads back token
ids only. Under dependent launch, kernel durations overlap and include
waits, so their sum is not GPU time.

## Correctness and quality

- `tests/test_qwen4_cuda` passes at every commit of this series on the
  RTX PRO 6000, and every commit builds for CUDA and Metal.
- On the candidate build: Compute Sanitizer memcheck and initcheck of the
  session-row kernels report no errors; the model-backed session tests
  cover n-gram read failure and recovery, mixed plain and MTP batches,
  forced draft rejections, lazy logits consumers, snapshots, rewinds,
  context limits and shared-arena growth, and plain and MTP verification at
  1, 8 and 16 sessions for 24 cycles; single-stream depth-3 MTP matches
  plain greedy decode for 24 cycles with forced first- and second-draft
  rejections.

Official-continuation scoring (`gguf-tools/quality-testing/score_official`,
100 cases, 5,696 target tokens; [`quality.csv`](quality.csv),
[`quality/`](quality/)):

| Sessions | Cases / tokens | Baseline mean NLL | Candidate mean NLL | Reference top-1 |
|---|---:|---:|---:|---:|
| 1 | 100 / 5,696 | 0.290426 | 0.290510 | 5,129 -> 5,127 of 5,568 |
| 8 | 32 / 2,030 | 0.293533 | 0.293617 | 1,871 -> 1,869 of 2,030 |
| 16 | 32 / 2,030 | 0.293534 | 0.293639 | 1,871 -> 1,869 of 2,030 |

The logits are not byte-identical between the two executables. On the
eight cases with the largest change, restoring the baseline's expert-GEMM
threshold and projection arithmetic reproduces every baseline score to nine
decimals, and restoring the threshold alone removes most of the largest
difference (case 079: 0.007557 -> 0.000011)
([`quality-dispatch.csv`](quality-dispatch.csv)). That threshold is already
at 32 rows in the batching series, so most of this difference is not
introduced here; the remaining differences come from shape-dependent
tensor-core reductions.
