# Qwen CUDA decode toward the roofline on RTX PRO 6000 Blackwell

This series of 15 code commits, on top of the fused-decode series ([`../qwen-cuda-6000`](../qwen-cuda-6000/README.md)), takes single-stream Qwen3.8-Flash-Next-Q4 decode at the bench host's 450 W limit from 140.60 to 170.51 t/s plain on the code prompt (+21.3%) and from 226.91 to 255.26 t/s with MTP (+12.5%). Over 10 prompts the mean gain is +21.05% plain and +12.45% MTP. At 600 W, the card's default limit, the same comparison gives +22.52% plain and +16.17% MTP, because at 450 W the stack is held back by the power cap and at 600 W it is not.

## What was measured

Like the fused-decode series, this series was developed on the combined build that the fused-decode series was later split from, and every number here was taken there, at the bench host's 450 W limit unless stated. Two builds appear throughout:

- **base**: that combined build, without this series (the build called the candidate in [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md#what-was-measured)).
- **main**: the same combined build with this series applied (its 15 commits in the order listed under [What changed](#what-changed)).

The commits here are main's commits rebased onto the fused-decode series. The kernels and host paths they add are the same; what differs is the rest of the tree, which lost the combined build's opt-in experiments and measurement switches when the fused-decode series was split out. Where main's code checked one of those switches, the rebased commit keeps only the structural condition the switch guarded, which is the path the combined build took by default. One difference matters for MTP: the combined build did not yet split each MTP verify row's attention like its one-token step (the first commit of the fused-decode series), so the MTP figures below were taken without that fix (see [Pre-existing base bugs](#pre-existing-base-bugs)). The per-feature figures come from a third build, the measured series, described under [Per-feature numbers](#per-feature-numbers). No new measurements were taken for the rebase apart from the kernel tests at every commit ([Exactness](#exactness)).

Hardware and software are the same as [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md): RTX PRO 6000 Blackwell Workstation Edition (GB202, sm_120a, 188 SMs), driver 615.71.09, CUDA 13.4, and the unmodified Qwen3.8-Flash-Next-Q4 GGUF. No weights or quantization changed.

Greedy output is unchanged. All 10 short prompts and both ~3K-token prompts give the same 256-token hash as the base, plain and MTP. Every change in the stack is bit-identical by construction; on main one of them, the catch-up row trim, could still change MTP draft rounding (never committed tokens), which it no longer can on top of the fused-decode series (see [Exactness](#exactness)).

Two items measured in the same round are **not** in this series. Each was kept on its own branch with one commit on top of it (see [Side branches](#side-branches)); both were dropped on 2026-10-10 and their numbers are kept here:
- `qwen-cuda-mtp-depth-ev`: an expected-value MTP draft-depth policy. This series keeps the existing fixed depth rule.
- `qwen-cuda-q8-split-k`: an opt-in split-K geometry for the 2560×6144 Q8 projections. It changes summation order.

Single-stream results are the acceptance criterion. Concurrency 8/16 results are secondary; they should not regress, and they do not.

## End to end at 450 W

These are cross-binary ABBA runs of main against the base, with each build in its own process running the in-process harness described under [Method](#method). Each process runs a warm-up block and then 2 pairs, 256 greedy tokens each, at `--ctx 4096`, with the MTP draft head limited to the first 65,536 vocabulary rows (`DS4_QWEN4_MTP_DRAFT_ROWS=65536`, the served draft prefix, as in [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md#single-stream); the default is the full vocabulary). The code and prose prompts ran in 4 processes per build (base, main, main, base, twice). The held-out and ~3K-token prompts ran in 2 processes per build. Every figure is the mean of the per-process means; the largest per-process spread is 0.51 t/s. Raw data is in [`end-to-end.csv`](end-to-end.csv) (`power_limit_w` = 450).

| Prompt | Plain base → main t/s | Change | MTP base → main t/s | Change |
|---|---:|---:|---:|---:|
| code (tuning) | 140.60 → 170.51 | +21.28% | 226.91 → 255.26 | +12.49% |
| prose (tuning) | 140.21 → 170.14 | +21.34% | 175.40 → 198.78 | +13.33% |
| explain | 140.44 → 170.07 | +21.10% | 196.62 → 222.37 | +13.09% |
| json | 140.40 → 169.90 | +21.01% | 242.87 → 271.22 | +11.67% |
| math | 140.35 → 169.89 | +21.05% | 216.16 → 243.06 | +12.45% |
| oped | 140.08 → 169.52 | +21.02% | 177.02 → 199.63 | +12.77% |
| rust | 140.08 → 169.50 | +21.00% | 209.25 → 235.51 | +12.55% |
| sql | 140.10 → 169.42 | +20.92% | 215.54 → 241.09 | +11.85% |
| table | 140.03 → 169.31 | +20.91% | 220.23 → 246.58 | +11.96% |
| translate | 139.94 → 169.14 | +20.86% | 185.63 → 208.50 | +12.32% |
| **10-prompt mean** | | **+21.05%** | | **+12.45%** |
| longcode (~3K tokens, sparse attention) | 134.06 → 164.75 | +22.89% | 176.95 → 203.72 | +15.13% |
| longdoc (~3K tokens, sparse attention) | 133.95 → 164.41 | +22.74% | 183.12 → 210.48 | +14.94% |

- **Held-out prompts:** the 8 prompts below the tuning pair average +20.98% plain and +12.33% MTP.
- **Tokens:** every row gives the same tokens on both builds, in both modes.
- **Prompts:** code and prose are [`../qwen-cuda-6000/prompts/`](../qwen-cuda-6000/prompts/); the eight held-out prompts and the two long ones are in [`prompts/`](prompts/). Each long prompt is a task plus the first 9,000 bytes of `ds4_qwen4_cuda.cuh` or all of `docs/QWEN38_FLASH_NEXT.md` as they stood in the base, so decode runs at positions around 3K on the sparse-attention path.
- **Why MTP gains less than plain:**
  - The trunk-side wins shrink in the 2–3-row verify steps.
  - At 450 W the card holds its limit by lowering SM clock, and the faster stack hits the limit much more often in MTP runs: the power-cap bit is set in 41–93% of busy samples for this stack against 3–38% for the base. The median SM clock is about 2,390–2,460 MHz against 2,590–2,650 MHz for the base (see [Power and clocks](#power-and-clocks-at-450-w)). At 600 W the MTP gain rises to +16.2%.

### Session harness

This uses `session_concurrency_bench` with 1 stream, 256 generated tokens, 16 warm-up cycles and 512-token prefill chunks. Each build ran twice as separate processes, in the order base, main, main, base. Raw data is in [`sessions.csv`](sessions.csv).

| Mode | Base t/s | Main t/s | Change | Step p50, base → main |
|---|---:|---:|---:|---:|
| ctx 256, plain | 140.9 / 141.1 | 170.5 / 170.7 | +21.0% | 7.09 → 5.86 ms |
| ctx 256, `--spec` (depth ≤ 2) | 154.5 / 154.6 | 175.5 / 175.7 | +13.6% | 9.10 → 8.01 ms |
| ctx 4096, plain (sparse path) | 134.6 / 134.3 | 164.8 / 164.8 | +22.6% | 7.43 → 6.07 ms |
| ctx 4096, `--spec` | 149.6 / 149.6 | 172.0 / 172.0 | +15.0% | 9.50 → 8.27 ms |

Tokens per cycle are identical on both builds: 1.41 at ctx 256 and 1.42 at ctx 4096. The session bench caps a cycle at 2 tokens, so its `--spec` numbers use at most depth 2 and understate CLI MTP.

Batched sessions: ctx 256, 64 generated tokens, same process order.

| Streams | Base aggregate t/s | Main aggregate t/s | Change |
|---:|---:|---:|---:|
| 1 | 141.3 / 141.2 | 171.6 / 171.7 | +21.5% |
| 8 | 442.4 / 440.6 | 448.6 / 448.4 | +1.6% |
| 16 | 590.3 / 587.9 | 596.6 / 596.6 | +1.3% |

At 8 and 16 streams the row kernels run instead of the single-token path, so these kernel changes mostly do not apply there. The gain comes from the batched GPU argmax. At 450 W both builds are also power-capped here (the cap bit is set in 64% of base and 79% of main samples); at 600 W the batched gain is +5.1% at 8 streams and +5.9% at 16.

**Prefill** (measured on the measured series, see [Per-feature numbers](#per-feature-numbers)):
- The first 544-token prefill of each process is about 2% slower: 1,289–1,300 t/s on base against 1,251–1,271 t/s with the stack.
- Later prefills in the same process, the 8- and 16-stream prefills of the batched run, differ by at most 0.8%, and the 4,096-token prefill by 0.2%.
- This points to the one-time conversion of the router and gate BF16 copies on first use (`bf16_copy`, up to 146 MB for the router and GDN alpha/beta copies, with a stream sync per tensor), not to a per-token cost. This explanation is inferred, not measured separately.

## At 600 W

The bench host runs the card at 450 W; 600 W is its default and maximum limit. One job measured both builds at 600 W and restored 450 W afterwards. It ran the measured series (see [Per-feature numbers](#per-feature-numbers)) with its expected-value depth policy switched off, which selects the old depth rule. With that setting it runs the same kernels and host paths as main; what it adds is the measurement switches' host-side `getenv` checks, and split-K is off by default there too. Its 450 W blocks agree with this stack's 450 W runs to within 0.1% (code plain 170.52 vs 170.51, MTP 255.24 vs 255.26; prose plain 170.14 vs 170.14, MTP 198.78 vs 198.78).

### What the higher limit gives each build

A power-limit ABBA per build: code and prose, the in-process harness with 2 pairs per block, blocks at 450, 600, 600 and 450 W. Raw data is in [`power-limit-abba.csv`](power-limit-abba.csv).

| Build, mode | 450 W t/s (code / prose) | 600 W t/s (code / prose) | 600 vs 450 W |
|---|---:|---:|---:|
| base, plain | 140.84 / 140.52 | 140.82 / 140.50 | −0.01% / −0.02% |
| base, MTP | 228.07 / 175.98 | 231.85 / 177.53 | +1.66% / +0.88% |
| main, plain | 170.52 / 170.14 | 172.39 / 171.97 | +1.10% / +1.08% |
| main, MTP | 255.24 / 198.78 | 269.61 / 205.83 | +5.63% / +3.55% |

Each 450 W figure is the mean of blocks 1 and 4, and each 600 W figure the mean of blocks 2 and 3. The base's plain decode is not limited by power at 450 W; the stack's is, slightly, and its MTP decode clearly. The base's MTP blocks 1 and 4 differ by 1% (229.17 and 226.97 t/s on code) as the card warmed from 53 to 62 °C, so the base MTP difference is less certain than the others.

### Main vs base at 600 W

Cross-binary ABBA as above: 4 processes per build for code and prose, 2 for the rest. Raw data is in [`end-to-end.csv`](end-to-end.csv) (`power_limit_w` = 600).

| Prompt | Plain base → main t/s | Change | MTP base → main t/s | Change |
|---|---:|---:|---:|---:|
| code | 140.72 → 172.36 | +22.49% | 231.80 → 269.64 | +16.32% |
| prose | 140.38 → 171.94 | +22.48% | 177.34 → 205.78 | +16.04% |
| explain | 140.52 → 172.17 | +22.52% | 199.56 → 231.82 | +16.16% |
| json | 140.52 → 172.15 | +22.51% | 251.31 → 292.39 | +16.35% |
| math | 140.55 → 172.26 | +22.56% | 221.27 → 256.89 | +16.10% |
| oped | 140.18 → 171.85 | +22.59% | 179.87 → 208.62 | +15.98% |
| rust | 140.23 → 171.91 | +22.59% | 213.53 → 247.97 | +16.13% |
| sql | 140.36 → 171.93 | +22.49% | 222.39 → 258.47 | +16.22% |
| table | 140.31 → 171.86 | +22.48% | 226.56 → 263.14 | +16.14% |
| translate | 140.30 → 171.88 | +22.51% | 190.00 → 220.83 | +16.23% |
| **10-prompt mean** | | **+22.52%** | | **+16.17%** |
| longcode | 134.05 → 166.55 | +24.25% | 178.87 → 211.68 | +18.34% |
| longdoc | 133.98 → 166.44 | +24.22% | 185.51 → 219.35 | +18.24% |

Tokens are identical across builds on every row, with the same hashes as at 450 W. The largest per-process spread is 0.35 t/s.

Session harness at 600 W (one process per build for the batched run, two for the others):

| Mode | Base t/s | Main t/s | Change |
|---|---:|---:|---:|
| ctx 256, plain | 140.9 / 140.9 | 172.1 / 172.1 | +22.1% |
| ctx 256, `--spec` | 155.5 / 155.5 | 180.3 / 180.3 | +16.0% |
| ctx 4096, plain | 134.5 / 134.4 | 165.8 / 165.8 | +23.3% |
| ctx 4096, `--spec` | 150.5 / 150.4 | 176.9 / 177.0 | +17.6% |
| batched, 1 stream | 141.2 | 172.4 | +22.1% |
| batched, 8 streams | 472.0 | 495.9 | +5.1% |
| batched, 16 streams | 636.0 | 673.8 | +5.9% |

The step p50 at ctx 256 plain is 7.09 → 5.81 ms.

### Clocks, power and temperature at 600 W

nvidia-smi sampled every 200 ms; busy samples only (utilisation ≥ 80%). Raw data is in [`power-clocks.csv`](power-clocks.csv).

| Run (600 W) | Build | Power median (W) | Power max, instantaneous (W) | SM clock median (MHz) | Max temp (°C) |
|---|---|---:|---:|---:|---:|
| code/prose, plain | base | 467.5 | 547.5 | 2797 | 66 |
| code/prose, plain | main | 532.3 | 548.8 | 2782 | 67 |
| code/prose, MTP | base | 502.3 | 555.4 | 2752 | 66 |
| code/prose, MTP | main | 548.1 | 571.2 | 2737 | 66 |
| held-out, plain | base | 493.0 | 601.6 | 2790 | 79 |
| held-out, plain | main | 564.4 | 603.3 | 2775 | 80 |
| held-out, MTP | base | 533.8 | 602.0 | 2745 | 79 |
| held-out, MTP | main | 575.5 | 603.6 | 2722 | 80 |
| ~3K prompts, MTP | base | 519.1 | 582.0 | 2752 | 73 |
| ~3K prompts, MTP | main | 549.6 | 582.0 | 2730 | 74 |

- **The stack is not power-limited at 600 W.** Its median draw is 532–576 W, below the limit, and the software power-cap bit is clear in every busy sample of both builds (apart from 0.3% of the base's short plain samples). No thermal-slowdown bit was set. At 450 W the cap bit was set in 41–93% of the stack's MTP samples.
- **Both builds run at the same SM clock at 600 W**, 2,720–2,800 MHz, within 25 MHz of each other in every cross-binary and power-ABBA group. At 450 W the stack ran 150–200 MHz below the base. The highest clock in any busy sample of the job is 2,820 MHz; nvidia-smi reports a 3,090 MHz maximum, which the card does not reach under this load.
- **The stack draws more power than the base** because it does the same work in less time: 532–576 W median against 467–534 W. Instantaneous peaks touch 603 W in the held-out runs of both builds.
- **Temperature** peaks at 79–80 °C in the held-out runs, the longest stretch of back-to-back decode in the job, against 73 °C at 450 W. Memory clock stays at 13,365 MHz in every busy sample at both limits.
- **What this says about the bottleneck:** giving the stack its full clock back (2,636 → 2,788 MHz, +6%, in the plain power-ABBA blocks) buys only +1.1% plain, so little of the plain token time scales with SM clock. MTP gains +3.6–5.6% from 2,464 → 2,737 MHz (+11%); its 2–3-row steps do more arithmetic per byte.

## What changed

Each feature below is one commit. During development each was measured against the previous path through a temporary run-time switch; the switches are not part of the commits. Each commit message records its standalone measurements, and [`features.csv`](features.csv) summarizes them.

| Commit | Area | Change |
|---|---|---|
| `945c25b` | host | GPU argmax for greedy decode. The logits stay on the device and are copied to the host only when a reader needs them (sampling, logprobs, copy, save). |
| `eb1bb46` | host | N-gram rows are staged through mapped pinned memory by a kernel, instead of by two `cudaDeviceSynchronize` calls inside the token. |
| `00ea158` | host | Batched sessions pick their tokens with a device argmax instead of downloading each session's logits. |
| `9739c9c` | router | The router loads its shared-gate row before the PDL dependency wait (`router_pre`) and uses a warp-shuffle max. |
| `8781645` | router | The router and GDN gate projections read exact BF16 copies of their F32 weights. Every element has zero low 16 bits; a tensor that fails the check keeps F32. |
| `a906488` | HC | The HC down/up, router, gate and `hc_norm` weights are loaded before the PDL wait. The F16 pair uses TMA bulk copies; the others use volatile register preloads. Decode rows only (T ≤ 3). |
| `1e89c0f` | HC | `hc_norm` triggers its dependents after the norm reduction, so HC down launches earlier (T = 1). |
| `55c641f` | GDN | GDN alpha/beta are folded into the qkv+gate launch (`gdn_proj`), with the activation in its epilogue. |
| `f3fc2e6` | GDN | The causal conv and SiLU move into the qkv epilogue (conv width 4). |
| `af2ac5b` | MoE | The MXFP4 down kernel issues one load batch per warp. The shared expert runs as two rows dispatched first, and the x loads are hoisted. |
| `cd004b0` | MoE | The shared expert is prefetched into L2 from the gate/up blocks while they wait on the router. |
| `9509296` | attention | Block-parallel `idx_select` radix scans for the sparse indexer. The serial kernel is removed. |
| `21ed16a` | MTP | The 3-row verify HC gate/mix runs as one launch. The 2+1 split remains on Metal. |
| `4ef0df9` | MTP | Catch-up rows are staged in one launch and projected in one tensor-core dispatch (CUDA; Metal keeps the per-row path). |
| `6b42d29` | MTP | Only the last catch-up row runs the full layer. Earlier rows stop after their K/V and indexer-key appends. |

The previous paths are still used wherever the new ones do not apply: tensor-parallel runs, Metal, forward passes wider than the new kernels handle, other weight types and shapes, and (for the BF16 gate copies) weights that are not exact BF16 and graph capture.

## Per-feature numbers

All figures in this section were measured on the development series ("the measured series"): the same commits on the base, each carrying a temporary switch that turned its feature off. The switches were removed before main was cut, without changing any kernel. The MTP figures of that series used the expected-value depth policy that now lives on a side branch.

"Standalone" means each feature measured on its own development branch against the base. "Combined" means a leave-one-out with all features merged: control = everything on, candidate = that one feature off. The gain is on/off − 1, given as code / prose.

All runs used the in-process ABBA harness ([Method](#method)) with 3 pairs of 256 tokens. The standard deviation of the per-quad ABBA deltas was at most 0.08%. The combined figures come from the measured series' merged build before the last integration round, which also had the GDN state preload (dropped below) and split-K turned on. Raw data: [`features.csv`](features.csv), [`leave-one-out-integrated.csv`](leave-one-out-integrated.csv), [`in-process-ab.csv`](in-process-ab.csv).

| Feature | Plain standalone | Plain combined | MTP standalone | MTP combined |
|---|---:|---:|---:|---:|
| GPU argmax | +1.59 / +1.61 | +1.66 / +1.63 | −0.02 / −0.01 | +0.02 / +0.01 |
| N-gram staging without syncs | +0.48 / +0.48 | +0.47 / +0.46 | +0.43 / +0.47 | +0.41 / +0.41 |
| Router pre-wait load | +3.24 / +3.22 | +4.33 / +4.28 | +1.77 / +2.25 | +1.80 / +1.65 |
| BF16 gate copies | +1.26 / +1.27 | +1.26 / +1.24 | +0.72 / +0.94 | +0.72 / +0.70 |
| HC/router/gate pre-wait loads | +5.83 / +5.80 | +6.59 / +6.50 | +3.48 / +4.12 | +2.12 / +1.92 |
| hc_norm late trigger | +0.69 / +0.68 | +0.93 / +0.94 | 0.00 / 0.00 | +0.04 / −0.01 |
| GDN alpha/beta fold | +2.54 / +2.52 | +2.38 / +2.30 | +1.82 / +2.03 | +1.51 / +1.40 |
| GDN conv fold | +1.44 / +1.44 | +1.71 / +1.68 | +0.57 / +0.14 | +0.66 / +0.69 |
| MXFP4 down latency | +1.35 / +1.31 | +1.66 / +1.65 | +0.83 / +0.82 | +0.82 / +0.79 |
| Shared-expert prefetch | +0.49 / +0.47 | +1.06 / +1.04 | +0.23 / +0.32 | +0.38 / +0.34 |
| idx_select scan (longcode / longdoc) | +1.94 / +1.84 | +2.20 / +2.18 | +1.81 / +1.74 | +2.21 / +2.12 |
| 3-row verify mix | – | – | +1.36 / +1.53 | +0.38 / +0.46 |
| Catch-up staging in one launch | – | – | +0.65 / +0.48 | +0.67 / +0.50 |
| Catch-up row trim | – | – | +0.90 / +0.49 | +0.97 / +0.52 |
| *Dropped:* GDN state pre-wait load | +0.46 / +0.43, +0.33 / +0.35 | +0.32 / +0.33 | +0.15 / +0.15 | +0.12 / +0.13 |

- **Batched sessions:** the batched GPU argmax gives +1.1% decode at 8 streams and +0.9% at 16. Including token selection it gives +3.2% and +3.8%.
- **Sparse path, session harness at ctx 4096:** the idx_select scan gives +3.8% plain and +2.1% `--spec`.
- **Interactions:** the individual plain gains in the combined column add up to about 23.5%; the end-to-end gain is 21%. The shared-expert prefetch doubles in combination, because the router preload opens the window it prefetches into. The 3-row verify mix shrinks from 1.4% standalone to about 0.4% in combination; it is kept because it targets MTP only and was consistent in every block.
- **Switches restored the base:** with every switch set in one process, the measured series gave the base hashes and base speed within 0.1% (cross-binary: plain 140.46 vs 140.43 and 140.08 vs 140.08 t/s; MTP 226.36 vs 226.32 and 175.21 vs 175.06).

### Dropped during development

- **GDN state loaded before the PDL wait** (a commit of an earlier integration of this series).
  - In combination it measured +0.32 / +0.33% plain, below the ~0.4% bar the other items were held to, and +0.12 / +0.13% MTP, within noise (t < 1.1). nsys puts its saving at about 18–31 µs per token.
  - It was also the only change that loads mutable state (not weights) before the wait. That is legal only because the scan runs directly after `gdn_prep`, and applying it to the generic scan had already broken chunked prefill once. The small gain does not pay for that constraint.

Items dropped earlier, on their own development branches:
- mapped staging of R/pos and the MTP rows: +0.16% plain
- router top-k fused into the last GEMV block: −0.21% plain, −1.5% MTP
- q/k norm folded into the scan: −0.15% with the preload
- `ssm_out` L2 prefetch from the scan: −0.8 to −2.2%
- `attn_output` L2 prefetch: −0.07%
- `#pragma unroll 12` in the one-warp Q8 GEMV: loses to split-K at every T
- MXFP4 down with the expert id loaded before the wait: −0.5% MTP
- a 4-row-per-warp down kernel: −0.35%

## Per-kernel cost (nsys)

These captures were taken on the measured series with Nsight Systems: the session harness with 1 stream, PDL on, and 32 captured steps, of which 29 are steady. The plain captures take the same code paths as this stack. Under PDL the per-kernel cost is the incremental cost, end_i − max(end_<i); kernel durations overlap and are not costs. Raw data: [`kernel-classes.csv`](kernel-classes.csv).

| µs per token | Base | Stack | Change |
|---|---:|---:|---:|
| ctx 256, plain | 7,172.9 (899 kernels) | 5,833.4 (794 kernels) | −1,339.5 (−18.7%) |
| ctx 4096, plain | 7,505.8 (935) | 6,053.9 (830) | −1,451.9 (−19.3%) |
| ctx 256, `--spec`, per cycle (depth ≤ 2) | 9,120.0 (983) | 7,979.5 (875) | −1,140.5 (−12.5%) |

The `--spec` capture used the expected-value policy, which never drafts a chain step when a cycle is capped at 2 tokens; the old rule still drafts one. The session `--spec` rates of the two builds agree (175.4–175.5 t/s measured series, 175.5–175.7 this stack).

By class, at ctx 256 plain (µs per token, µs per launch in parentheses):

| Class | Base | Stack | Change | Main cause |
|---|---:|---:|---:|---|
| HC mix (up) | 594.7 (6.13) | 294.5 (3.04) | −300 | pre-wait TMA copies |
| router top-k | 378.4 (7.88) | 152.6 (3.18) | −226 | router pre-wait load |
| GDN conv/prep/scan/out | 373.2 (144 launches) | 206.1 (108) | −167 | conv fold |
| GDN alpha/beta GEMV | 147.0 (72 launches) | – | −147 | folded into `gdn_proj` |
| router GEMV | 213.0 (4.44) | 70.5 (1.47) | −143 | BF16 copy and pre-wait load |
| HC down F16 320×10240 | 539.2 (5.56) | 422.0 (4.35) | −117 | pre-wait TMA copies, late trigger |
| MoE down | 563.8 (11.75) | 461.3 (9.61) | −103 | latency rework |
| HC norm | 714.1 (7.36) | 621.0 (6.40) | −93 | gamma/inject preload, host bubble gone |
| MoE gate/up | 798.8 (16.64) | 737.3 (15.36) | −62 | shared-expert prefetch |
| PLE / n-gram projection | 50.4 | 20.1 | −30 | no device sync before it |
| GDN qkv+gate (`gdn_proj`) | 1,125.7 (23.45) | 1,151.9 (24.00) | +26 | now also computes alpha/beta and the conv |
| attention prep/core/merge | 155.5 | 169.0 | +14 | lower SM clock (power cap) |
| Q8 2560×6144 GEMV | 672.4 (13.72) | 674.7 (13.77) | +2 | unchanged (split-K, on its side branch: 12.47 µs per launch) |
| output head (675 MB Q8) | 413.0 | 412.8 | 0 | already at bandwidth |
| *of which launch gaps* | 220.6 | 41.7 | −179 | GPU argmax, no intra-token syncs |

At ctx 4096 the indexer falls from 261.1 to 133.9 µs per token (6.64 → 3.41 µs per launch). The spec capture shows the same pattern. The one class that rises there is HC norm, from 467.6 to 564.3 µs, because its pre-wait loads share DRAM with HC down's copies. Even so, the three HC classes together fall from 1,782 to 1,442 µs.

## Exactness

Every change in this stack is bit-identical by construction: it moves loads, fuses launches with the same per-row arithmetic, or changes host scheduling. The one exception on main was the catch-up row trim (`6b42d29`): there the draft's last row moved from the 2-row attention kernel to the one-token kernel, which did not yet round alike, so a draft could change (committed tokens cannot, and the acceptance sequences on code and prose were identical in both arms). On top of the fused-decode series, whose first commit makes each row of a multi-row decode attention compute what its one-token step computes, the trim is bit-identical too. Evidence for main, unless marked otherwise:

- **Greedy hashes:** 256 tokens per prompt from the in-process harness. All 10 short prompts and both ~3K-token prompts match the base in plain and MTP mode (24 of 24), in the dedicated exactness run and again in every block of the 450 W cross-binary runs. The 600 W runs of the equivalent build match too. longdoc is `a8aa66ea20369b32`.
- **Kernel tests:** `tests/test_qwen4_cuda` passed at every commit of main and at both side-branch heads, and passes again at every commit of this series as rebased onto the fused-decode series and at both rebased side branches (one run over per-commit builds; see [`validation.csv`](validation.csv)). It includes byte-exact tests for every new kernel: `test_presync_load`, `test_router_paths`, `test_gdn_front_exact`, the MXFP4 down comparison, `test_idx_select_exact_all`, `test_hc_mix3_split`, `test_mtp_stage_rows` and the batched catch-up projection, `test_host_argmax` and `test_host_staging`. Their references are:
  - for the router, dense GEMV, HC norm/mix/combine and MXFP4 down: tests-only `*_ref_tensor` entry points in `ds4_qwen4_cuda.cuh` that run the previous kernels and that the engine never calls;
  - for the GDN front: the separate kernels, with each fold turned off by shape (F16 alpha/beta weights, conv width 3);
  - for the shared-expert prefetch: the non-prefetching `moe_mid` entry point;
  - for idx_select: a CPU model of the exact output layout, checked slot by slot, since the serial kernel is gone.
- **Host path oracle:** `tests/test_qwen4_host_path` (model-backed) passed in plain, batch and `--mtp` modes at every commit of main and at both side-branch heads; it was not rerun after the rebase. It runs a reference session in lockstep that copies its logits to the host after every step, so its readers run on the host row, and compares every lazy-logits reader of the candidate bit for bit, including at temperatures 0 and 0.8.
- **Sanitizers:** compute-sanitizer memcheck on the full kernel test: 0 errors. On the measured series, racecheck on the ROWS, GDN_FRONT and PRESYNC subsets (and on its merged build, the ROUTER, IDX_SELECT_ONLY and DECODE_FUSIONS subsets): 0 hazards.
- **Session `--verify`** (ctx 256, gen 32, warm-up 4): plain batched equals sequential at 1/4/8/16 streams. `--spec` equals plain greedy at 1 and 16 streams and fails at 8 streams exactly as the base does (known bug 1 below). On the measured series, plain batched also equals sequential at ctx 4096 (1/4/8 streams).
- **Pre-wait loads** (measured series):
  - The SASS audit shows that only immutable weights are loaded before `griddepcontrol.wait` (ACQBULK), apart from the intended pinned-host n-gram load.
  - No kernel stores before the wait, and every trigger comes after it.
  - `DS4_CUDA_NO_PDL=1` changes no token.
  - `mtp_stage_rows` has no `__restrict__` on its two device-written inputs, so read-only (`.nc`) loads of them cannot be scheduled above the wait.
- **MTP verify rows** (this series as rebased, not main or the base): on top of the first commit of the fused-decode series, which gives every verify row its own attention key split (`test_attn_decode_rows`), rows T ≤ 3 round exactly like single-token decode, because every kernel this series adds or changes also computes each row of a T ≤ 3 step on its own; its byte-exact tests run T = 1..3. Main and the base predate that commit, and their 2- and 3-row verify rows did not always round like single-token decode (bug 2 below).

## Roofline context

At ctx 256 the base moved 6,936 MB per token: 6,702 MB of weights, 226.5 MB of GDN state read plus write, and 6.3 MB of KV. The achievable decode roofline is therefore:
- **236.2 t/s** at 1,638 GB/s, the measured streaming read on this card;
- **246.7 t/s** at 1,711 GB/s, the peak at the 13,365 MHz memory clock the card actually holds under load (never 14,001 MHz).

The BF16 router copies remove 126 MB per token, which moves the roofline to 240.5–251.2 t/s. Compute is irrelevant: at 2 FLOP/B the arithmetic intensity is 35× below the ridge.

This stack's plain decode runs at 170.5 t/s at 450 W (5.86 ms per step) and 172.1 t/s at 600 W, which is **71–72% of the 240.5 t/s achievable figure**; the base ran at 60% (140.9 of 236.2) at either limit. About 4.2 ms of the 5.83 ms token is byte-roofline time.

**At or near bandwidth** (ctx 256):
- the output head: 412.8 µs for 675 MB, 1.64 TB/s
- the GDN qkv+gate projection
- HC down: 6.55 MB in 4.35 µs, about 92%
- the attention q/k/v projections
- the router GEMV

**Still latency-bound**, largest first (nsys, ctx 256 plain, 450 W):

| Item | µs per token | Note |
|---|---:|---|
| Q8 2560×6144 GEMV, one warp per row | 675 (13.77 × 49) | 16.7 MB per call is 74% of stream bandwidth. Split-K brings it to about 12.5 µs but changes summation order (side branch). |
| hc_norm | 621 (6.40 × 97) | Moves little data per call (one 4×2560 residual row); the largest single kernel class left. Deferring its RMS scaling into HC down was not attempted because it conflicts with the pre-wait loads. |
| MoE down | 461 | Against a byte roofline of about 306 µs (65%). |
| GDN prep/scan/out | 206 | Against 138 µs of state traffic. |
| attention prep/core/merge | 169 | 242 µs at ctx 4096. |
| router top-k | 153 | |
| MoE reduce | 130 | |
| indexer (ctx 4096 only) | 134 | |
| launch gaps and host bubble | 42 | |

The 600 W run suggests that more SM clock does little for them: +6% SM clock bought +1.1% plain. Most of their time is DRAM latency and waiting along the dependency chain between launches, so the next wins are more likely to come from removing launches and serial steps than from clock or bandwidth.

**MTP:** the third verify row costs about 20% of a 2-row cycle (+1.78 ms over 9.07 ms, from a fit on the base build). The expert-union and 3-row kernels of the T = 3 verify are the next MTP target.

## Power and clocks at 450 W

nvidia-smi sampled every 200 ms during the 450 W cross-binary job; busy samples only (utilisation ≥ 80%). Raw data: [`power-clocks.csv`](power-clocks.csv) (`power_limit_w` = 450).

| Run | Build | Power median (W) | SM clock median (MHz) | SW power-cap bit set |
|---|---|---:|---:|---:|
| short prompts, plain | base | 450.2 | 2790 | 0% |
| short prompts, plain | main | 450.0 | 2640 | 1.6% |
| held-out prompts, plain | base | 450.0 | 2752 | 1.0% |
| held-out prompts, plain | main | 450.0 | 2595 | 1.2% |
| short prompts, MTP | base | 449.9 | 2647 | 2.9% |
| short prompts, MTP | main | 450.0 | 2460 | 40.6% |
| held-out prompts, MTP | base | 449.9 | 2587 | 7.9% |
| held-out prompts, MTP | main | 450.0 | 2392 | 86.0% |
| ~3K prompts, MTP | base | 451.1 | 2632 | 38.1% |
| ~3K prompts, MTP | main | 450.4 | 2460 | 93.3% |
| batched 1/8/16 streams | base | 450.1 | 2512 | 63.8% |
| batched 1/8/16 streams | main | 450.0 | 2407 | 78.7% |

- **Both builds decode at the 450 W board limit.** The governor gives up SM clock, never memory clock: memory stays at 13,365 MHz in every busy sample. No thermal slowdown was recorded; the maximum was 73 °C.
- **Same power, lower SM clock:** at the same power this stack runs about 150 MHz lower in plain decode (−6%) and about 190 MHz lower in MTP (−7%).
- **Consequence:** at 450 W, further bandwidth gains are partly traded against SM clock, and the MTP and batched numbers in this document are limited by power. Work that removes instructions or launches, rather than moving bytes faster, is worth more at the limit.

## Side branches

Both were measured on top of main with the in-process harness, 3 pairs, at 450 W. Each branch keeps its own switch because it is still under evaluation. Raw data: [`side-branches.csv`](side-branches.csv).

### Expected-value MTP depth policy (`qwen-cuda-mtp-depth-ev`)

The existing rule engages the chained second draft after a perfect 8-cycle window of first-draft accepts and leaves after two consecutive second-draft rejects; once out, a session can stay latched at depth 2. The branch drafts the chain when it pays in expectation: with p1 and p2 the running first- and second-draft acceptance rates, a 3-row cycle gains p1·p2 − k·(1 + p1) over a 2-row cycle. It also never drafts a chain step when a cycle is capped at 2 tokens. The cost constant k = 0.25 was fitted to cycle times of the base build. `DS4_QWEN4_NO_DEPTH_EV=1` restores the old rule.

Gain = EV / old rule − 1, MTP, on top of this stack. Outputs were identical in every block.

| code | prose | explain | json | math | oped | rust | sql | table | translate | Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +0.45 | −3.20 | +3.64 | 0.00 | +6.41 | −3.37 | +5.84 | +1.53 | +6.23 | −2.77 | +1.48 |

**Why it is not in this stack:** it costs 2.8–3.4% on the three prose-like prompts (prose, oped, translate). In the faster stack, k fitted to the base's cycle times over-engages depth 3 on low-acceptance text. Raising k does not fix it: on the measured series, k = 0.35 gave a mean of −0.8% and k = 0.5 −3.1%, with code, json and table losing 4–10%. The rule needs rework (for example, cycle costs measured at run time) before it is revisited; the branch was dropped. Its lead has also shrunk: on its own on the base it gained +7.4% on explain, +8.8% on math and +8.9% on rust, against +3.6%, +6.4% and +5.8% here.

### Q8 split-K (`qwen-cuda-q8-split-k`)

The 2560-row, 6144-wide Q8 projections (36 GDN `ssm_out`, 12 attention outputs, and the MTP layer's attention output) run one warp per row. The branch sends them through the existing split-K kernel (eight warps per row) on GPUs with at least 128 SMs, when `DS4_QWEN4_Q8_SPLIT_K=1` is set. A Q8 GEMV microbenchmark on the combined build (not in this repository) puts it at 11.48 against 12.86 µs per call at T = 1, and 14.61 against 22.44 at T = 3.

Gain = on / off − 1, on top of this stack:

| Prompt | Plain off → on t/s | Plain | MTP off → on t/s | MTP | Greedy output unchanged |
|---|---:|---:|---:|---:|---|
| code | 169.71 → 171.68 | +1.16% | 252.38 → 258.90 | +2.59% | yes |
| prose | 169.26 → 171.22 | +1.15% | 196.70 → 203.17 | +3.29% | yes |
| longcode | 164.17 → 165.98 | +1.10% | 202.18 → 208.53 | +3.14% | yes |
| longdoc | 163.88 → 165.14 | +0.77% | 208.52 → 215.86 | +3.52% | **no** |

One split-K block of the longdoc plain run was slow (range 162.68–165.70 t/s), so +0.77% is the least reliable figure. The MTP runs were power-capped in 86–97% of samples, so at 600 W the MTP gain is likely larger; that was not measured.

**Why it is not in this stack:** it changes summation order and fails the exactness gate used for default decode changes (greedy output unchanged on the test prompts, logprobs moving by at most about 1e-3):
- **Greedy flip:** on longdoc the 256-token hash changes from `a8aa66ea` to `daf80c24` in all 6 blocks, plain and MTP. The first differing token is 109, where the top two logits are 0.0146 apart (18.722 for "QA", 18.708 for "ated"); split-K reverses their order.
- **Logprob movement:** over the 109 shared tokens before that point, the top-1 logprob moves by up to 9.2e-3 through the session API. The CLI measurement in the commit message gave up to 2.2e-3 (median 7.6e-7); the two tools and token ranges differ and the figures have not been reconciled.
- **Verify harness:** it moves which concurrencies hit the near-tie at stream 0, token 33 in `--spec --verify` (known bug 1): off, 2 and 8 streams fail; on, 3, 4 and 8 fail.
- **Quality:** NLL is neutral. On 100 cases of qwen38-flash-alibaba-100 at ctx 4096 (measured series), NLL is 0.290510 off and 0.290498 on, with API top-1 5127/5568 and first_match 83 both ways.

Enabling it by default is a one-line change in `split_matvec_shape` if the gate is relaxed for it; otherwise the branch should be dropped.

**Re-measured on 2026-10-10 and dropped.** The staged-projection series ([`../qwen-cuda-6000-staging`](../qwen-cuda-6000-staging/README.md)) now loads these projections' weights before their dependency wait, which takes most of the latency split-K removed. On main after that series, in-process ABBA at 600 W (the bench host's limit from that day), 3 pairs, split-K on against off: plain +0.12%, +0.09%, −0.33% and −0.41%, MTP +0.35%, +0.77%, −0.03% and +0.66% on code, prose, longcode and longdoc, with longdoc's output changed. Not worth relaxing the exactness gate for.

## Pre-existing base bugs

These were found on the base and are unchanged by this series. No fix was attempted here; the second has since been fixed by the fused-decode series.

1. **`session_concurrency_bench --spec --verify` mismatch.** At ctx 256, gen 32, warm-up 4, stream 0 token 33 is a near-tie (tokens 67 and 428). Which side wins depends on the batch composition. On main it fails at 2 and 8 streams and passes at 1, 3, 4 and 16 (2, 3 and 4 were run on the split-K side branch with split-K off, which runs main's kernels). The base fails at 8 and passes at 1 and 16 in the same job, and failed at 2 and 8 and passed at 1, 3, 4 and 16 in the measured series' validation.
2. **MTP depth 2 does not round like plain decode on oped** ([`prompts/oped.txt`](prompts/oped.txt)). With the automatic depth rule the text diverges near generated token 30; with depth forced to 2, the first argmax flip is at generated token 111, as recorded in "Split each Qwen CUDA decode attention row like its one-token step".
   - Plain gives `f9a1f845`; MTP with the auto rule gives `c3e13776`, on both builds. On the measured series, forced depth 3 matches plain and forced depth 2 does not.
   - It persists with the MTP changes and GPU verify turned off, so the 2-row verify or its rewind does not match single-token rounding at a near-tie. That contradicted `verify_rows_exact`.
   - Cause and fix: the CUDA decode attention sized the key split of a multi-row decode from its last row, so a verify row split its keys differently from the one-token step at the same position. The first commit of the fused-decode series, "Split each Qwen CUDA decode attention row like its one-token step", gives every row its own split; the base predates it.
   - Evidence for the fix: on the base with that commit applied, oped gives `f9a1f845` in all four modes (plain, automatic depth, forced depth 2 and forced depth 3). The *MTP verify rows* section of [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md#mtp-verify-rows) records every compared verify row on the op-ed prompt bit-identical to plain decode (254 of 254 at depth 2 and at depth 3), and `tests/test_qwen4_mtp_exact.py` reproducing the plain greedy continuation at depths 2 and 3 on ten benchmark prompts. Main with the fix, which is this series as rebased, was not measured on oped.
   - With split-K enabled, depth 2 also matches plain on the measured series.
3. **`tests/test_cuda_session_batch` fails for Qwen** on the bit-equality of tensor-core batch rows vs single-row kernels: max_abs 3.11e-5 with 243,146 differing values, identical on the base and the measured series.

Also noted but not changed:
- The BF16 gate copies are released by the Q8→F16 cache's out-of-memory path as well as on model-map changes, and are then rebuilt on the next use.
- The BF16 copies (up to 146 MB) are not counted in the cache budget.
- compute-sanitizer synccheck was only partly conclusive: its barrier tracking overflowed on the full test.

See [`validation.csv`](validation.csv) for every check and its result.

## Method

- **In-process harness:** `qwen_decode_ab`, a small program built against the engine API that is kept with the bench-host tooling, not in this repository. It decodes the way the CLI does (chat prompt without thinking, 256 greedy tokens, `--ctx 4096`, MTP up to depth 3 with `--mtp` with the MTP draft head limited to the first 65,536 vocabulary rows (`DS4_QWEN4_MTP_DRAFT_ROWS=65536`, the served draft prefix, as in [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md#single-stream); the default is the full vocabulary)) in one process: a discarded warm-up block, then ABBA pairs of control and candidate blocks, a fresh session per block, and the token hash of every block. For a feature A/B the candidate blocks set the feature's temporary switch; without one it reports per-block rates and hashes. Cross-binary runs alternate two builds' processes (base, main, main, base).
- **GPU access:** every GPU run held an exclusive lease on the bench host, with the model server stopped and the page cache untouched.
- **Power limit:** 450 W unless stated. The 600 W job set 600 W itself and restored 450 W on exit.
- **Session harness:** `speed-bench/session_concurrency_bench`. It caps MTP at depth 2 (`--spec`), so depth-3 behaviour is measured with the in-process harness only.
- **nsys:** captures use `DS4_BENCH_CUDA_PROFILE_RANGE=8:32` and classify kernels by name and grid.

## Files

- [`end-to-end.csv`](end-to-end.csv): cross-binary base vs main at 450 W and 600 W, per prompt and mode, with process means and hashes.
- [`power-limit-abba.csv`](power-limit-abba.csv): each build at 450, 600, 600 and 450 W.
- [`sessions.csv`](sessions.csv): session harness runs at 450 W and 600 W (1 stream at ctx 256/4096, plain/spec; batched 1/8/16).
- [`power-clocks.csv`](power-clocks.csv): power, clocks, temperature and power-cap state per run group, at both limits.
- [`side-branches.csv`](side-branches.csv): the depth-policy and split-K ABBA runs on top of main.
- [`features.csv`](features.csv): per-feature standalone and combined gains (measured series), what was turned off to measure each, and status.
- [`leave-one-out-integrated.csv`](leave-one-out-integrated.csv): the raw leave-one-out ABBA results from the merged build (measured series).
- [`in-process-ab.csv`](in-process-ab.csv): the measured series' final-build in-process ABBA runs (every switch set, split-K, depth policy).
- [`kernel-classes.csv`](kernel-classes.csv): nsys incremental cost by kernel class, base vs stack (`stack_*` columns), for the three captures.
- [`validation.csv`](validation.csv): tests, sanitizers, verify runs and known-bug checks.
- [`prompts/`](prompts/): the eight held-out prompts and the two ~3K-token prompts (code and prose are in [`../qwen-cuda-6000/prompts/`](../qwen-cuda-6000/prompts/)).
