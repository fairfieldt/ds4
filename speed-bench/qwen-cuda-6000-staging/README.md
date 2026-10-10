# Qwen CUDA decode: staged projections and the HC norm, RTX PRO 6000 Blackwell

Two commits on top of the roofline series ([`../qwen-cuda-6000-roofline`](../qwen-cuda-6000-roofline/README.md)), measured on the same host and model (RTX PRO 6000 Blackwell Workstation Edition, 450 W limit, CUDA 13.4, Qwen3.8-Flash-Next-Q4). Every change is bit-identical by construction and the greedy output is unchanged on every prompt; the kernel tests pass at both commits.

| Commit | Change |
|---|---|
| `af43ac1` | The decode HC normalization (`hc_norm_pre`): the injection partials load as one batch, the first batches of the residual row go out before the injection barrier, and the four partial sums take one reduction round. Also batches the injection loads of `moe_reduce` and `hc_combine`. |
| `a4c5fa6` | The Q8 output projections (GDN and attention output 2560×6144, attention q 12288×2560) copy their rows to shared memory with one bulk (TMA) copy before the dependency wait (`matvec_q8_bulk`, two rows per block). The small kernels in front of them trigger their dependents before their own wait (`pdl_enter_early`: gdn_prep, gdn_scan, gdn_out, attn_prep, block_key, the indexer kernels, the attention core and merge), so the projection is resident and streaming while they run. `attn_prep` and `gdn_out` load their norm weights before the wait; `attn_merge` requests its per-split values sixteen at a time. |

## End to end at 450 W

Cross-binary ABBA ([`../qwen-cuda-6000-roofline/README.md#method`](../qwen-cuda-6000-roofline/README.md#method)): each build in its own process, `qwen_decode_ab` with 2 pairs of 256 greedy tokens after a warm-up block, `--ctx 4096`, `DS4_QWEN4_MTP_DRAFT_ROWS=65536`, processes in the order main, series, series, main. Raw data in [`end-to-end.csv`](end-to-end.csv).

| Prompt | Plain main → series t/s | Change | MTP main → series t/s | Change |
|---|---:|---:|---:|---:|
| code | 170.96 → 177.85 | +4.03% | 255.95 → 261.36 | +2.11% |
| prose | 170.53 → 177.33 | +3.98% | 199.83 → 206.00 | +3.09% |
| longcode (~3K tokens, sparse attention) | 164.72 → 171.97 | +4.40% | 204.94 → 211.85 | +3.37% |
| longdoc (~3K tokens, sparse attention) | 164.28 → 171.41 | +4.34% | 211.66 → 218.77 | +3.36% |

The first commit alone: code 170.81 → 175.82 (+2.93%), prose 170.37 → 175.36 (+2.93%) plain; 255.10 → 257.38 (+0.89%) and 199.37 → 201.37 (+1.00%) MTP.

Held-out prompts ([`../qwen-cuda-6000-roofline/prompts/`](../qwen-cuda-6000-roofline/prompts/)), one pair per process, processes main, series, series, main:

| Prompt | Plain main → series t/s | Change | MTP main → series t/s | Change |
|---|---:|---:|---:|---:|
| explain | 170.53 → 177.03 | +3.81% | 224.32 → 231.26 | +3.09% |
| json | 170.38 → 176.86 | +3.81% | 272.89 → 277.79 | +1.80% |
| math | 170.38 → 176.88 | +3.82% | 245.23 → 251.68 | +2.63% |
| oped | 169.97 → 176.37 | +3.77% | 207.05 → 213.61 | +3.17% |
| rust | 169.88 → 176.30 | +3.78% | 237.67 → 244.17 | +2.73% |
| sql | 169.81 → 176.25 | +3.79% | 242.68 → 247.99 | +2.19% |
| table | 169.72 → 176.12 | +3.77% | 248.34 → 254.13 | +2.33% |
| translate | 169.50 → 176.24 | +3.97% | 210.76 → 216.72 | +2.82% |
| **8-prompt mean** | | **+3.81%** | | **+2.59%** |

## Per-kernel cost

Nsight Systems, session bench with one stream, 29 steady tokens, incremental cost per token (end_i − max(end_<i)); raw data in [`kernel-classes.csv`](kernel-classes.csv). Both captures of main include cold n-gram page reads at `stage_host` (see below), which are excluded from the totals.

| µs per token (excl. `stage_host`) | main | series |
|---|---:|---:|
| ctx 256 (dense attention) | 5,807 | 5,581 |
| ctx 4096 (sparse attention) | 6,041 | 5,763 |

The classes that moved, ctx 256, µs per launch:

| Kernel | main | series | Note |
|---|---:|---:|---|
| GDN / attention output projection (`matvec_q8` → `matvec_q8_bulk`, 2560 rows) | 13.75 | 5.31 | rows staged during gdn_prep/scan/out or the attention core |
| attention q projection (12288 rows) | 22.51 | 16.15 | first wave staged during the HC chain |
| `hc_norm` combine variant / plain variant | 7.60 / 4.30 | 6.0 / 4.5 | injection batch, earlier row loads, one reduction |
| `gdn_proj` | 30.39 | 34.21 | the GDN output projection's rows now stream during its last wave (zero-sum DRAM) |
| `gdn_scan` | 2.98 | 4.02 | its state reads share DRAM with that stream |
| `attn_prep` | 3.72 | 10.63 | its cache appends run under the attention output projection's stream (see below) |
| `moe_reduce` | 2.64 | 2.33 | injection batch |

## What the stream costs the kernels under it

A bulk stream of 16.7 MB issued at once by 1,280 resident blocks saturates the memory queues for about 10 µs, and every dependent DRAM read made by a kernel running meanwhile waits several µs: `attn_prep`'s per-layer norm weights and rope positions, `attention_group`'s key/value tiles, `gdn_scan`'s state. Kernels whose reads hit L2 (`hc_norm`, `hc_mix`, `moe_reduce`) are not slowed. Moving the trigger only moves the cost: with the attention core triggering after its wait, `attn_prep` returns to 3.0 µs and `attention_group` goes from 8.0 to 15.4 µs; with `attn_merge` triggering after its wait, the merge goes from 2.0 to 9.4 µs. The three placements agree within 0.3% on the short prompts; the one kept (every chain kernel triggers early) is 0.6% faster on the ~3K-token prompts ([`end-to-end.csv`](end-to-end.csv), `triggers` rows). Loading the weights a kernel needs before its wait removes that kernel's share: `gdn_out` went back from 1.9 to 1.2 µs with its gamma preloaded; `attn_prep` did not, because its remaining reads are the projection outputs it consumes.

## Exactness

- `tests/test_qwen4_cuda` passes at both commits. `test_presync_load` now also compares the staged Q8 rows with the plain kernel byte for byte at T = 1..3 on 2560×6144, 12288×2560 and two odd row counts; `test_decode_fusions` and the presync HC cases cover `hc_norm_pre` against the reference `hc_norm`.
- Every ABBA block of every prompt gives the same 256-token hash on main and on the series, plain and MTP.
- The early triggers move only `griddepcontrol.launch_dependents`; every kernel still waits before reading what its predecessor wrote, and the staged kernels read only weights before the wait (verified in SASS: `UBLKCP` before `ACQBULK`, no stores before it).

## Tried and dropped

Each was built, tested byte-exact where it ran, and measured in the same harness.

- **The norm fused into the HC down launch** (every block of the 320-row projection re-normalizing the four residual streams, the first 32 blocks writing xn and the injection partials): −4.3% plain, −6.4% with the normalized operand formed before the bulk wait. These split-K projections are bound by their redundant activation reads out of L2 (320 blocks × 40 KB); the fusion doubles them (29 MB per launch against 13).
- **Two weight rows per block in the HC down bulk GEMV** (half the activation reads): +0.9 µs per launch.
- **A shared-memory copy of the residual row in `hc_norm_pre`** for the chunk pass: ptxas interleaves each store with the load that feeds it and serializes the loads; the chunk pass already hits L1. **cp.async staging** of the same row: slower than register batches.
- **A max-shared L1/shared-memory carveout for every decode kernel**: +0.3% plain, −0.5% MTP; 50% gave +0.3% / −0.9%.
- **`gdn_scan` triggering after its state loads** instead of before its wait: the compiler barrier needed to keep the loads ahead of the trigger made the scan 3.8 → 10.7 µs.
- **`attn_prep`'s rope positions loaded before the wait**: no change (its projection inputs are what wait behind the stream).
- **`moe_reduce` slot loads in one batch**: within noise.

## Cold n-gram reads

Both nsys captures of main show the session bench stalling 0.4–0.5 ms per token on average at `stage_host`, the n-gram rows read from the 95 GiB table: on this host (a VM with a virtio disk) 16 random rows cost 0.44 ms when queued with `posix_fadvise`, of which layer 0 hides about 0.1 ms. The ABBA harness does not see it because its repeated blocks keep the rows page-cached. It is a storage limit, not a code path: one row costs 0.16 ms, threads do not beat the queued reads, and the table does not fit the 78 GB of RAM.

## Method

- Builds: `rf-build` on the bench host from the branch, `CUDA_ARCH=native`, with `qwen_decode_ab` linked against the engine.
- End to end: `rf-xab A B --rounds 1 --pairs 2 [--mtp] --prompt-file ...`, i.e. A, B, B, A processes, each a warm-up block then 2 pairs; the table reports the mean of each build's two processes.
- Profiles: `rf-nsys` (session bench, 1 stream, 64 generated after 16 warm-up, PDL on, 32 captured steps), per-kernel incremental cost from `timeline.py`.
- GPU runs held an exclusive gpuq lease; the model server was stopped.
