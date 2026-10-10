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

# Second series: paced copies and router keys

Two more commits on top of `c120d4a`, measured on the same GPU after its move to bare metal (host `voyage`: RTX PRO 6000 Blackwell Workstation Edition at 450 W, NVMe, 96 GB; the tables above were taken in the `inf` VM, so absolute t/s differ slightly and nothing below is compared across hosts). Both changes are bit-identical by construction and the greedy output is unchanged on every prompt; the kernel tests pass.

| Commit | Change |
|---|---|
| paced copies | `matvec_q8_bulk` issues each block's copy as two sequential mbarrier phases (one weight row each) instead of one 13 KB copy, so at most half of a projection's 16.7 MB is in flight at once (`DS4_QWEN4_BULK_PHASES` selects 1, 2 or 4). |
| router keys | `router_pre` selects the ten experts on integer keys (`float_as_uint(p) + 1`, 0 for NaN or none) with one `__reduce_max_sync` and one `__reduce_min_sync` per pick instead of a five-round compare-and-swap shuffle; the same total order (probability descending, expert ascending, NaN never, slot index with weight 0 when nothing is left). |

## End to end

Cross-binary ABBA as in the first series, two rounds of A, B, B, A processes with 2 pairs each; raw data in [`end-to-end.csv`](end-to-end.csv), `final-voyage` rows.

| Prompt | Plain series → stack t/s | Change | MTP series → stack t/s | Change |
|---|---:|---:|---:|---:|
| code | 177.60 → 178.25 | +0.36% | 261.42 → 263.94 | +0.96% |
| prose | 177.06 → 177.80 | +0.42% | 206.19 → 207.51 | +0.64% |
| longcode (~3K tokens, sparse attention) | 171.98 → 172.20 | +0.13% | 212.11 → 213.49 | +0.65% |
| longdoc (~3K tokens, sparse attention) | 171.54 → 171.81 | +0.16% | 219.09 → 220.42 | +0.60% |


Each change alone, one round, same host (`end-to-end.csv`, rows named by the change):

| Change (A → B) | code plain | prose plain | code MTP | prose MTP | longcode plain | longdoc plain | longcode MTP | longdoc MTP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paced copies, 2 phases (series tip → +pace) | −0.04% | −0.04% | +0.70% | +0.34% | −0.20% | −0.14% | +0.23% | +0.25% |
| router keys (attention-tile build → +router) | +0.43% | +0.43% | +0.24% | +0.27% | | | | |

## Per-kernel cost

Nsight Systems as before (session bench, one stream, 29 steady tokens, incremental cost per token), series tip and stack captured in the same job on `voyage`; raw data in [`kernel-classes-voyage.csv`](kernel-classes-voyage.csv).

The stack's per-token totals are 5612.8 µs at ctx 256 and 5811.9 at ctx 4096. The series-tip captures of this job were cold (0.3–0.7 ms per token of n-gram page reads in `stage_host`, which also lands on the `hc_norm_pre` row), so their totals are not comparable; the warm series-tip captures of the earlier jobs on this host read 5593.7 and 5791.2. The classes that moved, µs per launch, series tip → stack:

| Kernel | ctx 256 series | ctx 256 stack | ctx 4096 series | ctx 4096 stack | Note |
|---|---:|---:|---:|---:|---|
| GDN / attention output projection (`matvec_q8_bulk`, 2560 rows; the 12288-row q projection shares the class) | 7.62 | 11.77 | 7.76 | 11.79 | its copy lands in two round trips |
| `router_pre` | 3.17 | 2.64 | 3.18 | 2.65 | key-based selection |
| `attn_prep` | 10.62 | 6.60 | 6.83 | 4.92 | under the attention output projection's stream |
| `gdn_prep` | 3.21 | 1.90 | 2.97 | 1.72 | under the GDN output projection's stream |
| `gdn_scan` | 4.09 | 3.74 | 4.04 | 3.69 | same |
| `gdn_out` | 1.31 | 1.19 | 1.30 | 1.19 | same |
| `attention_group` | 8.22 | 8.01 | 12.08 | 11.40 | its key/value tiles under the stream |
| `idx_select` |  |  | 7.43 | 6.73 | sparse path only |
| `attn_merge` | 2.27 | 2.07 | 5.26 | 5.18 |  |

The profile and the end-to-end numbers disagree by about 20 µs per token on the sign: in the profile the projection's +250 µs per token is paid back by the kernels under its stream and by the router only within the capture noise, while the ABBA, which is the arbiter, is positive on every prompt and mode.


## What pacing moves

Two phases per block halve the bytes in flight. At ctx 256 the kernels running under the attention output projection's stream recover most of what the first series charged them: `attn_prep` 10.2 → 6.3 µs per launch, `gdn_prep` 3.8 → 1.9, `gdn_scan` 4.1 → 3.7, `attention_group` 8.2 → 7.9; the projection itself rises from 7.6 to 11.8 µs per launch, because its own copy now takes two round trips and the window in front of it (the GDN chain, 5.6 µs) no longer hides all of it. Net −14 µs per token at ctx 256 and +7 at ctx 4096 in the profiles; end to end within ±0.2% plain and +0.23 to +0.70% MTP, whose verify rows run more of the small kernels under the stream. Four phases recover more (`attn_prep` 4.3, `gdn_prep` 1.2) but cost the projection 14.0 µs per launch: +7 µs per token at ctx 256 against the series tip, dropped.

## Exactness

- `tests/test_qwen4_cuda` passes. `test_router_paths` compares `router_pre` with the reference `router()` byte for byte at T = 1..13 and now also on tie rows: the top probability shared by a fifth of the experts, every expert equal, three experts above a floor of zeros, −inf logits at every even expert, and an all-NaN row (one NaN logit makes the softmax sum NaN, so both kernels select experts 0..9 with weight 0). `test_presync_load` compares the paced copies with the plain kernel byte for byte at the three phase counts.
- Every ABBA block of every prompt gives the same 256-token hash on the series tip and on the stack, plain and MTP.

## Tried and dropped

- **The attention value tile requested together with the key tile** (`attention_group`: a second 16.5 KB tile, so the key and value rows of a tile go out as two `cp.async` groups and the tile costs one DRAM round trip; 55 KB per block with the static part, dynamic shared memory with the opt-in above 48 KiB). Byte-identical, but no faster: at ctx 256 the kernel went from 8.5 to 8.2 µs per launch (−10 µs per token in the profile, ±0.0% end to end), and at ctx 4096 it first went from 12.3 to 16.0 µs. The request order is not the reason: a build that selects it per launch (values after the softmax as before, both before the query scaling, both after it) gives 196.4, 195.6 and 195.4 µs per token, all the same. The reason is the L1/shared-memory carveout: for a 55 KB block the driver picks a smaller carveout than the one the staged projections beside it run under (up to seven 13 KB blocks per SM), and the SM reconfigures between the two kernels; with `cudaFuncAttributePreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared` on the kernel the cost returns to 12.5 µs per launch. Even so the 55 KB footprint halves the resident blocks once the MTP verify rows' grid exceeds the SM count (3 rows × 2 KV heads × 64 splits), and the full ABBA against the series tip sums negative: ±0.0% plain and +0.16/+0.17% MTP on the short prompts, −0.04/−0.08% plain and −0.54/−0.63% MTP on the ~3K-token prompts (`end-to-end.csv`, `kv-carveout-voyage`; the `kv-no-carveout-inf` rows are the same change before the carveout fix, −0.5% plain and −0.9% MTP on the long prompts). The carveout lesson stands for any decode kernel above 48 KB of shared memory.
- **The indexer's three launches as one** (`idx_fused`: 256-thread blocks score eight pooled block keys each, count their arrival, and the last block to arrive runs the radix select and the expansion, reading the scores through L2). Byte-identical to the three kernels, but the one launch costs 12.6 µs at ctx 4096 against 7.2 + 2.5 + 2.0 for the three: the select starts only after the last scoring block's fence and atomic, where before it was resident and waiting at its dependency; end to end −0.12/−0.33% plain and −0.09/−0.09% MTP on the ~3K-token prompts (`fused-indexer-voyage` rows).
- **Four copy phases**: see above.
