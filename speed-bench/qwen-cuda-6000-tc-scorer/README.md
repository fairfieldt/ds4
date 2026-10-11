# Qwen CUDA indexer scores on tensor cores, RTX PRO 6000 Blackwell

One commit on main (`f19a88b`), after the prefill series ([`../qwen-cuda-6000-prefill`](../qwen-cuda-6000-prefill/README.md)), on the same host and model (RTX PRO 6000 Blackwell Workstation Edition, now at its 600 W limit, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage bare metal). Prefill chunks whose rows select from more than 8192 blocks (past 32K tokens of context) score the indexer on tensor cores and rescore exactly every block the top-512 selection can tell apart, so the selection equals the one from exact scores, slot for slot: the greedy text matches main at every frontier below, decode and MTP tokens match, and the kernel tests pass.

| Build | Commit | Change |
|---|---|---|
| main | `f19a88b` | The parent. |
| tc | `5dd0ac8` | Tensor-core indexer scores with exact rescoring (`idx_tc_prep`, `idx_score_tc`, `idx_tc_exact`). |

## Why the selection does not change

Let a row have approximate scores a and exact scores s (the arithmetic of `idx_score_body`), with |a − s| ≤ δ for every block, and let τ be at most its 512th largest approximate tile maximum (8-block tiles). Then 512 blocks have a ≥ τ, so the row's 512th largest exact key thr is at least τ − δ, and a block with a < τ − 2δ has s < thr and a < thr. Those blocks keep their approximate scores; every other visible block gets its exact score, and the tiles holding them get their maxima rewritten, so every tile maximum is the largest key of its tile (tiles without rescored blocks keep their approximate maxima rather than 0). The row holds the exact key of every block at or above thr and only keys below thr elsewhere, with consistent tile maxima, so `idx_select_wide` (unchanged) picks the same slots, ties included.

δ for a row is max |k| Σ_h (|r_h| + 2⁻¹¹ |q_h|), with margins, plus 2⁻⁵⁰ (max |k| + 1) for subnormals flushed to zero. |r_h| is the norm of the rounding error of head h's half query (each row is scaled by a power of two so its largest element lands in [2¹⁴, 2¹⁵)), which bounds the query rounding by Cauchy–Schwarz. The 2⁻¹¹ |q_h| term covers the rest. The tensor cores multiply halves exactly and add 16 products plus the accumulator per step, aligned to the largest term with 25 fraction bits and truncated (Khattak and Mikaitis, [arXiv 2512.07004](https://arxiv.org/abs/2512.07004), measured on this GPU), which keeps their error under 2⁻¹⁷ Σ|q_i k_i| over the eight steps; the reference's fused multiply-adds and butterfly round each product at most nine times; the four relu'd heads add six roundings. That leaves 2⁻¹² for the tensor cores, 32 times their bound. The kernel tests measure the accumulation at most 13.2 · 2⁻²⁴ Σ|q_i k_i| on products spanning 25 binades, with and without cancellation, and |a − s| at most 0.04 δ on random rows.

## Where it went

Nsight Systems, one 8192-row chunk, incremental cost (end_i − max(end_<i)); raw data in [`kernels.csv`](kernels.csv):

| ms per chunk (12 layers) | main, 122,880 | tc, 122,880 | main, 245,760 | tc, 245,760 |
|---|---:|---:|---:|---:|
| block scores | 75.9 | 24.5 | 148.9 | 37.2 |
| — `idx_score_tc` | | 12.5 | | 24.7 |
| — `idx_tc_exact` | | 11.7 | | 12.0 |
| — `idx_tc_prep` | | 0.4 | | 0.5 |
| block selection | 5.3 | 5.3 | 6.7 | 6.7 |
| **chunk** | **2,455.5** | **2,412.6** | **2,536.8** | **2,428.7** |

The other kernels match within 2.5%. At 600 W, main's scorer is 149 ms of the chunk at 245,760 tokens; the earlier record's 194 ms was at 450 W.

## The kernels

- `idx_tc_prep` rounds each row's queries to halves in the order of the `mma.sync` fragments and computes the row's error coefficient; a second set of blocks transposes the keys into the order `idx_tc_exact` reads and takes their largest norm.
- `idx_score_tc` stages 64 keys at a time with `cp.async`. Each warp keeps the four heads of eight rows as A fragments in registers (`m16n8k16`, FP32 accumulation), sums the relu'd heads, and writes approximate scores (8-byte streaming stores) and tile maxima. Blocks no row of a 64-row panel sees get −3·10³⁸ without any products.
- `idx_tc_exact` (one block of 256 threads per row) finds τ by bisection on 22 bits of the tile maxima, held in registers. It then gathers the tiles reaching τ − 2δ, reads their approximate scores, and rescores the blocks that reach the cut, two lanes a block: each lane replays half of `idx_score_mm`'s binary counter over bit-reversed lane values, and the pair adds the halves and heads in the reference's order. Tile maxima are rewritten from shared-memory maxima.

`test_idx_score_topk_all` compares the selected slots with those from exact scores byte for byte: random rows (row scales 2⁻⁶ to 2⁶, half the blocks visible, selection groups of two tiles at 70,001 blocks), heavy ties (40 distinct keys), 3,000 near-equal keys straddling the threshold (several batches of tiles), rows with 509 to 525 positive scores, zero queries, and infinite or NaN queries. It also checks that every key at or above a row's 512th largest exact key is exact, every other key stays below it, and every tile maximum is the largest key of its tile; `test_idx_tc_bound` checks |a − s| ≤ δ and the tensor cores' accumulation against 2⁻¹² Σ|q_i k_i|.

Decode and the MTP verify rows (under 16 rows) keep their kernels. `DS4_QWEN4_IDX_TC_MIN` sets the smallest block count that takes this path (default 8192; 0 turns it off), and `DS4_QWEN4_IDX_SCORE_MM=0` turns off both prefill scorers. Metal keeps `ds4_gpu_qwen4_idx_score_tensor`; the CUDA call site uses the new `ds4_gpu_qwen4_idx_score_topk_tensor`, which takes the selection's top-k.

## Real rows

Inputs of real chunks (a layer's queries and pooled keys), dumped from `ds4-bench` runs over the same prompt and analysed with an FP32 emulation of the half queries ([`candidates.csv`](candidates.csv)):

| Chunk at | Layer | δ / thr (p50) | thr − τ (p50) | Candidate tiles a row | Candidate blocks a row (mean, max) |
|---:|---:|---:|---:|---:|---:|
| 245,760 | 0 | 0.48% | 4.6% | 569 of 7,936 | 985, 1,546 |
| 245,760 | 5 | 0.37% | 2.7% | 558 | 821, 1,370 |
| 245,760 | 11 | 0.48% | 3.9% | 565 | 929, 1,365 |
| 122,880 | 5 | 0.41% | 4.8% | 549 of 4,096 | 940, 1,792 |
| 57,344 | 5 | 0.43% | 6.7% | 543 of 2,048 | 1,050, 1,723 |

The largest error seen was 0.08 δ. τ, taken over tiles, sits 3–7% below the block-level threshold because the top blocks cluster in tiles (the median row's top 512 blocks fill 305–371 tiles), so 1.6–2 times as many blocks are rescored as are selected. A tile's eight blocks are rescored only where they reach the cut: rescoring every block of a candidate tile would take ~4,500 blocks a row.

## Microbenchmarks

`DS4_TEST_QWEN4_IDX_TOPK_BENCH=1`, one 8192-row chunk, 600 W ([`microbench.csv`](microbench.csv)); the real chunks' selections are identical for all 36 layers:

| Data | Blocks | `idx_score_mm` µs | tc µs | with selection: mm | tc |
|---|---:|---:|---:|---:|---:|
| chunk at 245,760, 12 layers | 63,488 | 12,023 | 3,097 | 12,925 | 3,624 |
| chunk at 122,880, 12 layers | 32,768 | 6,539 | 2,233 | 7,207 | 2,622 |
| chunk at 57,344, 12 layers | 16,384 | 3,211 | 1,561 | 3,584 | 1,934 |
| chunk at 57,344, first blocks | 9,000 | 1,569 | 1,285 | 1,959 | 1,635 |
| chunk at 57,344, first blocks | 12,288 | 2,205 | 1,265 | 2,595 | 1,626 |
| random rows | 63,488 | 12,410 | 3,586 | 13,848 | 4,118 |

Exclusive kernel times at 245,760 (PDL off): `idx_tc_prep` 33 µs, `idx_score_tc` 1,858 µs, `idx_tc_exact` 1,017 µs (the cut ~170, gathering ~220, rescoring ~630).

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64`, prefill rate of the newest 32K interval and steady greedy decode, processes main, tc, tc, main at 600 W ([`end-to-end.csv`](end-to-end.csv)):

| Context | Prefill main | tc | Change | Decode main | tc |
|---:|---:|---:|---:|---:|---:|
| 32K | 3,415 | 3,413 | −0.0% | 170.4 | 170.4 |
| 64K | 3,405 | 3,420 | +0.4% | 169.3 | 169.3 |
| 96K | 3,374 | 3,413 | +1.2% | 168.8 | 168.7 |
| 128K | 3,342 | 3,404 | +1.9% | 168.3 | 168.3 |
| 160K | 3,316 | 3,397 | +2.4% | 168.0 | 168.0 |
| 192K | 3,286 | 3,389 | +3.1% | 168.6 | 168.5 |
| 224K | 3,261 | 3,385 | +3.8% | 168.0 | 168.0 |
| 256K | 3,232 | 3,378 | +4.5% | 167.0 | 167.0 |

All four processes give the same greedy text at every frontier. The newest interval at 32K has no row past 8192 blocks. Prefill at 256K is now 1.0% below 32K, against 5.4% for main. Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: −0.02% and −0.03% plain, −0.02% and −0.03% MTP.

## Findings

- `idx_score_tc` is not bound by the tensor cores: without the products it takes as long (2.15 against 2.07 ms, PDL on), and without its 2 GB of score stores it takes 1.48 ms instead of 1.87 (PDL off). The unchanged selection needs every score written. 8-byte stores in place of 4-byte ones: 1.93 → 1.87 ms; 16-warp blocks (128 rows, half the key traffic) changed nothing.
- The rescoring issues at about a third of the SMs' rate, and neither two blocks per lane pair (128 registers), nor queries held in registers with eight lanes per block, nor sector-aligned keys moved it by more than 6%; what is left is fetching 256 scattered bytes of key per block from L2.
- Consecutive rows share few candidates: the union over 2 rows is 1.4× one row's tiles, over 32 rows 5–7×, so grouping rows to share key loads would multiply the work.
- The bisection cut costs the same as the two-round radix select it replaced (0.27 ms a layer in both, PDL on), with no histograms in shared memory.
- The first version found the cut in a kernel of its own (1024 threads a row, radix select), scanned the candidate tiles 256 at a time behind a barrier and rescored one block per thread: 4.2 ms against 3.3 ms for the fused kernel.

What remains at 245,760: the indexer costs 37 ms of the 2.43 s chunk; `matrix_half_tile` (675 ms) and `gdn_scan` (504 ms) lead it.
