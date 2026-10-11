# Qwen CUDA prefill at long context, RTX PRO 6000 Blackwell

Two commits on top of the long-context decode series ([`../qwen-cuda-6000-longctx`](../qwen-cuda-6000-longctx/README.md)), on the same host and model (RTX PRO 6000 Blackwell Workstation Edition, 450 W limit, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage bare metal). Both are bit-identical by construction: the greedy text matches the parent at every frontier below, decode and MTP tokens match, and the kernel tests pass at both commits.

| Build | Commit | Change |
|---|---|---|
| base | `e67a1c7` | The parent (the long-context decode series). |
| scorer | `0474e6c` | The indexer scores of prefill rows from shared-memory tiles (`idx_score_mm`). |
| final | `7bc8efc` | Prefill attention with its key and value gathers overlapped (`attention_group_pipe`). |

## Where long-context prefill went

A prefill chunk is 8192 rows. Its 12 attention layers each score every pooled 4-token block against every row (`idx_score*`), keep each row's top 512 blocks and attend their 2048 tokens plus the tail. Nsight Systems, the chunk at 24,576 or 245,760 tokens of context, incremental cost (end_i − max(end_<i)); raw data in [`kernels.csv`](kernels.csv):

| Kernel, ms per chunk | base, 24,576 | final, 24,576 | base, 245,760 | final, 245,760 |
|---|---:|---:|---:|---:|
| block scores | 214.9 | 20.8 | 894.4 | 193.8 |
| attention (2051 keys) | 127.5 | 105.6 | 149.9 | 116.6 |
| block selection | 10.8 | 13.6 | 7.7 | 7.6 |
| **chunk** | **2,784** | **2,660** | **3,566** | **2,854** |

The per-warp scorers read every key again for every row (16 MB a row at 245,760 tokens) and reduce each score with five shuffles. The base capture at 24,576 tokens was the session's first; the other kernels ran 1-4% slower in the final build's later capture (`matrix_half_tile` 711 → 739 ms), within run-to-run variation, and match within 1% at 245,760.

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64`: prefill rate of the newest 32K interval (8192-row chunks) and steady greedy decode rate at each frontier, processes base, scorer, final, final, scorer, base. Raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Prefill base | scorer | final | Change | Decode base | scorer | final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 3,021 | 3,152 | 3,183 | +5.4% | 169.8 | 169.7 | 169.4 |
| 64K | 2,981 | 3,128 | 3,160 | +6.0% | 168.7 | 168.6 | 168.5 |
| 96K | 2,851 | 3,091 | 3,122 | +9.5% | 168.1 | 168.0 | 167.9 |
| 128K | 2,730 | 3,055 | 3,087 | +13.1% | 167.5 | 167.4 | 167.3 |
| 160K | 2,622 | 3,021 | 3,051 | +16.3% | 167.1 | 166.9 | 166.9 |
| 192K | 2,521 | 2,989 | 3,018 | +19.7% | 167.4 | 167.5 | 167.3 |
| 224K | 2,428 | 2,961 | 2,992 | +23.2% | 166.9 | 166.9 | 166.7 |
| 256K | 2,343 | 2,933 | 2,960 | +26.3% | 163.5 | 163.5 | 163.4 |

All six processes give the same greedy text at every frontier. An earlier series on the same boot, base, final, final, base, agrees: +4.8% at 32K, +12.8% at 128K and +26.2% at 256K (2352 → 2967 t/s). Decode follows the order of the processes rather than the build (the first is the fastest, 170.0 t/s at 32K against 169.4 to 169.8 for the rest); `rf-xab` below measures it more finely.

At a 600 W limit (prefill only, processes base, final, final, base): base 3,248 → 3,416 t/s at 32K (+5.2%), 3,002 → 3,342 at 128K (+11.3%), 2,626 → 3,227 at 256K (+22.9%). Against the 450 W series base, final, final, base, the extra power lifts base more at 256K (+11.6%) than final (+8.8%): the old scorer was the most power-starved kernel.

## Short contexts

Prefill only, frontiers 2K to 32K (each row the newest interval), three interleaved processes per build ([`short-context.csv`](short-context.csv)):

| Context | scorer | final |
|---:|---:|---:|
| 2K | −0.36% | +0.56% |
| 4K | +0.56% | +1.89% |
| 8K | +1.43% | +2.89% |
| 16K | +3.64% | +4.98% |
| 32K | +6.69% | +8.26% |

No row of the 2K chunk reaches the scorer, so the scorer's −0.36% there is the order of the processes (each scorer process ran right after a base one, on a GPU that slowed slightly from one process to the next); the final build's 2K rows attend densely through `attention_group_pipe`. Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: scorer −0.03% and −0.01% plain, +0.00% and +0.04% MTP; final −0.02% and −0.03% plain, −0.00% and −0.03% MTP.

## The scorer

`idx_score_body` gives lane l of a warp the products of dims l, l+32, l+64 and l+96, in that order, as fused multiply-adds, then adds the 32 lane values with a butterfly over lanes (l, l^16), then (l, l^8), down to (l, l^1). Taken in bit-reversed lane order that butterfly is a binary counter: lane values 0 and 16 make a pair, then 8 and 24 a pair that merges with the first, and so on, so each lane value merges with the pending subtrees of its size as soon as it is computed. One thread can then follow the warp's exact sequence of roundings with at most four pending partial sums per half of the lanes (the even lanes, then the odd ones, whose two subtrees make the last add).

`idx_score_mm` stages a tile of 32 rows × 128 blocks: each lane value's four dims sit together in shared memory as floats (queries [l][row], one head at a time; keys [l][block], converted from half once). A thread scores 4 rows × 4 blocks per lane value (8 shared loads, 64 fused multiply-adds, 15.5 adds) and loads the next head's queries while it multiplies this head's; tiles that no row sees write −3·10³⁸ without loading anything. One block per SM (80 KB, 182 registers). Launches of 16 or more rows take it; decode and the MTP verify rows keep their kernels, and `DS4_QWEN4_IDX_SCORE_MM=0` keeps the per-warp kernels everywhere. One 8192-row chunk ([`microbench.csv`](microbench.csv)):

| Context | Per-warp kernel µs | `idx_score_mm` µs |
|---:|---:|---:|
| 8K | 3,824 | 198 |
| 32K | 15,599 | 1,311 |
| 128K | 32,204 | 6,993 |
| 248K | 65,381 | 13,845 |

`test_idx_score_exact_all` compares the scores and tile maxima with the per-warp kernels byte for byte, and the visible scores with the same arithmetic replayed on the CPU (`fmaf` chains and the butterfly), at up to 70,001 blocks, with masked tails and tiles past every row's blocks.

## The attention

`attention_group_body` handles a 32-key tile in turn: it waits for the tile's positions, then gathers its keys, then its values, so each tile pays three dependent memory latencies. `attention_group_pipe` does the same arithmetic in the same order for prefill rows (one split), but gathers the values during the tile's key products, the next tile's keys during its softmax and value products, and reads the positions two tiles ahead. Keys and values get a tile each; to stay at two blocks per SM (49 KB), the queries keep only their group's 12 rows (rows past it read a zero chunk) and the tiles swizzle their 16-byte chunks by row instead of padding. `test_attn_pipe_exact_all` compares the outputs with `attention_group` byte for byte, with empty rows, invalid and future positions; `DS4_QWEN4_ATTN_PIPE=0` restores `attention_group`. The prefill microbenchmark takes 8.05 → 6.00 ms per layer at 32K tokens and 8.35 → 6.24 at 248K; in the real chunk, 127.5 → 105.6 ms at 24,576 tokens and 149.9 → 116.6 at 245,760.

## Findings

- The scorers are power-bound at 450 W: their microbenchmarks ran at the cap (SW power cap) with the SM clock at 2.06 to 2.24 GHz and the die at 37 to 43 °C, and at 600 W `idx_score_mm` ran 12% faster (13.8 → 12.1 ms at 248K), as did the per-warp kernel. The attention kernels are not (6.00 ms at both limits). The driver's thermal slowdown counters stayed at zero throughout, at up to 86 °C in the 600 W runs.
- A first exact layout, keys as halves converted at each use with 4 rows × 2 blocks per thread step at two blocks per SM, took 16.2 ms at 248K against 13.8 for floats in shared memory and a 4 × 4 step at one block per SM: in the power-bound regime, instructions per score count more than occupancy.
- Tensor cores would score a chunk at 248K in 5.6 ms (queries rounded to half) or 7.0 ms (queries as hi + lo/4096 halves) against 13.8, about 100 ms of the 2.85 s chunk, but the sums round differently and the selections near the threshold change. Not taken: it would need the quality evidence an exact kernel does not.
- sm_120 has no paired FP32 FMA: `fma.rn.f32x2` compiles to two FFMAs (nvcc 13.4).
- A max-shared carveout for `attention_group` changed nothing (8.09 against 8.13 ms per layer): it already ran two blocks per SM.

What remains: at 256K the final build prefills 7.0% slower than at 32K. The scorer still costs 194 ms of the chunk at 245,760 tokens against 21 at 24,576, and the attention 117 against 106.
