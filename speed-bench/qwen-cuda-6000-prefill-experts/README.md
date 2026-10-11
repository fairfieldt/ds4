# Qwen CUDA prefill expert tiles, RTX PRO 6000 Blackwell

One commit on top of main (`5e1293a`, after the hyper-connection mixes in [`../qwen-cuda-6000-prefill-hc`](../qwen-cuda-6000-prefill-hc/README.md)) that runs the routed experts' GEMMs on tiles of 192 tokens instead of 64. Same host and model (RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage), 600 W. Every output takes the same m16n8k16 chain over k in order on the same halves; the kernel tests compare `matrix_half_tile`, `matrix_half_ws` and the new kernel byte for byte, and the greedy text matches main at every frontier to 256K.

| Build | Commit | Change |
|---|---|---|
| main | `5e1293a` | The parent. |
| new | `31292a4` | `matrix_half_wb`. |

## Where a prefill chunk went

Nsight Systems, an 8192-row chunk at 24,576 and at 245,760 tokens of context, incremental cost per kernel; all kernels in [`kernels.csv`](kernels.csv):

| Kernel, ms per chunk | main, 24,576 | new, 24,576 | main, 245,760 | new, 245,760 |
|---|---:|---:|---:|---:|
| routed experts (`matrix_half_ws` → `matrix_half_wb`) | 242.3 | 214.7 | 245.6 | 217.6 |
| everything else | 1,055.8 | 1,056.1 | 1,097.0 | 1,094.3 |
| **chunk** | **1,298.1** | **1,270.8** | **1,342.6** | **1,311.9** |

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64 --show-output`: prefill rate of the newest 32K interval and steady greedy decode rate at each frontier, processes main, new, new, main; raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Prefill main | new | Change | Decode main | new |
|---:|---:|---:|---:|---:|---:|
| 32K | 6,502 | 6,598 | +1.5% | 170.8 | 170.7 |
| 64K | 6,481 | 6,583 | +1.6% | 169.6 | 169.7 |
| 96K | 6,439 | 6,534 | +1.5% | 169.0 | 169.1 |
| 128K | 6,402 | 6,495 | +1.4% | 168.4 | 168.4 |
| 160K | 6,354 | 6,466 | +1.8% | 168.1 | 168.1 |
| 192K | 6,317 | 6,428 | +1.8% | 168.8 | 168.8 |
| 224K | 6,279 | 6,398 | +1.9% | 168.2 | 168.2 |
| 256K | 6,240 | 6,371 | +2.1% | 167.2 | 167.1 |

All four processes give the same greedy text at all eight frontiers. Short contexts, prefill only, frontiers 2K to 32K (each row the newest interval), processes main, new, new, main, main, new ([`short-context.csv`](short-context.csv)):

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 5,204 | 5,212 | +0.1% |
| 4K | 5,695 | 5,688 | −0.1% |
| 8K | 6,465 | 6,508 | +0.7% |
| 16K | 6,722 | 6,853 | +1.9% |
| 32K | 6,671 | 6,796 | +1.9% |

Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: +0.01% and +0.01% plain, +0.00% and +0.03% MTP.

## The kernel

`matrix_half_ws` (see [`../qwen-cuda-6000-prefill-kernels`](../qwen-cuda-6000-prefill-kernels/README.md)) dequantized a tile's weights once per 64 of an expert's tokens, so a weight was dequantized about three times per expert at 8192 rows, and that work shared the SM with the products ([`../qwen-cuda-6000-prefill-fusion`](../qwen-cuda-6000-prefill-fusion/README.md#findings)). `matrix_half_wb` takes 128 A rows (64 gate rows and the same up rows, or 128 down rows) by 192 tokens: four producer warps (a thread per A row) load and dequantize a step's weights into one of two A stages and copy its activations into one of two B stages (a zero-filling copy, without a branch), behind the same empty barrier; eight consumer warps take 32 A rows (gate/up: 16 of each) by 96 tokens and only multiply. The dequant per product falls by three. Microbenchmarks with uniform routing ([`microbench.csv`](microbench.csv)), one chunk of 8192 rows: gate/up 2,733 → 2,093 µs, down 1,800 → 1,399 µs; at 4096 rows 1,866 → 1,675 and 1,078 → 952 µs; at 2048 rows gate/up 1,703 → 1,553 µs but down 661 → 733 µs, so the down projections keep `matrix_half_ws` where an expert averages fewer than 64 tokens.

## Findings

- With the model's routing the expert kernels gained 11% in the chunk against 22% with uniform routing in the microbenchmark. A 192-token tile with 96 or fewer tokens leaves one of the two consumer warps of each sub-partition without columns, as a 64-token tile with 32 or fewer did, so the gain falls where experts get few tokens; the model's routing is probably less even than the uniform one (not measured here).

Dropped, with numbers:

- `matrix_half_md`: Q4_K gate/up on 64-token tiles with the dequant in the multiplying warps (raw 48-byte rows of a 64-k step in three cp.async stages, one barrier a step, each lane's mma fragments built from its rows' nibbles by byte permutes): exact, 2,872 against 2,735 µs. Per warp and step it issues 64 HMMAs, about 260 instructions of dequant and 280 of integer and branch work; two such warps a sub-partition fill about 80% of its issue slots and the HMMAs run at 47% of their rate.
- `matrix_half_wb` with the activation rows brought in by the tensor memory accelerator (a 2D copy of 64 k by one token row, 192 a step from one producer warp, into the same swizzle): exact, gate/up 4,637 µs, down 3,725 µs.
- FP32x2 arithmetic for the dequant: on sm_120 `fma.rn.f32x2` compiles to two FFMAs.
