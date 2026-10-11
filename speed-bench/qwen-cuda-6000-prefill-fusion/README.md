# Qwen CUDA prefill projections, RTX PRO 6000 Blackwell

One commit on top of main (`beba91e`, after the context-independent kernel series in [`../qwen-cuda-6000-prefill-kernels`](../qwen-cuda-6000-prefill-kernels/README.md)) that runs the Q8 projections of a prefill chunk as one kernel instead of three cuBLAS GEMMs and a rescale. Same host and model (RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage), 600 W. The new kernel computes each output with the same operations in the same order as cuBLAS where cuBLAS does not split k, and keeps cuBLAS where it does; the kernel tests compare the two byte for byte, and the greedy text matches main at every frontier to 256K.

| Build | Commit | Change |
|---|---|---|
| main | `beba91e` | The parent. |
| new | `1d3f605` | `dense_split_tma`. |

## Where a prefill chunk went

Nsight Systems, an 8192-row chunk at 24,576 and at 245,760 tokens of context, incremental cost per kernel; all kernels in [`kernels.csv`](kernels.csv):

| Kernel, ms per chunk | main, 24,576 | new, 24,576 | main, 245,760 | new, 245,760 |
|---|---:|---:|---:|---:|
| Q8 projections (cuBLAS GEMMs and `dense_rescale2` → `dense_split_tma`) | 528.2 | 450.9 | 535.1 | 459.8 |
| weight packs (`pack_half_components`) | 8.9 | 11.9 | 9.1 | 12.0 |
| activation packs (`pack_half_rows`) | 103.8 | 101.9 | 105.8 | 105.6 |
| HC down projections (cuBLAS, F16, unchanged) | 39.8 | 39.8 | 40.5 | 40.7 |
| routed experts (`matrix_half_ws`) | 242.3 | 242.2 | 245.2 | 245.8 |
| delta-net scan (`gdn_scan_pipe`) | 91.1 | 91.1 | 92.4 | 92.7 |
| everything else | 426.5 | 425.5 | 450.5 | 450.3 |
| **chunk** | **1,440.7** | **1,363.3** | **1,478.5** | **1,406.9** |

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64 --show-output`: prefill rate of the newest 32K interval and steady greedy decode rate at each frontier, processes main, new, new, main; raw data in [`end-to-end.csv`](end-to-end.csv). (These runs used the build before the last change to the dispatch rule, which takes the same kernels for every chunk of this sweep.)

| Context | Prefill main | new | Change | Decode main | new |
|---:|---:|---:|---:|---:|---:|
| 32K | 5,856 | 6,153 | +5.1% | 170.7 | 170.6 |
| 64K | 5,856 | 6,129 | +4.7% | 169.7 | 169.7 |
| 96K | 5,817 | 6,094 | +4.8% | 169.1 | 169.1 |
| 128K | 5,780 | 6,052 | +4.7% | 168.5 | 168.4 |
| 160K | 5,744 | 6,025 | +4.9% | 168.2 | 168.2 |
| 192K | 5,716 | 5,993 | +4.8% | 168.8 | 168.7 |
| 224K | 5,688 | 5,970 | +5.0% | 168.2 | 168.1 |
| 256K | 5,660 | 5,953 | +5.2% | 167.2 | 167.1 |

All four processes give the same greedy text at all eight frontiers. Short contexts, prefill only, frontiers 2K to 32K (each row the newest interval), processes main, new, new, main, main, new ([`short-context.csv`](short-context.csv)):

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 4,930 | 5,054 | +2.5% |
| 4K | 5,521 | 5,515 | −0.1% |
| 8K | 5,812 | 6,050 | +4.1% |
| 16K | 5,890 | 6,247 | +6.0% |
| 32K | 5,866 | 6,189 | +5.5% |

The 2K and 4K rows are 2048-row chunks, where only the projections with 1,536 or more rows (of a tile) leave cuBLAS. Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: −0.04% and −0.06% plain, −0.00% and +0.03% MTP.

## The kernel

A Q8 projection of a prefill chunk split its input and weights into scaled half components (hi and lo, with power-of-two row scales) and ran three cuBLAS GEMMs into the output (2^-12 · wl·xh, then 2^-12 · wh·xl added, then wh·xh added), then `dense_rescale2` multiplied it by the two row scales: the output was written four times and read three times, and the weights went through in 32 MB tiles. `dense_split_tma` computes the three products of a 128-row by 96-token tile in one block, each an m16n8k16 chain over k in order (three accumulators per output), and its epilogue repeats the GEMMs' accumulation and the rescale's (out · xs) · ws before the one store. Thread 0 brings each 32-k stage of the four operands in with the tensor memory accelerator (2D tensor maps, the 64-byte swizzle the ldmatrix addressing expects), refilling the stage the warps finished a step earlier, and mbarriers hand the three stages over, so the eight warps only issue `ldmatrix` and `mma.sync`. The projections of one input (a delta-net layer's qkv and gate, an attention layer's keys, values and indexer projections, the shared expert's gate and up) share a launch, their rows packed together, and blocks walk groups of row tiles (up to 48 MB of weights) across all token tiles so the group's weights stay in L2: without that order the 168 MB of a qkv-and-gate launch went back to DRAM for every token tile and the kernel was no faster than cuBLAS (7.29 against 5.93 ms in the chunk). Microbenchmarks in [`microbench.csv`](microbench.csv): 6,587 → 5,569 µs for a delta-net layer's qkv and gate, 2,488 → 2,318 µs for its output projection, 366 → 248 µs for the shared expert's down projection, packs included.

## Findings

- Where cuBLAS does not split k, its hss GEMM (f16 inputs, FP32 accumulation, `CUBLAS_GEMM_DEFAULT`) equals an in-order m16n8k16 chain per output bit for bit. It splits k when the output is small: of 702 shapes with K = 640, 2560 and 6144 ([`cublas-order.csv`](cublas-order.csv)), every one with rows × tokens ≥ 2.3 M was exact and the largest that split was K = 6144, 3,264 rows by 700 tokens. The dispatch asks for 3 M per cuBLAS tile, so a full chunk takes the new kernel for every tile except the indexer keys' 128 rows. cuBLASLt's top heuristic does not predict the split: 68 shapes it reports without split k differ.
- Every product of two halves is a multiple of 2^-48, so no partial sum of the three GEMMs is subnormal and the flush-to-zero of the fast-math build cannot change the accumulation.
- `matrix_half_ws` is not held back by how it loads its weights. Loading every row's whole superblock with consecutive threads reading consecutive 16-byte pieces (a 144-byte piece per row reaches 5.5 TB/s from L2 against 0.9 TB/s for one 16-byte piece per row) made it slower (gate/up 2.74 → 2.81 ms, down 1.81 → 1.97 ms). With clock64 timers, its consumer warps spend 89% of their time in the multiply loop and under 1% waiting on producers, and the producers under 1% waiting on consumers; the loop runs at 40% of the HMMA rate. Removing parts in compiled variants (2.94 ms with timers): the weight loads → 2.36, the epilogue's stores → 2.40, the dequant → 2.60, the activation loads → 2.70, all producer work → 1.94 ms. The cost is the sum of everything that shares the SM with the HMMAs: a loop of 12 `ldmatrix` and 32 HMMAs per step runs at 96% of peak alone, 85% with a barrier per step, 76% with the A stores, 62% with all the kernel's memory traffic and producer arithmetic (a microbenchmark of one 512-thread block per SM: 1,046, 1,200, 1,340 and 1,648 cycles a step against 1,024). Getting closer to the tensor time needs the dequant in the multiplying warps (no A stage round trip, a barrier per many k steps).
- The fused projection kernel draws more power than the GEMMs it replaces; a 32K prefill runs at 2,440 to 2,470 MHz and 556 W against 2,485 to 2,510 MHz and 555 to 576 W for main.

Dropped, with numbers:

- `matrix_half_ws` with unit loads (above).
- `dense_split_pipe`: the same tiles loaded by every thread with cp.async and a `__syncthreads` per stage: 4,050 µs against 4,139 for cuBLAS (qkv), 2,948 against 2,488 (K = 6144); the copy addresses computed once per block changed nothing.
- A producer warp for the copies (nine warps): one sub-partition then holds three warps and its 16K registers cap each thread at 168, so the kernel spilled.
- The group's weight budget at 12, 24, 48 and 96 MB: within noise of each other.
- `dense_split_tma3` (on main `19137f2`): the chains split over twelve warps on 128 × 128 tiles, eight warps taking wl·xh and wh·xh on 32 rows by 64 tokens and four taking wh·xl on 64 × 64, the wh·xl sums through shared memory: exact; 1% to 7% faster in the microbenchmarks, the chunk 1,206.0 → 1,207.0 ms, prefill +0.7% to +0.9%.
