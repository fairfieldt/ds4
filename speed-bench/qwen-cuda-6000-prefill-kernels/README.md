# Qwen CUDA prefill kernels, RTX PRO 6000 Blackwell

Three commits on top of main (`6e527ba`, after the tensor-core indexer series in [`../qwen-cuda-6000-tc-scorer`](../qwen-cuda-6000-tc-scorer/README.md)) that rework the context-independent kernels of a prefill chunk: the delta-net convolutions and scan, the routed expert GEMMs and the dense projections' input packing. Same host and model as the earlier series (RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage), at the 600 W limit the GPU now runs at. Every new kernel computes each output with the same operations in the same order as the kernel it replaces, the kernel tests compare the two byte for byte, and the greedy text matches main at every frontier to 256K.

| Build | Commit | Change |
|---|---|---|
| main | `6e527ba` | The parent. |
| convs + scan | `023f9ad` | `conv_seg`, `ngram_conv_seg` and `gdn_scan_pipe`. |
| experts | `472878d` | `matrix_half_ws`. |
| new | `15502a8` | `pack_half_rows`, one pack per shared input, `dense_f16_pipe`. |

## Where a prefill chunk went

Nsight Systems, an 8192-row chunk at 24,576 and at 245,760 tokens of context, incremental cost per kernel (end_i − max(end_<i)); all kernels in [`kernels.csv`](kernels.csv):

| Kernel, ms per chunk | main, 24,576 | new, 24,576 | main, 245,760 | new, 245,760 |
|---|---:|---:|---:|---:|
| routed experts (`matrix_half_tile` → `matrix_half_ws` + half pack) | 674.6 | 252.3 | 674.0 | 254.9 |
| delta-net scan (`gdn_scan` → `gdn_scan_pipe`) | 499.9 | 91.3 | 500.7 | 92.4 |
| delta-net conv (`conv` → `conv_seg`) | 143.2 | 17.9 | 143.4 | 17.9 |
| projection input packs (`pack_half_components`, `pack_half_rows`) | 130.6 | 113.1 | 127.4 | 113.7 |
| HC up projections (`dense_f16_components` → `dense_f16_pipe`) | 64.6 | 44.5 | 65.7 | 45.0 |
| n-gram conv (`ngram_conv` → `ngram_conv_seg`) | 8.9 | 1.2 | 11.3 | 1.2 |
| cuBLAS GEMMs (unchanged) | 514.9 | 557.8 | 523.5 | 565.0 |
| everything else | 355.0 | 364.8 | 375.1 | 387.9 |
| **chunk** | **2,391.6** | **1,442.9** | **2,420.9** | **1,478.1** |

The untouched kernels ran up to 10% slower in the new build's captures (cuBLAS 8.3%, `attention_group_pipe` 9.6%): with less time waiting on loads the GPU draws more power and clocks down under the 600 W limit, an average SM clock of 2,447 MHz at 558 W during a 32K prefill against 2,683 MHz at 524 W for main.

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64 --show-output`: prefill rate of the newest 32K interval (8192-row chunks) and steady greedy decode rate at each frontier, processes main, new, new, main; raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Prefill main | new | Change | Decode main | new |
|---:|---:|---:|---:|---:|---:|
| 32K | 3,432 | 5,802 | +69.1% | 170.6 | 170.7 |
| 64K | 3,441 | 5,797 | +68.5% | 169.5 | 169.6 |
| 96K | 3,433 | 5,763 | +67.9% | 168.9 | 169.1 |
| 128K | 3,424 | 5,731 | +67.4% | 168.4 | 168.5 |
| 160K | 3,414 | 5,706 | +67.1% | 168.1 | 168.2 |
| 192K | 3,405 | 5,684 | +66.9% | 168.7 | 168.7 |
| 224K | 3,397 | 5,659 | +66.6% | 168.1 | 168.2 |
| 256K | 3,388 | 5,640 | +66.5% | 167.1 | 167.2 |

All four processes give the same greedy text at all eight frontiers. Short contexts, prefill only, frontiers 2K to 32K (each row the newest interval), processes main, new, new, main, main, new ([`short-context.csv`](short-context.csv)):

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 3,026 | 4,957 | +63.8% |
| 4K | 3,233 | 5,587 | +72.8% |
| 8K | 3,409 | 5,942 | +74.3% |
| 16K | 3,478 | 6,049 | +73.9% |
| 32K | 3,469 | 6,029 | +73.8% |

Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: −0.17% and −0.18% plain, +0.00% and +0.05% MTP. Single-token decode and the MTP verify rows (up to three) never reach the new kernels; the plain difference held across all four runs of each build and is not explained.

## The kernels

Microbenchmarks of one 8192-row chunk's launch, in the kernel test binary with the weights copied to the device ([`microbench.csv`](microbench.csv)).

**Convolutions.** `conv` walked a chunk with one thread per channel, 40 blocks for 8192 tokens of 10,240 channels, each thread waiting on a dependent load per token: 3.96 ms per GDN layer. `conv_seg` gives each channel's tokens to the 16 warps of a block in segments; every warp reads the three inputs before its segment (or the history) ahead of a barrier, then convolves the segment in place, loading eight tokens before storing any: 0.52 ms, near the 0.42 ms of moving the chunk through DRAM once. The PLE layer's n-gram conv (dilation 3, nine inputs of history) takes the same scheme.

**Delta-net scan.** `gdn_scan` keeps each state row in registers and walks the chunk's tokens, but reloaded the token's q, k, v and gates after the previous token's output store: every step waited on a memory latency, 13.9 ms per layer. `gdn_scan_pipe` stages them in shared memory two groups of eight tokens ahead (cp.async, 26 KB per block so all 384 blocks of a layer fit the GPU at once), gives a thread the quarters of lanes l and l + 16 of two rows, so the butterfly's first level (lane l's partial plus lane l^16's, the same sum either way round) is an add in registers, and finishes each token's output butterfly beside the next token's, off the state's dependency chain: 2.34 ms.

**Expert GEMMs.** `matrix_half_tile` dequantizes its weights through `value4`'s byte loads and reads each activation as a float, then multiplies, one k step at a time, so the tensor cores waited on dependent loads: 8.41 ms for one chunk's Q4_K gate/up GEMMs and 5.84 ms for the MXFP4 down GEMMs per layer. `matrix_half_ws` runs one persistent block per SM over tiles of 256 A rows (128 gate rows and the same up rows, or 256 down rows) by 64 tokens. Eight producer warps load the weights two k steps ahead, a thread per A row, dequantize them into one of two shared-memory A stages, and copy the activations (rounded to halves once per call, as `matrix_half_tile` rounds each one it loads) into one of three B stages a step ahead; eight consumer warps only issue `ldmatrix` and `mma.sync`, loading the next 16 k's fragments during the current products. Named barriers hand the stages over, and consumers skip the 8-token columns past an expert's last token. The operands are the same halves: `value4`'s arithmetic, except that a nibble becomes a float as the bit pattern 2^23 + n minus 2^23 (a byte permute and an add, exact) instead of the integer conversion, which issues at half rate. Each output takes the same m16n8k16 chain over k in order: 2.90 and 1.81 ms.

**Projections.** The Q8 and F16 projections split their FP32 inputs into scaled half components for cuBLAS. `pack_half_components` read each row twice with scalar loads; `pack_half_rows` keeps the row in registers between the maximum and the split. The projections of one input (the keys, values and indexer projections of an attention layer, qkv and gate of a GDN layer, the shared expert's gate and up) packed it once each; `multi_gemv` packs it once for all, each weight's tiling and GEMMs unchanged: 518 packs of a chunk become 410 (130.6 → 113.1 ms with the weight packs). `dense_f16_components` ran the HC up projections (K = 320 to 10,240 outputs) as a load-wait-multiply cycle per 64 × 64 tile; `dense_f16_pipe` takes 128 × 128 tiles with the next k step loading during the products: 652 → 424 µs per projection.

## Findings

- `mma.sync.m16n8k16` with FP32 accumulation peaks at 547 TFLOP/s on this GPU (16 cycles per instruction per SM sub-partition, measured with independent accumulators from one warp per sub-partition up). At that rate the expert GEMMs' tiles would take 48% (gate/up) and 38% (down) of their time now, cuBLAS's split-precision projections about half of theirs.
- L2-resident loads: 11.2 TB/s coalesced, but 1.4 TB/s when each lane reads 16 bytes of its own row 1440 bytes apart (a Q4_K expert row), 2.7 TB/s for 32 or 48 bytes per row and 3.9 TB/s for 80 bytes; 3.2 to 6.4 TB/s at a 352-byte stride. That is the likely limit of `matrix_half_ws`, at twice its tensor-core time: its producers fetch about 275 million such pieces per gate/up launch. Neither half alone is fast (an earlier revision: consumers with the dequant removed took 2.59 ms, producers with the products removed 2.32 ms).
- In the scan, once its loads are staged, the shuffle and shared-memory path is the limit: removing the butterflies alone took 2.23 → 1.47 ms per layer.
- Two 49 KB blocks per SM, or one 90 KB block, need `cudaSharedmemCarveoutMaxShared` and the dynamic-shared-memory opt-in.

Dropped, with numbers:

- `matrix_half_pipe`: two blocks per SM, every warp dequantizing its share of the next k step after multiplying the current one, single-buffered A: gate/up 3.33 ms, down 2.10 ms.
- The first `gdn_scan_pipe`, four stages and 128-thread blocks: 36 KB blocks fit two per SM, so 8 of 384 blocks ran in a second wave: 3.98 ms.
- `gdn_scan_pipe` with four rows per thread, the half-warp's butterflies scattered over the rows (shuffles per row 4 → 1.75) and 64-thread blocks: 2.49 against 2.23 to 2.32 ms. Fewer warps per SM left each step's dependency chain exposed.
- `matrix_half_ws` producers loading a row's whole Q4_K superblock (144 bytes, four k steps) or two MXFP4 steps (80 bytes) at a time: 2.87 and 1.85 ms, no change. Each lane still reads its own row.

What remains of the 24,576-token chunk's 1,443 ms: cuBLAS's split-precision projections 558 ms, the expert GEMMs 252 ms, the activation packs 113 ms and the scan 91 ms, and the DRAM-bound passes of the MoE reduction, the hyper-connections and the projections' rescale (about 205 ms together).
