# Qwen CUDA prefill hyper-connection mix, RTX PRO 6000 Blackwell

One commit on top of main (`c2cfecd`, after the fused projections in [`../qwen-cuda-6000-prefill-fusion`](../qwen-cuda-6000-prefill-fusion/README.md)) that runs a prefill chunk's hyper-connection mixes in four launches instead of seven, so the 335 MB of each mix's normalized streams are neither stored nor read back. Same host and model (RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage), 600 W. Every output is computed with the same operations in the same order as before; the kernel tests compare the two byte for byte, and the greedy text matches main at every frontier to 256K.

| Build | Commit | Change |
|---|---|---|
| main | `c2cfecd` | The parent. |
| new | `a5ad8d2` | `hc_norm_split`, `hc_up_rows`. |

## Where a prefill chunk went

A mix (two a layer: before the attention or delta-net block and before the experts) normalized the four 2560-wide streams of each token (`hc_norm_prefill`, writing them as `xn`), split `xn` into half components for the down projection (`pack_half_rows` over 10,240 values a token), ran the down projection (two cuBLAS GEMMs to rank 320 and a rescale), activated it (`hc_lo`), ran the up projection back to 4 × 2560 (`dense_f16_pipe`) and mixed the streams (`hc_rows`: the sum of sigmoid(up) · xn over the streams). Nsight Systems, an 8192-row chunk at 24,576 and at 245,760 tokens of context, incremental cost per kernel; all kernels in [`kernels.csv`](kernels.csv):

| Kernel, ms per chunk | main, 24,576 | new, 24,576 | main, 245,760 | new, 245,760 |
|---|---:|---:|---:|---:|
| norm (`hc_norm_prefill` → `hc_norm_split`) | 43.5 | 51.1 | 43.6 | 51.5 |
| down projection's pack (`pack_half_rows`) | 43.1 | — | 42.8 | — |
| down projection (cuBLAS, unchanged) | 40.7 | 42.4 | 41.6 | 43.3 |
| up projection and mix (`dense_f16_pipe` and `hc_rows` → `hc_up_rows`) | 87.8 | 55.9 | 88.5 | 57.0 |
| activation, up projection's pack, combine (unchanged) | 26.4 | 26.4 | 26.4 | 26.4 |
| everything else | 1,119.5 | 1,124.1 | 1,157.9 | 1,161.6 |
| **chunk** | **1,360.9** | **1,299.9** | **1,400.8** | **1,339.8** |

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64 --show-output`: prefill rate of the newest 32K interval and steady greedy decode rate at each frontier, processes main, new, new, main; raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Prefill main | new | Change | Decode main | new |
|---:|---:|---:|---:|---:|---:|
| 32K | 6,214 | 6,461 | +4.0% | 170.7 | 170.7 |
| 64K | 6,195 | 6,443 | +4.0% | 169.8 | 169.6 |
| 96K | 6,146 | 6,391 | +4.0% | 169.1 | 169.0 |
| 128K | 6,112 | 6,352 | +3.9% | 168.4 | 168.4 |
| 160K | 6,079 | 6,311 | +3.8% | 168.2 | 168.1 |
| 192K | 6,049 | 6,276 | +3.7% | 168.8 | 168.7 |
| 224K | 6,011 | 6,252 | +4.0% | 168.2 | 168.1 |
| 256K | 5,980 | 6,228 | +4.2% | 167.2 | 167.1 |

All four processes give the same greedy text at all eight frontiers. Short contexts, prefill only, frontiers 2K to 32K (each row the newest interval), processes main, new, new, main, main, new ([`short-context.csv`](short-context.csv)):

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 5,122 | 5,206 | +1.6% |
| 4K | 5,594 | 5,691 | +1.7% |
| 8K | 6,194 | 6,458 | +4.3% |
| 16K | 6,425 | 6,718 | +4.6% |
| 32K | 6,376 | 6,664 | +4.5% |

Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical: −0.03% and −0.02% plain, +0.01% and −0.11% MTP; decode and its verify rows never take the new kernels.

## The kernels

`hc_norm_split` gives a token's four streams to one 1,024-thread block, each 256-thread group doing what a `hc_norm_prefill` block did (the same partial sums of squares, the same injection partials in the same order), then splits the token's 10,240 values for the down projection as `pack_half_rows` does (their maximum, the power-of-two row scale, the half and the scaled residual) without storing them. `hc_up_rows` computes the up projection with `dense_f16_pipe`'s products and rounding on tiles of 32 columns of every stream by 64 tokens, arranged so a warp's four m16 tiles are the four streams of the same 16 columns: each thread holds the four values `hc_rows` sums for its outputs and sums them in the same order, after recomputing each normalized value from the stream as the norm computed it (R · inv · gamma, the stream's inverse RMS kept by `hc_norm_split`). Its 32-k stages take 32 KB, so two blocks share an SM and one block's epilogue reads run beside the other's products; the rows it reads start for L2 at the top of the block. Microbenchmark ([`microbench.csv`](microbench.csv)): 868 → 482 µs for the up projection and mix of a chunk. Mixes whose down projection has K ≤ 512, or whose weights are not F16, keep the old kernels.

## Findings

- The mixes are bound by DRAM. Each moved the 335 MB of a chunk's streams eight times (read the residual, write `xn`, read it to pack, write the halves, read them in the down GEMMs, write the up projection, read it and `xn` to mix); the new path reads the residual twice and writes the halves once, the down GEMMs read them, and the mix is written: `hc_norm_split` runs at 84% of the 0.45 ms that reading the residual and writing the halves take at 1.5 TB/s.

Dropped, with numbers:

- `hc_up_rows` with 128-token tiles and sigmoid(up) through shared memory (64 KB), one block an SM: 639 µs, 529 µs with the L2 prefetch.
