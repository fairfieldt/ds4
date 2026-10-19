# DeepSeek V4.1 Flash 1017 Q2 on one RTX PRO 6000

Measured 2026-10-19 on voyage: RTX PRO 6000 Blackwell Workstation Edition (96 GB, 600 W), CUDA 13.4, driver 615.71.09, Ryzen 9 9950X, 192 GB DDR5, Samsung 990 PRO. Model `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` from `antirez/deepseek-v4.1-flash-gguf` (SHA-256 `5296e85f…ce93a`, 340.6 GiB including 188.8 GiB of disk-only Engram). Build: main after the October 19 upstream rebase, plus the CUDA SSD host tier (`DS4_CUDA_SSD_HOST_GB`).

The model needs SSD streaming here: 9.4 GiB of non-routed weights and a 59.6 GiB expert cache (6,425 of 15,360 experts, 9.49 MiB each) fit in VRAM; the 142.4 GiB of routed experts do not. Stock streaming reads every cache miss from the GGUF. The host tier pins all 40 layers' routed experts in RAM, so a miss is a PCIe copy.

```sh
DS4_CUDA_SSD_HOST_GB=auto ds4-bench --cuda -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --gen-tokens 256 --ctx-alloc 32768
```

Greedy decode, no speculation, automatic cache sizing, `DS4_CUDA_SSD_STATS=1` for the hit and miss counts. Each row is one run.

## Speed

| | Stock SSD streaming | Host tier |
|---|---:|---:|
| 2K prefill | 74.8 t/s | 509.6 t/s |
| Decode after 2K (256 tokens) | 20.7 t/s | 48.4 t/s |
| Decode cache hits | 94.1% | 94.1% |
| Cost per miss | 2.20 ms | 0.24 ms |
| Cold CLI, 17-token prompt, 64 tokens | 9.4 t/s | 39.9 t/s |

The greedy CLI text is byte-identical between the two. With `DS4_SSD_AUTO_CACHE_PCT=95` the cache grows only to 6,787 slots, held back by the runtime's 8 GiB VRAM reserve: 518.2 t/s prefill, 49.3 t/s decode ([`host-2k-cache95.csv`](host-2k-cache95.csv)).

Longer prompts, host tier, 64K allocated context ([`host-8k-32k.csv`](host-8k-32k.csv)):

| Prompt | Prefill | Decode after (128 tokens) |
|---|---:|---:|
| 8K fresh | 1,370 t/s | 46.3 t/s |
| 8K to 32K continued | 1,465 t/s | 43.8 t/s |

`ds4-server --cuda --ssd-streaming --ctx 65536` with the host tier was ready 16 s after start (warm page cache) and decoded at 52-53 t/s. Pinning 142.4 GiB takes 30 s from a cold page cache and 11-16 s warm.

## Where decode time goes

nsys over 64 decode tokens after a 512-token prompt (host tier): 18.9 ms per token, the GPU busy 15.4 ms of it, about 2,550 kernel launches.

| Kernel | Per token |
|---|---:|
| `matmul_q8_0_preq_warp8` (attention, shared experts) | 4.30 ms |
| `moe_gate_up_mid_decode_lut_qwarp32` (IQ2_XXS gate/up) | 3.20 ms |
| `moe_down_sum_qwarp32` (Q2_K down) | 1.10 ms |
| `rms_norm_weight` | 1.07 ms |
| `grouped_q8_0_a_preq_warp8` | 0.98 ms |
| Host-to-device expert copies | 1.49 ms |

The gate/up kernel reads 35 MB per layer in 80 us, about 0.44 TB/s, a quarter of the card's bandwidth. It is the clearest kernel to optimize next.

## Cache size

The stock 2K run's expert requests, recorded with `DS4_CUDA_SSD_TRACE=FILE` and replayed through LRU caches of other sizes by [`sim_cache.py`](sim_cache.py):

| Slots | GiB | Decode hits |
|---:|---:|---:|
| 4,000 | 37.1 | 90.0% |
| 6,425 | 59.5 | 94.1% |
| 8,000 | 74.1 | 95.9% |
| 10,000 | 92.7 | 98.3% |

With 0.24 ms misses, the extra VRAM slots that would fit are worth about 1 ms per token.

## Quality

V4.1 general-100 fixture, 4K context, host tier ([`general-100.tsv`](general-100.tsv)): NLL 0.392, first-token match 80/100, API top-1 2,689/2,994. The fixture's continuations came from the API on 2026-09-10, before DeepSeek released the 1017 checkpoint, so this is a cross-checkpoint score. The earlier Q2's band on a Spark is NLL 0.363 and 2,696/2,994.

The V4.1 CUDA tests pass on sm_120: `test_deepseek41_cuda` (and `--attention-output-large`), `test_cuda_q8_rows`, `test_cuda_ssd_cache` (also under Compute Sanitizer and with `--prefetch-exit`), `test_cuda_ssd_batch`.

## Roofline

Measured ceilings on this machine: GPU DRAM 1,646 GB/s, host-to-device copies 44 GB/s (the speed of the DDR5-3600 host memory, four 48 GB DIMMs), INT8 tensor cores 1,020 TOPS, FP16 523 TFLOPS, NVMe 7.1 GB/s.

| | Q2 (this file) | 1017 MXFP4 (native precision) |
|---|---:|---:|
| Decode reads per token | 11.1 GB | 13.2 GB |
| Routed experts | 142.4 GiB | 268.9 GiB |
| Decode floor, every expert in VRAM | 6.8 ms (148 t/s) | 8.1 ms (124 t/s) |
| Prefill matmul work per token | 30.9 GFLOP | 30.9 GFLOP |

With the 60 GiB VRAM expert cache, decode also pays 2 to 7 ms per token in cache misses (0.24 ms each over PCIe), so its practical floor is about 9 to 10 ms. A prefill sweep copies every routed expert not in VRAM once, about 2 to 3.5 s.

## October 19 series

Eight changes, each measured ABBA against its predecessor and kept only if the frontier logits stayed byte-identical:

| Change | Effect |
|---|---|
| Read-ahead copies from the host tier in 8 MiB pieces | prefill +1.6..3.6% |
| Queue decode layers instead of draining the GPU per layer | decode +0.7% |
| IQ2_XXS decode gate/up, one warp per row (3.20 -> 1.33 ms/token) | decode +8..9% |
| Prefill sweeps keep what was cached before them | turns: prefill +24%, decode +11% |
| F16 projection rows in one cuBLASLt batch | prefill +6.5% (8K), +14.7% (32K) |
| RMS norm inputs held in registers | decode +4.6..5.6% |
| Indexer scores, one thread per score for large grids | prefill +7.5% (32K) |
| Exact Q8 MMA prefill kernel, two token groups per warp | prefill +4.2% (32K) |

Main against the series, ABBA in one session ([`series-abba.csv`](series-abba.csv)); "turns" is five 2K appends with 64 decode tokens after each:

| Workload | Prefill, main | Series | Decode, main | Series |
|---|---:|---:|---:|---:|
| 2K prompt, 128 tokens | 523 t/s | 531 t/s | 45.5 t/s | 52.2 t/s (+14.6%) |
| 8K prompt, 128 tokens | 1,396 t/s | 1,625 t/s (+16.4%) | 45.9 t/s | 52.8 t/s (+14.9%) |
| Turns | 577 t/s | 771 t/s (+33.7%) | 42.3 t/s | 52.8 t/s (+24.8%) |
| 32K prompt, 32 tokens | 2,296 t/s | 2,922 t/s (+27.3%) | 38.1 t/s | 42.5 t/s (+11.8%) |

The general-100 scores are byte-identical to main's. Bit-for-bit tests: `test_cuda_moe_decode_gate`, `test_cuda_v41_projection_rows`, `test_cuda_norm_reg`, `test_cuda_v41_index_scores`, `test_cuda_q8_mma_groups`.

Measured and dropped: a warp-per-row Q2_K down-sum kernel (-0.6%); prefetching the next layer's experts from its router applied to the current layer's input (43% of misses predicted, 65% of guesses already cached); CUDA graphs for the position-independent parts of each streaming decode layer (-1.2..1.4%); four token groups per warp in the exact Q8 kernel (162 registers, slower). Plain cuBLAS strided or pointer batching does not reproduce one-row results, nor can cuBLASLt batch the 5120 -> 32 indexer projection, which stays one call per row.

Left at 32K, from nsys: the exact attention-output kernel (about 2 s, bound by per-block float work rather than its tensor-core products), the routed MoE GEMMs and their per-sweep repacks (about 3 s), per-row indexer projections and cuBLASLt's per-row split-K reductions (about 1 s plus host overhead), and read-ahead copies not hidden by compute (about 2 s). Decode spends about 5.8 ms per token in small kernels and 3 to 7 ms in cache misses.
