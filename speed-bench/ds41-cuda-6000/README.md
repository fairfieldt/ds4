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
