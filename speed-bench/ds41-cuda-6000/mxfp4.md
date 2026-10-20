# DeepSeek V4.1 Flash 1017 MXFP4 on one RTX PRO 6000

Measured 2026-10-19/20 on voyage, the machine in [README.md](README.md), with the GPU power limit at 450 W (the Q2 record ran at 600 W). Model: antirez's `DeepSeek-V4.1-Flash-1017-MXFP4-experts` split parts from `antirez/deepseek-v4.1-flash-gguf`, merged into one 467.16 GiB GGUF with [`merge_split_gguf.py`](../../gguf-tools/merge_split_gguf.py). Every part matched the Hub's SHA-256 (the two Engram parts were rebuilt from the 1017 Q2 file), and every merged tensor was re-read against its source.

The routed experts take 268.9 GiB, 17.93 MiB each. They fit neither in VRAM nor in RAM: the VRAM cache holds about 3,900 of the 15,360 experts, and `DS4_CUDA_SSD_HOST_GB=auto` pins 24 of the 40 layers (161.4 GiB). A miss in a pinned layer is a 0.43 ms copy from RAM; any other miss reads the SSD, about 3 ms.

```sh
DS4_CUDA_SSD_HOST_GB=auto ds4-bench --cuda -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --gen-tokens 512 --ctx-alloc 32768
```

Greedy decode, no speculation, automatic cache sizing, `DS4_CUDA_SSD_STATS=1`.

## Speed

2K prompt, 512 tokens, after each change (ABBA where marked; every run is in [`mxfp4-steps.csv`](mxfp4-steps.csv), with the miss counts from `DS4_CUDA_SSD_STATS`):

| Build | Decode | VRAM slots | Misses per token, RAM / SSD |
|---|---:|---:|---:|
| main | 15.2 t/s | 3,040 | 17.3 / 10.5 |
| Cost-aware VRAM eviction | 17.0 t/s | 3,040 | 31.4 / 6.5 |
| A 3,600-slot cache request (old kernels; capped at 3,591) | 19.2 t/s | 3,591 | 24.1 / 5.5 |
| One-row MXFP4 decode kernels, cache sized from free VRAM | 22.3 t/s | 3,521 | 24.6 / 5.6 |
| Three parallel SSD readers (ABBA) | 23.4 t/s | 3,521 | 24.6 / 5.6 |
| Prefill staging inside the cache (ABBA) | 25.1 t/s | 3,905 | 23.5 / 4.8 |

The series head measured 24.9 and 24.7 t/s in the final validation session. Main against the series, ABBA in one session ([`mxfp4-abba.csv`](mxfp4-abba.csv)):

| Workload | Prefill, main | Series | Decode, main | Series |
|---|---:|---:|---:|---:|
| 2K prompt, 128 tokens | 83.4 t/s | 84.8 t/s | 12.8 t/s | 19.7 t/s (+54%) |
| 8K prompt, 128 tokens | 305 t/s | 311 t/s | 15.3 t/s | 23.8 t/s (+55%) |

The short runs decode slower than the 512-token one: they start with fewer cached experts. A 32K prompt at 64K context prefilled at 853 t/s and decoded 64 tokens at 21.9 t/s; peak VRAM was 96.1 of 97.9 GB. Prefill is bound by expert sweeps: each 8K-token chunk reads most of the 269 GiB of experts, about 9 to 25 s per chunk.

The frontier logits are byte-identical to main's on both workloads. Q2 gains from the larger cache; its logits and general-100 scores are byte-identical to main's:

| Q2 workload, 450 W | Prefill, main | Series | Decode, main | Series |
|---|---:|---:|---:|---:|
| 2K prompt, 128 tokens | 520 t/s | 515 t/s | 51.5 t/s | 53.8 t/s (+4.4%) |
| 8K prompt, 128 tokens | 1,553 t/s | 1,592 t/s | 52.1 t/s | 53.6 t/s (+2.9%) |
| Turns | 750 t/s | 781 t/s | 52.7 t/s | 53.7 t/s (+1.9%) |
| 32K prompt, 32 tokens | 2,597 t/s | 2,721 t/s | 42.3 t/s | 44.0 t/s (+4.1%) |

`ds4-server --cuda --ssd-streaming --ctx 65536` with the MXFP4 GGUF was ready 151 s after start, from a page cache that held part of the file, and decoded a short chat at about 35 t/s.

## Decode kernels

MXFP4 decode runs `ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel` and `ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel`, tuned on the Spark. On this card the down kernel gave each half-warp eight rows, 80 CTAs of four warps for 188 SMs; nsys put the two kernels at 10.6 ms of the 21 ms of GPU time per token. One row per half-warp keeps every row's arithmetic and order, so the bits match (`test_cuda_mxfp4_decode_moe`). V4.1 shapes, weights in VRAM:

| Gate/up rows | Down rows | Per decode MoE | Expert bandwidth |
|---:|---:|---:|---:|
| 4 | 8 (before) | 265.8 us | 424 GB/s |
| 4 | 1 | 103.6 us | 1,089 GB/s |
| 1 | 1 (now) | 81.4 us | 1,385 GB/s |

## Cache misses

Decode requests from three runs (the 2K bench, a 768-token C program, a 768-token essay), recorded with `DS4_CUDA_SSD_TRACE` and replayed by [`sim_tier.py`](sim_tier.py). Against the runs it replays, it matched the measured hit rate to 0.01% and the cost-aware eviction's miss counts within 5%. Miss cost per token at the final 3,905 VRAM slots and the 24 pinned layers (9,216 experts), with 0.43 ms RAM and 3.0 ms SSD misses:

| Host RAM holds | VRAM eviction | 2K bench | All three runs |
|---|---|---:|---:|
| Layers 0-23 | LRU | 29.2 ms | 42.2 ms |
| Layers 0-23 (kept) | cost-aware | 24.2 ms | 31.3 ms |
| An LRU of the experts that missed VRAM | LRU | 23.3 ms | 31.1 ms |
| Use counts, starting from layers 0-23 | cost-aware | 27.2 ms | 30.0 ms |
| The most used experts, chosen with hindsight | LRU | 9.2 ms | 26.1 ms |

The three runs touched 13,602 distinct experts in 2,132 tokens, nearly the whole model, so even hindsight cannot keep decode off the SSD across workloads. A RAM tier that follows decode was not built: the replay puts it within about 1 ms per token of the pinned layers once VRAM eviction is cost-aware. At the original 3,040 slots, cost-aware eviction alone took the 2K bench from 43.0 to 36.0 ms.

At 25 t/s a token takes 40 ms: about 25 ms of cache misses (23.5 RAM copies, 10 ms; 4.8 SSD reads, 14 ms) and 15 ms of GPU work and launches.

## Measured and not kept

Six SSD readers decoded no faster than three (23.2 against 23.4 t/s). The Spark's staged gate/up kernel ran at 254 us per decode MoE, against 81 us for one row per half-warp. An 8 GiB VRAM margin, the old default, left 0.8 GB free at the 32K peak; the cache now keeps 2.5 GiB free plus the layer that prefill stages through.

## Left

Cache misses are 60% of decode time. RAM copies run at 44 GB/s, the speed of this host's DDR5-3600, and the layer that needs them waits for them. Candidates: start the copies while the shared expert computes; prefetch a layer's likely experts during the previous layer (on Q2 its router, applied to the current layer's input, predicted 43% of misses); pin experts by use rather than whole layers.
