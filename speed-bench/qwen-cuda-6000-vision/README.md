# Qwen3.8 vision on CUDA, RTX PRO 6000 Blackwell

The measurements were taken while this series was developed on the combined build that the fused-decode series ([`../qwen-cuda-6000`](../qwen-cuda-6000/README.md)) was split from (the build called the candidate there, and the base of [`../qwen-cuda-6000-roofline`](../qwen-cuda-6000-roofline/README.md)), with this series' commits applied on top of it. The vision encoder, image decoding and resize code there is the code this series starts from; the text-decode changes of the fused-decode and roofline series do not touch it. "Before" is that build with only the latency bench and the repeat option of `tests/test_qwen4_vision` (`e9de627`); "after" is it with the whole series (`7d8abb5`). The CSVs label these builds `before` and `after`, and the build with only the flash-attention commit `flash`. Commit hashes in this document and its CSVs name the commits of this series that carry the measured code. Hardware and model as in [`../qwen-cuda-6000`](../qwen-cuda-6000/README.md): RTX PRO 6000 Blackwell Workstation Edition, 450 W limit, CUDA 13.4, Ryzen 9 9950X host, Qwen3.8-Flash-Next-Q4 GGUF. The vision encoder is `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` from `ggml-org/Qwen3.8-Flash-Next-GGUF` (0.617 GB, SHA-256 `b2e9b5e4…`): 27 layers, width 1,152, 16 heads of 72, FFN 4,304; Q8_0 qkv/out/up and merger weights, F16 `ffn_down`, F32 patch embedding. One image token is four 16x16 patches, so 64 / 256 / 1,024 image tokens are 256 / 1,024 / 4,096 patches.

## Encoder

`tests/test_qwen4_vision` with a repeat count: one process, eleven encodes of the same exact-size PNG (`earth` resized to 256, 512 and 1,024 pixels square), image decoding excluded, preprocessing, uploads and the read-back of the embeddings included. The first encode of a process is listed separately; the warm value is the median of the other ten, averaged over two rounds with the builds interleaved.

| Image tokens | Before, warm ms | After, warm ms | Speed-up | Before, first call ms | After, first call ms |
|---:|---:|---:|---:|---:|---:|
| 64 | 7.98 | 4.26 | 1.9x | 105.3 | 36.2 |
| 256 | 45.61 | 12.05 | 3.8x | 141.7 | 44.4 |
| 1,024 | 520.0 | 68.0 | 7.6x | 606.7 | 87.5 |

At 1,024 tokens the ten warm encodes of a run spread from 57 to 70 ms: the tensor-core work holds the board at its power limit and the clocks drop as it heats (the minimum over both rounds is 57.4 ms; before, 517.5 ms). Raw values: [`encode.csv`](encode.csv).

### Per commit

Same method, one build per commit, two interleaved rounds ([`encode-per-commit.csv`](encode-per-commit.csv)). The two host commits (`91e9dea`, `cf04151`) do not touch this measurement.

| Commit | Change | 64 tokens | 256 tokens | 1,024 tokens | First call, 64 tokens |
|---|---|---:|---:|---:|---:|
| `e9de627` | before | 7.94 | 44.19 | 507.7 | 103.7 |
| `6aaaa91` | flash attention on tensor cores | 6.43 | 15.44 | 76.2 | 102.5 |
| `5ca6dc9` | `vis_mm` projections, fused epilogues, split-k for small grids | 5.07 | 13.14 | 67.2 | 72.4 |
| `56b2ba2` | patch and position embedding on the GPU | 4.81 | 12.80 | 65.6 | 37.8 |
| `7d8abb5` | scratch arena kept across encodes | 4.23 | 11.94 | 64.9 | 35.8 |

A variant of the attention kernel with 4-warp blocks and 32-key tiles for up to 2,048 patches measured 4.11 / 12.08 / 64.85 ms (at every size: 4.11 / 11.96 / 69.6 ms) and was not kept.

### Where the time goes

Nsys, one warm encode at 1,024 tokens, incremental kernel time ([`kernel-profile.csv`](kernel-profile.csv)):

| Kernel | Before ms | After ms |
|---|---:|---:|
| attention | 468.5 (`vis_attention`, one warp per patch and head, scalar FP32) | 27.4 (`vis_flash`) |
| projections | 35.8 cuBLAS GEMMs (three FP16 products each, FP32 for the patch embedding) + 10.3 packing, rescale and bias/residual | 36.6 (`vis_mm`, epilogues and patch embedding included) |
| LayerNorm, qkv/rope, patch/position | 1.5 | 1.8 |
| GPU span | 516.2 | 65.8 |

Per layer at 4,096 patches the projections take qkv 331 us, out 151 us, up 423 us, down 418 us, about 195 TFLOP/s for the two-product form; cuBLAS reached 270–340 TFLOP/s for each of its three products. The attention pass does 245 GFLOP per layer (three products each for q.k and p.v) at 0.9–1.0 ms. At 64 tokens the GPU span fell from 7.1 ms to about 4 ms; before, 1.1 ms of it was weight packing and 2.0 ms attention.

## End to end

`speed-bench/qwen_vision_bench`: the engine with the Q4 model and the encoder loaded once; per image three fresh-session blocks after a discarded one. A block encodes the image from its file bytes (decode, preprocessing and GPU encode), prefills system prompt, image and "Describe this image in detail." without thinking, samples the first token, and decodes 128 tokens greedily the way the CLI does. TTFT is encode + prefill + first sample. Medians of three blocks ([`e2e.csv`](e2e.csv), every block in [`e2e-blocks.csv`](e2e-blocks.csv)).

| Image | Image tokens | Encode ms, before → after | Prefill ms, before → after | TTFT ms, before → after | TTFT, MTP, before → after | Decode t/s, plain | Decode t/s, MTP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256x256 PNG | 64 | 10.9 → 6.0 | 92.5 → 92.4 | 103.4 → 98.3 | 104.8 → 99.7 | 140.8 → 140.8 | 178.3 → 191.7 |
| 512x512 PNG | 256 | 54.5 → 18.8 | 166.0 → 164.1 | 220.5 → 182.9 | 223.1 → 186.4 | 140.8 → 141.0 | 181.1 → 180.3 |
| 1024x1024 PNG | 1,024 | 549.7 → 85.8 | 408.5 → 394.5 | 958.2 → 480.2 | 957.5 → 483.8 | 140.1 → 140.2 | 181.2 → 179.5 |
| 4000x3000 JPEG | 972 | 972.0 → 314.1 | 395.0 → 381.2 | 1,367.9 → 695.6 | 1,366.1 → 694.7 | 140.2 → 140.2 | 183.5 → 181.9 |

Decode speed with an image in context is unchanged (both builds predate the roofline series, so they decode at about 140 t/s plain). Greedy continuations are identical across blocks of a build; between the builds they are identical for the 1,024-token PNG and differ for the others, because the embeddings differ in the fourth decimal place and MTP acceptance follows the text (178 → 192 t/s at 64 tokens is a different reply, not a faster decoder). The encode column of the JPEG is now dominated by the host JPEG decoder (251 ms). Prefill of the image tokens is the text model's prefill; for a 94-token prompt its 92 ms are mostly the MoE prefill kernel `matrix_half_tile` (57 ms) and packing (21 ms).

### Host image path

Before the GPU sees an image the host decodes it, fingerprints the pixels (SHA-256) and resizes. Best of five on the Ryzen 9 9950X, gcc ([`host-image.csv`](host-image.csv)); outputs are bit-identical before and after for every case listed there, and on Apple silicon (clang). `tests/test_image_decode` (part of `make test`) now checks a subset at every build against the previous code's outputs: resized levels of the Qwen3.8 and GLM-5.3 preprocessing, fixture and stored PNG fingerprints, and rejection of a corrupted CRC or Adler-32.

| Case | Before ms | After ms |
|---|---:|---:|
| resize 12 MP JPEG to 1,024 Qwen tokens | 243.1 | 16.9 |
| resize 12 MP JPEG, GLM-5.3 budget | 292.7 | 32.7 |
| decode 1024x1024 photo PNG | 46.0 | 31.5 |
| decode 1400x900 screenshot PNG | 22.5 | 12.4 |
| decode 1024x768 stored (uncompressed) PNG | 21.0 | 10.4 |

The resize computes each axis's taps once and splits rows across threads (`91e9dea`); PNG decoding uses a table CRC, deferred Adler-32 reduction and a 9-bit Huffman lookup (`cf04151`). The JPEG decoder (251 ms for 12 MP) and the portable SHA-256 (about 9 ms per megapixel) were not changed.

## Accuracy

Embedding dumps of 14 cases (fixture images at 54 to 1,050 image tokens, including 640x480 at forced 64 and 1,024 tokens, a 24x16 grayscale and a 64x48 progressive JPEG, and the three exact-size PNGs) against the Metal tower, and the exact-size PNGs against the Hugging Face tower (`transformers` 5.17 `qwen4_exp`, CPU, float32, the same GGUF weights dequantized; `tests/qwen4_vision_ref.py` functions with the snapshot's `config.json` and `preprocessor_config.json` only). Max error is the largest absolute difference relative to the largest reference value ([`accuracy.csv`](accuracy.csv)).

| Build | Worst per-token cosine vs Metal | Max error vs Metal | Worst per-token cosine vs HF | Max error vs HF |
|---|---:|---:|---:|---:|
| before (`e9de627`) | 0.999999 | 5.5e-4 | 1.000000 | 1.5e-4 |
| flash attention (`6aaaa91`) | 0.999998 | 7.1e-4 | 1.000000 | 2.1e-4 |
| after (`7d8abb5`) | 0.999998 | 6.7e-4 | 1.000000 | 2.2e-4 |
| Metal | — | — | 1.000000 | 4.8e-5 |

The documented gate is a minimum per-token cosine of 0.99 against HF. Attention keeps about 22 bits of each operand (high and low FP16 parts for q, k, v and the probabilities); a single FP16 product per term measured 8.7e-3 against Metal and was not used. The projections multiply exact Q8_0 quants (or F16 weights) with both FP16 parts of the activations and accumulate in FP32.

With preprocessing included, HF and ds4 differ where the host resize departs from PIL's (identically on Metal and CUDA, unchanged here): worst per-token cosine 0.998 for a 640x480 image squeezed to 54 tokens and 0.917 for the 24x16 JPEG enlarged to 70 tokens.

## Other checks

- `tests/test_qwen4_cuda`: all kernel tests pass after the series on the combined build, and again at every commit of this series as rebased onto the roofline series.
- `tests/test_qwen4_cli_vision.py` with the Q4 model, `orbit.png` then `earth.jpg`: ordinary and MTP pass (four turns each; the replies read the text "ORBIT 4729" and describe the Earth image).
- Text-only decode, cross-build ABBA with the in-process harness described in [`../qwen-cuda-6000-roofline`](../qwen-cuda-6000-roofline/README.md#method) (code and prose prompts from [`../qwen-cuda-6000/prompts/`](../qwen-cuda-6000/prompts/), 256 tokens, plain, two blocks per arm per round): the combined build without this series (`before` in the CSV) 140.57 / 140.26 t/s, with it 140.58 / 140.31 t/s, identical tokens ([`text-noregression.csv`](text-noregression.csv)). These builds predate the roofline series, which is why both decode at about 140 t/s.
- Metal: the encoder dumps of all 14 cases are bit-identical before and after on Apple silicon.

## Method

- Encoder latency: `tests/test_qwen4_vision mmproj.gguf image.png out.bin 64 1024 11`.
- End to end: `speed-bench/qwen_vision_bench -m Qwen3.8-Flash-Next-Q4.gguf --vision mmproj.gguf --image a.png --image b.png@1024 [--mtp] --repeat 3 --encode-repeat 8 -n 128`; `@N` sets `DS4_QWEN4_IMAGE_MAX_TOKENS` for that image.
- Profiles: `nsys profile --trace=cuda` of `tests/test_qwen4_vision` with three encodes; kernel time is the incremental end-to-end cost, as in the text-decode profiles.
- GPU runs held an exclusive lease on the bench host; the board was otherwise idle.
