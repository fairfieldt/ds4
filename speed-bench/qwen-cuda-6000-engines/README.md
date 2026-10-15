# Qwen3.8 on ds4, llama.cpp and Strata, RTX PRO 6000 Blackwell

Three inference engines serving Qwen3.8-Flash-Next on one GPU, measured 2026-10-15 on voyage: RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, driver 615.71.09, 600 W, Ryzen 9 9950X, 192 GB RAM. One stream, greedy, thinking off, a 135,168-token context in every server. Each comparison of engines uses identical weights:

| | Engine | Weights |
|---|---|---|
| A | ds4 main `e50a190` | DS4 Q4: Q4_K gate/up, MXFP4 down, Q8_0 dense, BF16 n-grams (177.3 GB) |
| B | llama.cpp `b76b133` + two loader changes (below) | the same file |
| C | llama.cpp `b76b133` | Unsloth UD-Q4_K_XL `38bb39e`: Q4_K gate/up, Q5_1 down (Q8_0 in 5 layers), Q8_0 dense, IQ4_NL n-grams (111.3 GB) |
| D | Strata v0.1.44.1 (`296b59d`) | the same Unsloth files |

Suffix 0 is plain decoding, 1 is the engine's MTP speculative decoding.

Stock llama.cpp cannot load the DS4 file. It needs [ggml-org/llama.cpp#30316](https://github.com/ggml-org/llama.cpp/pull/30316) (`2e74fdb`), which defaults the missing `rope.dimension_sections` and accepts the u64 `attention.compress_ratios`. It also needs a 4-line change to `gguf_init_from_reader` that accepts an aligned offset past the expected one, since the n-gram table starts on a padded offset. Neither touches computation. Unsloth's files have no MTP block; C uses its `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (`766911a`) with `-md`.

## Method

Servers:

```sh
ds4-server --cuda -m Qwen3.8-Flash-Next-Q4.gguf --ctx 135168 [--mtp]
llama-server -m MODEL -ngl 999 -fa on -c 135168 -np 1 --jinja -b 4096 -ub 4096 \
    -ot 'per_layer_token_embd\.weight=CPU' [--spec-type draft-mtp --spec-draft-n-max 5 [-md MTP.gguf]]
strata --serve --pack PACK --native UD-Q4_K_XL-00001-of-00004.gguf --expert-profile data/expert-profile.bin \
    --expert-cache auto --prefill auto --max-context 135168 --kv fp16 \
    (--spec 4 --spec-min-p 0.5 --mtp DRAFT_LAYER | --spec 2 --suffix-draft 0)
```

- ds4 runs with its default 8,192-token prefill chunk and no steering vector.
- llama.cpp's draft length and ubatch come from sweeps ([`sweeps.csv`](sweeps.csv)). Draft lengths 2-5 on the prompt mix gave 178 / 191 / 191 / 193 t/s on the Unsloth file and – / 174 / 182 / 188 on the DS4 file (2 not run). Ubatch 1024 / 2048 / 4096 / 8192 gave 4,482 / 4,985 / 5,192 / 5,177 t/s prefill at 32K.
- Strata's MTP settings are what its setup writes: the Q2_0 draft layer plus suffix lookup. A native pack needs a verify window, so its plain run is `--spec 2 --suffix-draft 0` without a draft layer. It ran with fp16 KV (its setup picks int8 above 8K context), and every expert fit in VRAM.

The client streams `/v1/chat/completions` with `temperature: 0`, `chat_template_kwargs: {enable_thinking: false}` and 256 output tokens:

- Time to first token is measured to the first content chunk.
- Prefill t/s is prompt tokens divided by time to first token.
- Decode t/s is (completion tokens − 1) divided by the time from the first to the last chunk.

Two workloads, each run twice per server:

- **Prompt mix:** the ten prompts in [`prompts/`](prompts) (code, explanation, JSON, math, op-ed, prose, Rust, SQL, a table, an Italian-to-English translation).
- **Long prompts:** about 1K, 8K, 32K and 131K tokens of `speed-bench/promessi_sposi.txt`, asking for the story to continue in Italian. Each request starts with a random tag and a different slice of the text, so no engine reuses a cached prefix.

Each server read its GGUF once before measurement so the files were in the page cache. Raw data: [`speed.csv`](speed.csv), [`memory.csv`](memory.csv).

## Decode

| Config | Mix t/s, mean | Min | Max | After 1K | After 131K | Peak device MiB |
|---|---:|---:|---:|---:|---:|---:|
| A0 ds4 | 179.7 | 179.4 | 180.0 | 178.4 | 167.8 | 83,640 |
| B0 llama.cpp, DS4 Q4 | 106.1 | 105.8 | 106.4 | 105.1 | 92.4 | 78,502 |
| C0 llama.cpp, UD-Q4_K_XL | 112.8 | 112.3 | 113.2 | 111.9 | 97.2 | 87,810 |
| D0 Strata, UD-Q4_K_XL | 134.3 | 134.0 | 134.5 | 133.6 | 126.9 | 83,426 |
| A1 ds4, MTP | 242.6 | 212.9 | 271.2 | 201.6 | 195.8 | 83,950 |
| B1 llama.cpp, DS4 Q4, MTP | 187.7 | 116.9 | 246.9 | 93.5 | 58.2 | 84,464 |
| C1 llama.cpp, UD-Q4_K_XL, MTP | 192.8 | 119.5 | 258.3 | 103.9 | 68.2 | 95,628 |
| D1 Strata, UD-Q4_K_XL, MTP | 240.6 | 183.3 | 299.0 | 161.2 | 149.7 | 84,888 |

MTP decode per mix prompt, t/s (mean of 2):

| Prompt | A1 | B1 | C1 | D1 |
|---|---:|---:|---:|---:|
| code | 261 | 223 | 238 | 283 |
| explain | 231 | 189 | 164 | 241 |
| json | 259 | 247 | 257 | 299 |
| math | 252 | 204 | 217 | 255 |
| oped | 214 | 118 | 120 | 184 |
| prose | 224 | 122 | 120 | 183 |
| rust | 252 | 209 | 222 | 242 |
| sql | 251 | 223 | 231 | 253 |
| table | 255 | 207 | 216 | 264 |
| translate | 228 | 135 | 142 | 202 |

Identical 256-token outputs on the ten mix prompts:

- **MTP against plain on the same engine:** ds4 10/10, Strata 10/10, llama.cpp 2/10 on the DS4 file and 1/10 on Unsloth's.
- **Across engines on the same weights:** ds4 and llama.cpp 0/10, Strata and llama.cpp 1/10.

Every configuration repeated its own outputs exactly.

## Prefill

| Config | ~1K | 8K | 32K | 131K | Time to first token at 131K |
|---|---:|---:|---:|---:|---:|
| A0 ds4 | 3,566 | 6,732 | 6,843 | 6,880 | 19.0 s |
| B0 llama.cpp, DS4 Q4 | 2,267 | 4,847 | 5,130 | 4,961 | 26.4 s |
| C0 llama.cpp, UD-Q4_K_XL | 2,536 | 4,935 | 5,155 | 4,993 | 26.2 s |
| D0 Strata, UD-Q4_K_XL | 2,058 | 4,086 | 5,280 | 5,411 | 24.2 s |
| A1 ds4, MTP | 3,495 | 6,580 | 6,695 | 6,745 | 19.4 s |
| B1 llama.cpp, DS4 Q4, MTP | 2,226 | 4,266 | 4,189 | 3,857 | 34.0 s |
| C1 llama.cpp, UD-Q4_K_XL, MTP | 2,197 | 4,257 | 4,180 | 3,850 | 34.0 s |
| D1 Strata, UD-Q4_K_XL, MTP | 2,065 | 4,082 | 5,243 | 5,392 | 24.3 s |

Prompt token counts agree across engines within 3 tokens (1,107 / 8,265 / 32,976 / 130,963 on average).

The ~1K column for llama.cpp averages a normal first request (3,651 and 3,659 t/s plain) with one made right after a 131K request. That one waited 0.5-1.0 s longer before prefilling, while llama-server stored the previous context in its host prompt cache. The other engines show no such wait.

## Fidelity

Teacher-forced log-likelihood of the tracked Alibaba continuations of hosted Qwen3.8 Flash. These are the 100 short prompts in `gguf-tools/quality-testing/data/qwen38-flash-alibaba-100` and the 12 prompts of 2K-24K tokens in `qwen38-flash-alibaba-long`.

- **ds4 (A):** scored by `score_official` with `--dump-token-logprobs`, at context 4096 for the short set and 32768 for the long set.
- **llama.cpp (B and C):** scored by a small libllama program fed the exact prompt and target token IDs that `score_official` printed under `DS4_SCORE_DEBUG=1`. It uses the same schedule: prefill the prompt, then decode the target one token at a time. Its dump format is the same, so `compare_token_logprobs.py` compares the engines directly.

Strata's API returns no logprobs, so D is not scored. Per-case data: [`fidelity.csv`](fidelity.csv).

| Config | Short NLL | First token = reference | Mean greedy prefix | Long NLL | First token = reference |
|---|---:|---:|---:|---:|---:|
| A ds4, DS4 Q4 | 0.29044 | 83/100 | 16.0 | 0.12594 | 11/12 |
| B llama.cpp, DS4 Q4 | 0.29240 | 80/100 | 16.8 | 0.12577 | 10/12 |
| C llama.cpp, UD-Q4_K_XL | 0.28645 | 89/100 | 18.1 | 0.12517 | 11/12 |

`compare_token_logprobs.py`, short / long sets:

| Pair | Greedy agreement | Approx KL mean, nats | Cases where the second is worse |
|---|---:|---:|---:|
| A vs B (same weights) | 97.8% / 98.7% | 4.6e-3 / 3.1e-3 | 51/100, 5/12 |
| B vs C (same engine) | 96.2% / 97.7% | 1.2e-2 / 5.2e-3 | 36/100, 6/12 |
| A vs C | 96.8% / 98.2% | 9.9e-3 / 3.1e-3 | 29/100, 3/12 |

## Findings

- **ds4 decodes 1.69x faster than llama.cpp on the same weights without speculation, and 1.29x faster with it.** Strata decodes 1.19x faster than llama.cpp on its weights without speculation.
- **ds4 and Strata tie on the MTP prompt mix (242.6 and 240.6 t/s).** Strata leads on code, JSON, tables and explanation by 3-15%. ds4 leads on prose, op-ed and translation by 13-22%, and on the long Italian prompts by 23-31% at every depth. The other prompts are within 4%.
- **llama.cpp's MTP is slower than its plain decode on the long Italian prompts (58-104 against 92-112 t/s), and changes its greedy output.** Five draft tokens were best on the mix, but most are rejected on this text, and the loss grows at 131K. ds4 adapts its draft depth and Strata gates drafts on confidence; both stay above their plain speed with outputs identical to plain.
- **From 32K up ds4 prefills 27-39% faster than either other engine.** Its plain decode falls 6% from 1K to 131K, against 13% for llama.cpp and 5% for Strata. With MTP on, llama.cpp prefills about 20% slower than without.
- **On identical weights ds4 and llama.cpp are equally close to the hosted model.** Their numerics differ: 97.8% greedy agreement, KL 4.6e-3 nats, about 10 times the int8-against-BF16 n-gram difference measured on 2026-10-09. But neither is better per case (51/100).
- **Unsloth's UD-Q4_K_XL is closer to the hosted model than DS4 Q4.** Its short-set NLL is 2.0% lower on the same engine, and it is better on 64/100 cases (71/100 against ds4). Its down experts are Q5_1/Q8_0 where DS4 Q4 uses MXFP4; its n-gram table is 4-bit.
- **Peak device memory is 77-86 GiB except C1.** C1 reaches 93.4 GiB (llama.cpp with Unsloth's self-contained MTP head at a 135K context).
