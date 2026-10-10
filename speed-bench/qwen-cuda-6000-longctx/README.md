# Qwen CUDA decode at long context, RTX PRO 6000 Blackwell

Two commits on top of the second Qwen CUDA series ([`../qwen-cuda-6000-staging`](../qwen-cuda-6000-staging/README.md)), on the same host and model (RTX PRO 6000 Blackwell Workstation Edition, 450 W limit, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage bare metal). Both are bit-identical by construction: the greedy text matches `main` at every frontier below with either commit, the MTP tokens of tiles match at 200K, and the kernel tests pass at both commits.

| Build | Commit | Change |
|---|---|---|
| main | `152f77b` | The parent. |
| select | `46c87e1` | The indexer's top-512 block selection for rows of 2048 scores or more (`idx_select_wide`). |
| tiles | `dbfe197` | The indexer scorer writes 8-block tile maxima next to the scores (`idx_score_tiles`), which the selection reads for rows of more than 8192 scores. |

## Where long-context decode went

The sparse attention layers (12 a token) score every pooled 4-token block against the indexer query, keep the top 512 blocks and attend their 2048 tokens. Only the first two steps grow with the context. Nsight Systems, `ds4-bench`, 21 steady decode tokens, incremental cost per token (end_i − max(end_<i)); raw data in [`kernels.csv`](kernels.csv):

| Kernel | main, 4K | main, 200K | select, 200K | tiles, 200K |
|---|---:|---:|---:|---:|
| block selection | 82 | 3,113 | 258 | 156 |
| block scores | 29 | 187 | 180 | 163 |
| attention (2048 keys) | 138 | 152 | 136 | 138 |
| **token** | **5,797** | **8,955** | **6,080** | **5,963** |

At 200K tokens a row holds 50,000 scores. `idx_select` ran one 256-thread block per row making six latency-bound passes over them, 259 µs a launch. The rest of the token costs the same at 4K and 200K.

## End to end

`ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64`: prefill rate of the newest 32K interval (8192-token chunks) and steady greedy decode rate at each frontier, each build in its own process. Raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Decode main | select | tiles | Change | Prefill main | select | tiles | Change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 160.3 | 171.3 | 170.8 | +6.5% | 3,042 | 3,072 | 3,105 | +2.1% |
| 64K | 150.2 | 170.3 | 169.6 | +12.9% | 2,765 | 2,800 | 3,063 | +10.8% |
| 96K | 138.2 | 168.6 | 168.9 | +22.3% | 2,531 | 2,566 | 2,919 | +15.3% |
| 128K | 128.6 | 167.1 | 168.2 | +30.8% | 2,334 | 2,363 | 2,788 | +19.5% |
| 160K | 117.6 | 165.8 | 167.7 | +42.6% | 2,164 | 2,190 | 2,672 | +23.5% |
| 192K | 111.6 | 164.4 | 168.1 | +50.5% | 2,012 | 2,042 | 2,565 | +27.5% |
| 224K | 107.1 | 163.2 | 167.4 | +56.3% | 1,878 | 1,914 | 2,470 | +31.5% |
| 256K | 102.4 | 159.8 | 164.4 | +60.6% | 1,762 | 1,801 | 2,378 | +34.9% |

MTP, session benchmark with one stream at 200K tokens (8192-token prefill chunks, 128 tokens after 16 warm-up steps, full draft head; [`sessions.csv`](sessions.csv)): 130.2 → 172.4 t/s (+32.4%), p95 step 12.4 → 9.4 ms, 1.52 tokens per cycle in both, the same tokens; prefill 2,398 → 2,764 t/s.

The prefill gain is the scorer's. With `DS4_QWEN4_TIMING=2` (a sync after each stage), the attention stage of the 8192-token chunk at 245,760 tokens takes 2,363 ms on main, 2,259 on select and 1,161 on tiles, of about 4.7 s a chunk: a prefill chunk launched `idx_score` as about 130 million four-block CTAs a layer, `idx_score_tiles` as 8 million.

## Short contexts

Rows under 2048 scores (8K tokens) keep `idx_select`, prefill chunks keep it up to 8192 scores (32K tokens), and rows up to 8192 scores keep `idx_score`, so short contexts run the parent's kernels. Measured ([`short-context.csv`](short-context.csv)):

- `ds4-bench` 4K to 64K, processes main, tiles, tiles, main: decode −0.09% at 4K, +0.13% at 8K, +1.17% at 16K, +5.68% at 32K, +11.88% at 64K.
- Prefill only, 8K to 32K, three interleaved processes per build: +0.19%, +0.17%, +0.20%.
- `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts at `--ctx 4096`, tokens identical: plain −0.01% and −0.00%, MTP +0.03% and +0.05%.

## The selection

`idx_select_wide` gives the slots `idx_select` gives: the keys (the bits of max(score, 0)) above the 512th largest in index order, then the first keys equal to it, in index order. One 1024-thread block per row:

1. The largest key of each group of s scores, s the finest power of two leaving at most 8192 groups: from one coalesced pass over the row, or from the scorer's tile maxima (one per tile up to 65,536 scores, per 2 or 4 tiles past that).
2. At least 512 keys reach the 512th largest group maximum, so its 22-bit prefix τ bounds the threshold from below. Two radix rounds over the group maxima in shared memory find it.
3. The keys at or above τ are gathered, reading only the groups whose maximum reaches τ (one atomic per eight loads per warp).
4. Three radix rounds (12, 10 and 10 bits) over the gathered keys find the threshold; two bitmaps, scanned per warp in index order, place the slots; the slots are staged in shared memory and written out together.

More than 4096 gathered keys (heavy ties), or fewer than 512, resolve on the scores with three passes. `DS4_QWEN4_SELECT_WIDE_MIN` overrides the row length from which it runs (0 for every row). Microbenchmarks, one launch on random rows ([`microbench.csv`](microbench.csv)):

| Scores | `idx_select` µs | wide, own pass µs | wide, tile maxima µs |
|---:|---:|---:|---:|
| 2,048 | 8.0 | 7.3 | |
| 8,192 | 24.6 | 10.5 | |
| 16,384 | 46.0 | 12.9 | 7.6 |
| 50,000 | 196.2 | 20.2 | 11.5 |
| 65,536 | 262.1 | 22.4 | 12.2 |
| 262,144 | 1033.9 | 40.8 | 21.0 |

The tile scorer computes each score as `idx_score_body` does (eight blocks a warp, queries in registers, every key load issued first), so the scores are bit-identical; over twelve layers' keys in turn (DRAM, as in decode) it takes 12.5 against 14.3 µs at 50,000 blocks and 45.1 against 66.2 at 262,144, and 3.6 against 2.3 at 1024 (hence only past 8192).

## Findings

- Real indexer scores are clumpier than random ones. With 32-score groups at 200K tokens, 1056 to 3590 keys reached τ and 35% of the launches overflowed a 2048-key buffer into the slow path (33 µs a launch against 19 on random rows). With 8-score groups from the tile maxima and a 4096-key buffer, a launch costs 13 µs in decode at 200K against 11.5 on random rows.
- One block reads the scores from L2 at about 15 bytes a cycle (about 42 GB/s), so a full pass over 50,000 scores costs about 5 µs: the passes have to read less, not compute less.
- A histogram of every key is expensive in one block: `__match_any_sync` per 32 keys cost about 31 cycles a key group (48K cycles at 50,000 scores), plain shared atomics 9.
- A register cap on the tile scorer to fit the 200K grid in one wave (48 or 40 registers) was slower: 14.0 and 17.1 µs against 11.6 at 50,000 blocks.

What remains: the token at 200K is 166 µs (2.9%) slower than main's at 4K. The scorer costs 134 µs a token more than at 4K (it reads 12.8 MB a layer at about 0.9 TB/s) and the selection 73 µs more.
