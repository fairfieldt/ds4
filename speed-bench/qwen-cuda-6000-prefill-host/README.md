# Qwen CUDA prefill host staging, RTX PRO 6000 Blackwell

One commit on top of main (`8a06c04`, after the combine fusion in [`../qwen-cuda-6000-prefill-hc`](../qwen-cuda-6000-prefill-hc/README.md#the-combine)). It keeps the GPU busy while the host stages a prefill chunk's inputs. Same host and model (RTX PRO 6000 Blackwell Workstation Edition, CUDA 13.4, Qwen3.8-Flash-Next-Q4, voyage), 600 W.

The GPU computes the same values from the same inputs: the embedding rows it gathers are the bits the host wrote, and the n-gram rows come from the same reads. The checks:

- The kernel test compares the gathered rows with the host's bit for bit.
- The frontier logits of all four long-sweep processes are byte-identical to main's.
- The greedy text matches at every frontier to 256K.

| Build | Commit | Change |
|---|---|---|
| main | `8a06c04` | The parent. |
| new | `c6757e3` | Embedding rows gathered on the GPU; n-gram rows read a chunk ahead. |

## Where a chunk waited

On main, the GPU sat idle twice in every 8192-row chunk:

- **Before the first kernel, about 39 ms.** The host widened the BF16 embedding row of every token into its four 2560-float streams (335 MB) and copied them from pageable memory.
- **After layer 0, about 40 ms.** The host waited for the GPU (a device-wide synchronize), then read the chunk's 131,072 n-gram rows (one `pread` each, 16 threads; 33 ms with the rows in the page cache) and copied them from pageable memory (5 ms). Layer 1 needs them.

The new build changes three things:

- **Embedding rows.** The host copies the token ids, and `embed_rows_bf16` widens the rows into the four streams (0.2 ms). Chunks with image rows, and tables other than BF16, keep the host rows.
- **N-gram read-ahead.** While the GPU runs a chunk, the host reads the next chunk's n-gram rows into page-locked memory. By then the host has queued every kernel of the chunk and would otherwise only wait. The forward that uses the rows queues their copy behind layer 0 (84 MB in 1.6 ms).
- **No sync before the read.** A chunk whose rows were not read ahead (the first of each `ds4_session_sync`) reads them while layer 0 runs, not after it.

The read-ahead is matched on the graph, its position, its n-gram context and the chunk's tokens. Forwards of four rows or fewer (decode, MTP verification) take none of these paths.

The table below uses Nsight Systems captures of the newest frontier's prefill: one chunk at 24,576 and at 245,760 tokens of context, and two chunks from 24,576. Each capture starts at the bench's `cudaProfilerStart`, and about 6.5 ms pass before the first CUDA call in either build. All values in [`captures.csv`](captures.csv):

| ms | main, 1 chunk at 24,576 | new | main, 1 chunk at 245,760 | new | main, 2 chunks from 24,576 | new |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1,306.1 | 1,244.4 | 1,345.7 | 1,286.8 | 2,598.5 | 2,486.8 |
| before the first kernel | 39.2 | 6.6 | 38.0 | 6.5 | 38.7 | 7.0 |
| GPU idle between kernels | 38.4 | 16.8 | 42.8 | 17.8 | 111.9 | 14.5 |
| GPU busy | 1,224.8 | 1,217.4 | 1,260.9 | 1,259.0 | 2,443.0 | 2,457.8 |

Over two chunks, main's idle time between kernels splits into 38.8 ms for the first chunk's n-gram rows, 31.6 ms between the chunks and 40.6 ms for the second chunk's rows. On the new build these are 11.7, 0.2 and 1.6 ms. Each kernel's own time is in [`kernels.csv`](kernels.csv); kernels that overlap under programmatic dependent launch are counted in full. They are unchanged within 1%, and `embed_rows_bf16` takes 0.21 ms a chunk.

## End to end

`DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 32768 --step-incr 32768 --ctx-max 262016 --gen-tokens 64 --show-output --dump-frontier-logits-dir DIR`. The table gives the prefill rate of the newest 32K interval and the steady greedy decode rate at each frontier. Processes ran in the order main, new, new, main; raw data in [`end-to-end.csv`](end-to-end.csv).

| Context | Prefill main | new | Change | Decode main | new |
|---:|---:|---:|---:|---:|---:|
| 32K | 6,667 | 6,930 | +3.9% | 170.8 | 170.7 |
| 64K | 6,654 | 6,921 | +4.0% | 169.9 | 169.7 |
| 96K | 6,607 | 6,876 | +4.1% | 169.2 | 169.1 |
| 128K | 6,561 | 6,827 | +4.0% | 168.5 | 168.4 |
| 160K | 6,522 | 6,784 | +4.0% | 168.4 | 168.1 |
| 192K | 6,500 | 6,743 | +3.7% | 168.9 | 168.7 |
| 224K | 6,461 | 6,714 | +3.9% | 168.3 | 168.1 |
| 256K | 6,436 | 6,683 | +3.8% | 167.3 | 167.1 |

All four processes give the same greedy text and byte-identical frontier logits at all eight frontiers.

Short contexts, prefill only, frontiers 2K to 32K (each row the newest interval). Processes ran in the order main, new, new, main, main, new ([`short-context.csv`](short-context.csv)):

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 5,200 | 5,365 | +3.2% |
| 4K | 5,694 | 5,938 | +4.3% |
| 8K | 6,504 | 6,733 | +3.5% |
| 16K | 6,830 | 7,125 | +4.3% |
| 32K | 6,781 | 7,105 | +4.8% |

Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical:

- Plain: +0.02% and +0.03%.
- MTP: −0.09% and −0.16%; a repeat gave −0.10% and −0.14%.

## Findings

- **MTP decode is 0.1% slower; the cause was not found.**
  - Every MTP run measured −0.07% to −0.16%, against +0.02% to +0.06% for plain decode, with tokens identical.
  - The decode forwards run the same code as on main.
  - Three variants left the difference in place ([`xab.csv`](xab.csv)): allocating the token-id buffer and the read-ahead buffers after every other buffer, moving the new graph fields to the end of the structure, and dropping the page-locked rows altogether.
- **The kernels run slightly longer without idle gaps.** Over two chunks the GPU was busy 0.6% longer (1.1% in an earlier pair of captures), and the projections and expert GEMMs up to 1% each. With no idle time left, the 600 W cap holds the clock lower. Of the 127 ms of idle time removed from the two-chunk capture, 15 ms went back to longer kernels.
- **The first chunk of each sync still waits 12 to 17 ms.**
  - It cannot be read ahead. Its n-gram read (33 ms) outlasts layer 0 (21 ms).
  - The long sweep syncs 32K tokens at a time, so one chunk in four waits; a single prompt waits once.
  - With the rows not in the page cache, the read takes longer. The read-ahead then hides more and the first chunk waits longer (not measured).
- **The copy of the rows read ahead holds layer 1 back by 1.6 ms.** On a second stream it would overlap layer 0 (not tried).
