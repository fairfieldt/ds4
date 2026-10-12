# Qwen CUDA bulk copies without driver syscalls, RTX PRO 6000 Blackwell

One commit on top of main (`9b2e06d`) that gives the bulk (TMA) copies a `.shared::cta` destination instead of `.shared::cluster`. Same host and model (RTX PRO 6000 Blackwell Workstation Edition, 188 SMs, CUDA 13.4, driver 615.71.09, Qwen3.8-Flash-Next-Q4, voyage), 600 W. The kernels run without clusters, so both forms copy the same bytes into the same shared memory. The checks:

- The kernel tests pass.
- Frontier logits from 2K to 32K are byte-identical to main's in all four processes.
- Greedy text and decode tokens are identical in every run.

| Build | Commit | Change |
|---|---|---|
| main | `9b2e06d` | The parent. |
| new | `25f3e93` | `bulk_copy` and `tma_2d` use `.shared::cta`. |

## The syscalls

On sm_120, nvcc 13.4 compiles `cp.async.bulk.shared::cluster.global...` and `cp.async.bulk.tensor.2d.shared::cluster.global...` to calls into the driver: `__cuda_syscall_cp_async_bulk_unicast` and `__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast`. These show up in `EIATTR_EXTERNS` (`cuobjdump -elf`) and as `CALL.ABS.NOINC` in SASS. On main, nine kernels had them:

- `dense_split_tma`
- `hc_mix_f16_bulk`
- `matvec_q8_bulk<1,2,4>`
- `matvec_split_f16_bulk<1,2,4,8>`

The first launch of any of these (the first `dense_split_tma` in prefill, or the first bulk matvec in decode) raised the device stack limit from 1,024 to 14,624 bytes. The driver reserves that stack for every thread slot (188 SMs × 1,536 threads) for the rest of the process, about 3.8 GiB.

The `.shared::cta` forms (PTX 8.6) compile to native copies. In the new build `ds4_cuda.o` has no `EIATTR_EXTERNS`. `dense_split_tma` issues 8 `UTMALDG.2D` and no calls, where main had a `CALL.ABS.NOINC` next to each. The bulk matvecs issue `UBLKCP.S.G`. Toolkits older than CUDA 12.8 keep the cluster form.

## Memory

From `nvidia-smi` sampled every 200 ms (1 s for 6 × 100K), and an `LD_PRELOAD` shim that reads the stack limit around every launch. All values are in [`memory.csv`](memory.csv). The commands use `speed-bench/session_concurrency_bench --prompt-file speed-bench/promessi_sposi.txt --prefill-chunk 4096 --gen 64 --force`.

| Run | Stack limit | Peak device memory | Left of 97,887 MiB | Result |
|---|---|---:|---:|---|
| main, 6 streams × 64K | 1,024 → 14,624 B | 93,510 MiB | 4,377 MiB | ok |
| new, 6 streams × 64K | 1,024 B | 89,622 MiB | 8,265 MiB | ok |
| main, 6 streams × 100K | | | | out of memory at the first prefill kernel |
| new, 6 streams × 100K | 1,024 B | 96,172 MiB | 1,715 MiB | ok |
| new, 6 streams × 100K, `--spec` | 1,024 B | 97,322 MiB | 565 MiB | ok |

The main rows for the stack limit and 6 × 100K come from the batch capacity measurements on main the same day. At 6 × 64K the new build peaks 3,888 MiB lower.

6 × 100K on the new build:

| Mode | Prefill t/s | Last stream ready | Decode t/s, total | Per stream | Step p95 |
|---|---:|---:|---:|---:|---:|
| plain | 6,362 | 94.3 s | 399.7 | 66.6 | 15.1 ms |
| plain, stack shim loaded | 6,622 | 90.6 s | 400.1 | 66.7 | 15.2 ms |
| `--spec` (depth ≤ 2), stack shim loaded | 6,421 | 93.4 s | 400.0 | 66.7 | 23.0 ms |

## Speed

Prefill: `DS4_BENCH_FORCE_SNAPSHOT=1 ds4-bench --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --step-mul 2 --ctx-max 32768 --gen-tokens 64 --show-output --dump-frontier-logits-dir DIR`. Each row is the prefill rate of the newest interval. Processes ran main, new, new, main; raw data in [`short-context.csv`](short-context.csv).

| Context | main | new | Change |
|---:|---:|---:|---:|
| 2K | 5,246 | 5,301 | +1.0% |
| 4K | 5,841 | 5,859 | +0.3% |
| 8K | 6,536 | 6,567 | +0.5% |
| 16K | 6,882 | 6,945 | +0.9% |
| 32K | 6,866 | 6,938 | +1.0% |

Main's first process ran 1–2% below its second. Against the second alone, the change is −0.2% to +0.5%.

Decode, `rf-xab` ([method](../qwen-cuda-6000-roofline/README.md#method), [`xab.csv`](xab.csv)), code and prose prompts, tokens identical:

- Plain: +0.07% and +0.08%.
- MTP: +0.18% and +0.07%.

Session harness: 1 stream, 256 generated tokens, 16 warm-up cycles, 512-token prefill chunks. Processes ran main, new, new, main; raw data in [`sessions.csv`](sessions.csv).

| Mode | main t/s | new t/s | Change |
|---|---|---|---:|
| ctx 256, plain | 178.5 / 180.2 | 180.2 / 180.3 | +0.5% |
| ctx 4096, plain | 172.5 / 173.8 | 174.2 / 174.2 | +0.6% |
| ctx 256, `--spec` | 189.9 / 190.8 | 191.0 / 191.0 | +0.4% |
| ctx 4096, `--spec` | 185.9 / 186.4 | 187.0 / 187.0 | +0.5% |

Tokens per cycle are identical on both builds: 1.457 at ctx 256 and 1.469 at ctx 4096. Against main's second process alone, the change is +0.1% to +0.3%.

## Findings

- **No `cudaMalloc` hook or Nsight kernel record shows the reservation.** It appears only as device memory in use (`cudaMemGetInfo`, `nvidia-smi`) and as the stack limit (`cudaDeviceGetLimit`). A process measured right after startup looked 3.8 GiB lighter than it was after its first prefill.
- **Any new kernel that uses the cluster forms brings the reservation back.** So does anything else the driver implements as a syscall. Check new kernels for `EIATTR_EXTERNS` with `cuobjdump -elf ds4_cuda.o`.
- **`cuda/mmq/ds4_mmq.o` still has 50 `vprintf` externs.** None of the Qwen runs above moved the stack limit.
- **6 × 100K with `--spec` leaves 565 MiB**, where plain leaves 1.7 GiB.
