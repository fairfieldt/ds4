# Models Larger Than RAM

[README](../README.md)

Use resident inference when the model and runtime fit: it is faster.
SSD streaming keeps a bounded cache of routed experts and reads missing
experts from the GGUF. It trades speed for capacity; it does not remove the
memory needed for other weights, activations, scratch, and the context.

Metal supports streaming for DeepSeek and GLM. CUDA has streaming paths too,
and ROCm supports GLM 5.2/5.3 streaming. Do not infer support for every model
and tensor layout from the existence of the flag.

## Start with the automatic budget

```sh
./ds4 -m ds4flash.gguf --ssd-streaming
```

The startup report shows the effective cache, resident layers where applicable,
and memory requirements. Prefer the automatic budget for a first run. Use a
local SSD and leave memory for the OS and other applications.

Examples:

```sh
# Flash Q2 on a smaller Mac.
./download_model.sh ds4f-q2
./ds4 --ssd-streaming --ctx 32768 --nothink

# GLM 5.3 Flash Q4 on a 128 GB Mac.
./download_model.sh glm53-q4
./ds4 --ssd-streaming --ctx 4096

# PRO Q2 on a 128 GB Mac: usable for inspection, but slow.
./download_model.sh pro-q2-imatrix
./ds4 --ssd-streaming --ctx 32768 --nothink
```

Generation is usually more sensitive to cache misses than prefill. A large
model that starts successfully can still be too slow for interactive work.
Use a short generation before committing to a long task.

## Adjusting the cache

To leave more room for context or other sessions:

```sh
./ds4 --ssd-streaming --ssd-streaming-cache-experts 32GB
```

A byte budget is a target, not a guaranteed allocation. DwarfStar reserves
routed-prefill headroom and fits the cache to the remaining model, graph,
context, and backend budget. The effective value may be smaller than requested.
Non-routed weights and KV state are additional to that expert-cache budget.

A plain number, such as `--ssd-streaming-cache-experts 4000`, requests dynamic
expert slots rather than a byte budget. It is also subject to memory limits.

By default, GLM spends the cache budget on selected experts across all layers.
`--ssd-streaming-full-layers N` reserves full routed prefix layers instead.
Metal also keeps the non-routed weights resident when it can; they are needed
by every token, and paging them back in delays generation after a prompt.
An oversized expert cache can displace those weights and slow decoding.
More cache helps only while the rest of the working set still fits.

More context and more server sessions consume more memory. Metal GLM budgeting
includes planned and already resident sessions; a request can still be refused
when its mandatory allocations cannot fit. Disabling the memory guard is not
a remedy for insufficient RAM.

Leave expert preloading enabled for normal use. `--ssd-streaming-cold` and
`--ssd-streaming-preload-experts N` are mainly useful for controlled measurements.
See [benchmarking](PERFORMANCE.md) and the [release QA guide](../QA_BEFORE_RELEASES.md).

## Discrete CUDA GPUs

With DeepSeek V4.1 on a discrete NVIDIA GPU, the expert cache is in VRAM.

A host with spare RAM can pin routed experts, so a cache miss copies from RAM
instead of reading the SSD:

```sh
DS4_CUDA_SSD_HOST_GB=auto ./ds4 --cuda --ssd-streaming \
  -m gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

`auto` pins whole layers while 16 GiB of RAM stay available. A number caps
the pinned GiB instead. Q2's routed experts take 142.4 GiB. Other programs
cannot use that memory while DwarfStar runs. Startup reads and pins the
experts first: 30 seconds from a cold page cache and 11 to 16 seconds when
the file is already cached.

On an RTX PRO 6000 (96 GB) with 192 GB of RAM, pinning all of Q2's experts
took a 2K-token prefill from 75 to 510 t/s and decode from 21 to 48 t/s.

## M5 Max measurements

On a 128 GB M5 Max, with automatic cache sizing and no speculative decoding:

| Model | Initial prefill | Continued prefill | Generation after each | Runs |
| --- | ---: | ---: | ---: | --- |
| GLM 5.3 Flash Q4_K, 177.77 GiB | 121 t/s | 104 t/s | 11.9 / 14.9 t/s | Three-run median |
| DeepSeek Flash Vision Exp MXFP4, 145.26 GiB | 300 t/s | 263 t/s | 11.9 / 19.3 t/s | Single run |

September 6, 2026. GLM used a 2K prompt and a 1K append; DeepSeek used 8K
and a 4K append. Both generated 128 tokens at each frontier. DeepSeek's latest
run uses an 86.2 GiB expert cache and updated eviction priorities; it is not
directly comparable to the earlier 64-token measurement. Generation includes
the first-token wait. Cache reuse depends on the prompt, so these are
workload references, not a speed
guarantee for every model larger than RAM.

Full GLM 5.3 IQ2_XXS (196.58 GiB) also benefits from caching short tool-result
prompts. With an 8K context and a 61.35 GiB effective expert cache, a 16-token
append fell from 30.8 to 2.9 seconds; generation afterward rose from 4.09 to
5.11 t/s (three-run medians, 64 generated tokens). All compared logits and
generated text were identical. This is a short-append improvement, not an
equivalent gain in initial prefill or every generation workload.
