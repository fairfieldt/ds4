#!/usr/bin/env python3
"""Replay a DS4_CUDA_SSD_TRACE file ("layer slots expert...") against expert caches.

Reports, per cache size, the hit rate of decode requests (slots <= 64) under
LRU (the runtime's policy) and under a static oracle that keeps the most used
experts of the whole trace (an upper bound for a fixed hot set).
"""
import sys
from collections import Counter, OrderedDict


def load(path):
    reqs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            layer, slots = int(p[0]), int(p[1])
            reqs.append((slots > 64, [(layer, int(e)) for e in p[2:]]))
    return reqs


def lru(reqs, cap):
    cache = OrderedDict()
    hits = n = 0
    for prefill, keys in reqs:
        for k in keys:
            if k in cache:
                cache.move_to_end(k)
                if not prefill:
                    hits += 1
            else:
                cache[k] = 1
                if len(cache) > cap:
                    cache.popitem(last=False)
            if not prefill:
                n += 1
    return hits / n if n else 0.0


def main():
    reqs = load(sys.argv[1])
    sizes = [int(x) for x in sys.argv[2:]] or [3000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15360]
    decode = [keys for prefill, keys in reqs if not prefill]
    use = Counter(k for keys in decode for k in keys)
    n = sum(use.values())
    print(f"requests {len(reqs)} decode {len(decode)} decode expert uses {n} distinct {len(use)}")
    ranked = [c for _, c in use.most_common()]
    for cap in sizes:
        static = sum(ranked[:cap]) / n if n else 0.0
        print(f"cap {cap:6d} ({cap * 9.49 / 1024:6.1f} GiB): LRU decode hit {lru(reqs, cap) * 100:6.2f}%  "
              f"static-oracle {static * 100:6.2f}%")


if __name__ == "__main__":
    main()
