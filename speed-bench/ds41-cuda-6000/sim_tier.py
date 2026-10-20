#!/usr/bin/env python3
"""Replay a DS4_CUDA_SSD_TRACE ("layer slots expert...") against a VRAM expert cache backed by
a host RAM tier and the NVMe, and report the decode miss cost per token.

  sim_tier.py TRACE VRAM_SLOTS HOST_SLOTS [--host-ms 0.43] [--nvme-ms 2.65] [--layers 40] [--experts 384]

Host tier policies:
  layers   whole layers 0..k pinned (the current DS4_CUDA_SSD_HOST_GB behavior)
  popular  the HOST_SLOTS most used experts of the first half of the trace (scored on the second),
           or of the trace named by --profile (scored on the whole trace)
  oracle   the HOST_SLOTS most used decode experts of the whole trace
  lru      a dynamic LRU of host slots; NVMe misses are inserted, host hits refresh
  lfu      dynamic, frequency admission: an NVMe miss replaces the least used host entry if used more
  lfu-warm lfu starting from the whole layers that "layers" pins
VRAM policies: lru (current), and gd (GreedyDual: a slot's priority is the clock plus its
reload cost, so experts that only the NVMe holds stay longer).
"""
import heapq
import sys
from collections import Counter, OrderedDict


def load(path):
    reqs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            reqs.append((int(p[1]) > 64, [(int(p[0]), int(e)) for e in p[2:]]))
    return reqs


class StaticHost:
    def __init__(self, keys):
        self.keys = set(keys)

    def has(self, k):
        return k in self.keys

    def touch(self, k):
        pass

    def insert(self, k):
        pass


class LruHost:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def has(self, k):
        return k in self.d

    def touch(self, k):
        self.d.move_to_end(k)

    def insert(self, k):
        self.d[k] = 1
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


class LfuHost:
    """Frequency admission: every decode use counts; an NVMe miss replaces the least used host
    entry only when it has been used more often (prefill sweeps cannot flush hot experts)."""
    def __init__(self, cap):
        self.cap, self.keys, self.count, self.heap = cap, set(), Counter(), []

    def has(self, k):
        return k in self.keys

    def use(self, k):
        self.count[k] += 1
        if k in self.keys:
            heapq.heappush(self.heap, (self.count[k], k))

    def touch(self, k):
        pass

    def insert(self, k):
        if len(self.keys) < self.cap:
            self.keys.add(k)
            heapq.heappush(self.heap, (self.count[k], k))
            return
        while self.heap:
            c, v = self.heap[0]
            if v not in self.keys or c != self.count[v]:
                heapq.heappop(self.heap)
                continue
            if self.count[k] > c:
                heapq.heappop(self.heap)
                self.keys.discard(v)
                self.keys.add(k)
                heapq.heappush(self.heap, (self.count[k], k))
            return


def run(reqs, vram_cap, host, vram_policy, host_ms, nvme_ms, score_from=0):
    vram, heap = {}, []  # key -> (priority, seq); heap of (priority, seq, key), stale entries skipped
    clock, seq = 0.0, 0
    host_hits = nvme = vram_hits = decode_layers = 0

    def put(k, cost):
        nonlocal seq
        seq += 1
        pr = clock + (cost if vram_policy == "gd" else 0.0)
        vram[k] = (pr, seq)
        heapq.heappush(heap, (pr, seq, k))

    for idx, (prefill, keys) in enumerate(reqs):
        scored = not prefill and idx >= score_from
        decode_layers += scored
        for k in keys:
            if not prefill and hasattr(host, "use"):
                host.use(k)
            cost = host_ms if host.has(k) else nvme_ms
            if k in vram:
                put(k, cost)
                vram_hits += scored
                continue
            if host.has(k):
                host.touch(k)
                host_hits += scored
            else:
                host.insert(k)
                nvme += scored
            while len(vram) >= vram_cap:
                pr, sq, victim = heapq.heappop(heap)
                if vram.get(victim) == (pr, sq):
                    del vram[victim]
                    if vram_policy == "gd":
                        clock = pr
            put(k, cost)
        if vram_policy == "lru":
            clock += 1.0
    return decode_layers, vram_hits, host_hits, nvme


def main():
    a = sys.argv
    path, vram_cap, host_cap = a[1], int(a[2]), int(a[3])
    opt = lambda name, default: type(default)(a[a.index(name) + 1]) if name in a else default
    host_ms, nvme_ms = opt("--host-ms", 0.43), opt("--nvme-ms", 2.65)
    n_layers, n_exp = opt("--layers", 40), opt("--experts", 384)
    reqs = load(path)
    decode = [(i, keys) for i, (prefill, keys) in enumerate(reqs) if not prefill]
    half = decode[len(decode) // 2][0] if decode else 0
    use_all = Counter(k for _, keys in decode for k in keys)
    use_first = Counter(k for i, keys in decode if i < half for k in keys)
    profile = opt("--profile", "")
    if profile:
        # Popularity from another trace (a different workload); score the whole trace.
        use_first = Counter(k for prefill, keys in load(profile) if not prefill for k in keys)
        half = 0
    def warm(h):
        # Start from the whole layers the current host tier pins.
        for l in range(min(n_layers, host_cap // n_exp)):
            for e in range(n_exp):
                h.insert((l, e))
        return h

    hosts = {
        "layers": lambda: StaticHost((l, e) for l in range(min(n_layers, host_cap // n_exp)) for e in range(n_exp)),
        "popular": lambda: StaticHost(k for k, _ in use_first.most_common(host_cap)),
        "oracle": lambda: StaticHost(k for k, _ in use_all.most_common(host_cap)),
        "lru": lambda: LruHost(host_cap),
        "lfu": lambda: LfuHost(host_cap),
        "lfu-warm": lambda: warm(LfuHost(host_cap)),
        "lru-warm": lambda: warm(LruHost(host_cap)),
    }
    print(f"{path}: {len(decode)} decode layer requests, {len(use_all)} distinct decode experts; "
          f"VRAM {vram_cap} slots, host {host_cap} slots, host {host_ms} ms, NVMe {nvme_ms} ms per miss")
    only = opt("--hosts", "")
    for hname, mk in hosts.items():
        if only and hname not in only.split(","):
            continue
        for vp in ("lru", "gd"):
            score_from = half if hname == "popular" and half else 0
            layers, vh, hh, nv = run(reqs, vram_cap, mk(), vp, host_ms, nvme_ms, score_from)
            tokens = layers / n_layers
            uses = vh + hh + nv
            print(f"  host {hname:8s} vram {vp:3s}: VRAM hit {vh / uses * 100:5.1f}%  host {hh / tokens:5.1f}/tok  "
                  f"NVMe {nv / tokens:5.2f}/tok  miss cost {(hh * host_ms + nv * nvme_ms) / tokens:5.1f} ms/token"
                  + ("  (second half)" if score_from else ""))


if __name__ == "__main__":
    main()
