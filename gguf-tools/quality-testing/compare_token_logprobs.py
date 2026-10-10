#!/usr/bin/env python3
"""Compare two score_official --dump-token-logprobs files from the same manifest.

A is the reference (for example original BF16 n-grams), B the candidate. KL is
approximate: it covers A's top 20 tokens plus one bucket for the rest, and
uses B's 20th logprob for a token missing from B's top 20.
"""
import math
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            top = [(int(t), float(lp)) for t, lp in (x.split(':') for x in p[5:])]
            rows[(p[0], int(p[1]))] = (int(p[2]), float(p[3]), int(p[4]), top)
    return rows


def kl(top_a, top_b):
    b = dict(top_b)
    floor = top_b[-1][1]
    pa = [math.exp(lp) for _, lp in top_a]
    pb = [math.exp(b.get(t, floor)) for t, _ in top_a]
    tail_a, tail_b = max(1 - sum(pa), 1e-12), max(1 - sum(pb), 1e-12)
    return sum(x * math.log(x / y) for x, y in zip(pa, pb) if x > 0) + tail_a * math.log(tail_a / tail_b)


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: compare_token_logprobs.py REFERENCE.tsv CANDIDATE.tsv')
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(a.keys() & b.keys())
    if len(keys) != len(a) or len(keys) != len(b):
        sys.exit('token dumps cover different positions: %d vs %d, %d shared' % (len(a), len(b), len(keys)))
    if any(a[k][0] != b[k][0] for k in keys):
        sys.exit('token dumps have different targets')
    deltas = sorted(b[k][1] - a[k][1] for k in keys)
    absd = sorted(abs(d) for d in deltas)
    kls = sorted(kl(a[k][3], b[k][3]) for k in keys)
    n = len(keys)
    q = lambda xs, f: xs[min(n - 1, int(f * n))]
    cases = {}
    for k in keys:
        cases.setdefault(k[0], [0.0, 0.0])
        cases[k[0]][0] -= a[k][1]
        cases[k[0]][1] -= b[k][1]
    worse = sum(1 for x, y in cases.values() if y > x + 1e-9)
    print('tokens %d, cases %d' % (n, len(cases)))
    print('mean NLL: A %.6f  B %.6f  (B-A %+.6f)' % (-sum(a[k][1] for k in keys) / n,
          -sum(b[k][1] for k in keys) / n, -sum(deltas) / n))
    print('target logprob |B-A|: mean %.5f  p50 %.5f  p99 %.5f  max %.4f' %
          (sum(absd) / n, q(absd, .5), q(absd, .99), absd[-1]))
    print('greedy agreement: %d/%d (%.3f%%)' % (sum(a[k][2] == b[k][2] for k in keys), n,
          100 * sum(a[k][2] == b[k][2] for k in keys) / n))
    print('approx KL(A||B) nats: mean %.2e  p50 %.2e  p99 %.2e  max %.2e' %
          (sum(kls) / n, q(kls, .5), q(kls, .99), kls[-1]))
    print('cases where B has higher NLL: %d/%d' % (worse, len(cases)))


if __name__ == '__main__':
    main()
