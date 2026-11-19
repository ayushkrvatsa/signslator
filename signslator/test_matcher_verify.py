"""
Sampled check: two-stage matcher vs exhaustive search.

The prefilter already measured 100% recall at top-5, which implies the DTW-best
sign is always among the candidates — so the two-stage result should equal the
exhaustive result. This confirms it empirically on a sample.
"""

import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator.sign_to_speech import (
    load_all_templates, dtw_distance, find_best_match, _build_prefilter,
)

SAMPLE = 30
random.seed(7)


def exhaustive(live, templates):
    """Ground truth: best match by DTW over EVERY template (as live matching does)."""
    best, best_d = None, float("inf")
    for s, seqs in templates.items():
        for t in seqs:
            d = dtw_distance(live, t)
            if d < best_d:
                best, best_d = s, d
    return best


def main():
    print("Loading templates...")
    templates = load_all_templates()
    prefilter = _build_prefilter(templates)

    pairs = [(s, t) for s, seqs in templates.items() for t in seqs]
    sample = random.sample(pairs, min(SAMPLE, len(pairs)))
    print(f"checking {len(sample)} templates\n")

    agree = disagree = 0
    t_ex = t_ts = 0.0

    for true_sign, tmpl in sample:
        t0 = time.time()
        ex = exhaustive(tmpl, templates)
        t_ex += time.time() - t0

        t0 = time.time()
        ts, _, _ = find_best_match(tmpl, templates, prefilter=prefilter)
        t_ts += time.time() - t0

        if ex == ts:
            agree += 1
            print(f"  {true_sign:<14} same")
        else:
            disagree += 1
            print(f"  {true_sign:<14} DIFF (exhaustive={ex}, two-stage={ts})")

    n = agree + disagree
    print("\n" + "=" * 56)
    print(f"AGREEMENT: {agree}/{n} = {100*agree/n:.0f}%")
    print(f"  exhaustive: {t_ex/n*1000:.0f} ms/query")
    print(f"  two-stage : {t_ts/n*1000:.0f} ms/query   ({t_ex/t_ts:.0f}x faster)")
    print("=" * 56)


if __name__ == "__main__":
    main()
