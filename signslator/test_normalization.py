"""
A/B test: scale normalization on vs off, measured on leave-one-out accuracy.

Builds both feature sets from the cached .h5 files, runs leave-one-out on each,
and prints the comparison. This tells us whether the change is worth keeping.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

import signslator.sign_to_speech as sts

CACHE = os.path.join(os.path.dirname(__file__), "isl500_cache")
DATA = os.path.join(os.path.dirname(__file__), "data", "signs")


def build_templates(normalize: bool) -> dict:
    """Convert cached .h5 into templates with the given normalization setting."""
    import signslator.isl500_loader as loader
    import re
    out = {}
    for fname in sorted(os.listdir(CACHE)):
        if not fname.endswith(".h5"):
            continue
        word = loader.parse_word_from_filename(fname)
        m = re.search(r"session(\d+)", fname)
        if not word or not m:
            continue
        with h5py.File(os.path.join(CACHE, fname), "r") as f:
            h5 = f["intermediate"][:]
        feats = loader.ist500_to_features(h5, normalize=normalize)
        out.setdefault(word.upper(), []).append(feats)
    return out


def loo_accuracy(templates: dict) -> tuple:
    correct = total = 0
    per_sign = {}
    for s in sorted(templates):
        per_sign[s] = [0, 0]
        for i in range(len(templates[s])):
            pool = {}
            for k, v in templates.items():
                if k == s:
                    rem = [x for j, x in enumerate(v) if j != i]
                    if rem:
                        pool[k] = rem
                else:
                    pool[k] = v
            pred, d, _ = sts.find_best_match(templates[s][i], pool)
            ok = (pred == s)
            correct += ok
            total += 1
            per_sign[s][0] += ok
            per_sign[s][1] += 1
    return correct, total, per_sign


def main():
    print("Building feature sets...\n")
    raw = build_templates(normalize=False)
    norm = build_templates(normalize=True)

    print("Running leave-one-out on both...\n")
    c_raw, n_raw, ps_raw = loo_accuracy(raw)
    c_norm, n_norm, ps_norm = loo_accuracy(norm)

    a_raw = 100 * c_raw / n_raw if n_raw else 0
    a_norm = 100 * c_norm / n_norm if n_norm else 0

    print("=" * 60)
    print(f"  RAW coordinates:        {c_raw}/{n_raw} = {a_raw:.1f}%")
    print(f"  SCALE-NORMALIZED:       {c_norm}/{n_norm} = {a_norm:.1f}%")
    print(f"  Difference:             {a_norm - a_raw:+.1f} percentage points")
    print("=" * 60)

    print("\nPer-sign:")
    print(f"  {'SIGN':<10}{'RAW':>8}{'NORMALIZED':>14}")
    print("  " + "-" * 32)
    for s in sorted(ps_raw):
        r = ps_raw[s]
        n = ps_norm.get(s, [0, 0])
        print(f"  {s:<10}{r[0]}/{r[1]:<6}{n[0]}/{n[1]}")

    verdict = "KEEP normalization" if a_norm >= a_raw else "REVERT normalization"
    print(f"\n→ {verdict}")


if __name__ == "__main__":
    main()
