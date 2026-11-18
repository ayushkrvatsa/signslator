"""
Full-dataset accuracy evaluation across all downloaded signs.

Leave-one-out, but with subsampling of the *other* signs so it stays fast:
each held-out sample is compared against
  - every other sample of the SAME sign (must win)
  - one representative sample of every OTHER sign (the competition)

Reports top-1 accuracy and the most common confusions.

Usage:
  python run.py full-eval
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator.sign_to_speech import load_all_templates, find_best_match

random.seed(42)


def main():
    print("Loading templates...")
    templates = load_all_templates()
    signs = sorted(templates.keys())
    total_templates = sum(len(v) for v in templates.values())
    print(f"{len(signs)} signs, {total_templates} templates\n")

    if len(signs) < 2:
        print("Need at least 2 signs.")
        return

    # Build the competition pool: 1 representative per OTHER sign
    rep = {s: random.choice(v) for s, v in templates.items()}

    correct = 0
    total = 0
    per_sign = {}
    confusions = {}
    dists_correct = []
    dists_wrong = []

    for true_sign in signs:
        per_sign[true_sign] = [0, 0]
        seqs = templates[true_sign]
        for i in range(len(seqs)):
            # pool = same-sign (minus held out) + one rep of each other sign
            pool = {}
            rem = [x for j, x in enumerate(seqs) if j != i]
            if rem:
                pool[true_sign] = rem
            for s, r in rep.items():
                if s != true_sign:
                    pool.setdefault(s, []).append(r)

            pred, dist, _ = find_best_match(seqs[i], pool)
            ok = (pred == true_sign)
            correct += ok
            total += 1
            per_sign[true_sign][0] += ok
            per_sign[true_sign][1] += 1
            (dists_correct if ok else dists_wrong).append(dist)
            if not ok:
                confusions[(true_sign, pred)] = confusions.get((true_sign, pred), 0) + 1

        c, n = per_sign[true_sign]
        print(f"  {true_sign:<16} {c}/{n}")

    acc = 100 * correct / total if total else 0
    print("\n" + "=" * 60)
    print(f"FULL-DATASET ACCURACY: {correct}/{total} = {acc:.1f}%")
    print("=" * 60)

    if dists_correct:
        print(f"\nMean distance — correct: {np.mean(dists_correct):.3f}   "
              f"wrong: {np.mean(dists_wrong):.3f}" if dists_wrong else
              f"\nMean distance — correct: {np.mean(dists_correct):.3f}")

    # Weakest signs
    weak = sorted(per_sign.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0]))
    print("\nWeakest signs (worth reviewing):")
    for s, (c, n) in weak[:10]:
        print(f"  {s:<16} {c}/{n}")

    if confusions:
        print("\nMost common confusions:")
        for (t, p), k in sorted(confusions.items(), key=lambda x: -x[1])[:12]:
            print(f"  {t} → {p}   ({k}×)")

    return acc


if __name__ == "__main__":
    main()
