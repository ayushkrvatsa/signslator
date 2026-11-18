"""
Leave-one-out cross-validation on ISL500 templates.

For each template:
  - remove it from the template pool
  - classify it against the REMAINING templates
  - check if the predicted sign is correct

This is the real test: it measures generalization across sessions of the
same signer, which is what live recognition has to do.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator.sign_to_speech import load_all_templates, find_best_match

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "signs")


def main():
    print("Loading templates...")
    templates = load_all_templates()
    signs = sorted(templates.keys())

    print(f"\n{len(signs)} signs, "
          f"{sum(len(v) for v in templates.values())} total templates")
    for s in signs:
        print(f"  {s}: {len(templates[s])} templates")

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 72)

    correct = 0
    total = 0
    confusion = {}   # (true, pred) -> count
    per_sign = {}    # sign -> [correct, total]

    for true_sign in signs:
        per_sign[true_sign] = [0, 0]
        for i in range(len(templates[true_sign])):
            # Build pool without this template
            pool = {}
            for s, tlist in templates.items():
                if s == true_sign:
                    remaining = [t for j, t in enumerate(tlist) if j != i]
                    if remaining:
                        pool[s] = remaining
                else:
                    pool[s] = tlist

            held_out = templates[true_sign][i]
            pred_sign, pred_dist, all_dists = find_best_match(held_out, pool)

            ok = (pred_sign == true_sign)
            correct += int(ok)
            total += 1
            per_sign[true_sign][1] += 1
            if ok:
                per_sign[true_sign][0] += 1

            key = (true_sign, pred_sign)
            confusion[key] = confusion.get(key, 0) + 1

            status = "OK " if ok else "MISS"
            print(f"  [{status}] true={true_sign:<8} pred={str(pred_sign):<8} "
                  f"dist={pred_dist:6.3f}")

    acc = 100 * correct / total if total else 0
    print("\n" + "=" * 72)
    print(f"LEAVE-ONE-OUT ACCURACY: {correct}/{total} ({acc:.1f}%)")
    print("=" * 72)

    print("\nPer-sign accuracy:")
    for s in signs:
        c, t = per_sign[s]
        bar = "#" * int(20 * c / t) if t else ""
        print(f"  {s:<8} {c}/{t}  {bar}")

    # Show misses
    misses = {k: v for k, v in confusion.items() if k[0] != k[1]}
    if misses:
        print("\nConfusions (true → predicted):")
        for (true_s, pred_s), cnt in sorted(misses.items(), key=lambda x: -x[1]):
            print(f"  {true_s} → {pred_s}: {cnt}×")
    else:
        print("\nNo confusions — perfect generalization.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
