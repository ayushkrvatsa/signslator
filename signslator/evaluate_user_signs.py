"""
Leave-one-out accuracy on the user's own recordings.

For each template: nearest same-sign template vs nearest different-sign
template. Reports how often the correct sign wins, and the distance gap.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator.sign_to_speech import (
    load_all_templates, dtw_distance, _build_prefilter, _resample,
)


def main():
    templates = load_all_templates()
    if not templates:
        print("No recordings found.")
        return

    signs = sorted(templates)
    total = sum(len(v) for v in templates.values())
    print(f"{len(signs)} signs, {total} recordings\n")

    correct = 0
    wrong = []
    margins = []

    for true_sign in signs:
        for idx, seq in enumerate(templates[true_sign]):
            best_same = float("inf")
            for j, other in enumerate(templates[true_sign]):
                if j == idx:
                    continue
                d = dtw_distance(seq, other, max_len=64)
                best_same = min(best_same, d)

            best_diff, best_diff_sign = float("inf"), None
            for s in signs:
                if s == true_sign:
                    continue
                for other in templates[s]:
                    d = dtw_distance(seq, other, max_len=64)
                    if d < best_diff:
                        best_diff, best_diff_sign = d, s

            win = best_same < best_diff
            correct += win
            if best_diff > 0:
                margins.append(best_diff / best_same if best_same > 0 else 99)
            if not win:
                wrong.append((true_sign, idx, best_diff_sign, best_same, best_diff))

    n = total
    print(f"LEAVE-ONE-OUT ACCURACY: {correct}/{n} = {100*correct/n:.1f}%")
    if margins:
        m = np.array(margins)
        print(f"margin (diff/same): mean {m.mean():.2f}x  p10 {np.percentile(m,10):.2f}x")
    print()
    if wrong:
        print(f"confusions ({len(wrong)}):")
        for s, i, other, ds, dd in wrong[:15]:
            print(f"  {s}[{i}] -> {other}   same={ds:.2f} diff={dd:.2f}")
    else:
        print("no confusions")

    # Also: how well does the current threshold separate?
    print("\n--- threshold check ---")
    print("A threshold must sit between same-sign and different-sign distances.")


if __name__ == "__main__":
    main()
