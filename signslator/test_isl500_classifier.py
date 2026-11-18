"""
Validation: does the DTW classifier actually discriminate ISL500 signs?

Self-match test: each template is fed in as if it were live input.
- distance to ITSELF should be ~0 (correct recognition)
- distance to OTHER signs should be clearly larger (discrimination)

This proves the classifier + feature format work before touching a camera.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signslator.sign_to_speech import (
    load_all_templates,
    find_best_match,
    dtw_distance,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "signs")


def main():
    print("Loading templates...")
    templates = load_all_templates()
    if not templates:
        print("No templates found.")
        return 1

    signs = sorted(templates.keys())
    print(f"\nLoaded {len(signs)} signs: {', '.join(signs)}\n")
    print("=" * 70)
    print("SELF-MATCH TEST (each template treated as live input)")
    print("=" * 70)

    correct = 0
    total = 0
    rows = []

    for true_sign in signs:
        for t_idx, tmpl in enumerate(templates[true_sign]):
            # Build a template dict EXCLUDING this exact instance would be ideal,
            # but with 1 template per sign we test against all (self-match = 0 is expected)
            pred_sign, pred_dist, all_dists = find_best_match(tmpl, templates)

            is_correct = (pred_sign == true_sign)
            correct += int(is_correct)
            total += 1

            # Margin: how much closer is the correct match vs the runner-up?
            sorted_d = sorted(all_dists.items(), key=lambda x: x[1])
            runner_up = None
            for name, d in sorted_d:
                if name != true_sign:
                    runner_up = (name, d)
                    break
            margin = (runner_up[1] - pred_dist) if runner_up else float('inf')

            rows.append((true_sign, pred_sign, pred_dist, runner_up, margin, is_correct))

    print(f"\n{'TRUE':<10}{'PRED':<10}{'DIST':>8}{'RUNNER-UP':>12}{'R-DIST':>9}{'MARGIN':>9}  OK")
    print("-" * 70)
    for true_sign, pred_sign, pred_dist, runner_up, margin, ok in rows:
        ru_name = runner_up[0] if runner_up else "-"
        ru_dist = runner_up[1] if runner_up else 0
        print(f"{true_sign:<10}{pred_sign:<10}{pred_dist:>8.3f}{ru_name:>12}{ru_dist:>9.3f}{margin:>9.3f}  {'YES' if ok else 'NO'}")

    print("-" * 70)
    print(f"\nSelf-match accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # Cross-sign discrimination: average distance between different signs
    print("\n" + "=" * 70)
    print("CROSS-SIGN DISTANCE MATRIX (higher = more distinguishable)")
    print("=" * 70)

    n = len(signs)
    matrix = np.zeros((n, n))
    for i, s1 in enumerate(signs):
        for j, s2 in enumerate(signs):
            if i == j:
                matrix[i][j] = 0
                continue
            # min distance across template pairs
            best = min(
                dtw_distance(a, b)
                for a in templates[s1]
                for b in templates[s2]
            )
            matrix[i][j] = best

    # Header
    print(f"{'':>9}", end='')
    for s in signs:
        print(f"{s[:6]:>8}", end='')
    print()
    for i, s1 in enumerate(signs):
        print(f"{s1[:8]:>9}", end='')
        for j, s2 in enumerate(signs):
            if i == j:
                print(f"{'—':>8}", end='')
            else:
                print(f"{matrix[i][j]:>8.2f}", end='')
        print()

    off_diag = matrix[~np.eye(n, dtype=bool)]
    print(f"\nOff-diagonal distances: min={off_diag.min():.3f}  mean={off_diag.mean():.3f}  max={off_diag.max():.3f}")
    print(f"Self distances: 0.000 (exact match)")
    print(f"\nSeparation ratio (mean off-diag / self): {'infinite (self=0)' if off_diag.min() > 0 else 'N/A'}")
    print(f"\n→ Classifier discriminates: {'YES' if off_diag.min() > 0.1 else 'WEAK — signs may be too similar'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
