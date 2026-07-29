"""Paired-bootstrap significance for the improvement claim (05 §0.1).

Feed it per-item scores for baseline and ours on the SAME held-out items
(LLM-judge 1-5, or chrF). Reports the mean delta, 95% CI, and empirical p-value.
Claim bar: the 95% CI on the delta must EXCLUDE 0.

Usage (from a scores JSONL with {"base": x, "ours": y} per line):
    python eval/bootstrap.py --scores eval/out/scores_iterN.jsonl
"""
import argparse
import json

import numpy as np


def paired_bootstrap(base, ours, n=10_000, seed=0):
    base, ours = np.asarray(base, float), np.asarray(ours, float)
    assert len(base) == len(ours), "paired: base and ours must be same length / same items"
    d = ours - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "n_items": int(len(d)),
        "delta": float(d.mean()),
        "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        "p_le_0": float((boot <= 0).mean()),
        "significant": bool(np.percentile(boot, 2.5) > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSONL with base/ours per item")
    ap.add_argument("--key-base", default="base")
    ap.add_argument("--key-ours", default="ours")
    args = ap.parse_args()

    base, ours = [], []
    with open(args.scores, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            base.append(r[args.key_base])
            ours.append(r[args.key_ours])

    res = paired_bootstrap(base, ours)
    print(json.dumps(res, indent=2))
    print("\nHeadline (for the model card):")
    lo, hi = res["ci95"]
    verdict = "SIGNIFICANT" if res["significant"] else "NOT yet significant (CI includes 0)"
    print(f"  delta={res['delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p(<=0)={res['p_le_0']:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
