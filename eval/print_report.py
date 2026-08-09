#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print a concise human-readable evaluation report."
    )
    parser.add_argument(
        "--report",
        default="eval/reports/eval_report.json",
        help="Aggregate report JSON path.",
    )
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["overall"]

    print("Fasal evaluation report")
    print("=======================")
    print(f"Items evaluated : {report['n_items']}")
    print(f"Items dropped   : {report['n_dropped']}")
    print(f"Base mean score : {overall['mean_score_base']:.3f}")
    print(f"Ours mean score : {overall['mean_score_ours']:.3f}")
    print(f"Delta           : {overall['delta']:+.3f}")
    print(
        f"95% CI          : "
        f"[{overall['ci95_lo']:+.3f}, {overall['ci95_hi']:+.3f}]"
    )
    print(f"Ours win rate   : {overall['win_rate_ours']:.3f}")
    print(f"McNemar p       : {overall['mcnemar_p']:.4f}")
    print(f"p(delta <= 0)   : {overall['p_le_0']:.4f}")
    print(f"Dataset version : {report['dataset_version']}")
    print(f"Held-out hash   : {report['heldout_hash']}")

    if report.get("by_language"):
        print()
        print("By language")
        print("-----------")
        for row in report["by_language"]:
            print(
                f"{row['lang']}: "
                f"n={row['n']} "
                f"delta={row['delta']:+.3f} "
                f"95% CI=[{row['ci95_lo']:+.3f}, {row['ci95_hi']:+.3f}]"
            )


if __name__ == "__main__":
    main()
