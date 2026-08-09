#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar

from bootstrap import paired_bootstrap


def load_scores(path):
    rows = []
    dropped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "error" in row:
                dropped += 1
                continue

            rows.append(row)

    return rows, dropped


def bootstrap_result(rows, n_bootstrap, seed):
    base = [float(r["score_base"]) for r in rows]
    ours = [float(r["score_ours"]) for r in rows]

    result = paired_bootstrap(
        base,
        ours,
        n=n_bootstrap,
        seed=seed,
    )

    lo, hi = result["ci95"]

    return {
        "n": len(rows),
        "delta": float(result["delta"]),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_le_0": float(result["p_le_0"]),
    }


def win_counts(rows):
    ours_wins = sum(r.get("choice_final") == "ours" for r in rows)
    base_wins = sum(r.get("choice_final") == "base" for r in rows)
    ties = sum(r.get("choice_final") == "tie" for r in rows)

    return ours_wins, base_wins, ties


def mcnemar_p_value(ours_wins, base_wins):
    discordant = ours_wins + base_wins

    if discordant == 0:
        return 1.0

    table = [
        [0, base_wins],
        [ours_wins, 0],
    ]

    result = mcnemar(table, exact=True)

    return float(result.pvalue)


def read_metadata():
    # Reads the two identifiers that pin this report to an exact
    # dataset + held-out snapshot, so a judge can verify which data
    # this number was measured on (RULEBOOK / 05 cold-reproduce guarantee).
    repo_root = Path(__file__).resolve().parents[1]

    dataset_version_path = repo_root / "dataset" / "VERSION"
    hashes_path = repo_root / "eval" / "heldout" / "HASHES.md"

    dataset_version = dataset_version_path.read_text(
        encoding="utf-8"
    ).strip()

    hashes_text = hashes_path.read_text(encoding="utf-8")

    # HASHES.md format (frozen at build time), e.g.:
    #   - heldout: 1226 rows, hash=150979417f9c
    # Parse the real held-out hash out of that line instead of hashing
    # an unrelated file (a prior bug here hashed D3_FLAGGED.md, which
    # is a paraphrase-audit report, not the held-out set itself).
    match = re.search(r"heldout:.*?hash=([0-9a-fA-F]+)", hashes_text)

    if not match:
        raise ValueError(
            f"could not find a 'heldout: ... hash=<hex>' line in {hashes_path}"
        )

    heldout_hash = match.group(1)

    return dataset_version, heldout_hash


def build_report(rows, dropped, bootstrap_n, seed):
    overall_boot = bootstrap_result(
        rows,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )

    ours_wins, base_wins, ties = win_counts(rows)

    total = ours_wins + base_wins + ties

    win_rate_ours = (
        float(ours_wins / total)
        if total
        else 0.0
    )

    mcnemar_p = mcnemar_p_value(
        ours_wins,
        base_wins,
    )

    mean_base = sum(float(r["score_base"]) for r in rows) / len(rows)
    mean_ours = sum(float(r["score_ours"]) for r in rows) / len(rows)

    languages = sorted(
        {r.get("lang", "unknown") for r in rows}
    )

    by_language = []

    for lang in languages:
        lang_rows = [
            r for r in rows
            if r.get("lang", "unknown") == lang
        ]

        lang_boot = bootstrap_result(
            lang_rows,
            n_bootstrap=bootstrap_n,
            seed=seed,
        )

        by_language.append(
            {
                "lang": lang,
                "n": len(lang_rows),
                "delta": lang_boot["delta"],
                "ci95_lo": lang_boot["ci95_lo"],
                "ci95_hi": lang_boot["ci95_hi"],
            }
        )

    by_language.sort(
        key=lambda x: (-x["n"], x["lang"])
    )

    dataset_version, heldout_hash = read_metadata()

    return {
        "n_items": len(rows),
        "n_dropped": dropped,
        "overall": {
            "mean_score_base": float(mean_base),
            "mean_score_ours": float(mean_ours),
            "delta": overall_boot["delta"],
            "ci95_lo": overall_boot["ci95_lo"],
            "ci95_hi": overall_boot["ci95_hi"],
            "p_le_0": overall_boot["p_le_0"],
            "win_rate_ours": win_rate_ours,
            "mcnemar_p": mcnemar_p,
        },
        "by_language": by_language,
        "dataset_version": dataset_version,
        "heldout_hash": heldout_hash,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        default="eval/out/scores.jsonl",
    )

    parser.add_argument(
        "--out",
        default="eval/reports/eval_report.json",
    )

    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
    )

    args = parser.parse_args()

    try:
        rows, dropped = load_scores(args.scores)

        print(f"loaded_items={len(rows)} dropped_errors={dropped}")

        report = build_report(
            rows,
            dropped,
            args.bootstrap_n,
            args.seed,
        )

        output_path = Path(args.out)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

        overall = report["overall"]

        print(
            "overall "
            f"delta={overall['delta']:+.4f} "
            f"95%CI=[{overall['ci95_lo']:+.4f}, "
            f"{overall['ci95_hi']:+.4f}] "
            f"McNemar_p={overall['mcnemar_p']:.4f} "
            f"win_rate_ours={overall['win_rate_ours']:.4f}"
        )

        print("\nper-language:")
        print("lang\tn\tdelta\tci95_lo\tci95_hi")

        for item in report["by_language"]:
            print(
                f"{item['lang']}\t"
                f"{item['n']}\t"
                f"{item['delta']:+.4f}\t"
                f"{item['ci95_lo']:+.4f}\t"
                f"{item['ci95_hi']:+.4f}"
            )

    except Exception as exc:
        print(f"aggregate error: {exc}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
