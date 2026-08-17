#!/usr/bin/env python3
"""
M3 step 6 — human-anchor the judge.

Fills the 'judge' column in eval/reports/judge_anchor_scores.csv by asking
the SAME frozen judge (eval/judge.py's prompt + Groq call + key rotation)
to rate each gold answer. We never touch judge_prompt.md (frozen, per
A0_LOCKED) -- instead we pass the SAME answer as both "A" and "B", so
score_A == score_B and we get a single 1-5 quality rating per row.

Then, once you (Lead) and T3 have filled in the human_lead / human_t3
columns by hand (1-5, your own judgment of each answer's quality), this
script computes Kendall's tau between each human and the judge, and
writes eval/reports/judge_anchor.md with the verdict.

Usage:
  # Step 1: fill judge scores (needs GROQ_API_KEYS or GROQ_API_KEY set)
  python eval/judge_anchor.py --csv eval/reports/judge_anchor_scores.csv --fill-judge

  # Step 2 (after you and T3 hand-score the human_lead/human_t3 columns):
  python eval/judge_anchor.py --csv eval/reports/judge_anchor_scores.csv --report
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import KeyPool, build_user_message, load_prompt, request_judge  # noqa: E402
import os
import requests


def read_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: str, rows, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fill_judge_scores(csv_path: str) -> None:
    raw_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
    if not raw_keys:
        print("ERROR: set GROQ_API_KEYS or GROQ_API_KEY first.", file=sys.stderr)
        sys.exit(2)
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    key_pool = KeyPool(keys)
    print(f"judge_anchor.py: using {len(key_pool)} Groq key(s).")

    prompt_template = load_prompt()
    rows = read_rows(csv_path)
    session = requests.Session()

    filled = 0
    for row in rows:
        if row.get("judge", "").strip():
            continue  # already scored, don't re-spend budget
        user_message = build_user_message(
            prompt_template,
            row["question"],
            row["context"],
            row["output"],
            row["output"],  # same answer as A and B -> single quality score
        )
        result = request_judge(
            session=session,
            key_pool=key_pool,
            # 2026-08-17: was llama-3.3-70b-versatile; Groq removed it from
            # this account's model list (see eval/JUDGE_BUDGET.md addendum).
            # NOTE: the tau=0.481 human-anchor result already on record was
            # computed against the OLD model, not this one -- disclosed
            # limitation, not silently carried forward as still-valid.
            model="openai/gpt-oss-120b",
            temperature=0,
            seed=1729,
            user_message=user_message,
            max_retries=5,
            sleep_on_429=60,
        )
        row["judge"] = str(result["score_A"])
        filled += 1
        print(f"  {row['id']} ({row['lang']}) -> judge={row['judge']}")

    write_rows(csv_path, rows, fieldnames=["id", "lang", "question", "context", "output", "human_lead", "human_t3", "judge"])
    print(f"Filled {filled} new judge scores. Total rows: {len(rows)}.")


def write_report(csv_path: str, report_path: str) -> None:
    from scipy.stats import kendalltau

    rows = read_rows(csv_path)

    def paired(col: str):
        h, j = [], []
        for r in rows:
            hv, jv = r.get(col, "").strip(), r.get("judge", "").strip()
            if hv and jv:
                h.append(int(hv))
                j.append(int(jv))
        return h, j

    lead_h, lead_j = paired("human_lead")
    t3_h, t3_j = paired("human_t3")

    tau_lead = kendalltau(lead_h, lead_j)[0] if len(lead_h) >= 2 else None
    tau_t3 = kendalltau(t3_h, t3_j)[0] if len(t3_h) >= 2 else None

    def fmt(t):
        return f"{t:.3f}" if t is not None else "N/A (not enough scored rows yet)"

    verdict = "accepted for steering"
    for t in (tau_lead, tau_t3):
        if t is not None and t < 0.4:
            verdict = "REVISED NEEDED — see judge_prompt.md, tau below 0.4"

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Judge anchor (M3 step 6)\n\n")
        f.write(f"N pairs scored (Lead): {len(lead_h)} / 40\n")
        f.write(f"N pairs scored (T3): {len(t3_h)} / 40\n\n")
        f.write(f"Kendall tau (Lead vs Llama-70B judge): {fmt(tau_lead)}\n")
        f.write(f"Kendall tau (T3   vs Llama-70B judge): {fmt(tau_t3)}\n\n")
        f.write(f"Verdict: {verdict}\n")

    print(f"Wrote {report_path}")
    print(f"tau(Lead)={fmt(tau_lead)}  tau(T3)={fmt(tau_t3)}  verdict={verdict}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval/reports/judge_anchor_scores.csv")
    parser.add_argument("--report-out", default="eval/reports/judge_anchor.md")
    parser.add_argument("--fill-judge", action="store_true", help="call Groq to fill the judge column")
    parser.add_argument("--report", action="store_true", help="compute tau + write judge_anchor.md")
    args = parser.parse_args()

    if not args.fill_judge and not args.report:
        print("Nothing to do — pass --fill-judge and/or --report.", file=sys.stderr)
        return 2

    if args.fill_judge:
        fill_judge_scores(args.csv)
    if args.report:
        write_report(args.csv, args.report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
