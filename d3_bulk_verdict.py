"""D3 triage helper — auto-fills 'remove' for near-certain duplicate pairs
(cosine >= --auto-threshold) so a human only needs to manually review the
genuinely ambiguous middle band, not all flagged rows one by one.

Why: with 4,569 flagged pairs, full row-by-row manual review isn't realistic
under the calendar. Most high-cosine pairs are the same question restated
(e.g. "asking about X" vs "asking about the X") — mechanically the same
judgment call repeated thousands of times. This script handles those
automatically and leaves only the ambiguous band for a human.

Usage:
    python d3_bulk_verdict.py --in eval/heldout/D3_FLAGGED.md \
        --auto-threshold 0.97 --sample-size 150 --seed 1729 \
        --out eval/heldout/D3_FLAGGED.md

After running: open the output file. Rows above --auto-threshold already say
"remove". A random --sample-size subset of the 0.90-0.97 band is marked
"REVIEW ME" — go through just those by hand and replace with keep/remove.
Once you've done that sample, if most of them came out "remove", it's fine
to bulk-mark the rest of the 0.90-0.97 band "remove" too (ask Lead to
confirm this call before doing it — it's a real judgment call, not a given).
"""
import argparse
import random
import re
import sys


def parse_rows(lines):
    header, rows, rest = [], [], []
    in_table = False
    for line in lines:
        if line.startswith("| cosine"):
            in_table = True
            header.append(line)
            continue
        if in_table and line.startswith("|---"):
            header.append(line)
            continue
        if in_table and line.startswith("|"):
            rows.append(line.rstrip("\n"))
            continue
        rest.append(line)
    return header, rows, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auto-threshold", type=float, default=0.97)
    ap.add_argument("--sample-size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        lines = f.readlines()

    header, rows, rest = parse_rows(lines)
    if not rows:
        sys.exit("No table rows found — check the file path/format.")

    mid_band = []
    auto_removed = 0
    out_rows = []
    for r in rows:
        parts = [p.strip() for p in r.strip("|").split("|")]
        if len(parts) != 4:
            out_rows.append(r)
            continue
        cosine_str, ht, tt, verdict = parts
        try:
            cosine = float(cosine_str)
        except ValueError:
            out_rows.append(r)
            continue
        if cosine >= args.auto_threshold:
            verdict = "remove (auto, cosine >= %.2f)" % args.auto_threshold
            auto_removed += 1
        else:
            mid_band.append(len(out_rows))
        out_rows.append(f"| {cosine_str} | {ht} | {tt} | {verdict} |")

    rng = random.Random(args.seed)
    sample_idx = set(rng.sample(mid_band, min(args.sample_size, len(mid_band))))
    for i in sample_idx:
        r = out_rows[i]
        parts = r.strip("|").split("|")
        parts[-1] = " REVIEW ME "
        out_rows[i] = "|" + "|".join(parts) + "|"

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(rest[:1])  # title line
        f.writelines(rest[1:])
        f.writelines(l if l.endswith("\n") else l + "\n" for l in header)
        for r in out_rows:
            f.write(r + "\n")

    print(f"Total flagged rows: {len(rows)}")
    print(f"Auto-marked 'remove' (cosine >= {args.auto_threshold}): {auto_removed}")
    print(f"Mid-band (0.90-{args.auto_threshold}) rows: {len(mid_band)}")
    print(f"Sampled for manual review (marked REVIEW ME): {len(sample_idx)}")
    print(f"\nNext: open {args.out}, search for 'REVIEW ME', replace each with keep/remove by hand.")


if __name__ == "__main__":
    sys.exit(main())
