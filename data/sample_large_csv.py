"""Reservoir-sample N rows from a huge CSV without loading it into memory.
For the 7.8GB kcc_dataset.csv (42.1M rows) — pandas.read_csv on the full
file will exhaust memory on a laptop. This streams the file once, keeping a
fixed-size random sample, and writes a small CSV that clean_kcc.py (or
pandas generally) can then handle normally.

Usage:
    python sample_large_csv.py --in /tmp/kcc-check/kcc_dataset.csv \
        --out data/raw/kcc.csv --n 100000 --seed 1729
"""
import argparse
import csv
import random
import sys

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir = []
    header = None
    total = 0
    malformed = 0
    n_cols = None

    with open(args.inp, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                malformed += 1
                continue

            if len(row) != n_cols:
                # desynced parse (usually an unescaped quote/newline upstream) —
                # skip rather than silently misalign columns
                malformed += 1
                continue

            total += 1
            if len(reservoir) < args.n:
                reservoir.append(row)
            else:
                j = rng.randint(0, total - 1)
                if j < args.n:
                    reservoir[j] = row
            if total % 2_000_000 == 0:
                print(f"  scanned {total:,} rows... ({malformed} malformed skipped)",
                      file=sys.stderr)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(reservoir)

    print(f"Scanned {total:,} good rows total ({malformed:,} malformed rows skipped), "
          f"sampled {len(reservoir):,} -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
