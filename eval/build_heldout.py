import argparse
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def sha256_lines(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-per-lang", type=int, default=40)
    ap.add_argument("--heldout-target", type=int, default=400)
    ap.add_argument("--reserve", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--langs", default="hi,gu,mr,ta,bn,kn,pa,en,mixed")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    with open(args.input, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)

            if not row.get("instruction") or not row.get("output") or not row.get("lang"):
                continue

            row["_uid"] = idx

            lang = row["lang"].lower()

            qtype = row.get("meta", {}).get("querytype", "other")
            qtype = qtype.lower() if qtype else "other"

            if qtype not in {
                "pest",
                "disease",
                "irrigation",
                "nutrient",
                "scheme",
                "weather",
            }:
                qtype = "other"

            row["_bucket"] = (lang, qtype)

            rows.append(row)

    buckets = defaultdict(list)

    for row in rows:
        buckets[row["_bucket"]].append(row)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    heldout = []
    selected = set()

    # Step 1: stratified sampling
    for bucket in buckets.values():
        take = min(args.min_per_lang, len(bucket))

        for row in bucket[:take]:
            if len(heldout) < args.heldout_target:
                heldout.append(row)
                selected.add(row["_uid"])

    # Step 2: fill remaining heldout
    remaining = [
        r for r in rows
        if r["_uid"] not in selected
    ]

    rng.shuffle(remaining)

    needed = args.heldout_target - len(heldout)

    for row in remaining[:needed]:
        heldout.append(row)
        selected.add(row["_uid"])

    # Step 3: reserve set
    leftover = [
        r for r in rows
        if r["_uid"] not in selected
    ]

    rng.shuffle(leftover)

    reserve = leftover[:args.reserve]

    def clean(row):
        row.pop("_bucket", None)
        row.pop("_uid", None)
        return row

    heldout = sorted(
        [clean(r) for r in heldout],
        key=lambda x: x.get("instruction", "")
    )

    reserve = sorted(
        [clean(r) for r in reserve],
        key=lambda x: x.get("instruction", "")
    )

    test_path = outdir / "test.jsonl"
    reserve_path = outdir / "reserve.jsonl"

    for path, data in [
        (test_path, heldout),
        (reserve_path, reserve),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def stats(data):
        lang = Counter()
        qtype = Counter()

        for r in data:
            lang[r["lang"]] += 1
            qtype[r.get("meta", {}).get("querytype", "other")] += 1

        return lang, qtype

    hl, hq = stats(heldout)
    rl, rq = stats(reserve)

    with open(outdir / "HASHES.md", "w", encoding="utf-8") as f:
        f.write("# Held-out and Reserve Hashes\n\n")
        f.write("| file | N | SHA-256 |\n")
        f.write("|---|---|---|\n")
        f.write(f"| test.jsonl | {len(heldout)} | {sha256_lines(test_path)} |\n")
        f.write(f"| reserve.jsonl | {len(reserve)} | {sha256_lines(reserve_path)} |\n\n")

        f.write("## Stratification\n\n")

        f.write("### Held-out language counts\n")
        for k, v in hl.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Held-out querytype counts\n")
        for k, v in hq.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Reserve language counts\n")
        for k, v in rl.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Reserve querytype counts\n")
        for k, v in rq.items():
            f.write(f"- {k}: {v}\n")

    print("Held-out:", len(heldout))
    print("Reserve:", len(reserve))


if __name__ == "__main__":
    main()
