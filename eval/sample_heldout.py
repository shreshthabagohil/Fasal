#!/usr/bin/env python3
"""eval/sample_heldout.py — deterministic, language-stratified subsampling of a
held-out eval set, so eval/run_eval.sh can enforce a fixed judge-call budget
(see eval/JUDGE_BUDGET.md) instead of relying on someone remembering to slice
the file by hand before running the pipeline.

Why this exists: eval/JUDGE_BUDGET.md locks two different N values (<=80 for
M3/M4 iterative comparisons, N=170 dual-order for the final M5 report,
340 total judge calls) but nothing in the codebase enforced them -- judge.py
processes every row of whatever --heldout file it is given, so the documented
budget was just a comment someone had to remember to honor by hand-editing a
file before each run. Wiring the budget into an explicit, reusable script
means: (a) MAX_N is a single env var in eval/run_eval.sh, not a manual step,
(b) the same seed always produces the same subset, so results are
reproducible run to run, and (c) the subset is stratified by language so a
small N doesn't accidentally drop or under-represent a low-resource language,
which a plain random.sample() over the full list could easily do.

Usage:
    python eval/sample_heldout.py --in eval/heldout/test.jsonl \
        --out eval/heldout/final_report_n170.jsonl --n 170 --seed 1729
"""
import argparse
import collections
import json
import sys


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(rows, n, seed):
    """Proportionally allocate n across languages present in rows, then
    seeded-shuffle within each language group before truncating -- so the
    same seed always yields the same subset, and every language keeps at
    least 1 row (if it has any) as long as n >= number of languages."""
    import random

    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r.get("lang", "unknown")].append(r)

    langs = sorted(by_lang)  # sorted for determinism independent of file order
    total = len(rows)
    if n >= total:
        return list(rows)

    rng = random.Random(seed)
    for lang in langs:
        rng.shuffle(by_lang[lang])

    # Largest-remainder proportional allocation so every language with rows
    # gets at least 1 slot (when n >= len(langs)), not just the biggest ones.
    raw_targets = {lang: (len(by_lang[lang]) / total) * n for lang in langs}
    targets = {lang: max(1, int(raw_targets[lang])) for lang in langs}
    allocated = sum(targets.values())

    remainders = sorted(langs, key=lambda l: raw_targets[l] - int(raw_targets[l]), reverse=True)
    i = 0
    while allocated < n and i < len(remainders):
        lang = remainders[i]
        if targets[lang] < len(by_lang[lang]):
            targets[lang] += 1
            allocated += 1
        i += 1
        if i >= len(remainders):
            i = 0
            if all(targets[l] >= len(by_lang[l]) for l in langs):
                break  # every language is fully exhausted, can't reach n

    sample = []
    for lang in langs:
        sample.extend(by_lang[lang][: targets[lang]])

    # Final seeded shuffle across languages so downstream order isn't
    # grouped-by-language (matters for e.g. rate-limit pacing patterns).
    rng.shuffle(sample)
    return sample[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    if not rows:
        print(f"ERROR: {args.inp} has no rows", file=sys.stderr)
        return 1

    sample = stratified_sample(rows, args.n, args.seed)
    write_jsonl(args.out, sample)

    by_lang = collections.Counter(r.get("lang", "unknown") for r in sample)
    print(f"[sample_heldout] {args.inp}: {len(rows)} rows -> {args.out}: {len(sample)} rows "
          f"(seed={args.seed})")
    print(f"[sample_heldout] per-language counts: {dict(sorted(by_lang.items(), key=lambda x: -x[1]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
