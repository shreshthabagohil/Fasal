"""Data gate tests (05_TESTING_AND_VALIDATION §1). Exit non-zero on any failure
so nothing dirty gets merged or trained on. Run after EVERY dataset regeneration.

Usage:
    python data/tests.py --dataset dataset/kcc_instruct_v1.jsonl \
        [--heldout eval/heldout/test.jsonl]
"""
import argparse
import json
import re
import sys

sys.path.insert(0, "src")
from clean_kcc import normalize, PHONE_RE, EMAIL_RE, AADHAAR_RE  # reuse the same logic

ALLOWED_LANGS = {"hi", "gu", "mr", "ta", "bn", "kn", "pa", "en", "mixed", "unknown"}
MIN_LANG_VOLUME = 200  # tune at M1; below this a language is dropped + documented


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def t_schema(rows):
    bad = [i for i, r in enumerate(rows)
           if not r.get("instruction") or not r.get("output") or r.get("lang") not in ALLOWED_LANGS]
    return (len(bad) == 0, f"{len(bad)} rows fail schema/lang")


def t_leakage(rows, heldout):
    if not heldout:
        return (True, "SKIP (no held-out set provided yet)")
    train_q = {normalize(r["instruction"]) for r in rows}
    test_q = {normalize(r["instruction"]) for r in heldout}
    overlap = train_q & test_q
    return (len(overlap) == 0, f"{len(overlap)} train/test overlaps (MUST be 0)")


def t_dedup(rows):
    keys = [normalize(r["instruction"]) + "||" + normalize(r["output"]) for r in rows]
    dups = len(keys) - len(set(keys))
    return (dups == 0, f"{dups} exact-normalized duplicate pairs remain")


def t_pii(rows):
    hits = 0
    for r in rows:
        blob = f"{r.get('instruction','')} {r.get('output','')}"
        if PHONE_RE.search(blob) or EMAIL_RE.search(blob) or AADHAAR_RE.search(blob):
            hits += 1
    return (hits == 0, f"{hits} rows still contain phone/email/ID (MUST be 0)")


def t_coverage(rows):
    counts = {}
    for r in rows:
        counts[r["lang"]] = counts.get(r["lang"], 0) + 1
    thin = {l: c for l, c in counts.items() if c < MIN_LANG_VOLUME and l not in ("en",)}
    msg = f"counts={dict(sorted(counts.items(), key=lambda x:-x[1]))}; below-threshold={thin}"
    return (True, msg)  # informational — you lock the language set from this at M1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heldout", default=None)
    args = ap.parse_args()

    rows = load(args.dataset)
    heldout = load(args.heldout) if args.heldout else None

    checks = [
        ("D1 schema", t_schema(rows)),
        ("D2 leakage=0", t_leakage(rows, heldout)),
        ("D4 dedup", t_dedup(rows)),
        ("D5 PII", t_pii(rows)),
        ("D6 coverage", t_coverage(rows)),
    ]
    failed = 0
    for name, (ok, msg) in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        failed += 0 if ok else 1
    print("\nNOTE: D3 (embedding paraphrase audit) + D7 (spot-read) are manual — see 05 §1.")
    if failed:
        print(f"\n{failed} gate(s) FAILED — do not train on this dataset.")
        return 1
    print("\nAll automated gates green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
