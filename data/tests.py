"""Data gate tests (05_TESTING_AND_VALIDATION §1). Exit non-zero on any failure
so nothing dirty gets merged or trained on. Run after EVERY dataset regeneration.

Usage:
    python data/tests.py --dataset dataset/kcc_instruct_v1.jsonl \
        [--heldout eval/heldout/test.jsonl]
"""
import argparse
import json
import sys

sys.path.insert(0, "data")  # clean_kcc.py lives here, not src/ (src/ only has seed.py) —
from clean_kcc import normalize, contains_pii  # corrected 2026-08-12; harmless before this
# fix only because Python auto-prepends the running script's own directory (data/) to
# sys.path, which happened to resolve the import anyway. Fixed for clarity, not behavior.

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


def t_leakage_cross_lingual(rows, heldout):
    """D2b — catches what D2 structurally cannot: 7 of 8 languages here are
    machine translations of the same English KCC source (see
    translate_to_indic.py), so the SAME underlying question can appear in
    train under one language's surface text and in held-out under another's
    — zero string overlap, but still the same example the model was trained
    on. meta.orig_instruction_en is the shared identity key that survives
    translation; D2 (above) only ever compares each row's own surface text,
    which is different per language by construction and can never catch
    this. Found for real 2026-08-12: 17 held-out rows leaked this way
    despite D2 and the D3 embedding audit both reporting clean — D3 uses a
    multilingual embedder (BGE-M3) but same-language paraphrases reliably
    score higher cosine similarity than genuine cross-lingual translations
    of identical meaning (the well-documented "language gap" in multilingual
    embedding spaces), so real translated duplicates can sit below D3's
    0.9 threshold precisely because they ARE cross-lingual. Full audit:
    eval/heldout/LEAKAGE_AUDIT.md.
    """
    if not heldout:
        return (True, "SKIP (no held-out set provided yet)")

    def source_key(r):
        src_text = (r.get("meta") or {}).get("orig_instruction_en") or r["instruction"]
        return normalize(src_text)

    train_src = {source_key(r) for r in rows}
    test_src = {source_key(r) for r in heldout}
    overlap = train_src & test_src
    return (len(overlap) == 0, f"{len(overlap)} train/test cross-lingual source overlaps (MUST be 0)")


def t_dedup(rows):
    keys = [normalize(r["instruction"]) + "||" + normalize(r["output"]) for r in rows]
    dups = len(keys) - len(set(keys))
    return (dups == 0, f"{dups} exact-normalized duplicate pairs remain")


def t_pii(rows):
    hits = 0
    for r in rows:
        blob = f"{r.get('instruction','')} {r.get('output','')}"
        if contains_pii(blob):
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
        ("D2b cross-lingual leakage=0", t_leakage_cross_lingual(rows, heldout)),
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
