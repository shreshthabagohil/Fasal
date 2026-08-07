"""Merge the machine-translated per-language files into the main dataset.
Appends, never overwrites the English rows. Assigns each row a stable id
(needed downstream by build_heldout.py and the gold-subset template, which
both assume an 'id' field that clean_kcc.py's output never actually had).

Runs the SAME dedup + PII safety-net checks clean_kcc.py runs on the English
set — translation can make two distinct source queries converge to identical
translated text (new duplicates) and can reformat digit spacing in ways that
expose fresh PII-shaped patterns the English source never had. Concatenating
without re-checking would silently ship rows that fail D4/D5.

Safe to re-run: always re-reads dataset/kcc_instruct_v1.jsonl + the translated
files fresh, dedupes the WHOLE combined set, and overwrites. Running it twice
in a row is a no-op on row count (the second run's "added" rows are exact
duplicates of what's already in the file and get dropped by the dedup pass).
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, "data")
from clean_kcc import normalize, contains_pii  # same logic tests.py uses


def make_id(rec: dict) -> str:
    blob = (rec.get("lang", "") + "|" + rec["instruction"] + "|" + rec["output"]).encode("utf-8")
    return "kcc-" + hashlib.sha256(blob).hexdigest()[:12]


def main():
    main_path = pathlib.Path("dataset/kcc_instruct_v1.jsonl")
    translated_dir = pathlib.Path("dataset/translated")

    rows = []
    with open(main_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    added = 0
    for f in sorted(translated_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                    added += 1

    seen = set()
    kept = []
    dup_dropped = 0
    pii_dropped = 0
    for r in rows:
        if contains_pii(r["instruction"] + " " + r["output"]):
            pii_dropped += 1
            continue
        key = normalize(r["instruction"]) + "||" + normalize(r["output"])
        if key in seen:
            dup_dropped += 1
            continue
        seen.add(key)
        if "id" not in r:
            r["id"] = make_id(r)
        kept.append(r)

    with open(main_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    langs = {}
    for r in kept:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1

    print(f"Read {added} rows from dataset/translated/*.jsonl + existing main file "
          f"({len(rows)} rows total before dedup/PII pass).")
    print(f"After re-checking the full merged set: {dup_dropped} duplicate + "
          f"{pii_dropped} PII rows dropped, {len(kept)} total kept.")
    print("per-language:", dict(sorted(langs.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    sys.exit(main())
