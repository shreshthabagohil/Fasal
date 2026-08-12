"""eval/heldout/build_test_set.py — builds eval/heldout/test.jsonl + HASHES.md
from dataset/gold/*.jsonl, filtering out rows that leak into the training set.

Why this exists: D3_FLAGGED.md's automated paraphrase audit reduced the original
1,226-row gold set down to the ~244 rows currently committed under dataset/gold/,
but that audit only caught same-language near-duplicates (embedding cosine >= 0.9
between held-out and train text). It did NOT catch cross-lingual leakage: the KCC
source data was machine-translated into multiple target languages from the same
underlying English query+answer, so a row can appear in dataset/gold/hi.jsonl
under one id/language and an equivalent translation of the SAME source record can
independently appear in the 69,670-row training set under a different id/language.
That's still leakage for eval purposes even though no single-language text or id
matches.

This script re-checks every dataset/gold/*.jsonl row against the live HF training
set (Algo-Nova/fasal-kcc-instruct) using three signals:
  1) exact id match
  2) exact normalized instruction-text match
  3) cross-lingual match: same meta.orig_instruction_en + same normalized output,
     translated into a different language than the training-set copy

Verified 2026-08-12: 17 of 244 gold rows leaked via signal (3) only (0 via 1/2).
Clean remainder: 227 rows across 8 languages.

Usage:
    python eval/heldout/build_test_set.py
"""
import json
import re
import unicodedata
import hashlib
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "dataset" / "gold"
OUT_TEST = Path(__file__).resolve().parent / "test.jsonl"
OUT_HASHES = Path(__file__).resolve().parent / "HASHES.md"
LANGS = ["bn", "en", "gu", "hi", "kn", "mr", "pa", "ta"]


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sऀ-ൿ]", "", t)
    return t


def main() -> None:
    print("[build_test_set] loading Algo-Nova/fasal-kcc-instruct (train) ...")
    train = load_dataset("Algo-Nova/fasal-kcc-instruct", split="train")

    train_ids = set(train["id"])
    train_instr = {norm(x) for x in train["instruction"]}
    train_cross = set()
    for row in train:
        oe = norm((row.get("meta") or {}).get("orig_instruction_en", ""))
        out = norm(row.get("output", ""))
        if oe and out:
            train_cross.add((oe, out))

    clean = []
    leaked_total = 0
    for lang in LANGS:
        path = GOLD_DIR / f"{lang}.jsonl"
        if not path.exists():
            print(f"[build_test_set] WARNING: missing {path}, skipping")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            rid = row.get("id")
            if not rid:
                rid = "gold-" + hashlib.sha256(
                    (norm(row.get("instruction", "")) + "||" + norm(row.get("output", ""))).encode()
                ).hexdigest()[:12]
            instr_n = norm(row.get("instruction", ""))
            oe = norm((row.get("meta") or {}).get("orig_instruction_en", ""))
            out = norm(row.get("output", ""))
            cross_hit = bool(oe and out and (oe, out) in train_cross)
            if lang == "en" and not cross_hit:
                cross_hit = (instr_n, out) in train_cross
            leaked = (rid in train_ids) or (instr_n in train_instr) or cross_hit
            if leaked:
                leaked_total += 1
                continue
            clean.append(
                {
                    "id": rid,
                    "instruction": row["instruction"],
                    "question": row["instruction"],  # judge.py expects "question"; infer.py expects "instruction"
                    "input": row.get("input", ""),
                    "output": row.get("output", ""),  # reference gold answer, not used by infer.py/judge.py
                    "lang": row.get("lang", lang),
                    "meta": row.get("meta", {}),
                }
            )

    print(f"[build_test_set] clean rows: {len(clean)}  leaked (excluded): {leaked_total}")

    with open(OUT_TEST, "w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    blob = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in clean)
    heldout_hash = hashlib.sha256(blob.encode()).hexdigest()[:12]

    with open(OUT_HASHES, "w", encoding="utf-8") as f:
        f.write("# Held-out set hashes\n\n")
        f.write(f"- heldout: {len(clean)} rows, hash={heldout_hash}\n\n")
        f.write(
            f"Built by eval/heldout/build_test_set.py from dataset/gold/*.jsonl (244 rows), "
            f"filtering out {leaked_total} rows that leak into the training set via cross-lingual "
            "translation of the same source query+answer. See eval/heldout/LEAKAGE_AUDIT.md for detail.\n"
        )

    print(f"[build_test_set] wrote {OUT_TEST} and {OUT_HASHES} (hash={heldout_hash})")


if __name__ == "__main__":
    main()
