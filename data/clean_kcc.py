"""KCC cleaning pipeline → instruction dataset (per 02_EXECUTION_PLAN §A3).

Turns raw KCC rows into {instruction, input, output, lang, meta} JSONL.
Steps: drop garbage → normalize → strip PII → language-tag → dedup.
Language-set locking and quality filtering are marked as TODO (need real data volumes at M1).

Usage:
    python data/clean_kcc.py --in data/raw/kcc.csv --out dataset/kcc_instruct_v1.jsonl
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata

# --- PII scrubbing (see 06_DATA_PRIVACY_AND_RELEASE_SAFETY §1) ---------------
PHONE_RE = re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b")
PHONE_SPACED_RE = re.compile(r"\b[6-9]\d{4}[\-\s]\d{5}\b")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def scrub_pii(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = AADHAAR_RE.sub("[ID]", text)
    text = PHONE_SPACED_RE.sub("[PHONE]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


# --- garbage / quality gates ------------------------------------------------
GARBAGE = {"na", "n/a", "nil", "test", "test call", "weather", "-", ".", "none"}


def is_garbage(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return (not a) or (a in GARBAGE) or (len(a.split()) < 2)


# --- surface clean (applied to stored text — folded in from PR #6's
#     scripts/clean_dataset.py: NFC-normalize + collapse whitespace on the
#     text we actually store, not just on the dedup/leakage key) -----------
def surface_clean(text: str) -> str:
    if not text:
        return text
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --- normalization (used for leakage/dedup keys — deliberately more
#     aggressive: lowercase + strip punctuation, per 05 §1 D2/D3) -----------
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\sऀ-෿]", "", t)  # keep Indic script ranges
    return t


# --- language tagging (placeholder; swap in fastText langid at M1) ----------
# NOTE: Marathi shares the Devanagari block with Hindi (0x0900-0x097F) and
# has no separate Unicode range, so script-range detection alone can NEVER
# distinguish mr from hi. Flagged for Lead — real KCC data includes Marathi;
# this placeholder will silently mistag every mr row as hi until a real
# langid model (e.g. fastText lid.176) is swapped in.
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF), "ta": (0x0B80, 0x0BFF), "kn": (0x0C80, 0x0CFF),
}


def detect_lang(text: str) -> str:
    counts = {lang: 0 for lang in SCRIPT_RANGES}
    latin = 0
    for ch in text or "":
        cp = ord(ch)
        if 0x41 <= cp <= 0x7A:
            latin += 1
            continue
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "en" if latin else "unknown"
    if latin > counts[best]:
        return "mixed"
    return best


def _first(row: dict, *keys: str) -> str:
    """Return the first non-empty value across possible column-name variants —
    different KCC mirrors use different schemas (QueryText/KccAns vs
    questions/answers seen so far). Avoids re-patching this file every time
    the raw source changes."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def row_to_record(row: dict) -> dict | None:
    q = surface_clean(scrub_pii(_first(row, "QueryText", "questions").strip()))
    a = surface_clean(scrub_pii(_first(row, "KccAns", "answers").strip()))
    if is_garbage(a) or not q:
        return None
    context = " | ".join(
        p for p in [
            f"State: {row.get('StateName','')}",
            f"Crop: {row.get('Crop','')}",
            f"Season: {row.get('Season','')}",
            f"Type: {row.get('QueryType','')}",
        ] if p.split(": ", 1)[1]
    )
    return {
        "instruction": q,
        "input": context,
        "output": a,
        "lang": detect_lang(q),
        "meta": {
            "state": row.get("StateName", ""),
            "district": row.get("DistrictName", ""),
            "crop": row.get("Crop", ""),
            "querytype": row.get("QueryType", ""),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.inp)
    seen = set()
    kept, dropped = [], 0
    for _, row in df.iterrows():
        rec = row_to_record(row.to_dict())
        if rec is None:
            dropped += 1
            continue
        key = normalize(rec["instruction"]) + "||" + normalize(rec["output"])
        if key in seen:               # exact-normalized dedup; fuzzy dedup is a TODO (datasketch)
            dropped += 1
            continue
        seen.add(key)
        kept.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    blob = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in kept)
    version_hash = hashlib.sha256(blob.encode()).hexdigest()[:12]
    langs: dict[str, int] = {}
    for r in kept:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    print(f"kept={len(kept)} dropped={dropped} version_hash={version_hash}")
    print("per-language:", dict(sorted(langs.items(), key=lambda x: -x[1])))
    print("TODO M1: lock final language set from these counts; add fuzzy dedup + quality filter.")


if __name__ == "__main__":
    sys.exit(main())
