"""Machine-translate a stratified sample of the cleaned English KCC set into
Indian languages, using facebook/nllb-200-distilled-600M (free, open-source,
CPU-feasible). Real English KCC content in, genuinely multilingual training
data out. Writes one JSONL per language plus a combined file.

Why this exists: both real KCC Kaggle mirrors turned out to be 100% English
(confirmed by an exhaustive non-ASCII byte scan across 42.1M lines) — KCC
call-center agents log queries in English regardless of the caller's spoken
language. This script is the documented, defensible fix: translate real
KCC semantic content into Hindi/Gujarati/Marathi/Tamil/Bengali/Kannada/Punjabi
rather than fabricate multilingual text from nothing.

Usage:
    pip install transformers sentencepiece torch --break-system-packages
    python translate_to_indic.py \
        --in dataset/kcc_instruct_v1.jsonl \
        --outdir dataset/translated \
        --sample-size 1200 --seed 1729

Resumable: re-running skips languages whose output file already has
--sample-size lines.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

LANGS = {
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "kn": "kan_Knda",
    "pa": "pan_Guru",
}
SRC_LANG = "eng_Latn"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
BATCH_SIZE = 8
MAX_LEN = 400


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stratified_sample(rows, n, seed):
    """Sample n rows spread across querytype buckets for diversity, not just
    random — mirrors the D7 spot-read / gold-subset diversity goal."""
    buckets = collections.defaultdict(list)
    for r in rows:
        raw_qt = r.get("meta", {}).get("querytype")
        qt = str(raw_qt).strip().lower() if raw_qt else "other"  # defensive: tolerate
        qt = qt or "other"                                        # any non-string/NaN upstream
        buckets[qt].append(r)
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)

    out, i = [], 0
    keys = list(buckets.keys())
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out[:n]


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-size", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--langs", default=",".join(LANGS.keys()))
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    for l in target_langs:
        if l not in LANGS:
            sys.exit(f"Unknown lang code {l!r} — must be one of {list(LANGS)}")

    rows = load_rows(args.inp)
    sample = stratified_sample(rows, args.sample_size, args.seed)
    print(f"Sampled {len(sample)} English rows (target {args.sample_size}) "
          f"from {len(rows)} cleaned rows.")

    # Lazy-import so --help doesn't require torch installed
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} (first run downloads ~2.4GB, cached after)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")

    def translate_batch(texts, tgt_code):
        tokenizer.src_lang = SRC_LANG
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=MAX_LEN,
            )
        return tokenizer.batch_decode(gen, skip_special_tokens=True)

    for lang in target_langs:
        tgt_code = LANGS[lang]
        outpath = outdir / f"{lang}.jsonl"
        if outpath.exists():
            existing = sum(1 for _ in open(outpath, encoding="utf-8"))
            if existing >= len(sample):
                print(f"[{lang}] already has {existing} rows, skipping.")
                continue
        print(f"[{lang}] translating {len(sample)} pairs -> {tgt_code} ...")
        done = 0
        with open(outpath, "w", encoding="utf-8") as fout:
            for batch in batched(sample, BATCH_SIZE):
                instr = [r["instruction"] for r in batch]
                outp = [r["output"] for r in batch]
                instr_t = translate_batch(instr, tgt_code)
                outp_t = translate_batch(outp, tgt_code)
                for r, qi, qo in zip(batch, instr_t, outp_t):
                    rec = {
                        "instruction": qi,
                        "input": r.get("input", ""),
                        "output": qo,
                        "lang": lang,
                        "meta": {**r.get("meta", {}), "src": "kcc-en-machine-translated",
                                 "mt_model": MODEL_NAME, "orig_instruction_en": r["instruction"]},
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += len(batch)
                print(f"  {done}/{len(sample)}", end="\r")
        print(f"\n[{lang}] wrote {outpath} ({done} rows)")

    print("\nDone. Merge into dataset/kcc_instruct_v1.jsonl with a separate step "
          "(don't overwrite the English set — concatenate).")


if __name__ == "__main__":
    sys.exit(main())
