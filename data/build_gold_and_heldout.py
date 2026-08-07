"""T07 + T08: build the multilingual gold subset and the held-out/reserve
eval split, in one pass, from the current dataset/kcc_instruct_v1.jsonl.

Why one script: gold subset, held-out, and reserve must all be drawn from the
SAME split logic so none of them overlap with train (D2 leakage=0) or with
each other. Doing them separately (as Gunn's PRs #6/#7 did, against the old
English-only dataset) risks exactly this kind of drift.

Split, per language:
  - gold:    20-40 hand-checkable pairs (T07) -> dataset/gold/<lang>.jsonl
             (small, for manual D7 spot-read / sanity, NOT used in training
             or as the eval metric)
  - heldout: held-out test set (T08) -> eval/heldout/test.jsonl
             (used for the real LLM-judge win-rate eval, must have ZERO
             overlap with train)
  - reserve: backup held-out, same size, untouched until/unless heldout is
             burned -> eval/heldout/reserve.jsonl
  - train:   everything else -> written back to dataset/kcc_instruct_v1.jsonl

All three (gold/heldout/reserve) are carved OUT of train (removed from it),
so leakage is impossible by construction, not just checked after the fact.

Usage:
    python build_gold_and_heldout.py \
        --in dataset/kcc_instruct_v1.jsonl \
        --gold-n 30 --heldout-n 150 --reserve-n 150 --seed 1729
"""
import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, "data")
from clean_kcc import normalize  # noqa: E402


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dataset/kcc_instruct_v1.jsonl")
    ap.add_argument("--gold-n", type=int, default=30, help="per-language gold pairs")
    ap.add_argument("--heldout-n", type=int, default=150, help="per-language held-out pairs")
    ap.add_argument("--reserve-n", type=int, default=150, help="per-language reserve pairs")
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    rows = load_rows(args.inp)
    print(f"Loaded {len(rows)} rows from {args.inp}")

    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    rng = random.Random(args.seed)
    gold, heldout, reserve, remaining = [], [], [], []

    # KCC data has many rows sharing IDENTICAL instruction text (the same
    # question asked in different districts with different answers). D2
    # leakage is defined on instruction text alone (see tests.py t_leakage),
    # so splitting individual ROWS can put the same question in both train
    # and held-out even though no single row is duplicated. Grouping by
    # normalized instruction PER LANGUAGE fixed most of this, but a handful
    # of instruction strings show up under more than one language tag (e.g.
    # short/generic text landing in "unknown"/"mixed" as well as "en") — so
    # the group key must be tracked GLOBALLY, not reset per language, or
    # those cross-language duplicates can still be split across sets.
    global_assignment = {}  # normalize(instruction) -> "gold"/"heldout"/"reserve"/"train"

    for lang, lrows in sorted(by_lang.items()):
        groups = collections.defaultdict(list)
        for r in lrows:
            groups[normalize(r["instruction"])].append(r)
        group_items = list(groups.items())
        rng.shuffle(group_items)

        need = args.gold_n + args.heldout_n + args.reserve_n
        if len(lrows) < need:
            print(f"[{lang}] only {len(lrows)} rows, need {need} for full gold+heldout+reserve "
                  f"-> scaling down proportionally for this language")
            frac = len(lrows) / need
            g_target = max(1, int(args.gold_n * frac)) if lang != "unknown" else 0
            h_target = max(1, int(args.heldout_n * frac)) if lang != "unknown" else 0
            r_target = max(0, len(lrows) - g_target - h_target)
        else:
            g_target, h_target, r_target = args.gold_n, args.heldout_n, args.reserve_n

        lang_gold, lang_heldout, lang_reserve, lang_remaining = [], [], [], []
        for key, grp in group_items:
            # If this instruction text was already assigned a split while
            # processing an earlier language, honor that assignment instead
            # of re-deciding — guarantees one instruction never lands in two
            # different buckets, regardless of which language(s) tag it.
            existing = global_assignment.get(key)
            if existing == "gold":
                lang_gold.extend(grp)
            elif existing == "heldout":
                lang_heldout.extend(grp)
            elif existing == "reserve":
                lang_reserve.extend(grp)
            elif existing == "train":
                lang_remaining.extend(grp)
            elif len(lang_gold) < g_target:
                lang_gold.extend(grp); global_assignment[key] = "gold"
            elif len(lang_heldout) < h_target:
                lang_heldout.extend(grp); global_assignment[key] = "heldout"
            elif len(lang_reserve) < r_target:
                lang_reserve.extend(grp); global_assignment[key] = "reserve"
            else:
                lang_remaining.extend(grp); global_assignment[key] = "train"

        gold.extend(lang_gold)
        heldout.extend(lang_heldout)
        reserve.extend(lang_reserve)
        remaining.extend(lang_remaining)

    # Belt-and-braces: assert zero normalized-instruction overlap between
    # train (remaining) and heldout, since D2 depends on this.
    train_q = {normalize(r["instruction"]) for r in remaining}
    heldout_q = {normalize(r["instruction"]) for r in heldout}
    overlap = train_q & heldout_q
    if overlap:
        sys.exit(f"REFUSING TO WRITE: {len(overlap)} train/heldout instruction overlaps "
                  f"found post-split — this should be impossible with disjoint slicing, "
                  f"investigate before proceeding.")

    # Write gold per-language files
    gold_by_lang = collections.defaultdict(list)
    for r in gold:
        gold_by_lang[r["lang"]].append(r)
    goldroot = pathlib.Path("dataset/gold")
    for lang, lrows in gold_by_lang.items():
        write_rows(goldroot / f"{lang}.jsonl", lrows)

    write_rows(pathlib.Path("eval/heldout/test.jsonl"), heldout)
    write_rows(pathlib.Path("eval/heldout/reserve.jsonl"), reserve)
    write_rows(pathlib.Path(args.inp), remaining)

    def content_hash(rs):
        blob = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rs)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    hashes_md = pathlib.Path("eval/heldout/HASHES.md")
    hashes_md.parent.mkdir(parents=True, exist_ok=True)
    with open(hashes_md, "w", encoding="utf-8") as f:
        f.write("# Held-out / reserve / train content hashes (frozen at build time)\n\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- train:   {len(remaining)} rows, hash={content_hash(remaining)}\n")
        f.write(f"- heldout: {len(heldout)} rows, hash={content_hash(heldout)}\n")
        f.write(f"- reserve: {len(reserve)} rows, hash={content_hash(reserve)}\n")
        f.write(f"- gold:    {len(gold)} rows, hash={content_hash(gold)}\n")
        f.write("\nper-language counts:\n\n")
        for lang in sorted(by_lang):
            g = len(gold_by_lang.get(lang, []))
            h = len([r for r in heldout if r["lang"] == lang])
            r_ = len([r for r in reserve if r["lang"] == lang])
            t = len([r for r in remaining if r["lang"] == lang])
            f.write(f"- {lang}: train={t} gold={g} heldout={h} reserve={r_}\n")

    print(f"\ngold={len(gold)} heldout={len(heldout)} reserve={len(reserve)} "
          f"train(remaining)={len(remaining)}")
    print(f"Wrote: dataset/gold/<lang>.jsonl, eval/heldout/test.jsonl, "
          f"eval/heldout/reserve.jsonl, {hashes_md}")
    print(f"Rewrote {args.inp} in place with heldout/reserve/gold rows removed.")
    print("\nVerified: zero train/heldout instruction overlap (D2 leakage=0 by construction).")


if __name__ == "__main__":
    sys.exit(main())
