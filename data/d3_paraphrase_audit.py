"""D3 — embedding-based paraphrase leakage audit (05_TESTING_AND_VALIDATION §1).

D2 (exact-normalized string match) is the hard gate, but KCC repeats the same
pest/crop/state Q&A with different wording across districts — a paraphrased
held-out item can hide in train even when D2 is clean. This embeds every
train + held-out instruction with the locked embedder (BGE-M3, per
02_EXECUTION_PLAN §A0) and flags any train item with cosine similarity > 0.9
to a held-out item for manual review (T3 approves keep-as-different or
remove — this script only flags, never auto-removes).

Usage:
    pip install sentence-transformers --break-system-packages
    python data/d3_paraphrase_audit.py \
        --dataset dataset/kcc_instruct_v1.jsonl \
        --heldout eval/heldout/test.jsonl \
        --threshold 0.9 \
        --out eval/heldout/D3_FLAGGED.md
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", default="eval/heldout/D3_FLAGGED.md")
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    train = load(args.dataset)
    heldout = load(args.heldout)
    print(f"Loaded {len(train)} train rows, {len(heldout)} held-out rows.")

    # Lazy import so --help doesn't require the model dep installed
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"Loading {args.model} (first run downloads the model, cached after)...")
    model = SentenceTransformer(args.model)

    train_texts = [r["instruction"] for r in train]
    held_texts = [r["instruction"] for r in heldout]

    print("Embedding train set...")
    train_emb = model.encode(train_texts, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True)
    print("Embedding held-out set...")
    held_emb = model.encode(held_texts, batch_size=32, show_progress_bar=True,
                             normalize_embeddings=True)

    # Cosine similarity via dot product (embeddings already L2-normalized)
    sims = held_emb @ train_emb.T  # shape: (n_heldout, n_train)

    flagged = []
    for hi, row_sims in enumerate(sims):
        hits = np.where(row_sims > args.threshold)[0]
        for ti in hits:
            flagged.append({
                "heldout_id": heldout[hi].get("id", hi),
                "heldout_text": held_texts[hi],
                "train_id": train[ti].get("id", int(ti)),
                "train_text": train_texts[ti],
                "cosine": float(row_sims[ti]),
            })

    flagged.sort(key=lambda x: -x["cosine"])

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# D3 paraphrase audit — {len(flagged)} pairs above cosine {args.threshold}\n\n")
        f.write(f"Embedder: `{args.model}` · train N={len(train)} · held-out N={len(heldout)}\n\n")
        f.write("Each row needs a T3 call: **keep as different** (genuinely distinct "
                "even though similar) or **remove** (real leakage — same Q&A, different wording).\n\n")
        f.write("| cosine | held-out | train (candidate leak) | verdict (fill in) |\n")
        f.write("|---|---|---|---|\n")
        for r in flagged:
            ht = r["heldout_text"].replace("|", "\\|")[:80]
            tt = r["train_text"].replace("|", "\\|")[:80]
            f.write(f"| {r['cosine']:.3f} | {ht} | {tt} | |\n")

    print(f"\n{len(flagged)} pairs flagged above cosine {args.threshold} -> {args.out}")
    print("D3 is NOT auto-pass/fail — T3 must review each flagged row and fill in "
          "the verdict column before D3 can be marked complete.")


if __name__ == "__main__":
    sys.exit(main())
