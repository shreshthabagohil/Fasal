#!/usr/bin/env bash
# End-to-end held-out evaluation:
# [optional: seeded stratified subsample to enforce the JUDGE_BUDGET.md N]
# -> base inference -> ours inference -> blind dual-order judge
# -> aggregate -> human-readable report.

set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL=<org/base>}"
: "${BASE_SHA:?export BASE_SHA=<pinned HF commit SHA>}"
: "${ADAPTER_REPO:?export ADAPTER_REPO=<hf-user/adapter-or-local-path>}"

HELDOUT="${HELDOUT:-eval/heldout/test.jsonl}"
OUT_DIR="${OUT_DIR:-eval/out}"
REPORT_DIR="${REPORT_DIR:-eval/reports}"
SEED="${SEED:-1729}"
# MAX_N enforces the judge-call budget locked in eval/JUDGE_BUDGET.md:
# unset/empty = full $HELDOUT (no behavior change from before this existed).
# Use MAX_N=80 for M3/M4 iterative comparisons, MAX_N=170 for the final M5
# report -- both values and the reasoning live in eval/JUDGE_BUDGET.md, this
# script just enforces whichever one you pass instead of relying on someone
# hand-slicing the held-out file.
MAX_N="${MAX_N:-}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

if [[ -n "$MAX_N" ]]; then
  SAMPLED_HELDOUT="$OUT_DIR/heldout_n${MAX_N}_seed${SEED}.jsonl"
  python eval/sample_heldout.py \
    --in "$HELDOUT" \
    --out "$SAMPLED_HELDOUT" \
    --n "$MAX_N" \
    --seed "$SEED"
  HELDOUT="$SAMPLED_HELDOUT"
fi

# 1) Base-only inference.
python eval/infer.py \
  --base "$BASE_MODEL" \
  --base-sha "$BASE_SHA" \
  --heldout "$HELDOUT" \
  --out "$OUT_DIR/base.jsonl" \
  --batch-size 8 \
  --max-new-tokens 512 \
  --seed "$SEED"

# 2) Base + adapter inference (ours).
python eval/infer.py \
  --base "$BASE_MODEL" \
  --base-sha "$BASE_SHA" \
  --adapter "$ADAPTER_REPO" \
  --heldout "$HELDOUT" \
  --out "$OUT_DIR/ours.jsonl" \
  --batch-size 8 \
  --max-new-tokens 512 \
  --seed "$SEED"

# 3) Blind A/B LLM judge, dual-order.
python eval/judge.py \
  --heldout "$HELDOUT" \
  --base "$OUT_DIR/base.jsonl" \
  --ours "$OUT_DIR/ours.jsonl" \
  --out "$OUT_DIR/scores.jsonl" \
  --model openai/gpt-oss-120b \
  --temperature 0 \
  --seed "$SEED" \
  --dual-order

# 4) Aggregate + significance.
python eval/aggregate.py \
  --scores "$OUT_DIR/scores.jsonl" \
  --out "$REPORT_DIR/eval_report.json" \
  --bootstrap-n 10000 \
  --seed "$SEED"

# 5) Print the report.
python eval/print_report.py \
  --report "$REPORT_DIR/eval_report.json"
