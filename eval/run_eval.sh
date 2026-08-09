#!/usr/bin/env bash
# End-to-end held-out evaluation:
# base inference -> ours inference -> blind dual-order judge
# -> aggregate -> human-readable report.

set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL=<org/base>}"
: "${BASE_SHA:?export BASE_SHA=<pinned HF commit SHA>}"
: "${ADAPTER_REPO:?export ADAPTER_REPO=<hf-user/adapter-or-local-path>}"

HELDOUT="${HELDOUT:-eval/heldout/test.jsonl}"
OUT_DIR="${OUT_DIR:-eval/out}"
REPORT_DIR="${REPORT_DIR:-eval/reports}"
SEED="${SEED:-1729}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

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
  --model llama-3.3-70b-versatile \
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
