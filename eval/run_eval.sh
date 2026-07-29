#!/usr/bin/env bash
# Runs the held-out eval and prints baseline-vs-ours with significance.
# Fill in the generation + judging steps once the base model + judge are chosen (M2).
set -euo pipefail

# 1) generate answers from BASE and OURS on the held-out set  (TODO: infer.py)
# python eval/infer.py --model base   --heldout eval/heldout/test.jsonl --out eval/out/base.jsonl
# python eval/infer.py --model ours   --heldout eval/heldout/test.jsonl --out eval/out/ours.jsonl

# 2) LLM-judge blind A/B -> per-item scores  (TODO: judge.py, uses judge_rubric.md)
# python eval/judge.py --base eval/out/base.jsonl --ours eval/out/ours.jsonl --out eval/out/scores.jsonl

# 3) significance
python eval/bootstrap.py --scores eval/out/scores.jsonl

echo "TODO: implement infer.py + judge.py once base model + judge model are locked (M2)."
