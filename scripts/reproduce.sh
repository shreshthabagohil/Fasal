#!/usr/bin/env bash
# The 3 commands a judge runs to reproduce our headline number (05 §6).
# Built around the LoRA ADAPTER + pinned base SHA (small, avoids the 240GB
# full-weights download problem reported on Adaption's platform).
# Test this COLD on a fresh machine during freeze. If it doesn't reproduce, it's a release blocker.
set -euo pipefail

# 1) install pinned deps
pip install -r requirements.txt

# 2) pull released adapter + pinned base (fill in once released)
export BASE_MODEL="<org/base-model>"          # e.g. sarvamai/sarvam-1
export BASE_SHA="<exact-commit-sha>"          # pin the revision, not just the name
export ADAPTER_REPO="<hf-username/fasal-kcc-adapter>"  # our released LoRA adapter on HF

# 3) run eval, print the number
bash eval/run_eval.sh
echo "Expected headline: +__._% win-rate over ${BASE_MODEL} (see model_card/MODEL_CARD.md)"
