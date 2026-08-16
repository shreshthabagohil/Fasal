#!/usr/bin/env bash
# The 3 commands a judge runs to reproduce our headline number (05 §6).
# Built around the LoRA ADAPTER + pinned base SHA (small, avoids the 240GB
# full-weights download problem reported on Adaption's platform).
# Test this COLD on a fresh machine during freeze. If it doesn't reproduce, it's a release blocker.
set -euo pipefail

# 1) install pinned deps
pip install -r requirements.txt

# 2) pull released adapter + pinned base. Defaults below are the real,
# published values (see train/A0_LOCKED.md) -- override any of the three
# if you're reproducing against a different adapter/base.
export BASE_MODEL="${BASE_MODEL:-sarvamai/sarvam-1}"
export BASE_SHA="${BASE_SHA:-e9607337286ddf496d4a2562b194e489dcf3feea}"
export ADAPTER_REPO="${ADAPTER_REPO:-Algo-Nova/fasal-sarvam1-lora}"

# 3) run eval, print the number
bash eval/run_eval.sh
echo "Expected headline: win-rate over ${BASE_MODEL} -- see model_card/MODEL_CARD.md for the current number"
