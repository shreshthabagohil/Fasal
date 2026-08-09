# A0 — Locked decisions (Fasal)

_Mirror of `02_EXECUTION_PLAN.md` §A0. Changes require a Planning-session decision + an entry in `08_MILESTONE_LOG.md`. Do NOT silently edit._

## Base model
- A/B two at M3: `sarvamai/sarvam-1` vs `google/gemma-2-2b`. Keep the winner (highest 95% CI lower bound on iter-0 delta).
- Pin the exact HF commit SHA of whichever base wins. Never load a floating tag.
- `LOCKED: <base>@<sha>` — filled at M3 step 12.

## Fine-tune method
- QLoRA SFT · 4-bit NF4 base · LoRA `r=16, alpha=32, dropout=0.05, target_modules="all-linear", bias="none"`.
- DoRA (`use_dora=true`) is opt-in at M4 iter-4 only if the leading iter's CI lower bound < +2 pp.

## Evaluation
- Primary metric: LLM-judge win-rate (blind A/B, dual-order, temp 0, seed 1729).
- Judge model: Llama 3.3 70B via Groq free tier. Cross-family from both Sarvam and Gemma.
- Judge prompt + rubric: `repo/eval/judge_prompt.md` (frozen at M3 step 1; do NOT change mid-event).
- Companion metric: chrF.
- Significance: paired bootstrap over held-out items (10k resamples, 95% CI) + McNemar for win/loss.
- Final candidate: retrain on 3 seeds {1729, 2027, 3141}; median-seed ships. Claim bar: CI excludes 0, McNemar p < 0.05, all 3 seeds sign-consistent.

## Release format
- Primary artifact: LoRA adapter (small, exportable).
- Optional: merged fp16 checkpoint if the AutoScientist export path works (M3 step 10 dry-run decides).
- Published to: Hugging Face (org `Algo-Nova` — note: actual HF slug is `Algo-Nova`, capital A/N with hyphen, not `algonova` as originally written in `02`/`10`; corrected at M0-T08, 2026-07-30) + Kaggle Models (both, with hash-parity).
- Dataset published to: Hugging Face Datasets + Kaggle Datasets.
- Pinned base commit SHA and dataset content hash both cited in the model card.

## Repo hygiene
- Fully pinned `requirements.txt` (== everywhere) + `requirements.lock` (`pip freeze` output).
- Global seed 1729 propagated to python/numpy/torch/transformers.
- `.gitignore` blocks raw data, weights, `.env`, `kaggle.json`, generated JSONL, `wandb/`.
- `scripts/reproduce.sh` = the 3 commands a judge runs; tested cold on a fresh machine during REL.

## Licences (gate)
- KCC data: GODL-India (derivative redistribution + attribution permitted). Documented in `repo/SOURCES.md`.
- Base licence: whichever base wins the M3 A/B, its licence flows through to merged weights. Full text in `repo/LICENSE_BASE_*.md`. Note: Sarvam-1 ships under the Sarvam AI Research License (non-commercial-flavored — permission required for uses not expressly authorized); Gemma-2 2B ships under the Gemma Terms of Use (redistribution permitted with use restrictions + required Notice file). Both fetched and filed at M0-T13 (2026-07-30).

## Experiment tracking (M0 deviation, logged)
- W&B: no separate team was created — the account's Pro trial was downgraded to Free immediately (zero-budget rule) and W&B Teams are a paid feature. Runs log to the personal entity `shreshthabagohil-personal`, project `fasal` (`wandb.ai/shreshthabagohil-personal/fasal`), not `algonova/fasal` as originally written in `02`/`10`. Teammates get visibility via shared/invited access on this entity, not a Team object.

_Changes require Planning-session decision + `08` log entry._
