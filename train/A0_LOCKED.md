# A0 — Locked decisions (Fasal)

_Mirror of `02_EXECUTION_PLAN.md` §A0. Changes require a Planning-session decision + an entry in `08_MILESTONE_LOG.md`. Do NOT silently edit._

## ⭐ CANONICAL IDENTIFIERS — single source of truth (confirmed; do not guess)
These four namespaces look alike but are DIFFERENT and were confused in early drafts. Corrected across all docs in the Planning session of 2026-07-31. Use exactly these:

| Thing | Canonical value | Confirmed from | Why / note |
|---|---|---|---|
| **Hugging Face org** | `Algo-Nova` (capital A/N, hyphen) | actual org created at M0-T08; `08` M0 log | Everything on HF: `Algo-Nova/fasal-kcc-instruct`, `Algo-Nova/fasal-sarvam1-lora`, Space `Algo-Nova/fasal-advisor-web`. **NOT** `algonova`. |
| **Kaggle username** | `shreshthabagohil` | Lead's profile `kaggle.com/shreshthabagohil` (screenshot, 2026-07-31) | Kaggle has no free orgs → datasets/models publish under the Lead's user: `shreshthabagohil/fasal-kcc-*`. **NOT** `algonova`. |
| **W&B entity** | `shreshthabagohil-personal` (personal) | M0 decision; A0 §Experiment tracking | No paid W&B Team (zero-budget). Project `wandb.ai/shreshthabagohil-personal/fasal`. Teammates log to their OWN entity + screenshot. There is **no** `algonova` W&B entity. |
| **HackIndia team name** | `AlgoNova` (one word, capital N) | HackIndia registration | Teammates register under this exact string. Different from the HF org spelling. |
| **GitHub repo** | `github.com/shreshthabagohil/Fasal` | Lead's repo | Our real submission repo. The auto-created `HackIndiaXYZ/…-algonova` repo is HackIndia's, not ours — leave it as their string. |

**Session note:** at the release milestone (REL), verify each of these still resolves before publishing links. The Kaggle namespace especially — confirm `shreshthabagohil/…` is claimable for the dataset + model before REL-T01/T05.

## Base model
- **LOCKED: `sarvamai/sarvam-1` @ `e9607337286ddf496d4a2562b194e489dcf3feea`** (re-confirmed live 2026-08-12 via `https://huggingface.co/api/models/sarvamai/sarvam-1`; `lastModified` 2024-11-08, so this SHA has been stable throughout the event).
- **Deviation from the original A0 plan, logged here per the "do not silently edit" rule:** the planned M3 A/B against `google/gemma-2-2b` was abandoned mid-event under GPU/Colab-quota and time pressure (see `08_MILESTONE_LOG.md`) — this is a forced single-base commitment, not a completed comparison with a measured winner. Gemma's commit SHA is recorded here for provenance only, since it appeared in earlier drafts of this file and in `train/config.yaml`'s `base_candidates`: `c5ebcd40d208330abc697524c919956e692655cf` (`google/gemma-2-2b`, gated, license `gemma`). It was never trained or evaluated — do not cite it as an A/B loser, only as "not attempted."
- Never load a floating tag; training/eval code pins the exact SHA above.

## Fine-tune method
- QLoRA SFT · 4-bit NF4 base · LoRA `r=16, alpha=32, dropout=0.05, target_modules="all-linear", bias="none"`.
- DoRA (`use_dora=true`) is opt-in at M4 iter-4 only if the leading iter's CI lower bound < +2 pp.

## Evaluation
- Primary metric: LLM-judge win-rate (blind A/B, dual-order, temp 0, seed 1729).
- Judge model: Llama 3.3 70B via Groq free tier.
- Judge prompt + rubric: `repo/eval/judge_prompt.md` (frozen at M3 step 1; do NOT change mid-event).
- Companion metric: chrF.
- Significance: paired bootstrap over held-out items (10k resamples, 95% CI) + McNemar for win/loss.
- Final candidate: retrain on 3 seeds {1729, 2027, 3141}; median-seed ships. Claim bar: CI excludes 0, McNemar p < 0.05, all 3 seeds sign-consistent. **Not yet run as of 2026-08-12** — blocked on GPU access for the eval pass (see `08_MILESTONE_LOG.md`); this is a single-base run, so "cross-family from both Sarvam and Gemma" no longer applies (Gemma was never trained).

## Release format
- Primary artifact: LoRA adapter (small, exportable) — `Algo-Nova/fasal-sarvam1-lora` on HF.
- Optional: merged fp16 checkpoint if the AutoScientist export path works (M3 step 10 dry-run decides).
- Published to: Hugging Face (org `Algo-Nova` — note: actual HF slug is `Algo-Nova`, capital A/N with hyphen, not `algonova` as originally written in `02`/`10`; corrected at M0-T08, 2026-07-30) + Kaggle Models.
- Dataset published to: Hugging Face Datasets + Kaggle Datasets.
- Pinned base commit SHA (`e9607337286ddf496d4a2562b194e489dcf3feea`) and dataset content hash both cited in the model card.

## Repo hygiene
- Fully pinned `requirements.txt` (== everywhere) + `requirements.lock` (`pip freeze` output).
- Global seed 1729 propagated to python/numpy/torch/transformers.
- `.gitignore` blocks raw data, weights, `.env`, `kaggle.json`, generated JSONL, `wandb/`.
- `scripts/reproduce.sh` = the 3 commands a judge runs; tested cold on a fresh machine during REL.

## Licences (gate)
- KCC data: GODL-India (derivative redistribution + attribution permitted). Documented in `repo/SOURCES.md`.
- Base licence: `sarvamai/sarvam-1` is the sole trained/shipped base (see above), so only its licence flows through to the adapter and any merged weights: the Sarvam AI Research License (non-commercial-flavored — permission required for uses not expressly authorized). Full text in `repo/LICENSE_BASE_SARVAM.md`. `LICENSE_BASE_GEMMA.md` remains in the repo for provenance only (Gemma was fetched/filed at M0-T13 during A/B planning but never trained or shipped) — do not present it as a licence that applies to any released artifact.

## Experiment tracking (M0 deviation, logged)
- W&B: no separate team was created — the account's Pro trial was downgraded to Free immediately (zero-budget rule) and W&B Teams are a paid feature. Runs log to the personal entity `shreshthabagohil-personal`, project `fasal` (`wandb.ai/shreshthabagohil-personal/fasal`), not `algonova/fasal` as originally written in `02`/`10`. Teammates get visibility via shared/invited access on this entity, not a Team object.

_Changes require Planning-session decision + `08` log entry._
