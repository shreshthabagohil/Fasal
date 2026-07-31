# AutoScientist run notes

## What AutoScientist actually exposes (M0 de-risk finding)

Correction to the original assumption in `02`/`10`: AutoScientist's exposed knobs are
**data-recipe operations**, not training hyperparameters (learning_rate, batch_size,
etc. were never surfaced in the UI at this stage). The recipe screen ("Let's start
adapting") exposes:

- **Prompt Rephrase** — optimizes prompts for task + quality (off by default)
- **Prompt Deduplication** — detects/removes duplicate prompts (ON by default)
- **Prompt Metadata Injection** — adds extra context per prompt (off by default)
- **House Special** — Adaption's own combined recipe, "most powerful combination" (ON by default)
- **Reasoning traces** — gold-standard reasoning traces for interpretability (off by default)
- **Hallucination mitigation** — reduces hallucinations (off by default)
- **Code verification** — coming soon (locked)
- **Checklist verification** — coming soon (locked)

Training-hyperparameter search (LR, warmup, batch size, LoRA config) was NOT observed
in this pass — either it happens in a separate downstream "AutoScientist" tab per
dataset (not reached in this toy run) or is not part of this product surface. Re-verify
at M3 before assuming the AS-searches list in `A0_LOCKED.md` is accurate; flag to
Planning session if the training-recipe search step looks different in practice.

## What we fix (never searched by AutoScientist, per A0 lock)
- LoRA r=16, alpha=32, dropout=0.05, target_modules="all-linear"
- num_train_epochs = 3
- max_seq_length = 2048
- seed = 1729
- base model + commit SHA (locked at M3 step 12)

## How to trigger a run
1. Adaption UI → Datasets (grid icon) → "Adapt my data"
2. Import data via Link tab → "Import from Hugging Face" → paste HF dataset repo ID
3. Select file(s) if the repo has multiple → Continue
4. Choose dataset type: "Instruction dataset" (matches our QLoRA SFT method)
5. Skip "expand size" (Translate/Localize) unless deliberately growing multilingual coverage
6. Skip "brand guidelines / global constraints" (choose No) unless we have a specific rule to enforce
7. Map columns: bind `prompt` → Prompt column, `completion` → Completion column
8. Review "Data evaluation" screen (quality score/percentile — informational, not a gate)
9. Recipe screen: toggle desired operations, "Continue"
10. Summary screen shows credit cost — confirm balance, click "Launch"
11. Job runs async (~15 min ETA), completion notified by email
12. Screenshot the completed run's "Measure" tab (quality before/after, grade, percentile)

## Toy run (M0 de-risk)
- Date: 2026-07-30
- Dataset: `HuggingFaceH4/instruction-dataset` (HF import, `step3-eval.jsonl`), NOT KCC
- Dataset size: 327 rows in → 287 rows after Prompt Deduplication + House Special recipe
- Recipe used: Prompt Deduplication (ON, default) + House Special (ON, default); all other toggles left off
- Credit cost: 4 Adaption credits (well under the ~20-credit estimate)
- Result: Job Completed. Quality score 6.0 → 7.3 (21.7% relative improvement). Grade C → B. Percentile 8.2 → 10.8.
- Domains detected: writing-editing-communication (64%), code (6%), math (2%). Language: English only.
- W&B: no run logged for this job — this Adaption "data adaptation" stage does not appear to push to W&B; W&B integration likely applies to a later model-training stage, not reached in this toy run. Re-verify at M2/M3 when we run a real fine-tune.
- Screenshot: repo/model_card/screenshots/M0-toy-run.png
