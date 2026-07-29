# Fasal — Multilingual Indian-Farmer Advisory Model (KCC)

*Fasal (फसल) — "crop/harvest". Built by team algonova for the AutoScientist Challenge.*

> **Headline result:** +__._% win-rate over `<base model>` (LLM-judge, blind A/B) on our multilingual held-out test. 95% CI [+_._, +_._], p<0.0__. Stable across 3 seeds.
> *(fill this in once M2 produces the first real number — it goes here AND at the top of the model card)*

AutoScientist Challenge Part 2 · Agriculture track · HackIndia. We fine-tune an open model on the **Kisan Call Centre (KCC)** corpus to answer Indian farmers' questions across multiple languages, and openly release the model + adapted dataset.

**Live demo:** _(link — M4)_ · **HF model:** _(link)_ · **HF/Kaggle dataset:** _(link)_

## How you're scored (confirmed from Adaption Discord)
No fixed external baseline — your model is judged on **LLM-judge win-rate over the base model you trained on**, on Adaption's hidden tasks. So we A/B two bases and keep the bigger honest win. See `../02_EXECUTION_PLAN.md` §A2.

## Repo map
```
data/        raw KCC (gitignored) + cleaning pipeline + data tests + version hashes
dataset/     the released instruction dataset (versioned) + data card
train/       QLoRA + AutoScientist run configs
eval/        held-out test, metric + paired-bootstrap scripts, judge rubric
model_card/  MODEL_CARD.md, criteria-mapping table, reproduce block
scripts/     reproduce.sh (the 3 commands a judge runs)
src/         shared utils (fixed seed, etc.)
```

## Quickstart (tomorrow)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your real tokens into .env (NEVER commit .env)
# 1) put the raw KCC CSV in data/raw/  (gitignored)
python data/clean_kcc.py --in data/raw/kcc.csv --out dataset/kcc_instruct_v1.jsonl
python data/tests.py --dataset dataset/kcc_instruct_v1.jsonl   # must be all-green before training
```

## Non-negotiables (why we lost last time)
- **Leakage = 0** between train and held-out test (`data/tests.py`), re-run every regeneration.
- **PII scrubbed** before any public release (`data/clean_kcc.py`).
- **Pinned deps + fixed seed + base-model commit SHA** so the judge's cold-reproduce gives our number.
- **Freeze Aug 5**, submit Aug 9. No new training after M4.

See the planning docs one level up: `01_TEAM_BRIEF`, `02_EXECUTION_PLAN`, `05_TESTING_AND_VALIDATION`, `06_DATA_PRIVACY_AND_RELEASE_SAFETY`, `07_RESOURCE_TRACKER`.
