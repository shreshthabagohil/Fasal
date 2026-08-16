# Fasal — Multilingual Indian-Farmer Advisory Model (KCC)

*Fasal (फसल) — "crop/harvest". Built by team AlgoNova for the AutoScientist Challenge (HackIndia, Agriculture/KCC track).*

> **Headline result:** <!-- TODO(eval): fill after eval/reports/eval_report.json lands -->
> `+__._%` LLM-judge win-rate over the base model (`sarvamai/sarvam-1`), blind dual-order A/B on N=1,226 held-out
> farmer queries across 8 Indian languages. 95% CI [+_._, +_._]. Judge: Llama-3.3-70B via Groq.
> *(Run in progress — see `eval/reports/eval_report.json` once `fasal-eval-phase2-judge` finishes. Update this line
> and the matching line in `model_card/MODEL_CARD.md` together.)*

We fine-tune an open base model on the **Kisan Call Centre (KCC)** corpus — real questions Indian farmers called in
to ask, and the answers they got — so the model can give grounded agricultural advice in the farmer's own language
and script. Model, dataset, and adapter are all openly released.

- **Model (adapter):** [Algo-Nova/fasal-sarvam1-lora](https://huggingface.co/Algo-Nova/fasal-sarvam1-lora)
- **Dataset:** [Algo-Nova/fasal-kcc-instruct](https://huggingface.co/datasets/Algo-Nova/fasal-kcc-instruct) — 69,670 rows, 8 languages (bn, en, gu, hi, kn, mr, pa, ta)
- **Base model:** [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1), pinned commit `e9607337286ddf496d4a2562b194e489dcf3feea`
- **Live demo:** [huggingface.co/spaces/Shreshthabagohil/fasal-advisor-web](https://huggingface.co/spaces/Shreshthabagohil/fasal-advisor-web) <!-- TODO: currently redeploying, see Status below -->
- **Source:** this repo — [github.com/shreshthabagohil/Fasal](https://github.com/shreshthabagohil/Fasal)

## Status

- ✅ Dataset cleaned, deduped, PII-scrubbed, language-tagged, published (public on HF + Kaggle).
- ✅ Adapter trained (QLoRA, 4-bit NF4, `r=16 alpha=32`) and published.
- 🔄 Final held-out evaluation (N=1,226, dual-order LLM-judge) — running now, numbers land in `eval/reports/`.
- 🔄 Live demo Space — deployed, currently redeploying after a dependency-pin fix (see `demo/` if present, or Space logs).

## How it's evaluated

No fixed external baseline: the model is judged on **LLM-judge win-rate over the base model it was trained on** —
blind, dual-order A/B, temperature 0, fixed seed (1729), Llama-3.3-70B judge via Groq. Full methodology, rubric, and
significance testing (paired bootstrap, 10k resamples) are documented in `eval/judge_prompt.md`,
`eval/judge_rubric.md`, and `eval/aggregate.py`.

The submitted hackathon entry used a reduced N=170 sample for the final report (documented, not hidden — see
`eval/JUDGE_BUDGET.md`) to fit inside the Groq free-tier daily token budget before the deadline. Post-submission,
we're re-running the full N=1,226 held-out set for the fully-powered number used on this README and the model card.

## Repo map

```
data/        raw KCC (gitignored) + cleaning pipeline + data tests + version hashes
dataset/     the released instruction dataset (versioned) + data card
train/       QLoRA training config + locked decisions (train/A0_LOCKED.md)
eval/        held-out test (gitignored), inference/judge/aggregate scripts, judge rubric
model_card/  MODEL_CARD.md — criteria-mapping table, reproduce block
scripts/     reproduce.sh — the commands to reproduce a result from scratch
src/         shared utils (fixed seed, etc.)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your own tokens into .env (never commit .env)

# Reproduce the published eval number end-to-end
bash scripts/reproduce.sh
```

See `scripts/reproduce.sh` for the exact pinned commands (base SHA, adapter, seed) used to produce the headline
number above.

## Method

- **Base:** `sarvamai/sarvam-1` @ commit `e9607337286ddf496d4a2562b194e489dcf3feea` (pinned, never a floating tag).
- **Fine-tune:** QLoRA SFT, 4-bit NF4 quantization, LoRA `r=16, alpha=32, dropout=0.05, target_modules="all-linear"`,
  via plain `transformers.Trainer` (not `trl.SFTTrainer` — see engineering notes below).
- **Data:** 69,670-row instruction set built from real KCC transcripts, cross-lingual-leakage-safe held-out split
  (`eval/heldout/`, 1,226 rows, frozen and gitignored so it can never leak into training).
- **Reproducibility discipline:** pinned `requirements.txt`, fixed global seed (1729) propagated through
  python/numpy/torch/transformers, pinned base-model commit SHA, dataset content hashes recorded in
  `eval/heldout/HASHES.md`.

## Engineering notes (things that weren't obvious)

- 4-bit-quantized LoRA-adapter inference on a single GPU hits a real `accelerate`/`peft` version-compatibility
  trap: `accelerate>=0.34` is needed for newer `peft` adapter configs, but versions in between break `.to()` calls
  on quantized models. Working pin set for this project: `accelerate==0.34.2` + latest `peft`.
- `trl.SFTTrainer` broke across two different version bumps during development; training was moved to plain
  `transformers.Trainer` for stability.
- The held-out eval set is deliberately excluded from git (`.gitignore`) so it can never leak into anyone's
  training run, including our own future iterations.

## Non-negotiables

- **Leakage = 0** between train and held-out test (`data/tests.py`, cross-lingual-safe), re-run on every regeneration.
- **PII scrubbed** before any public release (`data/clean_kcc.py`).
- **Pinned deps + fixed seed + base-model commit SHA** so results are cold-reproducible.

## License

- Code: see `LICENSE_BASE_SARVAM.md` for the base model's license terms (Sarvam AI Research License,
  non-commercial-flavored) — this flows through to the released adapter.
- Data: KCC data via data.gov.in under GODL-India (attribution required — see `SOURCES.md`).
