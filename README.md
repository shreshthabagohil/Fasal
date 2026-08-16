# Fasal — Multilingual Indian-Farmer Advisory Model (KCC)

*Fasal (फसल) — "crop/harvest".* Built by team AlgoNova for HackIndia's AutoScientist Challenge (Agriculture track).

Fasal fine-tunes an open language model on real Kisan Call Centre (KCC) farmer queries so it can answer agricultural questions in the language a farmer actually speaks, not just English. Agricultural expertise in India is locked behind a language barrier — a farmer in rural Karnataka asking about pest control has to think in English before most AI tools are useful to them. Fasal closes that gap directly, across 8 Indian languages.

**Status:** Iteration 0 — **complete**. A QLoRA adapter was trained end-to-end on real KCC data and published to Hugging Face, with verified held-out outputs in the model card (link below). Iteration 1 — the same pipeline re-run on the full 69,670-row dataset, staged across multiple Kaggle sessions — is training now as a follow-up improvement.

**Model (trained + published, with sample outputs):** https://huggingface.co/Algo-Nova/fasal-sarvam1-lora
**Dataset:** https://huggingface.co/datasets/Algo-Nova/fasal-kcc-instruct (69,670 real KCC query-response pairs, 8 languages)
**Training notebook (iteration 1, in progress):** https://www.kaggle.com/code/shreshthabagohil/notebook51acbe7484

## How scoring works

Judged on **LLM-judge win-rate over the base model** (`sarvamai/sarvam-1`) on Adaption's hidden tasks, blind A/B, validated against a human-annotated anchor set (Kendall tau correlation).

## Repo map

```
data/         raw KCC (gitignored) + cleaning pipeline + data tests + version hashes
dataset/      the released instruction dataset (versioned) + data card
train/        QLoRA + AutoScientist run configs
eval/         held-out test, metric + paired-bootstrap scripts, judge rubric
model_card/   MODEL_CARD.md, criteria-mapping table, reproduce block
scripts/      reproduce.sh (the 3 commands a judge runs)
src/          shared utils (fixed seed, etc.)
```

## Architecture

- **Base model:** [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1) — chosen for native strength across Indian languages
- **Fine-tuning method:** QLoRA (4-bit NF4 quantization, LoRA r=16, alpha=32, dropout=0.05, `target_modules=all-linear`)
- **Training loop:** plain `transformers.Trainer` (not `trl.SFTTrainer` — see Engineering notes)
- **Data:** [`Algo-Nova/fasal-kcc-instruct`](https://huggingface.co/datasets/Algo-Nova/fasal-kcc-instruct), 69,670 real KCC query-response pairs across Hindi, Bengali, Gujarati, Kannada, Marathi, Punjabi, Tamil, and English

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then paste your real tokens into .env (NEVER commit .env)
# 1) put the raw KCC CSV in data/raw/ (gitignored)
python data/clean_kcc.py --in data/raw/kcc.csv --out dataset/kcc_instruct_v1.jsonl
python data/tests.py --dataset dataset/kcc_instruct_v1.jsonl  # must be all-green before training
```

See `scripts/reproduce.sh` for exact steps to reproduce training from the published adapter.

## Engineering notes

- **Why plain `Trainer` instead of `trl.SFTTrainer`:** `trl`'s SFTTrainer/SFTConfig API broke across two different pinned versions during development (a chunked cross-entropy patch bug, then a `formatting_func` signature mismatch). Standard `transformers.Trainer` + `DataCollatorForLanguageModeling` avoids this and has a stable, well-tested API.
- **Why staged training:** Kaggle's free GPU tier caps sessions at ~9 hours, and one pass over the full 69,670-row dataset needs roughly 19-20 hours. Training is split across multiple committed sessions, each continuing from the previous session's published adapter, until full dataset coverage is reached.
- **Why "Save & Run All (Commit)" over interactive sessions:** interactive Kaggle sessions can be killed by idle-timeout disconnects. Committed runs execute on Kaggle's infrastructure independent of the browser tab.

## Engineering standards

- **Leakage = 0** between train and held-out test (`data/tests.py`), re-run every regeneration.
- **PII scrubbed** before any public release (`data/clean_kcc.py`).
- **Pinned deps + fixed seed + base-model commit SHA** so a cold-reproduce reliably gives our number.

## Team

AlgoNova — Shreshthaba P Gohil (lead), Anwesha Bhagat, Aratrika Anwita, Diksha P, Gunn Diwan

## License

Code in this repo: Apache 2.0. Base model (Sarvam-1) and any adapters/derivatives fine-tuned from it: **Sarvam AI Research License** — non-commercial, research-use only, with its own attribution and redistribution terms. See `LICENSE_BASE_SARVAM.md` for the full text.
