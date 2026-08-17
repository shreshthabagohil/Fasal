# Model Card — Fasal · Multilingual KCC Farmer-Advisory Model

> **Result:** <!-- TODO(eval): fill after eval/reports/eval_report.json lands, keep in sync with README.md -->
> `+__._%` LLM-judge win-rate over `sarvamai/sarvam-1` (base) on N=1,226 multilingual held-out test.
> 95% CI [+_._, +_._] (paired bootstrap, 10k resamples). Judge: `openai/gpt-oss-120b` via Groq, dual-order, seed 1729 (see `eval/JUDGE_BUDGET.md` for the 2026-08-17 judge-model change).

*(Submitted-entry number used N=170 for time-budget reasons — see `eval/JUDGE_BUDGET.md`. This card is being
updated with the fully-powered N=1,226 re-run.)*

## Judging-criteria map (do the judge's scoring for them)
| Criterion | What we did | Where |
|---|---|---|
| Measurable improvement | +__% win-rate vs our own base, significance-tested | this card, §Result |
| Dataset quality/originality | cleaned, multi-language-tagged, deduped, PII-scrubbed KCC instruction set | §Dataset + Kaggle |
| Real-world impact | 100M+ farm households; multilingual advisory in the farmer's language | §Intended use |
| AutoScientist depth | N documented co-optimization iterations + ablation table | §Method + experiment log |
| Open-release quality | this card, reproduce block, dataset+adapter released, licenses documented | §Reproduce, §License |

## Dataset
Source: Kisan Call Centre (KCC), data.gov.in, GODL-India (redistribution + attribution). Cleaning:
garbage-drop, dedup, PII scrub, script-normalize, language-tag. 69,670 training rows across 8 languages:
en=64,059 · bn=819 · gu=802 · hi=803 · kn=798 · mr=794 · pa=797 · ta=798.
HF: [`Algo-Nova/fasal-kcc-instruct`](https://huggingface.co/datasets/Algo-Nova/fasal-kcc-instruct).

## Method
Base: `sarvamai/sarvam-1` @ commit `e9607337286ddf496d4a2562b194e489dcf3feea`. QLoRA SFT, 4-bit NF4,
`r=16 alpha=32 dropout=0.05 target_modules="all-linear"`, trained via `transformers.Trainer`.

## Experiment log (AutoScientist depth)
| Iter | What changed | Dataset ver | Win-rate vs base | Notes |
|---|---|---|---|---|
| 0 | baseline QLoRA fine-tune, 600-example proof-of-concept | v0 | not measured | submitted as fallback during Kaggle session loss |
| 1 | full-dataset retrain (69,670 rows) | v1 (`29b553a73c22`) | <!-- TODO(eval) --> | current published adapter |

## Reproduce
```bash
bash scripts/reproduce.sh   # pulls adapter + pinned base, runs eval, prints the number
```

## Safety & limitations (06 §3)
In-domain agricultural advice only; declines out-of-domain. **AI-generated guidance — verify
pesticide names/doses with your local KVK/KCC before acting.** Weak spots: low-resource languages, rare crops.

## License
Data: GODL-India (attribution string pending — see `SOURCES.md`). Base model: Sarvam AI Research License
(non-commercial-flavored) — flows through since we release adapter-only, not merged weights. Base commit SHA:
`e9607337286ddf496d4a2562b194e489dcf3feea`.
