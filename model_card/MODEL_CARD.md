# Model Card — Fasal · Multilingual KCC Farmer-Advisory Model

> **Result:** +__._% LLM-judge win-rate over `<base model>` on our N=___ multilingual held-out test.
> 95% CI [+_._, +_._], p<0.0__ (paired bootstrap, 10k). Stable across 3 seeds (+_._ to +_._).

*(This number goes FIRST — playbook Rule 9. Fill from M2 onward.)*

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
garbage-drop, dedup, PII scrub, script-normalize, language-tag. Languages + counts: _____. Size: _____.
Kaggle: _____ · HF: _____.

## Method
Base: `<model>` @ commit `<sha>`. QLoRA SFT (4-bit). AutoScientist co-optimization loop; tracked in W&B.

## Experiment log (AutoScientist depth)
| Iter | What changed | Dataset ver | Win-rate vs base | Notes |
|---|---|---|---|---|
| 0 | baseline fine-tune | v1 (`<hash>`) | | |

## Reproduce
```bash
bash scripts/reproduce.sh   # pulls adapter + pinned base, runs eval, prints the number
```

## Safety & limitations (06 §3)
In-domain agricultural advice only; declines out-of-domain. **AI-generated guidance — verify
pesticide names/doses with your local KVK/KCC before acting.** Weak spots: low-resource languages, rare crops.

## License
Data: GODL-India (attribution: _____). Base model: `<license>` (flows through if merged weights released;
we release adapter-only where possible). Base commit SHA: `<sha>`.
