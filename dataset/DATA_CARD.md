# Fasal Dataset Card — kcc_instruct_v1

## Dataset
- Name: KCC instruction dataset v1
- File: dataset/kcc_instruct_v1.jsonl
- Final training rows: 69670
- Languages: en, hi, gu, mr, ta, bn, kn, pa (7 machine-translated via NLLB, see notes)
- Dataset version hash: see `dataset/VERSION`

## Data Quality Gates

| Gate | Status |
|---|---|
| D1 Schema validation | PASS |
| D2 Train/heldout leakage | PASS (0 overlap) |
| D3 Paraphrase audit | Completed |
| D4 Exact duplicate check | PASS |
| D5 PII removal | PASS (0 detected) |
| D6 Language coverage | PASS |
| D7 Spot-read | Completed |

## Evaluation Split

Held-out evaluation data:
- File: `eval/heldout/test.jsonl`
- Reserve set: `eval/heldout/reserve.jsonl`
- Frozen hashes: `eval/heldout/HASHES.md`

## Notes

7 of 8 languages (hi/gu/mr/ta/bn/kn/pa) are machine-translated from English via facebook/nllb-200-distilled-600M, not native transcripts. English (en) is the original KCC text.

The dataset was cleaned using the KCC cleaning pipeline:
- garbage removal
- normalization
- PII scrubbing
- language tagging
- deduplication

The dataset is approved for progression to the next milestone.
