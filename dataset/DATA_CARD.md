# Fasal Dataset Card — kcc_instruct_v1

## Dataset
- Name: KCC instruction dataset v1
- File: dataset/kcc_instruct_v1.jsonl
- Final training rows: 135846
- Language: English (`en`)
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

The dataset was cleaned using the KCC cleaning pipeline:
- garbage removal
- normalization
- PII scrubbing
- language tagging
- deduplication

The dataset is approved for progression to the next milestone.
