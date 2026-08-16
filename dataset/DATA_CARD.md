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
| D2 Train/heldout leakage (same-language exact match) | PASS (0 overlap) |
| D2b Train/heldout leakage (cross-lingual, same source question translated into a different language) | 17 rows found and excluded from the held-out set 2026-08-12 — see `eval/heldout/LEAKAGE_AUDIT.md`. Automated gate added in `data/tests.py`; current `eval/heldout/test.jsonl` (227 rows) is clean per this check. |
| D3 Paraphrase audit | Completed |
| D4 Exact duplicate check | PASS |
| D5 PII removal | PASS (0 detected) |
| D6 Language coverage | PASS |
| D7 Spot-read | Completed |

**Note on D2 vs D2b:** D2 only compares each row's own surface text, which differs by construction across the 8 languages here (7 are machine translations of the same English source). It cannot catch the case where the same underlying KCC question appears in train under one language and in held-out under a translation of it — D2b closes that gap. Read D2 "PASS (0 overlap)" as same-language-only; the cross-lingual claim is D2b's row above, not D2's.

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
