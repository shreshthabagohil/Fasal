# Fasal Dataset Card — kcc_instruct_v1

## Dataset
- Name: KCC instruction dataset v1
- File: dataset/kcc_instruct_v1.jsonl
- Final training rows: 54313 (was 69,670 before the 2026-08-18 data-integrity fix — see Addendum below)
- Languages: en, hi, gu, mr, ta, bn, kn, pa (7 machine-translated via NLLB, see notes)
- Dataset version hash: see `dataset/VERSION`

## Data Quality Gates

| Gate | Status |
|---|---|
| D1 Schema validation | PASS |
| D2 Train/heldout leakage (same-language exact match) | PASS (0 overlap) |
| D2b Train/heldout leakage (cross-lingual, same source question translated into a different language) | Originally: 17 rows found and excluded from the held-out set 2026-08-12 (see `eval/heldout/LEAKAGE_AUDIT.md`, written against a 227-row held-out set). STALE — see Addendum 2026-08-18 below: re-run against the true final dataset found 668 phrasings / 13,862 rows still leaking; now fixed and re-verified PASS. |
| D3 Paraphrase audit | Completed |
| D4 Exact duplicate check | PASS |
| D5 PII removal | Originally: PASS (0 detected). STALE — see Addendum 2026-08-18 below: re-run against the true final dataset found 470 rows with residual PII; now fixed and re-verified PASS. |
| D6 Language coverage | PASS |
| D7 Spot-read | Completed |

**Note on D2 vs D2b:** D2 only compares each row's own surface text, which differs by construction across the 8 languages here (7 are machine translations of the same English source). It cannot catch the case where the same underlying KCC question appears in train under one language and in held-out under a translation of it — D2b closes that gap. Read D2 "PASS (0 overlap)" as same-language-only; the cross-lingual claim is D2b's row above, not D2's.

## Evaluation Split

Held-out evaluation data:
- File: `eval/heldout/test.jsonl`
- Reserve set: `eval/heldout/reserve.jsonl`
- Frozen hashes: `eval/heldout/HASHES.md`

## Addendum 2026-08-18 — data-integrity fix pass (D5 PII + D2b leakage were never actually gate-checked against the true final file)

`data/tests.py` (the automated D-gate script) had never been run against the
true final 69,670-row merged dataset -- only against an earlier, smaller
version, before the last Kaggle/Colab session's rows were appended. Running
it for the first time against the actual shipped file (downloaded fresh from
`Algo-Nova/fasal-kcc-instruct` on HF to confirm this wasn't an artifact of
local edits) surfaced two real, pre-existing failures:

- **D5 PII: 470 rows still contained live phone numbers.** Root cause: NLLB
  translation mangled the `[PHONE]` redaction placeholder itself into
  script-specific text (e.g. `[ফোন]` in Bengali) instead of preserving it
  literally, while a *second*, differently-formatted phone number in the
  same source row (one that didn't match the original PII regex shapes)
  survived untouched and got translated/copied through as live text.
- **D2b cross-lingual leakage: 668 unique question phrasings** (13,862
  individual rows, once every row sharing each phrasing is counted) were
  present in both train and `eval/heldout/test.jsonl`. This is a much larger
  number than the "17 rows" originally documented and fixed on 2026-08-12 --
  that fix was correct for the pool size at the time, but the final
  Kaggle/Colab session's additional ~14.7K rows were never re-audited
  against the frozen held-out set before shipping.

**Fix applied:** re-ran `is_garbage()` (now including the new
`is_call_redirect()` check, see `data/clean_kcc.py`), `contains_pii()`, and a
held-out-source-key filter against the full original 69,670-row dataset (not
incremental patches). Row accounting:

| Step | Rows removed | Reason |
|---|---:|---|
| Call-redirect / KCC noise | 1,025 | Misdials, boilerplate call-redirects, bare toll-free numbers -- see `data/clean_kcc.py`'s `is_call_redirect()` comment |
| Residual PII | 470 | Live phone numbers that survived translation (root cause above) |
| Cross-lingual leakage | 13,862 | Every row sharing one of the 668 question-phrasings also present in `eval/heldout/test.jsonl` |
| **Kept** | **54,313** | (was 69,670) |

Per-language after fix: en=52,081, bn=342, hi=333, gu=331, kn=315, mr=309,
pa=307, ta=295 -- all still above `data/tests.py`'s `MIN_LANG_VOLUME=200`
floor, but each translated language shrank by roughly 60% (from ~800 rows).
This was a deliberate choice (not a silent shortcut): the alternative was
leaving ~20% of the corrected leakage undisclosed. `eval/heldout/test.jsonl`
and `eval/heldout/reserve.jsonl` themselves were **not** modified -- the
fix removed the leaking rows from *train*, not from held-out, so the frozen
hashes in `eval/heldout/HASHES.md` and all prior D2/D2b-adjacent audits of
the held-out set itself remain valid.

Re-verified after the fix: `python data/tests.py --dataset
dataset/kcc_instruct_v1.jsonl --heldout eval/heldout/test.jsonl` reports all
6 automated gates PASS. New `dataset/VERSION` hash:
`e16b774cf3d8a7c27dcbf24db25e5acb92657fc79615a576d304fa4abc975341`.

**Not yet done:** the 470-PII-row and 13,862-leakage-row categories were
computed against the *original* dataset's translated slices (`dataset/translated/*.jsonl`,
the ~800-per-language stratified samples used to build the merged file) --
those source files were not separately re-cleaned, only their downstream
effect on the merged `kcc_instruct_v1.jsonl` was fixed. If `translate_to_indic.py`
is ever re-run to grow the translated languages back up, the same PII-placeholder-mangling
and leakage risks apply and must be re-checked, not assumed fixed by this pass.

## Notes

7 of 8 languages (hi/gu/mr/ta/bn/kn/pa) are machine-translated from English via facebook/nllb-200-distilled-600M, not native transcripts. English (en) is the original KCC text.

The dataset was cleaned using the KCC cleaning pipeline:
- garbage removal
- normalization
- PII scrubbing
- language tagging
- deduplication

The dataset is approved for progression to the next milestone.
