# data/

- `raw/` — put the downloaded KCC CSV here. **Gitignored** (may contain PII pre-scrub). Never commit raw.
- `clean_kcc.py` — cleaning pipeline → instruction JSONL in `../dataset/`.
- `tests.py` — the data gate tests (schema, leakage=0, dedup, PII, coverage). Must be green before training.

Flow: download → `python data/clean_kcc.py --in data/raw/kcc.csv --out ../dataset/kcc_instruct_v1.jsonl`
→ `python data/tests.py --dataset ../dataset/kcc_instruct_v1.jsonl`.
