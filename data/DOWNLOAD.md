# KCC Data — Download Runbook

Follow top to bottom. Goal: raw KCC data in `data/raw/`, then a clean instruction dataset that passes the tests. Needs NO Adaption credits — you can do this today.

## Option A — Kaggle mirror (fastest, already Q&A-cleaned) — recommended first
1. Make a free account at https://kaggle.com
2. Account → Settings → API → **Create New API Token** → downloads `kaggle.json`
3. Put it where the CLI expects it and lock permissions:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
4. From the repo root:
   ```bash
   pip install kaggle
   kaggle datasets download -d daskoushik/farmers-call-query-data-qa -p data/raw --unzip
   ```
5. You should now have a CSV in `data/raw/`. Rename it to `kcc.csv` if needed.

## Option B — data.gov.in (fuller raw archive, widest language coverage)
1. Register for a free API key at https://www.data.gov.in (My Account → API key).
2. KCC catalog: https://www.data.gov.in/dataset-group-name/kisan-call-centre
3. Use the resource API (JSON/CSV) or the bulk download on the resource page. This gives more states/months/languages — good for the multilingual goal.
4. **While here, note the licence** (GODL-India) and its attribution string → write it into `../SOURCES.md`. This is required before we publish the dataset.

> Tip: start with Option A to get moving today; pull Option B later if we need more low-resource-language volume.

## Then: clean + test (the part that matters)
```bash
# from repo root, with the venv active and deps installed
python data/clean_kcc.py --in data/raw/kcc.csv --out dataset/kcc_instruct_v1.jsonl
python data/tests.py --dataset dataset/kcc_instruct_v1.jsonl
```
- `clean_kcc.py` drops garbage, scrubs PII (phones/emails/IDs), tags language, dedupes, and prints per-language counts + a version hash.
- `tests.py` must be **all green** before we train on it. Leakage and PII checks are non-negotiable.
- Read the per-language counts it prints — at M1 we use those to **lock the final language set** (keep languages with enough volume, drop the too-sparse ones, document it).

## Do NOT
- Do not commit anything from `data/raw/` (it's gitignored — may contain PII pre-scrub).
- Do not train on `kcc_sample_ILLUSTRATIVE.csv` (that's a fake format reference, not real data).
- Do not skip the tests because "it looks fine" — that's how the last hackathon's numbers didn't reproduce.

Questions or a failing test → ping `s` (repo/reproducibility lane).
