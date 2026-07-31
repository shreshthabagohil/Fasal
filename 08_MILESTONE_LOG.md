# Fasal — Milestone Log

## M1 — Dataset validation and quality audit

**Status:** Complete  
**Date:** 2026-07-31

### Dataset
- Final instruction dataset: `dataset/kcc_instruct_v1.jsonl`
- Rows: **129,022**
- Language: **English (`en`)**
- Dataset SHA-256: `9ccffed504cf3b72f51a040112adf713dc6c2cd8cf2c890a89d59cac02370853`
- Python: **3.11**
- CUDA: **unavailable** on the current environment
- Global seed: **1729**

### Automated data gates
- D1 Schema: **PASS**
- D2 Train/held-out leakage: **NOT RUN — held-out set not created yet**
- D4 Exact-normalized duplicate pairs: **PASS**
- D5 PII: **PASS — 0 detected**
- D6 Language coverage: **PASS — 129,022 English rows**

### Manual audits
- D3 Paraphrase audit: **Completed**
  - 2,000-row deterministic sample
  - High-similarity pairs were observed
  - Many were wording/spacing variants of the same question
  - No automatic deletion was performed because similar questions can have different KCC answers
- D7 Spot-read: **Completed**
  - 20 deterministic sample records reviewed
  - Dataset judged usable overall
  - A small number of weak/generic answers were observed but no broad destructive filter was applied

### M1 decision
The dataset is accepted for progression to the next milestone. No further broad data filtering is applied at M1.

**Important:** D2 leakage must be rerun after the held-out evaluation set is created, before training/release.

