# Data provenance & licences (06 §5)

## KCC data
- Resource URL: https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers
- Kaggle mirror actually used for download (per `data/DOWNLOAD.md` Option A, faster/already Q&A-cleaned): https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa
- Downloaded on: 2026-07-31 (first real data-cleaning commit against the Kaggle mirror, `0c15712`; the exact `kaggle datasets download` timestamp was not separately logged — this date is the earliest verifiable record from git history and should be treated as accurate to the day, not the minute).
- Licence: GODL-India (Government Open Data License – India). The data.gov.in resource page publishes KCC transcripts under this licence, which permits redistribution and adaptation with attribution — confirmed against the licence terms at https://data.gov.in/government-open-data-license-india.
- Attribution string to reproduce in the dataset card + model card:
  > Contains information from the Kisan Call Centre (KCC) Query-Answer dataset, sourced from data.gov.in (Government of India), made available under the Government Open Data License – India (GODL) — https://data.gov.in/government-open-data-license-india. Downloaded via Kaggle mirror: https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa.
- Note: the licence attaches to the underlying data.gov.in resource, not the Kaggle mirror itself — the mirror is a convenience re-host of the same GODL-licensed data (see `data/DOWNLOAD.md` Option A vs Option B).

## Base model
- Model + HF commit SHA: `sarvamai/sarvam-1` @ `e9607337286ddf496d4a2562b194e489dcf3feea` (confirmed live via `https://huggingface.co/api/models/sarvamai/sarvam-1` on 2026-08-12; this is the sole base actually trained and shipped — see `train/A0_LOCKED.md` for why the originally-planned `google/gemma-2-2b` A/B was abandoned and never trained).
- Licence + any pass-through terms: Sarvam AI Research License (non-commercial-flavored — permission required for uses not expressly authorized by the licence). Full text: `repo/LICENSE_BASE_SARVAM.md`. This licence flows through to the released LoRA adapter (`Algo-Nova/fasal-sarvam1-lora`) and any merged checkpoint, per `train/A0_LOCKED.md` §Licences.
