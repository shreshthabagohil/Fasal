# train/

QLoRA + AutoScientist run configs go here (added at M1/M2 once the base model is chosen via the A/B).

- Keep every run's config committed (LR, rank, schedule, dataset version hash, seed) — it's the
  AutoScientist depth evidence and the reproducibility record.
- Log each run to W&B: one project, one run per iteration.
- QLoRA is locked (small adapter = exportable; avoids the platform's full-weights download problem).
