# Judge budget lane — locked at M3 step 3.5 (revised 2026-08-09, time-crunch pass)

Groq free-tier caps for llama-3.3-70b-versatile (verified 2026-07-27):
  30 RPM · 1,000 RPD · 12,000 TPM · 100,000 TPD (per key)

Judge call avg ~1,700 tokens.

## Chosen lane: B+ (multi-key pool + fixed final-report N)

## Reason
Lane B (shrink held-out to N<=80 for iterative evals) still stands for M3/M4
iteration comparisons. The ORIGINAL plan for the final M5 report was full
N=1,226 dual-order (2,452 calls) on a single key — the math for that
(~58 calls/day on 1 key) meant ~42 days, incompatible with the submission
deadline. Two changes fix this without weakening the final claim:

1. judge.py now rotates across multiple Groq API keys (up to 6 — Lead +
   up to 5 teammates, each a free account, zero cost). This multiplies the
   *daily* token budget by the number of keys pooled. Because the cap is a
   rolling-day total (not something you're forced to spread evenly across
   24h), a run that fits inside the pooled daily budget completes in
   minutes, limited only by RPM/TPM, not by "days."
2. The FINAL report uses N=170 (not the full 1,226), dual-order, chosen to
   fit safely inside a 6-key pooled budget with margin (170 x 2 x 1,700 =
   578,000 tokens; 6 keys = 600,000 TPD budget).

## Impact
- Held-out N used for iterative evals (M3/M4): <=80, deterministic subset, same as before.
- Held-out N used for final M5 report: N=170, dual-order (was: full N=1,226 dual-order — reduced for time, documented here per RULEBOOK "no silent shortcut" rule).
- Judge model + endpoint: llama-3.3-70b-versatile via Groq (https://api.groq.com/openai/v1/chat/completions)
- Keys pooled: set GROQ_API_KEYS as a comma-separated list of every teammate's free Groq key.
- Estimated total judge calls for the final report: 340 (170 items x 2 orders).
- Approved spend: none — still zero-budget, all keys are free tier.

## Known limitation to disclose in the model card
N=170 (not the full 1,226-row held-out) is used for the headline significance
claim, due to the hackathon's compute/time deadline, not a hidden design
choice. The full 1,226-row set stays frozen and available for anyone to
re-run post-event if they want the fully-powered number.

## Addendum 2026-08-16 — post-submission full-N attempt, multi-provider pool

With the submission deadline passed, attempted the full N=1,226 dual-order
report for the portfolio version of this project. First attempt used the
Groq-only 7-key pool from above and failed: N=1,226 dual-order needs
~4.17M tokens (2,452 calls x ~1,700 tokens), but 7 Groq keys only provide
7 x 100,000 = 700,000 TPD — about 6x short. The run burned a full 7-hour
Kaggle GPU session mostly retrying against exhausted keys and only scored
55/1,226 items before the notebook finished (no crash, just no data).

Fix: `judge.py` now pools keys across **two** free-tier providers instead
of one — Groq (`llama-3.3-70b-versatile`, 100K TPD/key) AND Cerebras Cloud
(`llama-3.3-70b`, 1,000,000 TPD/key + 14,400 RPD/key, OpenAI-compatible
endpoint, no card required). One Cerebras key alone is ~10x a single Groq
key's daily budget. Set `CEREBRAS_API_KEYS` (comma-separated, same pattern
as `GROQ_API_KEYS`) alongside the existing Groq keys; `judge.py` rotates
across the combined pool automatically. `--resume` was also added so a run
interrupted by a rate-limit storm can continue from the last
cleanly-scored item instead of re-spending budget on rows already done.

Mixing two providers for the same nominal model is a real, disclosed
tradeoff: Groq and Cerebras may not be bit-identical deployments of
Llama-3.3-70B even at temperature 0, so which provider judged a given
item is a (small, undocumented-magnitude) source of noise not present in
a single-provider run. Worth a sentence in the model card's limitations
section rather than treating the combined-provider number as identical
in rigor to a single-provider one.
