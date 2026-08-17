# Judge budget lane — locked at M3 step 3.5 (revised 2026-08-09, time-crunch pass; wiring added 2026-08-12)

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

## How the N budget is actually enforced (added 2026-08-12)
Until now this N was documentation only — judge.py processes every row of
whatever `--heldout` file it receives, so nothing in the codebase actually
stopped a run from re-processing the full 1,226-row set and blowing the
budget above. `eval/run_eval.sh` now takes a `MAX_N` env var that, when set,
runs `eval/sample_heldout.py` first to build a deterministic (seeded),
language-stratified subsample of `eval/heldout/test.jsonl` before handing it
to `infer.py`/`judge.py`, so the subset is reproducible run-to-run and every
language keeps representation even at N=170 or N=80 (not just the highest-
volume ones):

```bash
# M3/M4 iterative comparison (N<=80):
MAX_N=80  BASE_MODEL=sarvamai/sarvam-1 BASE_SHA=e9607337286ddf496d4a2562b194e489dcf3feea \
  ADAPTER_REPO=Algo-Nova/fasal-sarvam1-lora GROQ_API_KEYS=... bash eval/run_eval.sh

# Final M5 report (N=170, the locked number this doc names above):
MAX_N=170 BASE_MODEL=sarvamai/sarvam-1 BASE_SHA=e9607337286ddf496d4a2562b194e489dcf3feea \
  ADAPTER_REPO=Algo-Nova/fasal-sarvam1-lora GROQ_API_KEYS=... bash eval/run_eval.sh
```

Leaving `MAX_N` unset preserves the old behavior (full `$HELDOUT`) — useful
for a post-event, fully-powered re-run against the frozen 1,226-row set.

## Impact
- Held-out N used for iterative evals (M3/M4): <=80, deterministic subset (`MAX_N=80`).
- Held-out N used for final M5 report: N=170, dual-order (`MAX_N=170`) — was full N=1,226 dual-order, reduced for time, documented here per RULEBOOK "no silent shortcut" rule.
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

## Addendum 2026-08-17 — Cerebras dropped entirely; Groq's judge model also changed

The 2026-08-16 fix above turned out to be broken in two separate, unrelated
ways, both only discovered by hitting the live APIs directly rather than
trusting docs/memory of an earlier verification:

1. **Cerebras's `llama-3.3-70b` was itself deprecated on 2026-02-16** (six
   months before this was even attempted) — every Cerebras call 404'd for
   ~7 hours on Kaggle, burning most of a week's GPU quota for zero scored
   items beyond what Groq alone produced. Re-pointing Cerebras at its
   suggested replacement (`gpt-oss-120b`) then hit a second wall: as of
   **2026-08-17**, Cerebras's free tier itself now requires a verified
   payment method on file to unlock any credits (`402 payment_required`).
   That directly conflicts with this project's zero-budget rule regardless
   of model choice, so **Cerebras is dropped from the provider pool
   entirely** — not a code fix, a hard policy wall on their end.
2. Independently, live-querying this account's actual Groq model list
   (`GET /openai/v1/models`, not the docs page) showed
   `llama-3.3-70b-versatile` is no longer available on this account at
   all — every call returned `404 model_not_found`. The docs page listing
   it with rate limits was stale relative to the live API. **The judge
   model is now `openai/gpt-oss-120b`** (Groq's own migration
   recommendation, and it happens to carry a *higher* per-key daily budget
   than the old model: 200K TPD vs. 100K TPD).

Net effect on the post-submission full-N=1,226 attempt: back to
**Groq-only**, but with the better-budgeted model — 7 keys x 200,000 TPD =
1,400,000 TPD pooled, vs. 700,000 TPD under the old model. At ~1,700
tokens/call this needs ~4.17M tokens total (2,452 calls), so roughly
**3 daily `--resume` runs** to complete the full dual-order set, run
locally (no GPU needed — `judge.py` is pure HTTP, the GPU-heavy adapter
inference step was already completed and cached as `eval/out/ours.jsonl`
in an earlier Kaggle session).

Also worth flagging: the existing human-anchor validation
(`eval/judge_anchor.py`, tau=0.481, p=0.0003, see `eval/reports/judge_anchor.md`)
was run against the now-dead `llama-3.3-70b-versatile`, not
`gpt-oss-120b`. That correlation result technically no longer validates
*this* judge model's reliability — a disclosed limitation, not silently
carried forward as still-current. Re-running the anchor validation against
`gpt-oss-120b` would close this gap but hasn't been done yet.
