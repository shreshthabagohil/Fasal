# Judge budget lane — locked at M3 step 3.5

Groq free-tier caps for llama-3.3-70b-versatile (verified 2026-07-27):
  30 RPM · 1,000 RPD · 12,000 TPM · 100,000 TPD

Judge call avg ~1,700 tokens => TPD = ~58 calls/day. Full dual-order eval on N=1226 (real frozen held-out size, eval/heldout/HASHES.md) = up to 2,452 calls / ~4.17M tokens => far past the ~58 calls/day TPD cap; at 58 calls/day this alone would take ~42 days to run in one pass.

## Chosen lane: B (shrink held-out to N<=80 for iterative evals)

## Reason
Zero-budget rule (project hard constraint) rules out Lane C (paid Groq). Lane B is the smallest code change (subset the held-out at eval time, no new judge.py backend) and it protects the number that actually matters: the final M5 report still runs on the full N=1226 (paced across multiple days to respect the ~58 calls/day TPD cap), so the headline significance claim stays fully powered. Only the M3/M4 iteration-to-iteration comparisons get wider CIs, which is acceptable since we are not claiming significance on those anyway.

## Impact
- Held-out N used for iterative evals: <=80 (subset of the frozen N=1226 held-out set, sampled deterministically with the project seed)
- Held-out N used for final M5 report: full held-out (N=1226 per eval/heldout/HASHES.md)
- Judge model + endpoint: llama-3.3-70b-versatile via Groq (https://api.groq.com/openai/v1/chat/completions)
- Estimated total judge calls across M3+M4+M5: within Groq free-tier TPD budget at N<=80 dual-order per iteration (~160 calls/iteration); full N=1226 dual-order reserved for the single final M5 report
- Approved spend (if Lane C): none — Lane C not selected
