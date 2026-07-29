# LLM-judge rubric (frozen for the whole event — 05 §3)

Judge at **temperature 0**, **blind randomized A/B** (two anonymized answers, order counterbalanced),
judge model from a **different family** than our base where possible. Same rubric for every language.

Score each answer 1–5:

- **5** — correct + specific + actionable: names the input/dose/timing/variety, and is safe to follow.
- **4** — correct and mostly specific, minor vagueness.
- **3** — generally correct but vague or missing a key specific.
- **2** — partially wrong, or unsafe if followed literally.
- **1** — wrong, off-topic, hallucinated, or answered in the wrong language/script than asked.

**Auto-cap at ≤2:** confident-but-generic ("use good medicine"), wrong crop/pest, unsafe dose,
or wrong-language answer.

**Win-rate:** for each item, the higher-scored answer wins (ties = no win). Report ours' win-rate
over base overall (with paired-bootstrap CI) and per-language (directional, wide CIs — do not claim
per-language significance).

Anchor the judge on the hand-read gold subset (hi/gu) before trusting it on languages we can't read.
