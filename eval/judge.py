#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import requests


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# 2026-08-16: N=1,226 dual-order (2,452 calls, ~4.17M tokens) does not fit
# inside Groq's free-tier daily budget even pooled across 7 keys (7 x
# 100,000 TPD = 700,000 TPD) -- confirmed empirically, run died at 55/1226
# scored after a 7h wall-clock loop that mostly retried against exhausted
# keys. Cerebras Cloud's free tier gives 1,000,000 tokens/day + 14,400
# requests/day PER KEY on the same llama-3.3-70b model (OpenAI-compatible
# endpoint, no card required) -- ~10x the per-key budget of Groq. Pooling
# a few free Cerebras keys alongside the existing Groq keys is what makes
# the full N=1,226 run actually finish. See PROVIDERS / MODEL_BY_PROVIDER
# below -- this is a second *provider*, not just a second key.
PROVIDERS = {
    "groq": GROQ_URL,
    "cerebras": CEREBRAS_URL,
}
MODEL_BY_PROVIDER = {
    "groq": DEFAULT_MODEL,
    # Cerebras' model id for the same Llama-3.3-70B weights. Verify this
    # against https://inference-docs.cerebras.ai/ before a long run --
    # provider model ids drift independently of Groq's.
    "cerebras": "llama-3.3-70b",
}
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 1729
DEFAULT_MAX_RETRIES = 5
DEFAULT_SLEEP_ON_429 = 60
DEFAULT_QPS = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic pairwise LLM judging with Groq."
    )
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dual-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--sleep-on-429", type=float, default=DEFAULT_SLEEP_ON_429)
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If --out already exists, keep every row that scored cleanly "
            "(no 'error' key) and only re-run the ids that are missing or "
            "previously errored. Lets a run interrupted by a rate-limit "
            "storm continue instead of re-spending budget on already-scored "
            "items."
        ),
    )
    return parser.parse_args()


def load_scores_lenient(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Reads a previously-written --out file for --resume. Lenient by
    design: a truncated/half-written last line (e.g. the process was
    killed mid-write) is skipped rather than raising, since the whole
    point of --resume is recovering from an interrupted prior run."""
    rows: List[Dict[str, Any]] = []
    skipped = 0

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    except FileNotFoundError:
        return [], 0

    return rows, skipped


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
    return rows


def index_by_id(rows: List[Dict[str, Any]], path: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if "id" not in row:
            raise ValueError(f"{path}: row is missing 'id'")

        row_id = str(row["id"])

        if row_id in result:
            raise ValueError(f"{path}: duplicate id: {row_id}")

        result[row_id] = row

    return result


def load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent / "judge_prompt.md"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Frozen judge prompt not found: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")


def build_user_message(
    prompt_template: str,
    question: str,
    context: str,
    answer_a: str,
    answer_b: str,
) -> str:
    replacements = {
        "Question (verbatim, may be in Hindi, Gujarati, Marathi, Tamil, Bengali, Kannada, Punjabi, English, or code-mixed).":
            question,
        "Context (State / Crop / Season / QueryType).":
            context,
        "Answer A (anonymised).":
            answer_a,
        "Answer B (anonymised).":
            answer_b,
    }

    message = prompt_template

    for placeholder, value in replacements.items():
        message = message.replace(placeholder, value)

    return message


class JudgeOutputParseError(ValueError):
    """Raised when the judge response body is not valid/expected JSON.

    Carries the raw response text so the caller can log and persist it
    for debugging (T03 spec step 9: unparseable judge output must be
    written as {"id","error","raw"}).
    """

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def parse_judge_json(content: str) -> Dict[str, Any]:
    try:
        return _parse_judge_json_impl(content)
    except ValueError as exc:
        raise JudgeOutputParseError(str(exc), raw=content) from exc


def _parse_judge_json_impl(content: str) -> Dict[str, Any]:
    text = content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("judge output is not valid JSON")

        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"judge output is not valid JSON: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("judge output JSON must be an object")

    for key in ("score_A", "score_B", "choice", "rationale"):
        if key not in parsed:
            raise ValueError(f"judge output missing field: {key}")

    try:
        score_a = int(parsed["score_A"])
        score_b = int(parsed["score_B"])
    except (TypeError, ValueError) as exc:
        raise ValueError("score_A and score_B must be integers") from exc

    if not 1 <= score_a <= 5:
        raise ValueError(f"score_A outside 1-5: {score_a}")

    if not 1 <= score_b <= 5:
        raise ValueError(f"score_B outside 1-5: {score_b}")

    if parsed["choice"] not in ("A", "B", "tie"):
        raise ValueError(
            f"choice must be A, B, or tie; got {parsed['choice']!r}"
        )

    parsed["score_A"] = score_a
    parsed["score_B"] = score_b
    parsed["rationale"] = str(parsed["rationale"])

    return parsed


class KeyPool:
    """Round-robins across multiple API keys -- possibly from more than one
    provider (Groq, Cerebras, ...) -- so one key/provider hitting its
    rate/daily limit doesn't stop the whole eval run. When a key gets a
    429, it is put in cooldown and skipped until every other key has
    also been tried once (then cooldown resets so a key that recovered
    can be reused).

    Each entry is a (provider, key) pair, e.g. ("groq", "gsk_...") or
    ("cerebras", "csk_..."). Mixing providers in one pool is what lets the
    daily token budget scale past what a single provider's free tier
    allows -- see the PROVIDERS/MODEL_BY_PROVIDER note above judge.py's
    constants.
    """

    def __init__(self, entries: List[Tuple[str, str]]):
        if not entries:
            raise ValueError("KeyPool needs at least one (provider, key) entry")
        for provider, _ in entries:
            if provider not in PROVIDERS:
                raise ValueError(
                    f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}"
                )
        self._entries = entries
        self._index = 0
        self._cooldown: Set[int] = set()

    def get(self) -> Tuple[str, str]:
        if len(self._cooldown) >= len(self._entries):
            self._cooldown.clear()
        for _ in range(len(self._entries)):
            idx = self._index
            entry = self._entries[idx]
            self._index = (self._index + 1) % len(self._entries)
            if idx not in self._cooldown:
                return entry
        # Should not happen given the clear() above, but stay safe.
        return self._entries[0]

    def mark_limited(self, entry: Tuple[str, str]) -> None:
        if entry in self._entries:
            self._cooldown.add(self._entries.index(entry))

    def __len__(self) -> int:
        return len(self._entries)

    def provider_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for provider, _ in self._entries:
            counts[provider] = counts.get(provider, 0) + 1
        return counts


def request_judge(
    session: requests.Session,
    key_pool: KeyPool,
    model: str,
    temperature: float,
    seed: int,
    user_message: str,
    max_retries: int,
    sleep_on_429: float,
    model_override: bool,
) -> Dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        provider, api_key = key_pool.get()
        url = PROVIDERS[provider]
        effective_model = model if model_override else MODEL_BY_PROVIDER[provider]

        payload = {
            "model": effective_model,
            "temperature": temperature,
            "seed": seed,
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=120,
            )

            status = response.status_code

            if status == 429:
                # This key is rate/daily limited -- try a different key
                # (possibly a different provider entirely) on the very
                # next attempt instead of just sleeping on the same
                # exhausted key.
                key_pool.mark_limited((provider, api_key))

                if attempt >= max_retries:
                    response.raise_for_status()

                if len(key_pool) > 1:
                    continue  # immediately retry with the next key

                time.sleep(sleep_on_429)
                continue

            if status >= 500:
                if attempt >= max_retries:
                    response.raise_for_status()

                delay = 2 ** attempt
                time.sleep(delay)
                continue

            response.raise_for_status()

            body = response.json()

            choices = body.get("choices")
            if not choices:
                raise ValueError("Groq response contains no choices")

            content = choices[0].get("message", {}).get("content")

            if not isinstance(content, str):
                raise ValueError("Groq response has no textual message content")

            return parse_judge_json(content)

        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc

            if attempt >= max_retries:
                raise

            delay = 2 ** attempt
            time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("judge request failed without an error")


def translate_choice(
    choice: str,
    a_is_base: bool,
) -> str:
    if choice == "tie":
        return "tie"

    if a_is_base:
        return "base" if choice == "A" else "ours"

    return "ours" if choice == "A" else "base"


def aggregate_orders(
    first: Dict[str, Any],
    second: Dict[str, Any] | None,
) -> Tuple[float, float, str, str | None]:
    base_scores = []
    ours_scores = []

    base_scores.append(first["score_A"])
    ours_scores.append(first["score_B"])

    choice_o1 = translate_choice(first["choice"], a_is_base=True)

    choice_o2 = None

    if second is not None:
        ours_scores.append(second["score_A"])
        base_scores.append(second["score_B"])
        choice_o2 = translate_choice(second["choice"], a_is_base=False)

    score_base = sum(base_scores) / len(base_scores)
    score_ours = sum(ours_scores) / len(ours_scores)

    if second is None:
        choice_final = choice_o1
    elif choice_o1 == choice_o2:
        choice_final = choice_o1
    else:
        choice_final = "tie"

    return score_base, score_ours, choice_o1, choice_o2, choice_final


def validate_id_sets(
    heldout: Dict[str, Dict[str, Any]],
    base: Dict[str, Dict[str, Any]],
    ours: Dict[str, Dict[str, Any]],
) -> None:
    heldout_ids = set(heldout)
    base_ids = set(base)
    ours_ids = set(ours)

    if base_ids != ours_ids:
        missing_from_base = sorted(ours_ids - base_ids)
        missing_from_ours = sorted(base_ids - ours_ids)

        raise ValueError(
            "base/ours id sets do not match: "
            f"missing_from_base={missing_from_base[:10]}, "
            f"missing_from_ours={missing_from_ours[:10]}"
        )

    if heldout_ids != base_ids:
        missing_from_heldout = sorted(base_ids - heldout_ids)
        missing_from_outputs = sorted(heldout_ids - base_ids)

        raise ValueError(
            "heldout/output id sets do not match: "
            f"missing_from_heldout={missing_from_heldout[:10]}, "
            f"missing_from_outputs={missing_from_outputs[:10]}"
        )


def load_provider_keys(env_var: str, provider: str) -> List[Tuple[str, str]]:
    raw = os.environ.get(env_var)
    if not raw:
        return []
    return [(provider, k.strip()) for k in raw.split(",") if k.strip()]


def main() -> int:
    args = parse_args()

    # Pool keys across BOTH providers -- see the 2026-08-16 note above
    # PROVIDERS: Groq alone (7 keys x 100K TPD = 700K TPD) can't cover a
    # full N=1,226 dual-order run (~4.17M tokens) in one day. Cerebras
    # free tier (1M TPD/key) closes that gap. Either provider alone still
    # works with just its own env var set.
    entries: List[Tuple[str, str]] = []
    entries += load_provider_keys("GROQ_API_KEYS", "groq")
    entries += load_provider_keys("GROQ_API_KEY", "groq")
    entries += load_provider_keys("CEREBRAS_API_KEYS", "cerebras")
    entries += load_provider_keys("CEREBRAS_API_KEY", "cerebras")

    if not entries:
        print(
            "ERROR: set GROQ_API_KEYS and/or CEREBRAS_API_KEYS "
            "(comma-separated keys), or the singular *_API_KEY form for one key.",
            file=sys.stderr,
        )
        return 2

    key_pool = KeyPool(entries)
    counts = key_pool.provider_counts()
    print(
        f"judge.py: using {len(key_pool)} key(s) in rotation "
        f"({', '.join(f'{p}={n}' for p, n in sorted(counts.items()))}).",
        flush=True,
    )

    model_override = args.model != DEFAULT_MODEL

    if args.qps <= 0:
        print("ERROR: --qps must be greater than zero.", file=sys.stderr)
        return 2

    if args.max_retries < 0:
        print("ERROR: --max-retries cannot be negative.", file=sys.stderr)
        return 2

    prompt_template = load_prompt()

    heldout_rows = load_jsonl(args.heldout)
    base_rows = load_jsonl(args.base)
    ours_rows = load_jsonl(args.ours)

    heldout = index_by_id(heldout_rows, args.heldout)
    base = index_by_id(base_rows, args.base)
    ours = index_by_id(ours_rows, args.ours)

    validate_id_sets(heldout, base, ours)

    ids = list(heldout.keys())

    session = requests.Session()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    previous_call_time: float | None = None
    total_score_base = 0.0
    total_score_ours = 0.0
    completed = 0
    wins_base = 0
    wins_ours = 0

    # --resume: keep every row from a prior run that scored cleanly (no
    # "error" key) and only re-process the ids that are missing or
    # previously errored. Without this, a rate-limit storm that killed a
    # 7-hour run means re-spending the same budget on the same items.
    keep_rows: List[Dict[str, Any]] = []
    done_ids: Set[str] = set()

    if args.resume and Path(args.out).exists():
        existing_rows, _dropped = load_scores_lenient(args.out)
        for row in existing_rows:
            if "error" not in row and row.get("id") in heldout:
                keep_rows.append(row)
                done_ids.add(row["id"])

        print(
            f"judge.py: --resume found {len(keep_rows)} already-scored id(s) "
            f"in {args.out}; {len(ids) - len(keep_rows)} remaining.",
            flush=True,
        )

    ids_to_process = [i for i in ids if i not in done_ids]

    with open(args.out, "w", encoding="utf-8") as output_handle:
        for row in keep_rows:
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            completed += 1
            total_score_base += float(row["score_base"])
            total_score_ours += float(row["score_ours"])

            if row.get("choice_final") == "base":
                wins_base += 1
            elif row.get("choice_final") == "ours":
                wins_ours += 1

        output_handle.flush()

        for row_id in ids_to_process:
            master = heldout[row_id]
            base_row = base[row_id]
            ours_row = ours[row_id]

            # NOTE 2026-08-10: heldout rows use "instruction", not "question"
            # (verified against eval/heldout/test.jsonl schema). The old
            # "question" lookup silently fell back to "" for every row,
            # which would have sent the judge blank questions.
            question = str(master.get("instruction", ""))
            context = str(master.get("input", ""))

            base_answer = base_row.get("answer")
            ours_answer = ours_row.get("answer")

            if base_answer is None:
                base_answer = ""
            if ours_answer is None:
                ours_answer = ""

            try:
                # Order 1: A = base, B = ours.
                user_message_1 = build_user_message(
                    prompt_template,
                    question,
                    context,
                    str(base_answer),
                    str(ours_answer),
                )

                if previous_call_time is not None:
                    elapsed = time.monotonic() - previous_call_time
                    wait = (1.0 / args.qps) - elapsed
                    if wait > 0:
                        time.sleep(wait)

                first = request_judge(
                    session=session,
                    key_pool=key_pool,
                    model=args.model,
                    temperature=args.temperature,
                    seed=args.seed,
                    user_message=user_message_1,
                    max_retries=args.max_retries,
                    sleep_on_429=args.sleep_on_429,
                    model_override=model_override,
                )
                previous_call_time = time.monotonic()

                second = None

                # Order 2: A = ours, B = base.
                if args.dual_order:
                    user_message_2 = build_user_message(
                        prompt_template,
                        question,
                        context,
                        str(ours_answer),
                        str(base_answer),
                    )

                    elapsed = time.monotonic() - previous_call_time
                    wait = (1.0 / args.qps) - elapsed
                    if wait > 0:
                        time.sleep(wait)

                    second = request_judge(
                        session=session,
                        key_pool=key_pool,
                        model=args.model,
                        temperature=args.temperature,
                        seed=args.seed,
                        user_message=user_message_2,
                        max_retries=args.max_retries,
                        sleep_on_429=args.sleep_on_429,
                        model_override=model_override,
                    )
                    previous_call_time = time.monotonic()

                (
                    score_base,
                    score_ours,
                    choice_o1,
                    choice_o2,
                    choice_final,
                ) = aggregate_orders(first, second)

                result = {
                    "id": row_id,
                    "lang": master.get("lang", ""),
                    "score_base": score_base,
                    "score_ours": score_ours,
                    "choice_o1": choice_o1,
                    "choice_o2": choice_o2,
                    "choice_final": choice_final,
                }

                output_handle.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_handle.flush()

                completed += 1
                total_score_base += score_base
                total_score_ours += score_ours

                if choice_final == "base":
                    wins_base += 1
                elif choice_final == "ours":
                    wins_ours += 1

                avg_base = total_score_base / completed
                avg_ours = total_score_ours / completed
                decisive = wins_base + wins_ours
                win_rate = wins_ours / decisive if decisive else 0.0

                print(
                    f"items={completed}/{len(ids)} "
                    f"avg_score_base={avg_base:.3f} "
                    f"avg_score_ours={avg_ours:.3f} "
                    f"current_win_rate={win_rate:.3f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"ERROR id={row_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

                error_result = {
                    "id": row_id,
                    "error": str(exc),
                }

                # T03 spec step 9: unparseable judge output must also
                # carry the raw response text for debugging.
                raw = getattr(exc, "raw", None)
                if raw is not None:
                    error_result["raw"] = raw

                output_handle.write(
                    json.dumps(error_result, ensure_ascii=False) + "\n"
                )
                output_handle.flush()

    print(
        f"done items={completed}/{len(ids)} "
        f"avg_score_base="
        f"{(total_score_base / completed) if completed else 0.0:.3f} "
        f"avg_score_ours="
        f"{(total_score_ours / completed) if completed else 0.0:.3f} "
        f"base_wins={wins_base} "
        f"ours_wins={wins_ours}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
