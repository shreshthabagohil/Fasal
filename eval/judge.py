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
DEFAULT_MODEL = "llama-3.3-70b-versatile"
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
    return parser.parse_args()


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
d

class KeyPool:
    """Round-robins across multiple Groq API keys so one key hitting its
    rate/daily limit doesn't stop the whole eval run. When a key gets a
    429, it is put in cooldown and skipped until every other key has
    also been tried once (then cooldown resets so a key that recovered
    can be reused)."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("KeyPool needs at least one API key")
        self._keys = keys
        self._index = 0
        self._cooldown: Set[int] = set()

    def get(self) -> str:
        if len(self._cooldown) >= len(self._keys):
            self._cooldown.clear()
        for _ in range(len(self._keys)):
            idx = self._index
            key = self._keys[idx]
            self._index = (self._index + 1) % len(self._keys)
            if idx not in self._cooldown:
                return key
        # Should not happen given the clear() above, but stay safe.
        return self._keys[0]

    def mark_limited(self, key: str) -> None:
        if key in self._keys:
            self._cooldown.add(self._keys.index(key))

    def __len__(self) -> int:
        return len(self._keys)


def request_judge(
    session: requests.Session,
    key_pool: KeyPool,
    model: str,
    temperature: float,
    seed: int,
    user_message: str,
    max_retries: int,
    sleep_on_429: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ],
    }

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        api_key = key_pool.get()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = session.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            status = response.status_code

            if status == 429:
                # This key is rate/daily limited -- try a different key
                # on the very next attempt instead of just sleeping on
                # the same exhausted key.
                key_pool.mark_limited(api_key)

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


def main() -> int:
    args = parse_args()

    raw_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
    if not raw_keys:
        print(
            "ERROR: set GROQ_API_KEYS (comma-separated, e.g. "
            "'gsk_key1,gsk_key2,gsk_key3') or, for a single key, GROQ_API_KEY.",
            file=sys.stderr,
        )
        return 2

    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        print("ERROR: GROQ_API_KEYS/GROQ_API_KEY is empty.", file=sys.stderr)
        return 2

    key_pool = KeyPool(keys)
    print(f"judge.py: using {len(key_pool)} Groq key(s) in rotation.", flush=True)

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

    with open(args.out, "w", encoding="utf-8") as output_handle:
        for row_id in ids:
            master = heldout[row_id]
            base_row = base[row_id]
            ours_row = ours[row_id]

            question = str(master.get("question", ""))
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
