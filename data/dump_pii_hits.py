import json
import re
import sys

sys.path.insert(0, "data")
from clean_kcc import PHONE_RE, PHONE_SPACED_RE, EMAIL_RE, _DIGIT_RUN_RE, contains_pii

n = 0
for line in open("dataset/kcc_instruct_v1.jsonl", encoding="utf-8"):
    r = json.loads(line)
    blob = r["instruction"] + " " + r["output"]
    if not contains_pii(blob):
        continue

    which = []
    if (m := EMAIL_RE.search(blob)):
        which.append(f"EMAIL:{m.group()!r}")
    if (m := PHONE_RE.search(blob)):
        which.append(f"PHONE:{m.group()!r}")
    if (m := PHONE_SPACED_RE.search(blob)):
        which.append(f"PHONE_SPACED:{m.group()!r}")
    for m in _DIGIT_RUN_RE.finditer(blob):
        if len(re.findall(r"\d{4}", m.group())) == 3:
            which.append(f"AADHAAR3GROUP:{m.group()!r}")

    print(" / ".join(which))
    print(repr(blob[:200]))
    print("---")
    n += 1
    if n >= 20:
        break

print(f"\n(showing first {n} hits)")
