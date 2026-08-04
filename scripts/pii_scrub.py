import json
import re

INPUT_FILE = "dataset/kcc_cleaned.jsonl"
OUTPUT_FILE = "dataset/kcc_pii_scrubbed.jsonl"

PHONE_RE = re.compile(r"\b\d{10}\b")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def scrub(text):
    text = PHONE_RE.sub("[PHONE]", text)
    text = AADHAAR_RE.sub("[AADHAAR]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return text

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        row = json.loads(line)

        row["instruction"] = scrub(row.get("instruction", ""))
        row["input"] = scrub(row.get("input", ""))
        row["output"] = scrub(row.get("output", ""))

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

print("=" * 40)
print(f"Processed records : {count}")
print(f"Saved file        : {OUTPUT_FILE}")
print("=" * 40)
