import json
import unicodedata

INPUT_FILE = "dataset/kcc_instruct_v1.jsonl"
OUTPUT_FILE = "dataset/kcc_cleaned.jsonl"

def clean(text):
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    return text.strip()

count = 0
removed = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        row = json.loads(line)

        # Clean fields
        row["instruction"] = clean(row.get("instruction", ""))
        row["input"] = clean(row.get("input", ""))
        row["output"] = clean(row.get("output", ""))

        # Skip records with empty output
        if row["output"] == "":
            removed += 1
            continue

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

print("=" * 40)
print(f"Original records : {count + removed}")
print(f"Cleaned records  : {count}")
print(f"Removed records  : {removed}")
print(f"Saved file       : {OUTPUT_FILE}")
print("=" * 40)
