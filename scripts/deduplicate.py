import json

INPUT_FILE = "dataset/kcc_pii_scrubbed.jsonl"
OUTPUT_FILE = "dataset/kcc_deduplicated.jsonl"

seen = set()
total = 0
duplicates = 0
kept = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        total += 1
        row = json.loads(line)

        key = (
            row.get("instruction", "").strip().lower(),
            row.get("output", "").strip().lower()
        )

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1

print("=" * 40)
print(f"Total records      : {total}")
print(f"Duplicates removed : {duplicates}")
print(f"Remaining records  : {kept}")
print(f"Saved file         : {OUTPUT_FILE}")
print("=" * 40)
