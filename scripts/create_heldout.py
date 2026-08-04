import json
import random
from collections import defaultdict
import sys

sys.path.insert(0, "data")
from clean_kcc import normalize
INPUT = "dataset/kcc_instruct_v1.jsonl"
TRAIN = "dataset/kcc_train.jsonl"
HELDOUT = "dataset/kcc_heldout.jsonl"

random.seed(42)

# Group all records by normalized instruction
groups = defaultdict(list)

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        key = key = normalize(row["instruction"])
        groups[key].append(row)

keys = list(groups.keys())
random.shuffle(keys)

split = int(len(keys) * 0.95)

train_keys = set(keys[:split])

train_rows = []
heldout_rows = []

for key, records in groups.items():
    if key in train_keys:
        train_rows.extend(records)
    else:
        heldout_rows.extend(records)

with open(TRAIN, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(HELDOUT, "w", encoding="utf-8") as f:
    for r in heldout_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Unique instructions:", len(keys))
print("Training records:", len(train_rows))
print("Held-out records:", len(heldout_rows))
