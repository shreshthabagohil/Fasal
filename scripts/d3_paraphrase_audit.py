import json
import random

with open("dataset/kcc_train.jsonl", encoding="utf-8") as f:
    train = [json.loads(x) for x in f]

with open("dataset/kcc_heldout.jsonl", encoding="utf-8") as f:
    heldout = [json.loads(x) for x in f]

random.seed(42)

samples = random.sample(heldout, 20)

for i, sample in enumerate(samples, 1):
    print("=" * 80)
    print(f"Held-out #{i}")
    print(sample["instruction"])
    print()

    words = set(sample["instruction"].lower().split())

    matches = []

    for row in train:
        overlap = len(words & set(row["instruction"].lower().split()))
        if overlap >= max(3, len(words)//2):
            matches.append(row["instruction"])

    print("Possible similar train examples:")
    for m in matches[:5]:
        print("-", m)
    print()
