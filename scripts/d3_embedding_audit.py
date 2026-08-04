import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("dataset/kcc_train.jsonl", encoding="utf-8") as f:
    train = [json.loads(x)["instruction"] for x in f]

with open("dataset/kcc_heldout.jsonl", encoding="utf-8") as f:
    heldout = [json.loads(x)["instruction"] for x in f]

train_emb = model.encode(train, convert_to_tensor=True)

hits = 0

for text in heldout[:100]:
    emb = model.encode(text, convert_to_tensor=True)
    score = util.cos_sim(emb, train_emb).max().item()

    if score > 0.90:
        hits += 1
        print(f"{score:.3f} -> {text}")

print("\nPotential semantic overlaps:", hits)
