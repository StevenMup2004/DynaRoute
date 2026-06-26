import json

INPUT_FILE = "beavertail_policies.json"
OUTPUT_FILE = "beavertail_dynaguard.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []

for sample in data:
    converted.append({
        "policy": sample["policies"],
        "transcript": sample["dialogue"],
        "label": sample["label"],
        "source": "beavertail",
        "prediction": None
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"Saved {len(converted)} samples to {OUTPUT_FILE}")