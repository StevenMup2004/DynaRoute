from datasets import load_dataset
import json

hf_ds = load_dataset(
    "montehoover/DynaBench",
    split="test"
)

hf_records = [dict(x) for x in hf_ds]
for x in hf_records:
    x["source"] = "montehoover/DynaBench"

with open(
    "dynabench_latest.json",
    "r",
    encoding="utf-8"
) as f:
    augment_records = json.load(f)
for x in augment_records:
    x["source"] = "dynabench_latest"

merged = augment_records + hf_records

seen = set()
dedup = []

for item in merged:
    item_for_key = {k: v for k, v in item.items() if k != "source"}
    key = json.dumps(
        item_for_key,
        sort_keys=True,
        ensure_ascii=False
    )

    if key not in seen:
        seen.add(key)
        dedup.append(item)

print(f"Before: {len(merged)}")
print(f"After dedup: {len(dedup)}")

with open(
    "dynabench_merged_dedup.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        dedup,
        f,
        ensure_ascii=False,
        indent=2
    )