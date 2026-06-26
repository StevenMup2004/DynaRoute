import json
from collections import defaultdict

SMALL_FILE = (
    "log/"
    "tomg-group-umd/DynaGuard-1.7B/"
    "1781840603425104253-valid/outputs.jsonl"
)

LARGE_FILE = (
    "log/"
    "tomg-group-umd/DynaGuard-8B/"
    "1781840866798715444-valid/outputs.jsonl"
)

OUTPUT_FILE = "router_valid.json"


def normalize_pred(text):
    text = text.strip()

    if text.startswith("PASS"):
        return "PASS"

    if text.startswith("FAIL"):
        return "FAIL"

    return text.split()[0]


small_rows = []
large_rows = []

with open(SMALL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        small_rows.append(json.loads(line))

with open(LARGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        large_rows.append(json.loads(line))

assert len(small_rows) == len(large_rows)

dataset = []

stats = defaultdict(int)

for s, l in zip(small_rows, large_rows):

    sm = s["metadata"]
    lg = l["metadata"]

    assert sm["base_id"] == lg["base_id"]

    gt = sm["label"]

    small_pred = normalize_pred(
        s["output"]
    )

    large_pred = normalize_pred(
        l["output"]
    )

    small_ok = (small_pred == gt)
    large_ok = (large_pred == gt)

    router_label = int(
        (not small_ok)
        and
        large_ok
    )

    if small_ok and large_ok:
        case = "cc"

    elif small_ok and not large_ok:
        case = "cw"

    elif not small_ok and large_ok:
        case = "wc"

    else:
        case = "ww"

    sample = {
        "base_id": sm["base_id"],
        "source": sm.get("source", "unknown"),
        "policy": sm["policy"],
        "transcript": sm["transcript"],
        "harmfulness": int(gt == "FAIL"),
        "label": router_label,
        "case": case,
    }

    dataset.append(sample)

    stats[case] += 1

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        dataset,
        f,
        ensure_ascii=False,
        indent=2
    )

print("=" * 60)
print("VALID DATASET")
print("=" * 60)

for k in ["cc", "cw", "ww", "wc"]:
    print(f"{k:5s}: {stats[k]}")

print()

hard = stats["wc"]

print(f"Hard: {hard}")
print(f"Total: {len(dataset)}")
print(f"Hard ratio: {hard/len(dataset):.4f}")

print()
print(f"Saved: {OUTPUT_FILE}")