from datasets import load_dataset
from collections import Counter

ds = load_dataset(
    "yushaohan/ProGuard-data",
    split="train",
)

category_counter = Counter()
turn_counter = Counter()

total_text = 0
multi_turn = 0

examples = []

for sample in ds:
    if sample.get("modality") != "text":
        continue

    total_text += 1

    category = sample.get("category", "UNKNOWN")
    messages = sample.get("message", [])

    num_messages = len(messages)

    category_counter[category] += 1
    turn_counter[num_messages] += 1

    if num_messages > 2:
        multi_turn += 1

        if len(examples) < 5:
            examples.append(sample)

print("=" * 80)
print(f"TEXT SAMPLES: {total_text}")
print(f"MULTI-TURN TEXT SAMPLES: {multi_turn}")

print("\nTURN DISTRIBUTION")
for n, cnt in sorted(turn_counter.items()):
    print(f"{n} messages: {cnt}")

print("\nCATEGORIES")
for cat, cnt in category_counter.most_common():
    print(f"{cat}: {cnt}")

print("\nUNIQUE CATEGORIES")
print(sorted(category_counter.keys()))

print("\nMULTI-TURN EXAMPLES")

for i, ex in enumerate(examples, 1):
    print("\n" + "=" * 80)
    print(f"Example {i}")
    print(f"Category: {ex['category']}")
    print(f"Messages: {len(ex['message'])}")

    for msg in ex["message"]:
        print(f"\n[{msg['role']}]")
        print(str(msg["content"])[:300])