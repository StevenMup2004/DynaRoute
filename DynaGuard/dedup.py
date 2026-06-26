import json

FILE_1B = "routing_dataset-1_7b.json"
FILE_8B = "routing_dataset-8b.json"

OUT_1B = "routing_dataset-1_7b_dedup.json"
OUT_8B = "routing_dataset-8b_dedup.json"

with open(FILE_1B, "r", encoding="utf-8") as f:
    data_1b = json.load(f)

with open(FILE_8B, "r", encoding="utf-8") as f:
    data_8b = json.load(f)

assert len(data_1b) == len(data_8b)

seen = set()

new_1b = []
new_8b = []

removed = 0

# Router statistics
hard = 0
both_correct = 0
oneb_only = 0
both_wrong = 0

for s1, s2 in zip(data_1b, data_8b):

    assert s1["policy"] == s2["policy"]
    assert s1["transcript"] == s2["transcript"]
    assert s1["label"] == s2["label"]
    assert s1["source"] == s2["source"]

    key = (
        s1["policy"].strip(),
        s1["transcript"].strip()
    )

    if key in seen:
        removed += 1
        continue

    seen.add(key)

    new_1b.append(s1)
    new_8b.append(s2)

    label = s1["label"]

    c1 = (s1["prediction"] == label)
    c2 = (s2["prediction"] == label)

    # hard sample = 1.7B sai nhưng 8B đúng
    if (not c1) and c2:
        hard += 1

    elif c1 and c2:
        both_correct += 1

    elif c1 and (not c2):
        oneb_only += 1

    else:
        both_wrong += 1

with open(OUT_1B, "w", encoding="utf-8") as f:
    json.dump(new_1b, f, ensure_ascii=False)

with open(OUT_8B, "w", encoding="utf-8") as f:
    json.dump(new_8b, f, ensure_ascii=False)

print("\n===== Dedup =====")
print(f"Original size : {len(data_1b)}")
print(f"Dedup size    : {len(new_1b)}")
print(f"Removed       : {removed}")

print("\n===== Router Stats =====")
print(f"Total         : {len(new_1b)}")
print(f"Hard          : {hard}")
print(f"Both correct  : {both_correct}")
print(f"1.7B only     : {oneb_only}")
print(f"Both wrong    : {both_wrong}")

print("\n===== Ratios =====")
print(f"Hard ratio         : {100*hard/len(new_1b):.2f}%")
print(f"Both correct ratio : {100*both_correct/len(new_1b):.2f}%")
print(f"1.7B only ratio    : {100*oneb_only/len(new_1b):.2f}%")
print(f"Both wrong ratio   : {100*both_wrong/len(new_1b):.2f}%")

print("\nSaved:")
print(OUT_1B)
print(OUT_8B)
