# import json

# PROGUARD_FILE = "proguard_text_multiturn.json"
# BARRED_FILE = "barred_augmented_with_pass-v2.json"
# DYNABENCH_FILE = "dynabench_augmented-30-70_pass-v2.json"

# OUTPUT_FILE = "proguard_merged.json"


# def normalize(entry, source):
#     return {
#         "policy": entry["policy"],
#         "transcript": entry["transcript"],
#         "label": entry["label"],
#         "source": source,
#     }


# # Load files
# with open(PROGUARD_FILE, "r", encoding="utf-8") as f:
#     proguard = json.load(f)

# with open(BARRED_FILE, "r", encoding="utf-8") as f:
#     barred = json.load(f)

# with open(DYNABENCH_FILE, "r", encoding="utf-8") as f:
#     dynabench = json.load(f)

# # Keep original ProGuard order
# merged = [normalize(x, "proguard") for x in proguard]

# # Append to the end
# merged.extend(normalize(x, "barred") for x in barred)
# merged.extend(normalize(x, "dynabench") for x in dynabench)

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(merged, f, ensure_ascii=False, indent=2)

# print(f"Saved {len(merged)} samples -> {OUTPUT_FILE}")
# print(f"ProGuard  : {len(proguard)}")
# print(f"Barred    : {len(barred)}")
# print(f"Dynabench : {len(dynabench)}")

















# import json

# PROGUARD_FILE = "proguard_text_multiturn.json"
# BARRED_FILE = "barred_augmented_with_pass-v2.json"
# DYNABENCH_FILE = "dynabench_augmented-30-70_pass-v2.json"

# PROGUARD_OUTPUT = "outputs-1b-proguard.jsonl"

# OUTPUT_FILE = "routing_dataset.json"


# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def extract_prediction(output_text):
#     """
#     Examples:
#         PASS
#         PASS\n</answer>
#         FAIL
#         FAIL\n</answer>
#     """
#     text = output_text.strip().upper()

#     if "PASS" in text:
#         return "PASS"

#     if "FAIL" in text:
#         return "FAIL"

#     return ""


# def normalize(entry, source):
#     return {
#         "policy": entry["policy"],
#         "transcript": entry["transcript"],
#         "label": entry["label"],
#         "source": source,
#         "prediction": ""
#     }


# # =====================================================
# # 1. Load datasets
# # =====================================================

# proguard = load_json(PROGUARD_FILE)
# barred = load_json(BARRED_FILE)
# dynabench = load_json(DYNABENCH_FILE)

# # =====================================================
# # 2. Build merged dataset
# #    Keep original ProGuard order
# # =====================================================

# merged = []

# for sample in proguard:
#     merged.append(normalize(sample, "proguard"))

# for sample in barred:
#     merged.append(normalize(sample, "barred"))

# for sample in dynabench:
#     merged.append(normalize(sample, "dynabench"))

# print(f"ProGuard   : {len(proguard)}")
# print(f"Barred     : {len(barred)}")
# print(f"Dynabench  : {len(dynabench)}")
# print(f"Total      : {len(merged)}")

# # =====================================================
# # 3. Build lookup for ProGuard samples
# # =====================================================

# lookup = {}

# for idx, sample in enumerate(merged):
#     if sample["source"] != "proguard":
#         continue

#     key = (
#         sample["policy"].strip(),
#         sample["transcript"].strip()
#     )

#     lookup[key] = idx

# # =====================================================
# # 4. Inject predictions from outputs-1b-proguard.jsonl
# # =====================================================

# matched = 0
# not_found = 0

# with open(PROGUARD_OUTPUT, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()

#         if not line:
#             continue

#         row = json.loads(line)

#         meta = row["metadata"]

#         key = (
#             meta["policy"].strip(),
#             meta["transcript"].strip()
#         )

#         pred = extract_prediction(row["output"])

#         if key in lookup:
#             merged[lookup[key]]["prediction"] = pred
#             matched += 1
#         else:
#             not_found += 1

# print(f"Matched predictions : {matched}")
# print(f"Not found           : {not_found}")

# # =====================================================
# # 5. Save
# # =====================================================

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(merged, f, ensure_ascii=False, indent=2)

# print(f"Saved to {OUTPUT_FILE}")


















import json

# PROGUARD_FILE = "VHDang/Guardrail/fix-ver/DynaGuard/proguard_text_multiturn.json"
# BARRED_FILE = "VHDang/Guardrail/fix-ver/DynaGuard/barred_augmented_with_pass-v2.json"
# DYNABENCH_FILE = "VHDang/Guardrail/fix-ver/DynaGuard/dynabench_augmented-30-70_pass-v2.json"

# # Guard outputs
# PROGUARD_OUTPUT = "VHDang/Guardrail/fix-ver/DynaGuard/outputs-1b-proguard.jsonl"
# BARRED_OUTPUT = "VHDang/Guardrail/fix-ver/DynaGuard/log/tomg-group-umd/DynaGuard-1.7B/1781776479741380211-barred/outputs.jsonl"
# DYNABENCH_OUTPUT = "VHDang/Guardrail/fix-ver/DynaGuard/log/tomg-group-umd/DynaGuard-1.7B/1781776589047635797-dynabench/outputs.jsonl"

# OUTPUT_FILE = "VHDang/Guardrail/fix-ver/DynaGuard/routing_dataset.json"


import json

PROGUARD_FILE = "proguard_text_multiturn.json"
BARRED_FILE = "barred_augmented_with_pass-v2.json"
DYNABENCH_FILE = "dynabench_augmented-30-70_pass-v2.json"

# Guard outputs
# PROGUARD_OUTPUT = "outputs-1b-proguard.jsonl"
# BARRED_OUTPUT = "log/tomg-group-umd/DynaGuard-1.7B/1781776479741380211-barred/outputs.jsonl"
# DYNABENCH_OUTPUT = "log/tomg-group-umd/DynaGuard-1.7B/1781776589047635797-dynabench/outputs.jsonl"


PROGUARD_OUTPUT = "log/tomg-group-umd/DynaGuard-8B/1781769807447064958-proguard-full/outputs.jsonl"
BARRED_OUTPUT = "log/tomg-group-umd/DynaGuard-8B/1781799387892416662-barred/outputs.jsonl"
DYNABENCH_OUTPUT = "log/tomg-group-umd/DynaGuard-8B/1781800629059255154-dyna/outputs.jsonl"
OUTPUT_FILE = "routing_dataset-8b.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_prediction(output_text):
    text = output_text.strip().upper()

    if "PASS" in text:
        return "PASS"
    if "FAIL" in text:
        return "FAIL"

    return ""


def normalize(entry, source):
    return {
        "policy": entry["policy"],
        "transcript": entry["transcript"],
        "label": entry["label"],
        "source": source,
        "prediction": ""
    }


def inject_predictions(dataset, output_file, source_name):
    """
    Update predictions for a given source.
    Skip silently if output file does not exist yet.
    """

    import os

    if not os.path.exists(output_file):
        print(f"[SKIP] {output_file} not found")
        return

    lookup = {}

    for idx, sample in enumerate(dataset):
        if sample["source"] != source_name:
            continue

        key = (
            sample["policy"].strip(),
            sample["transcript"].strip()
        )

        lookup[key] = idx

    matched = 0
    not_found = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            meta = row["metadata"]

            key = (
                meta["policy"].strip(),
                meta["transcript"].strip()
            )

            pred = extract_prediction(row["output"])

            if key in lookup:
                dataset[lookup[key]]["prediction"] = pred
                matched += 1
            else:
                not_found += 1

    print(
        f"[{source_name}] matched={matched}, not_found={not_found}"
    )


# =====================================================
# Build merged dataset
# =====================================================

proguard = load_json(PROGUARD_FILE)
barred = load_json(BARRED_FILE)
dynabench = load_json(DYNABENCH_FILE)

merged = []

# Keep original order
merged.extend(normalize(x, "proguard") for x in proguard)

# Append at the end
merged.extend(normalize(x, "barred") for x in barred)
merged.extend(normalize(x, "dynabench") for x in dynabench)

print(f"ProGuard  : {len(proguard)}")
print(f"Barred    : {len(barred)}")
print(f"Dynabench : {len(dynabench)}")
print(f"Total     : {len(merged)}")

# =====================================================
# Inject predictions
# =====================================================

inject_predictions(
    merged,
    PROGUARD_OUTPUT,
    "proguard"
)

inject_predictions(
    merged,
    BARRED_OUTPUT,
    "barred"
)

inject_predictions(
    merged,
    DYNABENCH_OUTPUT,
    "dynabench"
)

# =====================================================
# Save
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        merged,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nSaved to: {OUTPUT_FILE}")