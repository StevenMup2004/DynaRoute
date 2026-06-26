import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ============================================================
# CONFIG
# ============================================================

SMALL_PATH = Path(
    "routing_dataset-1_7b_dedup.json"
)

LARGE_PATH = Path(
    "routing_dataset-8b_dedup.json"
)

PRIORITY_SOURCES = {
    "barred",
    "dynabench",
    "plan",
}

HARD_RATIO = 3
EASY_RATIO = 4

OUTPUT_PATH = "train_router_3_4_balanced.json"

# ============================================================
# LOAD
# ============================================================

with open(SMALL_PATH, "r", encoding="utf-8") as f:
    small_data = json.load(f)

with open(LARGE_PATH, "r", encoding="utf-8") as f:
    large_data = json.load(f)

assert len(small_data) == len(large_data)

# ============================================================
# BUILD DATA
# ============================================================

hard = []

priority_easy = []

cc_other = []
cw_other = []
ww_other = []

raw_stats = defaultdict(int)
raw_source_stats = defaultdict(int)

for idx, (s, l) in enumerate(zip(small_data, large_data)):

    gt = s["label"]

    small_pred = s["prediction"]
    large_pred = l["prediction"]

    small_ok = small_pred == gt
    large_ok = large_pred == gt

    router_label = int(
        (not small_ok)
        and
        large_ok
    )

    sample = {
        "base_id": f"{s['source']}_{idx}",
        "source": s["source"],
        "policy": s["policy"],
        "transcript": s["transcript"],
        "harmfulness": int(gt == "FAIL"),
        "label": router_label,
    }

    source = s["source"].lower()

    raw_source_stats[source] += 1

    # =====================================================
    # HARD
    # =====================================================

    if router_label == 1:

        sample["case"] = "wc"

        hard.append(sample)

        raw_stats["wc"] += 1
        continue

    # =====================================================
    # EASY TYPE
    # =====================================================

    if small_ok and large_ok:

        case = "cc"

    elif small_ok and not large_ok:

        case = "cw"

    else:

        case = "ww"

    sample["case"] = case

    raw_stats[case] += 1

    if source in PRIORITY_SOURCES:

        priority_easy.append(sample)

    else:

        if case == "cc":
            cc_other.append(sample)

        elif case == "cw":
            cw_other.append(sample)

        else:
            ww_other.append(sample)

# ============================================================
# REPORT RAW
# ============================================================

print("=" * 80)
print("RAW DISTRIBUTION")
print("=" * 80)

for k in ["cc", "cw", "ww", "wc"]:
    print(f"{k:5s}: {raw_stats[k]}")

total_raw = sum(raw_stats.values())

print(f"\nTotal: {total_raw}")
print(f"Hard ratio: {raw_stats['wc']/total_raw:.4f}")

# ============================================================
# TARGET
# ============================================================

hard_count = len(hard)

target_easy = int(
    hard_count * EASY_RATIO / HARD_RATIO
)

print("\n" + "=" * 80)
print("TARGET")
print("=" * 80)

print(f"Hard count : {hard_count}")
print(f"Target easy: {target_easy}")

# ============================================================
# TAKE ALL PRIORITY EASY
# ============================================================

selected_easy = list(priority_easy)

print("\n" + "=" * 80)
print("PRIORITY EASY")
print("=" * 80)

priority_case_stats = defaultdict(int)

for x in priority_easy:
    priority_case_stats[x["case"]] += 1

for k, v in sorted(priority_case_stats.items()):
    print(f"{k:5s}: {v}")

print(f"\nPriority easy total: {len(priority_easy)}")

# ============================================================
# NEED MORE?
# ============================================================

need = target_easy - len(selected_easy)

print(f"Need additional easy: {max(0, need)}")

if need > 0:

    # target per class
    target_cc = need // 3
    target_cw = need // 3
    target_ww = need - target_cc - target_cw

    print("\nAdditional easy target:")
    print(f"CC: {target_cc}")
    print(f"CW: {target_cw}")
    print(f"WW: {target_ww}")

    add_cc = random.sample(
        cc_other,
        min(target_cc, len(cc_other))
    )

    add_cw = random.sample(
        cw_other,
        min(target_cw, len(cw_other))
    )

    add_ww = random.sample(
        ww_other,
        min(target_ww, len(ww_other))
    )

    selected_easy.extend(add_cc)
    selected_easy.extend(add_cw)
    selected_easy.extend(add_ww)

    # nếu vẫn thiếu do một nhóm không đủ
    remain = target_easy - len(selected_easy)

    if remain > 0:

        leftovers = []

        used = {
            id(x)
            for x in (
                add_cc
                + add_cw
                + add_ww
            )
        }

        for pool in [cc_other, cw_other, ww_other]:

            for item in pool:

                if id(item) not in used:
                    leftovers.append(item)

        if len(leftovers) > 0:

            selected_easy.extend(
                random.sample(
                    leftovers,
                    min(remain, len(leftovers))
                )
            )

# ============================================================
# TRIM IF TOO MANY
# ============================================================

if len(selected_easy) > target_easy:

    priority_part = list(priority_easy)

    remain_quota = target_easy - len(priority_part)

    if remain_quota < 0:

        selected_easy = random.sample(
            priority_part,
            target_easy
        )

    else:

        non_priority = [
            x
            for x in selected_easy
            if x not in priority_part
        ]

        selected_easy = (
            priority_part
            + random.sample(
                non_priority,
                min(remain_quota, len(non_priority))
            )
        )

# ============================================================
# FINAL
# ============================================================

final_dataset = hard + selected_easy

random.shuffle(final_dataset)
# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)

hard_final = sum(
    x["label"]
    for x in final_dataset
)

easy_final = len(final_dataset) - hard_final

print(f"Hard : {hard_final}")
print(f"Easy : {easy_final}")
print(f"Total: {len(final_dataset)}")

print(
    f"Hard ratio: "
    f"{hard_final/len(final_dataset):.4f}"
)

# ============================================================
# CASE DISTRIBUTION
# ============================================================

case_stats = defaultdict(int)

for x in final_dataset:
    case_stats[x["case"]] += 1

print("\nCase distribution")

for k in ["cc", "cw", "ww", "wc"]:
    print(f"{k:5s}: {case_stats[k]}")

print("\nCase ratios")

for k in ["cc", "cw", "ww", "wc"]:
    print(
        f"{k:5s}: "
        f"{case_stats[k]/len(final_dataset):.4f}"
    )

# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

source_stats = defaultdict(int)

for x in final_dataset:
    source_stats[x["source"]] += 1

print("\nSource distribution")

for k, v in sorted(source_stats.items()):
    print(f"{k:20s}: {v}")

# ============================================================
# PRIORITY EASY COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("PRIORITY EASY COVERAGE")
print("=" * 80)

priority_ok = True

for src in sorted(PRIORITY_SOURCES):

    raw_cnt = sum(
        1
        for x in priority_easy
        if x["source"].lower() == src
    )

    selected_cnt = sum(
        1
        for x in selected_easy
        if x["source"].lower() == src
    )

    status = "OK" if raw_cnt == selected_cnt else "MISSING"

    if raw_cnt != selected_cnt:
        priority_ok = False

    print(
        f"{src:12s}: "
        f"raw={raw_cnt:<6d} "
        f"selected={selected_cnt:<6d} "
        f"[{status}]"
    )

print()

if priority_ok:
    print("✓ All priority easy samples were preserved.")
else:
    print("✗ Some priority easy samples were dropped.")

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

expected_hard = len(hard)
actual_hard = hard_final

expected_easy = int(
    len(hard) * EASY_RATIO / HARD_RATIO
)

actual_easy = easy_final

print(f"Expected hard : {expected_hard}")
print(f"Actual hard   : {actual_hard}")

print(f"Expected easy : {expected_easy}")
print(f"Actual easy   : {actual_easy}")

print()

if expected_hard == actual_hard:
    print("✓ Hard count correct")
else:
    print("✗ Hard count mismatch")

if expected_easy == actual_easy:
    print("✓ Easy count correct")
else:
    print("✗ Easy count mismatch")

print("\nValidation complete.")

# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        final_dataset,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved:", OUTPUT_PATH)