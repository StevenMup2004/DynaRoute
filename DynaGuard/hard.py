import json
from tqdm import tqdm

FILE_1B = "log/tomg-group-umd/DynaGuard-1.7B/1781808254422808341/outputs.jsonl"
FILE_8B = "log/tomg-group-umd/DynaGuard-8B/1781808358632066939/outputs.jsonl"

def get_pred(output):
    output = output.upper()

    if "FAIL" in output:
        return "FAIL"
    if "PASS" in output:
        return "PASS"

    return None

hard_samples = []

with open(FILE_8B) as f8, open(FILE_1B) as f1:
    for idx, (l8, l1) in enumerate(zip(f8, f1)):
        d8 = json.loads(l8)
        d1 = json.loads(l1)

        gt = d8["metadata"]["label"]

        pred8 = get_pred(d8["output"])
        pred1 = get_pred(d1["output"])

        correct8 = pred8 == gt
        correct1 = pred1 == gt

        if correct8 and not correct1:
            hard_samples.append({
                "idx": idx,
                "label": gt,
                "pred_8b": pred8,
                "pred_1b": pred1,
                "sample": d8["metadata"]
            })

print("Hard samples:", len(hard_samples))

with open("hard_samples_8b_win.json", "w") as f:
    json.dump(hard_samples, f, indent=2, ensure_ascii=False)

print("Saved hard_samples_8b_win.json")




# import json

# FILE_8B = "routing_dataset-8b.json"
# FILE_1B = "routing_dataset-1_7b.json"

# with open(FILE_8B, "r", encoding="utf-8") as f:
#     data8 = json.load(f)

# with open(FILE_1B, "r", encoding="utf-8") as f:
#     data1 = json.load(f)

# if len(data8) != len(data1):
#     raise ValueError(
#         f"Dataset size mismatch: {len(data8)} vs {len(data1)}"
#     )

# hard_samples = []
# easy_samples = []

# both_correct = 0
# both_wrong = 0
# hard_count = 0
# small_win = 0

# for idx, (d8, d1) in enumerate(zip(data8, data1)):
#     gt = d8["label"]

#     pred8 = d8["prediction"]
#     pred1 = d1["prediction"]

#     correct8 = pred8 == gt
#     correct1 = pred1 == gt

#     sample = {
#         "idx": idx,
#         "policy": d8["policy"],
#         "transcript": d8["transcript"],
#         "label": gt,
#         "source": d8.get("source", ""),
#         "prediction_8b": pred8,
#         "prediction_1_7b": pred1,
#     }

#     # Hard sample: 8B đúng, 1.7B sai
#     if correct8 and not correct1:
#         sample["route_label"] = 1
#         hard_samples.append(sample)
#         hard_count += 1

#     # Easy sample: cả hai đều đúng
#     elif correct8 and correct1:
#         sample["route_label"] = 0
#         easy_samples.append(sample)
#         both_correct += 1

#     elif not correct8 and correct1:
#         small_win += 1

#     else:
#         both_wrong += 1

# print(f"Total samples : {len(data8)}")
# print(f"Both correct  : {both_correct}")
# print(f"Hard samples  : {hard_count}")
# print(f"1.7B wins     : {small_win}")
# print(f"Both wrong    : {both_wrong}")

# with open("hard_samples_8b_win.json", "w", encoding="utf-8") as f:
#     json.dump(hard_samples, f, indent=2, ensure_ascii=False)

# with open("easy_samples.json", "w", encoding="utf-8") as f:
#     json.dump(easy_samples, f, indent=2, ensure_ascii=False)

# print(f"Saved {len(hard_samples)} hard samples")
# print(f"Saved {len(easy_samples)} easy samples")