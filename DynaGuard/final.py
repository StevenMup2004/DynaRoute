import json
from tqdm import tqdm

# FILE_1B = "log/tomg-group-umd/DynaGuard-1.7B/1781808254422808341-beavertail/outputs.jsonl"
# FILE_8B = "log/tomg-group-umd/DynaGuard-8B/1781808358632066939-bevertail/outputs.jsonl"

FILE_1B = "log/tomg-group-umd/DynaGuard-1.7B/1781805070766507357-plan/outputs.jsonl"
FILE_8B = "log/tomg-group-umd/DynaGuard-8B/1781804913690600586-plan/outputs.jsonl"
ROUTER_1B = "routing_dataset-1_7b.json"
ROUTER_8B = "routing_dataset-8b.json"


def extract_prediction(output_text):
    text = output_text.strip().upper()

    if "PASS" in text:
        return "PASS"

    if "FAIL" in text:
        return "FAIL"

    return ""


def append_beavertail(router_file, output_file):
    print(f"\nProcessing {router_file}")

    with open(router_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_size = len(dataset)

    with open(output_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip():
                continue

            row = json.loads(line)

            meta = row["metadata"]

            dataset.append(
                {
                    "policy": meta["policy"],
                    "transcript": meta["transcript"],
                    "label": meta["label"],
                    "source": "plan",
                    "prediction": extract_prediction(
                        row["output"]
                    ),
                }
            )

    print(
        f"Added {len(dataset) - original_size} BeaverTails samples"
    )

    with open(router_file, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {router_file}")


append_beavertail(
    ROUTER_1B,
    FILE_1B,
)

append_beavertail(
    ROUTER_8B,
    FILE_8B,
)