# """Evaluate an LPG checkpoint on DynaBench with DynaGuard-style labels.

# Example:
#     python eval_lpg_dynabench.py \
#         --lpg_repo_path ../Latent_Policy_Guard \
#         --model_path Qwen/Qwen3-4B \
#         --ckpt_dir VHDang/Guardrail/fix-ver/Latent_Policy_Guard/training/outputs/lpg_qwen3_4b/Qwen3-4B/ep_3/lr_5e-05/seed_11/checkpoint-6453 \
#         --dataset_path tomg-group-umd/DynaBench \
#         --subset DynaBench \
#         --split test
# """

# import argparse
# import csv
# import json
# import logging
# import os
# import re
# import sys
# import time
# from collections.abc import Iterable
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import datasets
# from tqdm import tqdm


# DEFAULT_LPG_REPO_PATH = Path(__file__).resolve().parents[1] / "Latent_Policy_Guard"
# DEFAULT_CKPT_DIR = (
#     "VHDang/Guardrail/fix-ver/Latent_Policy_Guard/training/outputs/"
#     "lpg_qwen3_4b/Qwen3-4B/ep_3/lr_5e-05/seed_11/checkpoint-6453"
# )


# def configure_logging(log_level: Optional[str]) -> None:
#     if log_level is None:
#         return
#     logging.basicConfig(level=getattr(logging, log_level.upper()))


# def add_lpg_repo_to_path(lpg_repo_path: str) -> None:
#     repo_path = Path(lpg_repo_path).expanduser().resolve()
#     if not repo_path.exists():
#         raise FileNotFoundError(f"LPG repo path does not exist: {repo_path}")
#     if not (repo_path / "evaluation" / "models" / "latent_policy_guard.py").exists():
#         raise FileNotFoundError(
#             "Could not find evaluation/models/latent_policy_guard.py under "
#             f"{repo_path}"
#         )
#     eval_path = repo_path / "evaluation"
#     for path in (repo_path, eval_path):
#         if str(path) not in sys.path:
#             sys.path.insert(0, str(path))


# def load_lpg_model_class(lpg_repo_path: str):
#     add_lpg_repo_to_path(lpg_repo_path)
#     from models.latent_policy_guard import LatentPolicyGuardModel

#     return LatentPolicyGuardModel


# def load_dynabench_dataset(
#     dataset_path: str,
#     subset: Optional[str],
#     split: str,
# ):
#     if os.path.exists(dataset_path):
#         return datasets.load_dataset(
#             "json",
#             data_files={"eval": dataset_path},
#             split="eval",
#         )
#     if subset:
#         return datasets.load_dataset(dataset_path, subset, split=split)
#     return datasets.load_dataset(dataset_path, split=split)


# def filter_labels(dataset, label_col: str):
#     def is_valid_row(example):
#         label = example.get(label_col)
#         if label is None:
#             return False
#         label = str(label).strip()
#         return bool(label) and label.lower() != "null"

#     return dataset.filter(is_valid_row)


# def maybe_select_examples(dataset, num_examples: int, seed: int):
#     if num_examples <= 0 or num_examples >= len(dataset):
#         return dataset
#     return dataset.shuffle(seed=seed).select(range(num_examples))


# def parse_jsonish(value: Any) -> Dict[str, Any]:
#     if isinstance(value, dict):
#         return value
#     if value is None:
#         return {}
#     if not isinstance(value, str):
#         return {}
#     value = value.strip()
#     if not value:
#         return {}
#     try:
#         return json.loads(value)
#     except json.JSONDecodeError:
#         return {}


# def as_int(value: Any) -> Optional[int]:
#     if value is None:
#         return None
#     try:
#         return int(value)
#     except (TypeError, ValueError):
#         return None


# def first_int_from_sequence(value: Any) -> Optional[int]:
#     if value is None:
#         return None
#     if isinstance(value, str):
#         matches = re.findall(r"-?\d+", value)
#         return int(matches[0]) if matches else None
#     if isinstance(value, Iterable):
#         for item in value:
#             parsed = as_int(item)
#             if parsed is not None:
#                 return parsed
#     return as_int(value)


# def detect_policy_index_offset(policy_text: str) -> int:
#     for line in str(policy_text).splitlines():
#         match = re.match(r"^(\d+)[.:]\s+.*$", line)
#         if match:
#             return 1 if int(match.group(1)) == 1 else 0
#     return 0


# def normalize_policy_block_for_lpg(policy_text: str) -> str:
#     lines = str(policy_text).splitlines()
#     first_header = None
#     for line in lines:
#         match = re.match(r"^(\d+)[.:]\s+(.*)$", line)
#         if match:
#             first_header = int(match.group(1))
#             break

#     if first_header not in (0, 1):
#         return str(policy_text)

#     expected = first_header
#     offset = 1 if first_header == 1 else 0
#     for idx, line in enumerate(lines):
#         match = re.match(r"^(\d+)[.:]\s+(.*)$", line)
#         if not match:
#             continue
#         rule_number = int(match.group(1))
#         if rule_number != expected:
#             continue
#         lines[idx] = f"{rule_number - offset}: {match.group(2)}"
#         expected += 1
#     return "\n".join(lines)


# def get_ground_truth_rule_number(example: Dict[str, Any]) -> Optional[int]:
#     metadata = parse_jsonish(example.get("metadata"))
#     for key in ("rules_violated", "violated_rules", "policy_indices"):
#         parsed = first_int_from_sequence(metadata.get(key))
#         if parsed is not None:
#             return parsed
#     for key in ("rule_violated", "violated_rule", "policy_index"):
#         parsed = as_int(metadata.get(key))
#         if parsed is not None:
#             return parsed
#     return None


# def get_ground_truth_policy_index(
#     example: Dict[str, Any],
#     policy_col: str,
# ) -> Optional[int]:
#     rule_number = get_ground_truth_rule_number(example)
#     if rule_number is None:
#         return None
#     return max(rule_number - detect_policy_index_offset(example.get(policy_col, "")), 0)


# def get_predicted_policy_index(prediction: Dict[str, Any]) -> Optional[int]:
#     parsed = as_int(prediction.get("policy_index"))
#     if parsed is not None:
#         return parsed
#     parsed = first_int_from_sequence(prediction.get("policy_indices"))
#     if parsed is not None:
#         return parsed
#     return first_int_from_sequence(prediction.get("violated_policy_indices"))


# def prediction_to_label(prediction: Dict[str, Any]) -> str:
#     return "PASS" if bool(prediction.get("safe", True)) else "FAIL"


# def normalize_label(value: Any, pos_label: str, neg_label: str) -> str:
#     if value is None:
#         return "null"
#     label = str(value).strip()
#     if label == pos_label:
#         return "FAIL"
#     if label == neg_label:
#         return "PASS"
#     return "null"


# def get_dataset_labels(dataset, label_col: str, pos_label: str, neg_label: str) -> List[str]:
#     return [
#         normalize_label(example.get(label_col), pos_label, neg_label)
#         for example in dataset
#     ]


# def get_stats(ground_truth_labels: List[str], predicted_labels: List[str]) -> Dict[str, Any]:
#     if len(ground_truth_labels) != len(predicted_labels):
#         raise ValueError(
#             "Ground truth labels and predicted labels must have the same length. "
#             f"Got {len(ground_truth_labels)} and {len(predicted_labels)}."
#         )

#     false_negatives = []
#     false_positives = []
#     true_positives = []
#     true_negatives = []
#     nulls = []

#     for idx, (ground_truth_label, predicted_label) in enumerate(
#         zip(ground_truth_labels, predicted_labels)
#     ):
#         if ground_truth_label not in {"PASS", "FAIL"} or predicted_label not in {"PASS", "FAIL"}:
#             nulls.append(idx)
#             continue
#         if predicted_label == "PASS" and ground_truth_label == "FAIL":
#             false_negatives.append(idx)
#         elif predicted_label == "FAIL" and ground_truth_label == "PASS":
#             false_positives.append(idx)
#         elif predicted_label == "FAIL" and ground_truth_label == "FAIL":
#             true_positives.append(idx)
#         elif predicted_label == "PASS" and ground_truth_label == "PASS":
#             true_negatives.append(idx)

#     valid_count = len(ground_truth_labels) - len(nulls)
#     tp = len(true_positives)
#     fp = len(false_positives)
#     tn = len(true_negatives)
#     fn = len(false_negatives)
#     accuracy = (tp + tn) / valid_count if valid_count else 0.0
#     precision = tp / (tp + fp) if (tp + fp) else 0.0
#     recall = tp / (tp + fn) if (tp + fn) else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
#     percent_pass = ground_truth_labels.count("PASS") / len(ground_truth_labels)

#     return {
#         "accuracy": accuracy,
#         "f1_score": f1,
#         "precision": precision,
#         "recall": recall,
#         "false_positives": false_positives,
#         "false_negatives": false_negatives,
#         "true_positives": true_positives,
#         "true_negatives": true_negatives,
#         "nulls": nulls,
#         "percent_pass": percent_pass,
#     }


# def safe_model_name(name: str) -> str:
#     return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lpg"


# def append_summary_csv(
#     csv_path: Path,
#     row: Dict[str, Any],
# ) -> None:
#     csv_path.parent.mkdir(parents=True, exist_ok=True)
#     fieldnames = [
#         "model_name",
#         "test_set",
#         "num_examples",
#         "accuracy",
#         "f1_score",
#         "recall",
#         "false_positive_rate",
#         "false_positive_rate_over_all",
#         "missing_labels",
#         "policy_accuracy",
#         "avg_inference_time",
#         "output_dir",
#     ]
#     file_exists = csv_path.exists()
#     with csv_path.open("a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow({key: row.get(key) for key in fieldnames})


# def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
#     with path.open("w", encoding="utf-8") as f:
#         for row in rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def compute_policy_accuracy(
#     ground_truth_labels: List[str],
#     gt_policy_indices: List[Optional[int]],
#     pred_policy_indices: List[Optional[int]],
#     predicted_labels: List[str],
# ) -> Tuple[Optional[float], int, int]:
#     total = 0
#     correct = 0
#     for gt_label, gt_idx, pred_idx, pred_label in zip(
#         ground_truth_labels,
#         gt_policy_indices,
#         pred_policy_indices,
#         predicted_labels,
#     ):
#         if gt_label != "FAIL" or gt_idx is None:
#             continue
#         total += 1
#         if pred_label == "FAIL" and pred_idx == gt_idx:
#             correct += 1
#     if total == 0:
#         return None, correct, total
#     return correct / total, correct, total


# def build_model(args):
#     LatentPolicyGuardModel = load_lpg_model_class(args.lpg_repo_path)
#     model = LatentPolicyGuardModel(
#         args.model_path,
#         ckpt_dir=args.ckpt_dir,
#         adapter_path=args.adapter_path,
#         lora_r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         num_latent_per_stage=args.num_latent_per_stage,
#         stage_names=args.stage_names,
#         use_prj=args.use_prj,
#         prj_dim=args.prj_dim,
#         greedy=args.greedy,
#         remove_eos=args.remove_eos,
#         model_max_length=args.model_max_length,
#     )
#     model.load()
#     return model


# def main(args):
#     configure_logging(args.log_level)

#     print(f"Loading DynaBench: {args.dataset_path} / {args.subset} / {args.split}")
#     dataset = load_dynabench_dataset(args.dataset_path, args.subset, args.split)
#     dataset = filter_labels(dataset, args.label_col)
#     dataset = maybe_select_examples(dataset, args.num_examples, args.seed)

#     missing_cols = [
#         col for col in (args.policy_col, args.transcript_col, args.label_col)
#         if col not in dataset.column_names
#     ]
#     if missing_cols:
#         raise ValueError(
#             f"Missing columns {missing_cols}. Available columns: {dataset.column_names}"
#         )

#     print(f"Loading LPG checkpoint: {args.ckpt_dir or args.adapter_path}")
#     model = build_model(args)

#     ground_truth_labels = get_dataset_labels(
#         dataset,
#         label_col=args.label_col,
#         pos_label=args.pos_label,
#         neg_label=args.neg_label,
#     )

#     rows = []
#     predicted_labels = []
#     gt_policy_indices = []
#     pred_policy_indices = []
#     inference_times = []

#     for idx, example in enumerate(tqdm(dataset, desc="Evaluating LPG on DynaBench")):
#         raw_policy = example[args.policy_col]
#         lpg_policy = normalize_policy_block_for_lpg(raw_policy)
#         output = model.generate(
#             system_prompt="",
#             content=example[args.transcript_col],
#             policies=lpg_policy,
#             dataset_type="multi_policy",
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#         )
#         prediction = output.prediction or {"safe": True}
#         pred_label = prediction_to_label(prediction)
#         gt_rule_number = get_ground_truth_rule_number(example)
#         gt_policy_idx = get_ground_truth_policy_index(example, args.policy_col)
#         pred_policy_idx = get_predicted_policy_index(prediction)

#         predicted_labels.append(pred_label)
#         gt_policy_indices.append(gt_policy_idx)
#         pred_policy_indices.append(pred_policy_idx)
#         inference_times.append(output.inference_time)

#         rows.append(
#             {
#                 "index": idx,
#                 "sample_id": example.get("base_id", idx),
#                 "ground_truth_label": ground_truth_labels[idx],
#                 "predicted_label": pred_label,
#                 "ground_truth_rule_number": gt_rule_number,
#                 "ground_truth_policy_index": gt_policy_idx,
#                 "predicted_policy_index": pred_policy_idx,
#                 "correct": pred_label == ground_truth_labels[idx],
#                 "policy_correct": (
#                     pred_label == "PASS"
#                     if ground_truth_labels[idx] == "PASS"
#                     else pred_label == "FAIL" and pred_policy_idx == gt_policy_idx
#                 ),
#                 "prediction": prediction,
#                 "raw_output": output.raw_output,
#                 "inference_time": output.inference_time,
#                 "policy": raw_policy,
#                 "lpg_policy": lpg_policy,
#                 "transcript": example[args.transcript_col],
#                 "metadata": parse_jsonish(example.get("metadata")),
#             }
#         )

#     stats = get_stats(ground_truth_labels, predicted_labels)
#     num_safe = ground_truth_labels.count("PASS")
#     num_total = len(ground_truth_labels)
#     false_positive_count = len(stats["false_positives"])
#     false_positive_rate = false_positive_count / num_safe if num_safe else 0.0
#     false_positive_rate_over_all = false_positive_count / num_total if num_total else 0.0
#     policy_accuracy, policy_correct, policy_total = compute_policy_accuracy(
#         ground_truth_labels,
#         gt_policy_indices,
#         pred_policy_indices,
#         predicted_labels,
#     )
#     avg_inference_time = (
#         sum(inference_times) / len(inference_times) if inference_times else 0.0
#     )

#     run_name = safe_model_name(args.run_name or f"lpg_{Path(args.ckpt_dir or args.adapter_path).name}")
#     output_dir = Path(args.output_dir) / run_name / str(time.time_ns())
#     output_dir.mkdir(parents=True, exist_ok=True)

#     metrics = {
#         "model_name": run_name,
#         "model_path": args.model_path,
#         "ckpt_dir": args.ckpt_dir,
#         "adapter_path": args.adapter_path,
#         "dataset_path": args.dataset_path,
#         "subset": args.subset,
#         "split": args.split,
#         "num_examples": num_total,
#         "accuracy": stats["accuracy"],
#         "f1_score": stats["f1_score"],
#         "recall": stats["recall"],
#         "false_positive_rate": false_positive_rate,
#         "false_positive_rate_over_all": false_positive_rate_over_all,
#         "false_positives": stats["false_positives"],
#         "false_negatives": stats["false_negatives"],
#         "missing_label_examples": stats["nulls"],
#         "missing_labels": len(stats["nulls"]),
#         "percent_pass": stats["percent_pass"],
#         "policy_accuracy": policy_accuracy,
#         "policy_correct": policy_correct,
#         "policy_total": policy_total,
#         "avg_inference_time": avg_inference_time,
#     }

#     write_jsonl(output_dir / "outputs.jsonl", rows)
#     with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2, ensure_ascii=False)

#     append_summary_csv(
#         Path(args.summary_csv),
#         {
#             **metrics,
#             "test_set": args.subset or args.dataset_path,
#             "output_dir": str(output_dir),
#         },
#     )

#     print("\nLPG on DynaBench")
#     print(f"Examples: {num_total}")
#     print(f"Accuracy: {stats['accuracy']:.2%}")
#     print(f"F1: {stats['f1_score']:.2%}")
#     print(f"Recall: {stats['recall']:.2%}")
#     print(f"False positive rate: {false_positive_rate:.2%}")
#     print(f"False positive rate over all examples: {false_positive_rate_over_all:.2%}")
#     if policy_accuracy is not None:
#         print(f"Policy accuracy: {policy_accuracy:.2%} ({policy_correct}/{policy_total})")
#     print(f"Missing labels: {len(stats['nulls'])}")
#     print(f"Average inference time: {avg_inference_time:.4f}s")
#     print(f"Outputs saved to: {output_dir / 'outputs.jsonl'}")
#     print(f"Metrics saved to: {output_dir / 'metrics.json'}")


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Evaluate a Latent Policy Guard checkpoint on DynaBench."
#     )
#     parser.add_argument(
#         "--lpg_repo_path",
#         default=str(DEFAULT_LPG_REPO_PATH),
#         help="Path to the Latent_Policy_Guard repo.",
#     )
#     parser.add_argument(
#         "--model_path",
#         default="Qwen/Qwen3-4B",
#         help="Base model path or HF id used to train LPG.",
#     )
#     parser.add_argument(
#         "--ckpt_dir",
#         default=DEFAULT_CKPT_DIR,
#         help="LPG checkpoint directory containing model.safetensors or pytorch_model.bin.",
#     )
#     parser.add_argument(
#         "--adapter_path",
#         default=None,
#         help="Optional PEFT adapter directory. If set, ckpt_dir can hold extra weights.",
#     )
#     parser.add_argument(
#         "--dataset_path",
#         default="tomg-group-umd/DynaBench",
#         help="Local DynaBench JSON/JSONL file or HF dataset id.",
#     )
#     parser.add_argument("--subset", default="DynaBench", help="HF dataset config/subset.")
#     parser.add_argument("--split", default="test", help="HF dataset split.")
#     parser.add_argument("--policy_col", default="policy")
#     parser.add_argument("--transcript_col", default="transcript")
#     parser.add_argument("--label_col", default="label")
#     parser.add_argument("--pos_label", default="FAIL")
#     parser.add_argument("--neg_label", default="PASS")
#     parser.add_argument("--num_examples", default=-1, type=int)
#     parser.add_argument("--seed", default=42, type=int)
#     parser.add_argument("--output_dir", default="log/lpg_dynabench")
#     parser.add_argument("--summary_csv", default="log/summary_lpg.csv")
#     parser.add_argument("--run_name", default=None)
#     parser.add_argument("--log_level", default=None)

#     parser.add_argument("--max_new_tokens", default=160, type=int)
#     parser.add_argument("--temperature", default=0.1, type=float)
#     parser.add_argument("--model_max_length", default=1024, type=int)
#     parser.add_argument("--num_latent_per_stage", default="4,6")
#     parser.add_argument("--stage_names", default="intent,risk")
#     parser.add_argument("--lora_r", default=128, type=int)
#     parser.add_argument("--lora_alpha", default=32, type=int)
#     parser.add_argument("--use_prj", default=True, action=argparse.BooleanOptionalAction)
#     parser.add_argument("--prj_dim", default=2560, type=int)
#     parser.add_argument("--greedy", default=True, action=argparse.BooleanOptionalAction)
#     parser.add_argument("--remove_eos", default=True, action=argparse.BooleanOptionalAction)
#     return parser.parse_args()


# if __name__ == "__main__":
#     main(parse_args())


"""Evaluate an LPG checkpoint on DynaBench with DynaGuard-style labels.

Example:
    python eval_lpg_dynabench.py \
        --lpg_repo_path ../Latent_Policy_Guard \
        --model_path Qwen/Qwen3-4B \
        --ckpt_dir VHDang/Guardrail/fix-ver/Latent_Policy_Guard/training/outputs/lpg_qwen3_4b/Qwen3-4B/ep_3/lr_5e-05/seed_11/checkpoint-6453 \
        --dataset_path tomg-group-umd/DynaBench \
        --subset DynaBench \
        --split test
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
from tqdm import tqdm


DEFAULT_LPG_REPO_PATH = Path(__file__).resolve().parents[1] / "Latent_Policy_Guard"
DEFAULT_CKPT_DIR = (
    "VHDang/Guardrail/fix-ver/Latent_Policy_Guard/training/outputs/"
    "lpg_qwen3_4b/Qwen3-4B/ep_3/lr_5e-05/seed_11/checkpoint-6453"
)


def configure_logging(log_level: Optional[str]) -> None:
    if log_level is None:
        return
    logging.basicConfig(level=getattr(logging, log_level.upper()))


def add_lpg_repo_to_path(lpg_repo_path: str) -> None:
    repo_path = Path(lpg_repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"LPG repo path does not exist: {repo_path}")
    if not (repo_path / "evaluation" / "models" / "latent_policy_guard.py").exists():
        raise FileNotFoundError(
            "Could not find evaluation/models/latent_policy_guard.py under "
            f"{repo_path}"
        )
    eval_path = repo_path / "evaluation"
    for path in (repo_path, eval_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_lpg_model_class(lpg_repo_path: str):
    add_lpg_repo_to_path(lpg_repo_path)
    from models.latent_policy_guard import LatentPolicyGuardModel

    return LatentPolicyGuardModel


def load_dynabench_dataset(
    dataset_path: str,
    subset: Optional[str],
    split: str,
):
    if os.path.exists(dataset_path):
        return datasets.load_dataset(
            "json",
            data_files={"eval": dataset_path},
            split="eval",
        )
    if subset:
        return datasets.load_dataset(dataset_path, subset, split=split)
    return datasets.load_dataset(dataset_path, split=split)


def filter_labels(dataset, label_col: str):
    def is_valid_row(example):
        label = example.get(label_col)
        if label is None:
            return False
        label = str(label).strip()
        return bool(label) and label.lower() != "null"

    return dataset.filter(is_valid_row)


def maybe_select_examples(dataset, num_examples: int, seed: int):
    if num_examples <= 0 or num_examples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(num_examples))


def maybe_filter_metadata_tokens(dataset, max_metadata_tokens: int):
    if max_metadata_tokens <= 0:
        return dataset

    def is_within_limit(example):
        metadata = parse_jsonish(example.get("metadata"))
        num_tokens = metadata.get("num_tokens")
        return isinstance(num_tokens, int) and num_tokens <= max_metadata_tokens

    return dataset.filter(is_within_limit)


def parse_jsonish(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if not isinstance(value, str):
        return {}
    value = value.strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_int_from_sequence(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        matches = re.findall(r"-?\d+", value)
        return int(matches[0]) if matches else None
    if isinstance(value, Iterable):
        for item in value:
            parsed = as_int(item)
            if parsed is not None:
                return parsed
    return as_int(value)


def detect_policy_index_offset(policy_text: str) -> int:
    for line in str(policy_text).splitlines():
        match = re.match(r"^(\d+)[.:]\s+.*$", line)
        if match:
            return 1 if int(match.group(1)) == 1 else 0
    return 0


def normalize_policy_block_for_lpg(
    policy_text: str,
    preserve_blank_lines: bool = False,
) -> str:
    lines = str(policy_text).splitlines()
    first_header = None
    for line in lines:
        match = re.match(r"^(\d+)[.:]\s+(.*)$", line)
        if match:
            first_header = int(match.group(1))
            break

    if first_header not in (0, 1):
        return str(policy_text)

    expected = first_header
    offset = 1 if first_header == 1 else 0
    normalized_lines = []
    for idx, line in enumerate(lines):
        if not preserve_blank_lines and not line.strip():
            continue
        match = re.match(r"^(\d+)[.:]\s+(.*)$", line)
        if not match:
            normalized_lines.append(line)
            continue
        rule_number = int(match.group(1))
        if rule_number != expected:
            normalized_lines.append(line)
            continue
        normalized_lines.append(f"{rule_number - offset}: {match.group(2)}")
        expected += 1
    return "\n".join(normalized_lines)


def normalize_transcript_for_lpg(transcript: str, quote_roles: bool = False) -> str:
    transcript = str(transcript)
    if not quote_roles:
        return transcript
    return (
        transcript
        .replace("'User':", "User:")
        .replace("'Agent':", "Agent:")
        .replace("User:", "'User':")
        .replace("Agent:", "'Agent':")
    )


def build_lpg_annotation_input(policy_block: str, transcript: str) -> str:
    return (
        "**Safety policies (indexed from 0):**\n"
        f"{policy_block}"
        "**Content to evaluate:**\n"
        f"{transcript}"
    )


def get_ground_truth_rule_number(example: Dict[str, Any]) -> Optional[int]:
    metadata = parse_jsonish(example.get("metadata"))
    for key in ("rules_violated", "violated_rules", "policy_indices"):
        parsed = first_int_from_sequence(metadata.get(key))
        if parsed is not None:
            return parsed
    for key in ("rule_violated", "violated_rule", "policy_index"):
        parsed = as_int(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def get_ground_truth_policy_index(
    example: Dict[str, Any],
    policy_col: str,
) -> Optional[int]:
    rule_number = get_ground_truth_rule_number(example)
    if rule_number is None:
        return None
    return max(rule_number - detect_policy_index_offset(example.get(policy_col, "")), 0)


def get_predicted_policy_index(prediction: Dict[str, Any]) -> Optional[int]:
    parsed = as_int(prediction.get("policy_index"))
    if parsed is not None:
        return parsed
    parsed = first_int_from_sequence(prediction.get("policy_indices"))
    if parsed is not None:
        return parsed
    return first_int_from_sequence(prediction.get("violated_policy_indices"))


def prediction_to_label(prediction: Dict[str, Any]) -> str:
    return "PASS" if bool(prediction.get("safe", True)) else "FAIL"


def normalize_label(value: Any, pos_label: str, neg_label: str) -> str:
    if value is None:
        return "null"
    label = str(value).strip()
    if label == pos_label:
        return "FAIL"
    if label == neg_label:
        return "PASS"
    return "null"


def get_dataset_labels(dataset, label_col: str, pos_label: str, neg_label: str) -> List[str]:
    return [
        normalize_label(example.get(label_col), pos_label, neg_label)
        for example in dataset
    ]


def get_stats(ground_truth_labels: List[str], predicted_labels: List[str]) -> Dict[str, Any]:
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError(
            "Ground truth labels and predicted labels must have the same length. "
            f"Got {len(ground_truth_labels)} and {len(predicted_labels)}."
        )

    false_negatives = []
    false_positives = []
    true_positives = []
    true_negatives = []
    nulls = []

    for idx, (ground_truth_label, predicted_label) in enumerate(
        zip(ground_truth_labels, predicted_labels)
    ):
        if ground_truth_label not in {"PASS", "FAIL"} or predicted_label not in {"PASS", "FAIL"}:
            nulls.append(idx)
            continue
        if predicted_label == "PASS" and ground_truth_label == "FAIL":
            false_negatives.append(idx)
        elif predicted_label == "FAIL" and ground_truth_label == "PASS":
            false_positives.append(idx)
        elif predicted_label == "FAIL" and ground_truth_label == "FAIL":
            true_positives.append(idx)
        elif predicted_label == "PASS" and ground_truth_label == "PASS":
            true_negatives.append(idx)

    valid_count = len(ground_truth_labels) - len(nulls)
    tp = len(true_positives)
    fp = len(false_positives)
    tn = len(true_negatives)
    fn = len(false_negatives)
    accuracy = (tp + tn) / valid_count if valid_count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    percent_pass = ground_truth_labels.count("PASS") / len(ground_truth_labels)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "nulls": nulls,
        "percent_pass": percent_pass,
    }


def safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "lpg"


def append_summary_csv(
    csv_path: Path,
    row: Dict[str, Any],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "test_set",
        "num_examples",
        "accuracy",
        "f1_score",
        "recall",
        "false_positive_rate",
        "false_positive_rate_over_all",
        "missing_labels",
        "policy_accuracy",
        "avg_inference_time",
        "output_dir",
    ]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_policy_accuracy(
    ground_truth_labels: List[str],
    gt_policy_indices: List[Optional[int]],
    pred_policy_indices: List[Optional[int]],
    predicted_labels: List[str],
) -> Tuple[Optional[float], int, int]:
    total = 0
    correct = 0
    for gt_label, gt_idx, pred_idx, pred_label in zip(
        ground_truth_labels,
        gt_policy_indices,
        pred_policy_indices,
        predicted_labels,
    ):
        if gt_label != "FAIL" or gt_idx is None:
            continue
        total += 1
        if pred_label == "FAIL" and pred_idx == gt_idx:
            correct += 1
    if total == 0:
        return None, correct, total
    return correct / total, correct, total


def build_model(args):
    LatentPolicyGuardModel = load_lpg_model_class(args.lpg_repo_path)
    model = LatentPolicyGuardModel(
        args.model_path,
        ckpt_dir=args.ckpt_dir,
        adapter_path=args.adapter_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_latent_per_stage=args.num_latent_per_stage,
        stage_names=args.stage_names,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        greedy=args.greedy,
        remove_eos=args.remove_eos,
        model_max_length=args.model_max_length,
    )
    model.load()
    return model


def main(args):
    configure_logging(args.log_level)

    print(f"Loading DynaBench: {args.dataset_path} / {args.subset} / {args.split}")
    dataset = load_dynabench_dataset(args.dataset_path, args.subset, args.split)
    dataset = filter_labels(dataset, args.label_col)
    dataset = maybe_filter_metadata_tokens(dataset, args.max_metadata_tokens)
    dataset = maybe_select_examples(dataset, args.num_examples, args.seed)

    missing_cols = [
        col for col in (args.policy_col, args.transcript_col, args.label_col)
        if col not in dataset.column_names
    ]
    if missing_cols:
        raise ValueError(
            f"Missing columns {missing_cols}. Available columns: {dataset.column_names}"
        )

    metadata_num_tokens = [
        parse_jsonish(example.get("metadata")).get("num_tokens")
        for example in dataset
    ]
    metadata_num_tokens = [
        value for value in metadata_num_tokens
        if isinstance(value, int)
    ]
    num_over_model_max_length = sum(
        value > args.model_max_length for value in metadata_num_tokens
    )
    if num_over_model_max_length:
        print(
            "WARNING: "
            f"{num_over_model_max_length}/{len(dataset)} examples have metadata.num_tokens "
            f"> model_max_length={args.model_max_length}. They may be truncated."
        )

    print(f"Loading LPG checkpoint: {args.ckpt_dir or args.adapter_path}")
    model = build_model(args)

    ground_truth_labels = get_dataset_labels(
        dataset,
        label_col=args.label_col,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
    )

    rows = []
    predicted_labels = []
    gt_policy_indices = []
    pred_policy_indices = []
    inference_times = []

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating LPG on DynaBench")):
        raw_policy = example[args.policy_col]
        lpg_policy = normalize_policy_block_for_lpg(
            raw_policy,
            preserve_blank_lines=args.preserve_policy_blank_lines,
        )
        transcript = normalize_transcript_for_lpg(
            example[args.transcript_col],
            quote_roles=args.quote_roles,
        )
        user_input = build_lpg_annotation_input(lpg_policy, transcript)
        output = model.generate(
            system_prompt="",
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        prediction = output.prediction or {"safe": True}
        pred_label = prediction_to_label(prediction)
        gt_rule_number = get_ground_truth_rule_number(example)
        gt_policy_idx = get_ground_truth_policy_index(example, args.policy_col)
        pred_policy_idx = get_predicted_policy_index(prediction)

        predicted_labels.append(pred_label)
        gt_policy_indices.append(gt_policy_idx)
        pred_policy_indices.append(pred_policy_idx)
        inference_times.append(output.inference_time)

        rows.append(
            {
                "index": idx,
                "sample_id": example.get("base_id", idx),
                "ground_truth_label": ground_truth_labels[idx],
                "predicted_label": pred_label,
                "ground_truth_rule_number": gt_rule_number,
                "ground_truth_policy_index": gt_policy_idx,
                "predicted_policy_index": pred_policy_idx,
                "correct": pred_label == ground_truth_labels[idx],
                "policy_correct": (
                    pred_label == "PASS"
                    if ground_truth_labels[idx] == "PASS"
                    else pred_label == "FAIL" and pred_policy_idx == gt_policy_idx
                ),
                "prediction": prediction,
                "raw_output": output.raw_output,
                "inference_time": output.inference_time,
                "policy": raw_policy,
                "lpg_policy": lpg_policy,
                "transcript": transcript,
                "annotation_input": user_input,
                "metadata": parse_jsonish(example.get("metadata")),
            }
        )

    stats = get_stats(ground_truth_labels, predicted_labels)
    num_safe = ground_truth_labels.count("PASS")
    num_total = len(ground_truth_labels)
    false_positive_count = len(stats["false_positives"])
    false_positive_rate = false_positive_count / num_safe if num_safe else 0.0
    false_positive_rate_over_all = false_positive_count / num_total if num_total else 0.0
    policy_accuracy, policy_correct, policy_total = compute_policy_accuracy(
        ground_truth_labels,
        gt_policy_indices,
        pred_policy_indices,
        predicted_labels,
    )
    avg_inference_time = (
        sum(inference_times) / len(inference_times) if inference_times else 0.0
    )

    run_name = safe_model_name(args.run_name or f"lpg_{Path(args.ckpt_dir or args.adapter_path).name}")
    output_dir = Path(args.output_dir) / run_name / str(time.time_ns())
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": run_name,
        "model_path": args.model_path,
        "ckpt_dir": args.ckpt_dir,
        "adapter_path": args.adapter_path,
        "dataset_path": args.dataset_path,
        "subset": args.subset,
        "split": args.split,
        "num_examples": num_total,
        "accuracy": stats["accuracy"],
        "f1_score": stats["f1_score"],
        "recall": stats["recall"],
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_over_all": false_positive_rate_over_all,
        "false_positives": stats["false_positives"],
        "false_negatives": stats["false_negatives"],
        "missing_label_examples": stats["nulls"],
        "missing_labels": len(stats["nulls"]),
        "percent_pass": stats["percent_pass"],
        "policy_accuracy": policy_accuracy,
        "policy_correct": policy_correct,
        "policy_total": policy_total,
        "avg_inference_time": avg_inference_time,
        "max_metadata_tokens": args.max_metadata_tokens,
        "model_max_length": args.model_max_length,
        "metadata_num_tokens_over_model_max_length": num_over_model_max_length,
        "preserve_policy_blank_lines": args.preserve_policy_blank_lines,
        "quote_roles": args.quote_roles,
    }

    write_jsonl(output_dir / "outputs.jsonl", rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    append_summary_csv(
        Path(args.summary_csv),
        {
            **metrics,
            "test_set": args.subset or args.dataset_path,
            "output_dir": str(output_dir),
        },
    )

    print("\nLPG on DynaBench")
    print(f"Examples: {num_total}")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    print(f"F1: {stats['f1_score']:.2%}")
    print(f"Recall: {stats['recall']:.2%}")
    print(f"False positive rate: {false_positive_rate:.2%}")
    print(f"False positive rate over all examples: {false_positive_rate_over_all:.2%}")
    if policy_accuracy is not None:
        print(f"Policy accuracy: {policy_accuracy:.2%} ({policy_correct}/{policy_total})")
    print(f"Missing labels: {len(stats['nulls'])}")
    print(f"Average inference time: {avg_inference_time:.4f}s")
    print(f"Outputs saved to: {output_dir / 'outputs.jsonl'}")
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Latent Policy Guard checkpoint on DynaBench."
    )
    parser.add_argument(
        "--lpg_repo_path",
        default=str(DEFAULT_LPG_REPO_PATH),
        help="Path to the Latent_Policy_Guard repo.",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-4B",
        help="Base model path or HF id used to train LPG.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default=DEFAULT_CKPT_DIR,
        help="LPG checkpoint directory containing model.safetensors or pytorch_model.bin.",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional PEFT adapter directory. If set, ckpt_dir can hold extra weights.",
    )
    parser.add_argument(
        "--dataset_path",
        default="tomg-group-umd/DynaBench",
        help="Local DynaBench JSON/JSONL file or HF dataset id.",
    )
    parser.add_argument("--subset", default="DynaBench", help="HF dataset config/subset.")
    parser.add_argument("--split", default="test", help="HF dataset split.")
    parser.add_argument("--policy_col", default="policy")
    parser.add_argument("--transcript_col", default="transcript")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--pos_label", default="FAIL")
    parser.add_argument("--neg_label", default="PASS")
    parser.add_argument("--num_examples", default=-1, type=int)
    parser.add_argument(
        "--max_metadata_tokens",
        default=-1,
        type=int,
        help=(
            "Optional filter on DynaBench metadata.num_tokens. "
            "Use 800 to match the LPG training script token filter."
        ),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="log/lpg_dynabench")
    parser.add_argument("--summary_csv", default="log/summary_lpg.csv")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_level", default=None)

    parser.add_argument("--max_new_tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--num_latent_per_stage", default="4,6")
    parser.add_argument("--stage_names", default="intent,risk")
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--use_prj", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--prj_dim", default=2560, type=int)
    parser.add_argument("--greedy", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--remove_eos", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--preserve_policy_blank_lines",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Preserve blank separator lines in DynaBench policies. Default compacts them to match LPG training data.",
    )
    parser.add_argument(
        "--quote_roles",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Convert User:/Agent: transcript role tags to 'User':/'Agent': before inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
