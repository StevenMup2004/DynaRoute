import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import DYNAGUARD_CONTENT_TEMPLATE, DYNAGUARD_SYSTEM_PROMPT


DEFAULT_DATASET_PATH = Path(r"VHDang/Guardrail/fix-ver/DynaGuard/proguard_text_multiturn.json")
DEFAULT_MODEL_ID = "tomg-group-umd/DynaGuard-8B"


def normalize_transcript(text: str) -> str:
    """Match the formatting used by the repo's DynaGuard prompt templates."""
    text = str(text)
    text = text.replace("User:", "'User':")
    text = text.replace("Agent:", "'Agent':")
    return text


def build_prompt(tokenizer, policy: str, dialogue: str) -> str:
    conversation = normalize_transcript(dialogue)
    user_content = DYNAGUARD_CONTENT_TEMPLATE.format(
        policy=policy.strip(),
        conversation=conversation.strip(),
    )
    messages = [
        {"role": "system", "content": DYNAGUARD_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": "<answer>\n"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=False,
        )
        if isinstance(prompt, str) and prompt.endswith("<|im_end|>\n"):
            prompt = prompt[:-11]
        return prompt


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Unsupported dataset format: expected a list of records or {data: [...]}." )
    if not isinstance(data, list):
        raise ValueError("Unsupported dataset format: expected a JSON array.")
    return data


def extract_label(text: str) -> str:
    match = re.search(r"\b(PASS|FAIL)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    text = text.strip().upper()
    if text.startswith("PASS"):
        return "PASS"
    if text.startswith("FAIL"):
        return "FAIL"
    return "UNKNOWN"


def compute_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for pred, gold in zip(preds, labels):
        if gold == "FAIL" and pred == "FAIL":
            tp += 1
        elif gold == "PASS" and pred == "FAIL":
            fp += 1
        elif gold == "PASS" and pred == "PASS":
            tn += 1
        elif gold == "FAIL" and pred == "PASS":
            fn += 1

    total = len(labels) if labels else 1
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision_fail": precision,
        "recall_fail": recall,
        "f1_fail": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval DynaGuard-8B on Beavertail policies.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_path", default="VHDang/Guardrail/fix-ver/DynaGuard/saferoute_pku_test_split_responses-out3.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    records = load_dataset(dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    device_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=device_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    ).eval()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_labels: List[str] = []
    pred_labels: List[str] = []

    with output_path.open("w", encoding="utf-8") as fout, torch.inference_mode():
        for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
            batch = records[start : start + args.batch_size]
            prompts = [
                build_prompt(
                    tokenizer,
                    policy=str(item.get("policies", item.get("policy", ""))),
                    dialogue=str(item.get("dialogue", item.get("transcript", ""))),
                )
                for item in batch
            ]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = enc["input_ids"].shape[1]
            new_tokens = generated[:, prompt_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for item, raw_text in zip(batch, decoded):
                gold = str(item.get("label", item.get("is_safe", ""))).strip()
                if gold in {"True", "true", "1"}:
                    gold = "PASS"
                elif gold in {"False", "false", "0"}:
                    gold = "FAIL"
                else:
                    gold = gold.upper()
                if gold not in {"PASS", "FAIL"}:
                    gold = "PASS" if bool(item.get("is_safe", True)) else "FAIL"

                pred = extract_label(raw_text)
                if pred not in {"PASS", "FAIL"}:
                    pred = "PASS"

                gold_labels.append(gold)
                pred_labels.append(pred)

                out_row = {
                    "id": item.get("id"),
                    "label": gold,
                    "prediction": pred,
                    "correct": pred == gold,
                    "raw_output": raw_text,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    metrics = compute_metrics(pred_labels, gold_labels)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
