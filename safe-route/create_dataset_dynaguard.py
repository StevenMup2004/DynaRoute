import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DYNAGUARD_DIR = REPO_ROOT / "DynaGuard"
if str(DYNAGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(DYNAGUARD_DIR))

from constants import DYNAGUARD_CONTENT_TEMPLATE, DYNAGUARD_SYSTEM_PROMPT  # noqa: E402
from hf_model_wrapper import HfModelWrapper  # noqa: E402


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}.")
    return data


def format_user_agent_tags(transcript: str, user_tag: str = "'User':", agent_tag: str = "'Agent':") -> str:
    return (
        transcript.replace("User:", user_tag)
        .replace("Agent:", agent_tag)
        .replace("'User':", user_tag)
        .replace("'Agent':", agent_tag)
    )


def build_messages(dataset: List[dict], wrapper: HfModelWrapper, think: bool = False) -> List[str]:
    messages: List[str] = []
    for row in dataset:
        policy = str(row.get("policy", "")).strip()
        transcript = str(row.get("transcript", "")).strip()

        # Keep transcript tags consistent with the DynaGuard prompt format.
        transcript = format_user_agent_tags(transcript, user_tag="'User':", agent_tag="'Agent':")
        content = DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript)
        prompt = wrapper.apply_chat_template(
            DYNAGUARD_SYSTEM_PROMPT,
            content,
            enable_thinking=think,
        )
        messages.append(prompt)
    return messages


@torch.inference_mode()
def extract_features(
    dataset: List[dict],
    model_name: str,
    batch_size: int,
    layer_idx: int,
    think: bool,
    max_length: int | None,
):
    wrapper = HfModelWrapper(
        model_name,
        batch_size=batch_size,
        custom_name="dynaguard_router_extractor",
    )
    wrapper.model.config.use_cache = False

    messages = build_messages(dataset, wrapper, think=think)
    features = []

    for start in tqdm(range(0, len(messages), batch_size), desc="Extract hidden states"):
        batch_messages = messages[start : start + batch_size]
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if max_length is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_length

        inputs = wrapper.tokenizer(batch_messages, **tokenizer_kwargs).to(wrapper.model.device)
        outputs = wrapper.model(
            **inputs,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[layer_idx][:, -1, :].detach().cpu().float()
        features.append(hidden)

        del inputs, outputs, hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(features, dim=0)


def save_split(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    layer_idx: int,
    think: bool,
    max_length: int | None,
):
    dataset = load_json_list(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_features(
        dataset=dataset,
        model_name=model_name,
        batch_size=batch_size,
        layer_idx=layer_idx,
        think=think,
        max_length=max_length,
    )
    labels = torch.tensor([int(row["label"]) for row in dataset], dtype=torch.float32)

    torch.save(features, output_dir / "features.pt")
    torch.save(labels, output_dir / "labels.pt")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": str(input_path),
                "model_name": model_name,
                "batch_size": batch_size,
                "layer_idx": layer_idx,
                "think": think,
                "max_length": max_length,
                "num_samples": len(dataset),
                "feature_shape": list(features.shape),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(dataset)} samples to {output_dir}")
    print(f"Feature shape: {tuple(features.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Build router features for DynaGuard from JSON files.")
    parser.add_argument(
        "--train_path",
        default=str(REPO_ROOT / "safe-route" / "train_router.json"),
        help="Path to train_router.json.",
    )
    parser.add_argument(
        "--valid_path",
        default=str(REPO_ROOT / "safe-route" / "valid_router.json"),
        help="Path to valid_router.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "safe-route" / "data" / "dynaguard_1p7b_8b"),
        help="Directory where features.pt/labels.pt will be written.",
    )
    parser.add_argument(
        "--model_name",
        default="tomg-group-umd/DynaGuard-1.7B",
        help="HF model id for the small/router model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for hidden-state extraction.",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Which hidden-state layer to extract.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode when building prompts.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional truncation length.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_path).expanduser().resolve()
    valid_path = Path(args.valid_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    train_dir = output_dir / "train"
    valid_dir = output_dir / "val"

    save_split(
        input_path=train_path,
        output_dir=train_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        layer_idx=args.layer_idx,
        think=args.think,
        max_length=args.max_length,
    )
    save_split(
        input_path=valid_path,
        output_dir=valid_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        layer_idx=args.layer_idx,
        think=args.think,
        max_length=args.max_length,
    )

    # Convenience copies that match the filenames expected by safe-route/train_router.py.
    torch.save(torch.load(train_dir / "features.pt", map_location="cpu"), output_dir / "train_features.pt")
    torch.save(torch.load(train_dir / "labels.pt", map_location="cpu"), output_dir / "train_labels.pt")
    torch.save(torch.load(valid_dir / "features.pt", map_location="cpu"), output_dir / "val_features.pt")
    torch.save(torch.load(valid_dir / "labels.pt", map_location="cpu"), output_dir / "val_labels.pt")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_path": str(train_path),
                "valid_path": str(valid_path),
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "layer_idx": args.layer_idx,
                "think": args.think,
                "max_length": args.max_length,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Done.")
    print(f"Ready for safe-route/train_router.py at: {output_dir}")


if __name__ == "__main__":
    main()
