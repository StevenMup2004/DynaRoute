# import argparse
# import json
# from pathlib import Path

# import torch
# from tqdm import tqdm

# from constants import DYNAGUARD_CONTENT_TEMPLATE, DYNAGUARD_SYSTEM_PROMPT
# from helpers import format_user_agent_tags
# from hf_model_wrapper import HfModelWrapper


# def load_dataset(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         data = json.load(f)
#     if not isinstance(data, list):
#         raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}.")
#     return data


# def build_messages(dataset, wrapper, think=False):
#     messages = []

#     for row in dataset:
#         policy = str(row.get("policy", "")).strip()
#         transcript = str(row.get("transcript", "")).strip()
#         # Keep the transcript format aligned with DynaGuard training data.
#         # This is idempotent if the transcript is already tagged with
#         # User:/Agent: or 'User':'Agent' markers.
#         transcript = format_user_agent_tags(transcript, user_tag="'User':", agent_tag="'Agent':")

#         content = DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript)
#         message = wrapper.apply_chat_template(
#             DYNAGUARD_SYSTEM_PROMPT,
#             content,
#             enable_thinking=think,
#         )
#         messages.append(message)

#     return messages


# def main():
#     parser = argparse.ArgumentParser(description="Extract DynaGuard hidden states for a JSON dataset.")
#     parser.add_argument(
#         "--dataset_path",
#         default=r"routing_dataset-1_7b.json",
#         help="Path to the JSON dataset file.",
#     )
#     parser.add_argument(
#         "--model_path",
#         default="tomg-group-umd/DynaGuard-1.7B",
#         help="HF model path or repo id.",
#     )
#     parser.add_argument(
#         "--output_dir",
#         default=r"hidden_states_out",
#         help="Directory to write hidden-state outputs.",
#     )
#     parser.add_argument(
#         "--layer_idx",
#         type=int,
#         default=-1,
#         help="Which hidden state layer to extract.",
#     )
#     parser.add_argument(
#         "--think",
#         action="store_true",
#         help="Use thinking mode when building the prompt.",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="Batch size for hidden-state extraction.",
#     )
#     args = parser.parse_args()

#     dataset_path = Path(args.dataset_path).expanduser().resolve()
#     output_dir = Path(args.output_dir).expanduser().resolve()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     dataset = load_dataset(dataset_path)
#     wrapper = HfModelWrapper(
#         args.model_path,
#         batch_size=args.batch_size,
#         custom_name="dynaguard_hidden_state_extractor",
#     )

#     messages = build_messages(dataset, wrapper, think=args.think)

#     hidden_states = []
#     prompt_lengths = []

#     for start in tqdm(range(0, len(messages), args.batch_size), desc="Extract hidden states"):
#         batch_messages = messages[start : start + args.batch_size]
#         inputs = wrapper.tokenizer(batch_messages, return_tensors="pt", padding=True).to(wrapper.model.device)
#         prompt_lengths.extend(inputs.attention_mask.sum(dim=1).tolist())

#         with torch.no_grad():
#             outputs = wrapper.model(
#                 **inputs,
#                 output_hidden_states=True,
#             )

#         batch_hidden = outputs.hidden_states[args.layer_idx][:, -1, :]
#         hidden_states.append(batch_hidden)

#     hidden_states = torch.cat(hidden_states, dim=0)

#     tensor_path = output_dir / "hidden_states.pt"
#     meta_path = output_dir / "hidden_state_metadata.jsonl"
#     summary_path = output_dir / "summary.json"

#     torch.save(
#         {
#             "hidden_states": hidden_states,
#             "layer_idx": args.layer_idx,
#             "model_path": args.model_path,
#             "dataset_path": str(dataset_path),
#             "think": args.think,
#         },
#         tensor_path,
#     )

#     with meta_path.open("w", encoding="utf-8") as f:
#         for i, row in enumerate(dataset):
#             record = {
#                 "index": i,
#                 "base_id": row.get("base_id"),
#                 "label": row.get("label"),
#                 "prediction": row.get("prediction"),
#                 "source": row.get("source"),
#                 "policy": row.get("policy"),
#                 "prompt_length": prompt_lengths[i],
#                 "hidden_state_shape": list(hidden_states[i].shape),
#             }
#             f.write(json.dumps(record, ensure_ascii=False) + "\n")

#     with summary_path.open("w", encoding="utf-8") as f:
#         json.dump(
#             {
#                 "num_samples": len(dataset),
#                 "hidden_state_shape": list(hidden_states.shape),
#                 "layer_idx": args.layer_idx,
#                 "model_path": args.model_path,
#                 "dataset_path": str(dataset_path),
#                 "output_tensor": str(tensor_path),
#                 "output_metadata": str(meta_path),
#             },
#             f,
#             indent=2,
#             ensure_ascii=False,
#         )

#     print(f"Saved hidden states to: {tensor_path}")
#     print(f"Saved metadata to: {meta_path}")
#     print(f"Saved summary to: {summary_path}")
#     print(f"Tensor shape: {tuple(hidden_states.shape)}")


# if __name__ == "__main__":
#     main()




import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from constants import DYNAGUARD_CONTENT_TEMPLATE, DYNAGUARD_SYSTEM_PROMPT
from helpers import format_user_agent_tags
from hf_model_wrapper import HfModelWrapper


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}.")
    return data


def build_messages(dataset, wrapper, think=False):
    messages = []

    for row in dataset:
        policy = str(row.get("policy", "")).strip()
        transcript = str(row.get("transcript", "")).strip()
        # Keep the transcript format aligned with DynaGuard training data.
        # This is idempotent if the transcript is already tagged with
        # User:/Agent: or 'User':'Agent' markers.
        transcript = format_user_agent_tags(transcript, user_tag="'User':", agent_tag="'Agent':")

        content = DYNAGUARD_CONTENT_TEMPLATE.format(policy=policy, conversation=transcript)
        message = wrapper.apply_chat_template(
            DYNAGUARD_SYSTEM_PROMPT,
            content,
            enable_thinking=think,
        )
        messages.append(message)

    return messages


def resolve_decoder_layers(model):
    """
    Find the decoder block list for common HF causal-LM architectures.

    Qwen/Llama-style models usually expose ``model.layers``.
    GPT-NeoX-style models use ``transformer.h``.
    """
    candidates = [
        ("model.layers", ("model", "layers")),
        ("model.model.layers", ("model", "model", "layers")),
        ("transformer.h", ("transformer", "h")),
        ("gpt_neox.layers", ("gpt_neox", "layers")),
    ]

    for name, attrs in candidates:
        obj = model
        ok = True
        for attr in attrs:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None:
            return name, obj

    raise AttributeError(
        "Could not find decoder layers on this model. "
        "Expected something like model.layers or transformer.h."
    )


def get_layer_hidden_state(model, inputs, layer_idx):
    """
    Capture just one transformer block output via a forward hook.

    This is much lighter than ``output_hidden_states=True`` because it avoids
    materializing the full hidden-state stack for every layer.
    """
    _, layers = resolve_decoder_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers.")

    captured = {}

    def hook(_, __, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured["hidden"] = hidden

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture hidden state from the target layer.")

    return captured["hidden"]


def main():
    parser = argparse.ArgumentParser(description="Extract DynaGuard hidden states for a JSON dataset.")
    parser.add_argument(
        "--dataset_path",
        default=r"routing_dataset-1_7b.json",
        help="Path to the JSON dataset file.",
    )
    parser.add_argument(
        "--model_path",
        default="tomg-group-umd/DynaGuard-1.7B",
        help="HF model path or repo id.",
    )
    parser.add_argument(
        "--output_dir",
        default=r"hidden_states_out_final",
        help="Directory to write hidden-state outputs.",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Which hidden state layer to extract.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Use thinking mode when building the prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for hidden-state extraction.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional truncation length. Leave unset to use the model's full context window.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    wrapper = HfModelWrapper(
        args.model_path,
        batch_size=args.batch_size,
        custom_name="dynaguard_hidden_state_extractor",
    )
    wrapper.model.config.use_cache = False

    messages = build_messages(dataset, wrapper, think=args.think)

    hidden_states = []
    prompt_lengths = []

    for start in tqdm(range(0, len(messages), args.batch_size), desc="Extract hidden states"):
        batch_messages = messages[start : start + args.batch_size]
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if args.max_length is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = args.max_length

        inputs = wrapper.tokenizer(batch_messages, **tokenizer_kwargs).to(wrapper.model.device)
        prompt_lengths.extend(inputs.attention_mask.sum(dim=1).tolist())

        batch_hidden = get_layer_hidden_state(wrapper.model, inputs, args.layer_idx)
        hidden_states.append(batch_hidden[:, -1, :].detach().cpu())

        del inputs, batch_hidden
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hidden_states = torch.cat(hidden_states, dim=0)

    tensor_path = output_dir / "hidden_states.pt"
    meta_path = output_dir / "hidden_state_metadata.jsonl"
    summary_path = output_dir / "summary.json"

    torch.save(
        {
            "hidden_states": hidden_states,
            "layer_idx": args.layer_idx,
            "model_path": args.model_path,
            "dataset_path": str(dataset_path),
            "think": args.think,
        },
        tensor_path,
    )

    with meta_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            record = {
                "index": i,
                "base_id": row.get("base_id"),
                "label": row.get("label"),
                "prediction": row.get("prediction"),
                "source": row.get("source"),
                "policy": row.get("policy"),
                "prompt_length": prompt_lengths[i],
                "hidden_state_shape": list(hidden_states[i].shape),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(dataset),
                "hidden_state_shape": list(hidden_states.shape),
                "layer_idx": args.layer_idx,
                "model_path": args.model_path,
                "dataset_path": str(dataset_path),
                "output_tensor": str(tensor_path),
                "output_metadata": str(meta_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved hidden states to: {tensor_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Tensor shape: {tuple(hidden_states.shape)}")


if __name__ == "__main__":
    main()
