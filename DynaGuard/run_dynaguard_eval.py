# """Run DynaGuard evaluation on a custom dataset.

# Example:
#     python evaluation/scripts/run_dynaguard_eval.py
# """

# from __future__ import annotations

# import argparse
# import subprocess
# import sys
# from pathlib import Path

# SCRIPT_PATH = Path(__file__).resolve()
# REPO_ROOT = SCRIPT_PATH.parents[2]
# EVAL_DIR = REPO_ROOT / "evaluation"

# def main() -> int:
#     parser = argparse.ArgumentParser(description="Evaluate DynaGuard on a custom dataset.")
#     parser.add_argument(
#         "--model_path",
#         default="tomg-group-umd/DynaGuard-4B",
#         help="Model path or Hugging Face id. Default: tomg-group-umd/DynaGuard-1.7B",
#     )
#     parser.add_argument(
#         "--dataset_path",
#         default=str(REPO_ROOT.parents[1] / "dynabench_format_output_v2.json"),
#         help="Path to the evaluation dataset.",
#     )
#     parser.add_argument(
#         "--output",
#         default=str(EVAL_DIR / "results" / "dynaguard_eval_results.json"),
#         help="Evaluation result JSON path.",
#     )
#     parser.add_argument(
#         "--max_samples",
#         type=int,
#         default=None,
#         help="Evaluate only the first N samples.",
#     )
#     args = parser.parse_args()

#     dataset_path = Path(args.dataset_path).resolve()
#     if not dataset_path.exists():
#         print(f"Error: Dataset not found at {dataset_path}")
#         return 1

#     output_path = Path(args.output).resolve()
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     cmd = [
#         sys.executable,
#         str(EVAL_DIR / "evaluate.py"),
#         "--model",
#         "dynaguard",
#         "--model_path",
#         args.model_path,
#         "--dataset",
#         "dynabench",
#         "--dataset_path",
#         str(dataset_path),
#         "--output",
#         str(output_path),
#         "--use_system_prompt",
#         "True",
#     ]
#     if args.max_samples is not None:
#         cmd.extend(["--max_samples", str(args.max_samples)])

#     print("DynaGuard Evaluation")
#     print(f"Model:      {args.model_path}")
#     print(f"Dataset:    {dataset_path}")
#     print(f"Output:     {output_path}")
#     print("\nCommand:")
#     print(subprocess.list2cmdline(cmd))

#     subprocess.run(cmd, cwd=str(EVAL_DIR), check=True)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
"""Run DynaGuard evaluation on a custom dataset.

Example:
    python evaluation/scripts/run_dynaguard_eval.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent.parent # The 'evaluation' directory
REPO_ROOT = EVAL_DIR.parent

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DynaGuard on a custom dataset.")
    parser.add_argument(
        "--model_path",
        default="tomg-group-umd/DynaGuard-8B",
        help="Model path or Hugging Face id. Default: tomg-group-umd/DynaGuard-8B",
    )
    parser.add_argument(
        "--dataset_path",
        default=str(REPO_ROOT.parents[1] / "proguard_text_multiturn.json"),
        help="Path to the evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(EVAL_DIR / "results" / "dynaguard_eval_results.json"),
        help="Evaluation result JSON path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return 1

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVAL_DIR / "evaluate.py"),
        "--model",
        "dynaguard",
        "--model_path",
        args.model_path,
        "--dataset",
        "dynabench",
        "--dataset_path",
        str(dataset_path),
        "--output",
        str(output_path),
        "--use_system_prompt",
        "True",
    ]
    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])

    print("DynaGuard Evaluation")
    print(f"Model:      {args.model_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Output:     {output_path}")
    print("\nCommand:")
    print(subprocess.list2cmdline(cmd))

    subprocess.run(cmd, cwd=str(EVAL_DIR), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
