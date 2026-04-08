#!/usr/bin/env python
"""
Evaluate a sequence of continual-learning checkpoints on multiple benchmarks.

Assumes checkpoints are stored as:

    FOLDER/
      task_0/model/
      task_1/model/
      ...

Currently supported evaluations:
- evalplus (code benchmarks via the existing call_evalplus.py helper)
- gsm8k accuracy (via scripts/evaluate_model_accuracy.py)
- pubmedqa accuracy
- mmlu-style accuracy

Example:
    python scripts/eval_continual_sequence.py \\
        --root_dir ./runs/continual_20250301_deck \\
        --judge_model Qwen/Qwen2-7B-Instruct \\
        --evals evalplus gsm8k pubmedqa mmlu \\
        --output_dir ./runs/continual_20250301_deck/eval_results
"""

import argparse
import os
import subprocess
from datetime import datetime
from typing import List, Sequence


def discover_task_models(root_dir: str) -> List[str]:
    """
    Find task_* subfolders under root_dir and return their `model` subdirectories,
    ordered by task index.
    """
    task_dirs = []
    for entry in os.listdir(root_dir):
        if not entry.startswith("task_"):
            continue
        full = os.path.join(root_dir, entry)
        if not os.path.isdir(full):
            continue
        try:
            suffix = entry.split("_", 1)[1]
            task_id = int(suffix)
        except (IndexError, ValueError):
            # Ignore folders that don't follow task_<int>
            continue
        model_dir = os.path.join(full, "model")
        if os.path.isdir(model_dir):
            task_dirs.append((task_id, model_dir))
    # Sort by task id
    task_dirs.sort(key=lambda x: x[0])
    return [m for _, m in task_dirs]


def run_evalplus(
    model_paths: Sequence[str],
    datasets: Sequence[str],
    passk: Sequence[int],
    backend: str,
    greedy: bool,
    output_dir: str,
) -> None:
    """
    Run EvalPlus evaluation for each model and dataset combination.
    Results (stdout/stderr) are written to per-model log files in `output_dir`.
    """
    if not model_paths:
        print("[evalplus] No model paths to evaluate.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for model_dir in model_paths:
        model_name = os.path.basename(os.path.dirname(os.path.normpath(model_dir)))
        log_path = os.path.join(output_dir, f"evalplus_{model_name}.log")
        print(f"[evalplus] Evaluating model: {model_dir}")
        with open(log_path, "w", encoding="utf-8") as log_file:
            for dataset in datasets:
                cmd = [
                    "evalplus.evaluate",
                    "--model",
                    model_dir,
                    "--dataset",
                    dataset,
                    "--backend",
                    backend,
                ]
                if greedy:
                    cmd.append("--greedy")
                else:
                    cmd.extend(["--temperature", "0.8", "--n_samples", "200"])

                log_file.write(f"Running command: {' '.join(cmd)}\n")
                log_file.flush()
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                    process.wait()
                    log_file.write(
                        f"Command {' '.join(cmd)} completed with return code {process.returncode}\n"
                    )
                    log_file.flush()
                except Exception as e:
                    err_msg = f"Error running command {' '.join(cmd)}: {e}\n"
                    print(f"[evalplus] {err_msg.strip()}")
                    log_file.write(err_msg)
                    log_file.flush()


def run_accuracy_eval(
    model_paths: Sequence[str],
    dataset_name: str,
    dataset_subset: str,
    data_split: str,
    num_samples: int,
    judge_model: str,
    max_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    judge_gpu_memory_utilization: float,
    seed: int,
    output_dir: str,
) -> None:
    """
    Run accuracy evaluation for a list of model paths using the shared
    evaluate_models helper.
    """
    from scripts.evaluate_model_accuracy import evaluate_models

    if not model_paths:
        print(f"[{dataset_name}] No model paths to evaluate.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"[{dataset_name}] Evaluating {len(model_paths)} models on "
        f"{dataset_name} ({data_split} split, {num_samples} samples)."
    )

    evaluate_models(
        model_paths=model_paths,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        data_split=data_split,
        num_samples=num_samples,
        judge_model=judge_model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_file=output_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        judge_gpu_memory_utilization=judge_gpu_memory_utilization,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a continual-learning sequence of models on multiple benchmarks."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing task_<id>/model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store evaluation outputs. "
        "Default: <root_dir>/eval_results_<timestamp>.",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        choices=["evalplus", "gsm8k", "pubmedqa", "mmlu"],
        default=["evalplus", "gsm8k", "pubmedqa", "mmlu"],
        help="Which evaluations to run.",
    )
    parser.add_argument(
        "--evalplus_datasets",
        nargs="+",
        default=["humaneval", "mbpp"],
        help="EvalPlus datasets to evaluate (e.g., humaneval mbpp).",
    )
    parser.add_argument(
        "--evalplus_passk",
        nargs="+",
        type=int,
        default=[1],
        help="pass@k values to report for EvalPlus.",
    )
    parser.add_argument(
        "--evalplus_backend",
        type=str,
        default="vllm",
        help="EvalPlus backend (e.g., vllm or hf).",
    )
    parser.add_argument(
        "--evalplus_no_greedy",
        dest="evalplus_greedy",
        action="store_false",
        help="Disable greedy decoding flag for EvalPlus.",
    )
    parser.set_defaults(evalplus_greedy=True)
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Judge model for accuracy-based evaluations "
        "(required if gsm8k/pubmedqa/mmlu are requested).",
    )
    parser.add_argument(
        "--gsm8k_dataset_name",
        type=str,
        default="modelscope/gsm8k",
        help="ModelScope dataset name for GSM8K-style evaluation.",
    )
    parser.add_argument(
        "--pubmedqa_dataset_name",
        type=str,
        default="modelscope/pubmedqa",
        help="ModelScope dataset name for PubMedQA evaluation.",
    )
    parser.add_argument(
        "--pubmedqa_subset_name",
        type=str,
        default="pqa_artificial",
        help="Default subset to use for PubmedQA",
    )
    parser.add_argument(
        "--mmlu_dataset_name",
        type=str,
        default="modelscope/MMLU-Pro",
        help="ModelScope dataset name for MMLU(-Pro) evaluation.",
    )
    parser.add_argument(
        "--gsm8k_num_samples",
        type=int,
        default=200,
        help="Number of samples to evaluate on GSM8K (0 = all).",
    )
    parser.add_argument(
        "--pubmedqa_num_samples",
        type=int,
        default=200,
        help="Number of samples to evaluate on PubMedQA (0 = all).",
    )
    parser.add_argument(
        "--mmlu_num_samples",
        type=int,
        default=50,
        help="Number of samples per category for MMLU-style datasets (0 = all per category).",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        help="Data split to use for accuracy-based evaluations (train/test/val).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum generation tokens for accuracy-based evaluations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for accuracy-based evaluations.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.2,
        help="Fraction of GPU memory for evaluated models (0.0-1.0).",
    )
    parser.add_argument(
        "--judge_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="Fraction of GPU memory for the judge model (0.0-1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used across evaluations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = os.path.join(root_dir, f"eval_results_{timestamp}")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover continual-learning checkpoints
    model_paths = discover_task_models(root_dir)
    if not model_paths:
        raise RuntimeError(
            f"No task_<id>/model folders found under {root_dir}. "
            "Expected structure: root_dir/task_0/model, task_1/model, ..."
        )

    print(f"Discovered {len(model_paths)} task models:")
    for idx, m in enumerate(model_paths):
        print(f"  [task {idx}] {m}")

    # Sanity check for judge model when needed
    needs_judge = any(e in args.evals for e in ("gsm8k", "pubmedqa", "mmlu"))
    if needs_judge and not args.judge_model:
        raise ValueError(
            "--judge_model is required when running gsm8k/pubmedqa/mmlu evaluations."
        )

    # 1) EvalPlus
    if "evalplus" in args.evals:
        evalplus_out = os.path.join(output_dir, "evalplus")
        run_evalplus(
            model_paths=model_paths,
            datasets=args.evalplus_datasets,
            passk=args.evalplus_passk,
            backend=args.evalplus_backend,
            greedy=args.evalplus_greedy,
            output_dir=evalplus_out,
        )

    # 2) GSM8K
    if "gsm8k" in args.evals:
        gsm8k_out = os.path.join(output_dir, "gsm8k")
        run_accuracy_eval(
            model_paths=model_paths,
            dataset_name=args.gsm8k_dataset_name,
            dataset_subset=None,
            data_split=args.data_split,
            num_samples=args.gsm8k_num_samples,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
            seed=args.seed,
            output_dir=gsm8k_out,
        )

    # 3) PubMedQA
    if "pubmedqa" in args.evals:
        pubmedqa_out = os.path.join(output_dir, "pubmedqa")
        run_accuracy_eval(
            model_paths=model_paths,
            dataset_name=args.pubmedqa_dataset_name,
            dataset_subset=args.pubmedqa_subset_name,
            data_split="train",
            num_samples=args.pubmedqa_num_samples,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
            seed=args.seed,
            output_dir=pubmedqa_out,
        )

    # 4) MMLU
    if "mmlu" in args.evals:
        mmlu_out = os.path.join(output_dir, "mmlu")
        run_accuracy_eval(
            model_paths=model_paths,
            dataset_name=args.mmlu_dataset_name,
            dataset_subset=None,
            data_split=args.data_split,
            num_samples=args.mmlu_num_samples,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
            seed=args.seed,
            output_dir=mmlu_out,
        )

    print("\nAll requested evaluations completed.")
    print(f"Results written under: {output_dir}")


if __name__ == "__main__":
    main()

