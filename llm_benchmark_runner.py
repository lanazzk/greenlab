#!/usr/bin/env python3
"""Run one Inspect benchmark and save LLM-specific metrics for Experiment Runner."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def value(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def numeric(item: Any) -> Optional[float]:
    if isinstance(item, bool):
        return float(item)
    if isinstance(item, (int, float)):
        return float(item)
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def eval_logs(log_dir: Path) -> Iterable[Path]:
    return sorted(log_dir.rglob("*.eval"), key=lambda path: path.stat().st_mtime)


def extract_token_limit_hits(log_path: Path, max_tokens: int):
    """Return (number hitting the output cap, number of samples)."""
    try:
        from inspect_ai.log import read_eval_log
        log = read_eval_log(str(log_path))
    except Exception:
        return None, None
    samples = list(value(log, "samples", []) or [])
    hits = 0
    for sample in samples:
        output = value(sample, "output")
        usage = value(output, "usage")
        output_tokens = numeric(value(usage, "output_tokens"))
        stop_reason = str(value(output, "stop_reason", value(output, "finish_reason", ""))).lower()
        if (output_tokens is not None and output_tokens >= max_tokens) or stop_reason in {
            "length", "max_tokens", "max_tokens_exceeded", "stop_length"
        }:
            hits += 1
    return hits, len(samples)


def extract_metrics(log_path: Path):
    """Read Inspect's structured log without relying on console formatting."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return None, None, None, "inspect_ai.log.read_eval_log is unavailable"
    try:
        log = read_eval_log(str(log_path))
    except Exception as exc:  # Keep raw Inspect files available for later inspection.
        return None, None, None, f"could not read {log_path.name}: {exc}"

    input_tokens = 0
    output_tokens = 0
    token_data_found = False
    score_values = []
    for sample in value(log, "samples", []) or []:
        output = value(sample, "output")
        usage = value(output, "usage")
        current_input = numeric(value(usage, "input_tokens"))
        current_output = numeric(value(usage, "output_tokens"))
        if current_input is not None:
            input_tokens += current_input
            token_data_found = True
        if current_output is not None:
            output_tokens += current_output
            token_data_found = True
        scores = value(sample, "scores", {}) or {}
        score_items = scores.values() if isinstance(scores, dict) else scores
        for score in score_items:
            score_value = numeric(value(score, "value", score))
            if score_value is not None:
                score_values.append(score_value)

    accuracy = sum(score_values) / len(score_values) if score_values else None
    # Inspect stores aggregate benchmark results under EvalLog.results.scores.
    results = value(log, "results")
    for score in value(results, "scores", []) or []:
        metrics = value(score, "metrics", {}) or {}
        items = metrics.items() if isinstance(metrics, dict) else []
        for metric_name, metric in items:
            if metric_name.lower() in {"accuracy", "acc", "exact_match"}:
                metric_value = numeric(value(metric, "value", metric))
                if metric_value is not None:
                    accuracy = metric_value
                    break
        if accuracy is not None:
            break
    if accuracy is not None and 0 <= accuracy <= 1:
        accuracy *= 100
    return (input_tokens if token_data_found else None,
            output_tokens if token_data_found else None,
            accuracy, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--num-ctx", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--task-arg", action="append", default=[],
                        help="Inspect task argument in name=value form; may be repeated")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # num_ctx is configured on the Ollama server with OLLAMA_CONTEXT_LENGTH.
    command = [
        "inspect", "eval", args.task,
        "--model", f"ollama/{args.model}",
        "--limit", str(args.limit),
        "--sample-shuffle", str(args.seed),
        "--max-connections", "1",
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
        "--max-retries", "0",
        "--log-dir", str(args.log_dir),
    ]
    for task_arg in args.task_arg:
        command.extend(["-T", task_arg])
    started = time.monotonic()
    completed = subprocess.run(command, text=True, stdout=sys.stdout, stderr=sys.stderr, check=False)
    inference_time = time.monotonic() - started
    logs = list(eval_logs(args.log_dir))
    input_tokens = output_tokens = accuracy = None
    extraction_error = None
    token_limit_hits = token_limit_total = None
    if logs:
        input_tokens, output_tokens, accuracy, extraction_error = extract_metrics(logs[-1])
        token_limit_hits, token_limit_total = extract_token_limit_hits(logs[-1], args.max_tokens)
    else:
        extraction_error = "Inspect did not create an .eval log"

    metrics = {
        "benchmark": args.benchmark,
        "model": args.model,
        "sample_seed": args.seed,
        "sample_limit": args.limit,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "task_args": args.task_arg,
        "inspect_command": command,
        "inspect_returncode": completed.returncode,
        "inference_time_s": round(inference_time, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "throughput_tokens_s": round(output_tokens / inference_time, 6) if output_tokens is not None and inference_time else None,
        "accuracy": accuracy,
        "token_limit_hits": token_limit_hits,
        "token_limit_total": token_limit_total,
        "token_limit_rate": (token_limit_hits / token_limit_total
                              if token_limit_hits is not None and token_limit_total else None),
        "eval_log": str(logs[-1]) if logs else None,
        "extraction_error": extraction_error,
    }
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
