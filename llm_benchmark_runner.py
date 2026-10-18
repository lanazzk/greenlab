#!/usr/bin/env python3
"""Run one Inspect benchmark and save its metrics for Experiment Runner."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def value(obj: Any, name: str, default=None):
    """Read a field from either an Inspect object or a decoded mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def eval_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.rglob("*.eval"), key=lambda path: path.stat().st_mtime)


def extract_token_limit_hits(log: Any, max_tokens: int) -> tuple[int, int]:
    """Return the number of capped samples and the total sample count."""
    samples = list(value(log, "samples", []) or [])
    limit_stop_reasons = {
        "length",
        "max_tokens",
        "max_tokens_exceeded",
        "stop_length",
    }
    hits = 0

    for sample in samples:
        output = value(sample, "output")
        if output is None:
            continue

        usage = value(output, "usage")
        output_tokens = value(usage, "output_tokens")
        stop_reason = str(value(output, "stop_reason", value(output, "finish_reason", ""))).lower()

        reached_limit = stop_reason in limit_stop_reasons
        reached_limit |= (output_tokens is not None and int(output_tokens) >= max_tokens)
        hits += reached_limit

    return hits, len(samples)


def extract_metrics(log: Any) -> tuple[float, float, float]:
    """Extract total input tokens, output tokens, and benchmark accuracy."""
    input_tokens = 0.0
    output_tokens = 0.0

    for sample in value(log, "samples", []) or []:
        output = value(sample, "output")
        if output is None:
            continue

        usage = value(output, "usage")
        if value(usage, "input_tokens") is not None:
            input_tokens += float(value(usage, "input_tokens"))
        if value(usage, "output_tokens") is not None:
            output_tokens += float(value(usage, "output_tokens"))

    accuracy = None
    results = value(log, "results")
    for score in value(results, "scores", []) or []:
        for name, metric in value(score, "metrics", {}).items():
            if name.lower() in {"accuracy", "acc", "exact_match"}:
                accuracy = float(value(metric, "value", metric))
                break
        if accuracy is not None:
            break

    if accuracy is None:
        raise ValueError("Inspect log does not contain an accuracy metric")
    if 0 <= accuracy <= 1:
        accuracy *= 100

    return input_tokens, output_tokens, accuracy


def extract_eval_time(log: Any) -> float:
    """Return the duration between Inspect's start and completion timestamps."""
    stats = value(log, "stats")
    started = datetime.fromisoformat(str(value(stats, "started_at")).replace("Z", "+00:00"))
    completed = datetime.fromisoformat(str(value(stats, "completed_at")).replace("Z", "+00:00"))
    return (completed - started).total_seconds()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "inspect",
        "eval",
        args.task,
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
    return command


def run_inspect(args: argparse.Namespace, command: list[str]) -> tuple[int, float]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        text=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    return completed.returncode, time.monotonic() - started


def make_metrics(
    args: argparse.Namespace,
    command: list[str],
    returncode: int,
    run_time: float,
    log_path: Path,
) -> dict[str, object]:
    log = read_eval_log(str(log_path))
    input_tokens, output_tokens, accuracy = extract_metrics(log)
    token_limit_hits, token_limit_total = extract_token_limit_hits(log, args.max_tokens)
    inference_time = extract_eval_time(log)

    return {
        "benchmark": args.benchmark,
        "model": args.model,
        "sample_seed": args.seed,
        "sample_limit": args.limit,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "task_args": args.task_arg,
        "inspect_command": command,
        "inspect_returncode": returncode,
        "inference_time_s": round(inference_time, 6),
        "run_time_s": round(run_time, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "throughput_tokens_s": round(output_tokens / inference_time, 6),
        "eval_log": str(log_path),
        "accuracy": accuracy,
        "token_limit_rate": token_limit_hits / token_limit_total,
    }


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--task-arg", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args)
    returncode, run_time = run_inspect(args, command)
    logs = eval_logs(args.log_dir)
    if not logs:
        raise RuntimeError("Inspect did not create an .eval log")

    metrics = make_metrics(args, command, returncode, run_time, logs[-1])
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
