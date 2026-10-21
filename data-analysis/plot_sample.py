"""Aggregate raw system sampler data into 100 ms summaries.

The script reads every ``system_samples.csv`` below the experiment directory.
It writes to data-analysis/data/ by default (run from the repository root):

    python3 data-analysis/plot_sample.py

Outputs:
    sampler_100ms_summary.csv      mean/min/max/std per model/benchmark/quantization/bin
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = {
    "gpu_temperature_c": "gpu_temperature_c",
    "cpu_temperature_c": "cpu_temperature_c",
    "gpu_utilization_pct": "gpu_utilization_pct",
    "cpu_utilization_pct": "cpu_utilization_pct",
    "ollama_rss_mb": "peak_ram_mb",
    "gpu_memory_used_mb": "peak_gpu_memory_mb",
}

REQUIRED_SAMPLER_COLUMNS = {"monotonic_s", *METRICS}


def find_sampler_files(experiment_dir: Path) -> list[Path]:
    # Exclude idle baselines and other non-experimental sampler folders.
    run_pattern = re.compile(r"^run_\d+_repetition_\d+$")
    return sorted(
        path for path in experiment_dir.rglob("system_samples.csv")
        if run_pattern.match(path.parent.name)
    )


def aggregate_run(path: Path, bin_ms: int) -> pd.DataFrame:
    run_id = path.parent.name
    frame = pd.read_csv(path)
    missing = REQUIRED_SAMPLER_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["monotonic_s"] = pd.to_numeric(frame["monotonic_s"], errors="coerce")
    frame = frame.dropna(subset=["monotonic_s"])
    if frame.empty:
        return pd.DataFrame()

    frame["elapsed_ms"] = (frame["monotonic_s"] - frame["monotonic_s"].iloc[0]) * 1000
    frame["time_bin_ms"] = (frame["elapsed_ms"] / bin_ms).round().astype(int) * bin_ms

    numeric_metrics = list(METRICS)
    for column in numeric_metrics:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    result = frame.groupby("time_bin_ms", as_index=False)[numeric_metrics].agg("mean")
    result.insert(0, "run_id", run_id)
    result.insert(1, "source_file", str(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/llm_quantization_main_360_v3"),
        help="Directory containing run folders and system_samples.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data-analysis/data"))
    parser.add_argument("--bin-ms", type=int, default=100)
    args = parser.parse_args()
    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")

    files = find_sampler_files(args.experiment_dir)
    if not files:
        raise FileNotFoundError(f"No system_samples.csv files found under {args.experiment_dir}")

    per_run = pd.concat(
        [aggregate_run(path, args.bin_ms) for path in files], ignore_index=True
    ).rename(columns=METRICS)
    if per_run.empty:
        raise ValueError("No usable sampler rows were found.")

    run_table_path = args.experiment_dir / "run_table.csv"
    if not run_table_path.exists():
        raise FileNotFoundError(f"Missing run metadata file: {run_table_path}")
    metadata = pd.read_csv(run_table_path, usecols=[
        "__run_id", "model_family", "benchmark", "quantization"
    ]).rename(columns={"__run_id": "run_id"})
    per_run = per_run.merge(metadata, on="run_id", how="left", validate="many_to_one")
    if per_run[["model_family", "benchmark", "quantization"]].isna().any().any():
        raise ValueError("Some sampler runs could not be matched to run_table.csv")

    metric_columns = list(METRICS.values())
    group_columns = ["model_family", "benchmark", "quantization", "time_bin_ms"]
    summary = (
        per_run.groupby(group_columns, as_index=False)[metric_columns]
        .agg(["mean", "max", "min", "std", "count"])
    )
    summary.columns = [
        column if isinstance(column, str)
        else column[0] if column[1] in ("", None)
        else f"{column[0]}_{column[1]}"
        for column in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sampler_100ms_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Processed {len(files)} sampler files")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
