#!/usr/bin/env python3
"""Create a separate result.csv from Experiment Runner data and Inspect scores."""

import argparse
import csv
import json
from pathlib import Path

from llm_benchmark_runner import extract_metrics, extract_token_limit_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=Path,
                        help="Directory containing run_table.csv and run_* folders")
    args = parser.parse_args()
    table_path = args.experiment_path / "run_table.csv"
    result_path = args.experiment_path / "result.csv"
    with table_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("run_table.csv has no rows")
    columns = list(rows[0].keys())
    for column in ("accuracy", "token_limit_hits", "token_limit_total", "token_limit_rate"):
        if column not in columns:
            columns.append(column)
    for row in rows:
        metrics_path = args.experiment_path / row["__run_id"] / "llm_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            accuracy = metrics.get("accuracy")
            if accuracy is None and metrics.get("eval_log"):
                _, _, accuracy, _ = extract_metrics(Path(metrics["eval_log"]))
            if accuracy is not None:
                row["accuracy"] = accuracy
            hits = metrics.get("token_limit_hits")
            total = metrics.get("token_limit_total")
            if (hits is None or total is None) and metrics.get("eval_log"):
                hits, total = extract_token_limit_hits(Path(metrics["eval_log"]), int(metrics.get("max_tokens", 1024)))
            row["token_limit_hits"] = hits
            row["token_limit_total"] = total
            row["token_limit_rate"] = hits / total if hits is not None and total else None
    with result_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
