"""Extract question-level correctness from the Inspect .eval logs.

Inspect stores its logs as zstd-compressed zip files, which R cannot read
without extra system libraries, so this small step is done in Python.

Usage (from the repository root):
    pip install zipfile-zstd pandas
    python3 data-analysis/prepare_question_level.py

Output:
    data-analysis/data/question_level.csv
      one row per run x question: run_id, model, benchmark, sample_id,
      target, answer, correct (1/0), output_tokens
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import zipfile_zstd  # noqa: F401  registers zstd support in zipfile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install zipfile-zstd first: pip install zipfile-zstd") from exc

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "llm_quantization_main_360_v3"
OUT = Path(__file__).resolve().parent / "data" / "question_level.csv"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def select_log(run_dir: Path) -> Path:
    """Pick the log written during the run's measurement window.

    A few run folders hold two logs because an attempt was interrupted and
    restarted. Incomplete logs (no header.json) are skipped; if two complete
    logs remain, the one whose evaluation started inside the recorded
    measurement window belongs to the measured run.
    """
    logs = sorted((run_dir / "inspect_logs").glob("*.eval"))
    # Logs of interrupted attempts have no header.json and are skipped.
    complete = [log for log in logs if "header.json" in zipfile.ZipFile(log).namelist()]
    if len(complete) == 1:
        return complete[0]
    meta = json.loads((run_dir / "phase_metadata.json").read_text())
    start = parse_time(meta["measurement_started_at"])
    end = parse_time(meta["measurement_finished_at"])
    for log in complete:
        header = json.loads(zipfile.ZipFile(log).read("header.json"))
        started = parse_time(header["stats"]["started_at"])
        if start <= started <= end:
            return log
    raise RuntimeError(f"No log inside the measurement window for {run_dir.name}")


def main() -> None:
    run_table = pd.read_csv(EXP / "run_table.csv").rename(columns={"__run_id": "run_id"})
    rows = []
    for run in run_table.itertuples():
        run_dir = EXP / run.run_id
        log = select_log(run_dir)
        samples = json.loads(zipfile.ZipFile(log).read("summaries.json"))
        for s in samples:
            score = next(iter(s["scores"].values()))
            usage = next(iter(s.get("model_usage", {}).values()), {})
            rows.append({
                "run_id": run.run_id,
                "model": run.model,
                "model_family": run.model_family,
                "quantization": run.quantization,
                "benchmark": run.benchmark,
                "sample_id": s["id"],
                "target": s["target"],
                "answer": score.get("answer"),
                "correct": int(score["value"] == "C"),
                "output_tokens": usage.get("output_tokens"),
                "log_file": log.name,
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
