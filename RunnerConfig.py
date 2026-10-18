import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.OperationType import OperationType
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from EventManager.EventSubscriptionController import EventSubscriptionController
from EventManager.Models.RunnerEvents import RunnerEvents
from ProgressManager.Output.OutputProcedure import OutputProcedure as output


def counter_energy_j(series: pd.Series, units_per_joule: float = 1.0) -> float:
    differences = pd.to_numeric(series, errors="coerce").dropna().diff()
    return float(differences[differences > 0].sum() / units_per_joule)


class RunnerConfig:
    ROOT_DIR = Path(__file__).resolve().parent

    name = "llm_quantization_main_360_v3"
    results_output_path = ROOT_DIR / "experiments"
    operation_type = OperationType.AUTO
    time_between_runs_in_ms = 0

    sample_seed = 42

    sample_limit = 100
    cooldown_seconds = 120

    dry_run_sample_limit = 10
    dry_run_cooldown_seconds = 30
    post_dry_run_cooldown_seconds = 120

    sampler_interval_ms = 100
    run_timeout_seconds = 7200
    pre_experiment_warmup_seconds = 120
    idle_baseline_seconds = 120

    def __init__(self) -> None:
        random.seed(self.sample_seed)

        EventSubscriptionController.subscribe_to_multiple_events(
            [
                (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
                (RunnerEvents.START_RUN, self.start_run),
                (RunnerEvents.START_MEASUREMENT, self.start_measurement),
                (RunnerEvents.INTERACT, self.interact),
                (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
                (RunnerEvents.STOP_RUN, self.stop_run),
                (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            ]
        )
        self.run_table_model = None
        self.profiler = None
        self.sampler = None
        self.energibridge_log = None
        self.phase_metadata = {}
        output.console_log("Custom config loaded")

    def before_experiment(self) -> None:
        marker = self.experiment_path / "pre_experiment_checks.json"
        output.console_log("Running pre-experiment checks for this experiment start")

        models = [
            "gemma2:9b-instruct-q4_K_M",
            "gemma2:9b-instruct-fp16",
            "qwen2.5:7b-instruct-q4_K_M",
            "qwen2.5:7b-instruct-fp16",
            "llama3.1:8b-instruct-q4_K_M",
            "llama3.1:8b-instruct-fp16",
        ]
        benchmarks = {
            "mmlu": ("mmlu_0_shot", []),
            "mmlu_pro": ("mmlu_pro", ["fewshot=0", "shuffle=false"]),
            "race_h": ("race_h", ["shuffle=false"]),
        }

        output.console_log("Measuring idle baseline before dry runs")
        idle_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        idle_dir = self.experiment_path / "idle_baseline" / idle_timestamp
        idle_dir.mkdir(parents=True, exist_ok=True)
        sampler_output = idle_dir / "system_samples.csv"
        energy_output = idle_dir / "energibridge.csv"
        energy_log = idle_dir / "energibridge.log"
        console_output = idle_dir / "command_output.txt"
        sampler = subprocess.Popen([
            sys.executable, str(self.ROOT_DIR / "system_sampler.py"),
            "--output", str(sampler_output),
            "--interval-ms", str(self.sampler_interval_ms),
        ])
        idle_started = datetime.now(timezone.utc).isoformat()
        with energy_log.open("w", encoding="utf-8") as log:
            profiler = subprocess.Popen([
                "energibridge", "--gpu", "--summary",
                "--output", str(energy_output),
                "--command-output", str(console_output),
                "--", "sleep", str(self.idle_baseline_seconds),
            ], stdout=log, stderr=subprocess.STDOUT)
            return_code = profiler.wait(timeout=self.idle_baseline_seconds + 30)
        if sampler.poll() is None:
            sampler.terminate()
            sampler.wait(timeout=10)
        if return_code != 0:
            raise RuntimeError(f"Idle EnergiBridge baseline failed with return code {return_code}")
        idle_finished = datetime.now(timezone.utc).isoformat()

        output.console_log("Running 18 end-to-end dry runs")
        dry_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dry_dir = self.experiment_path / "dry_runs" / dry_timestamp
        dry_dir.mkdir(parents=True, exist_ok=True)
        dry_run_started = datetime.now(timezone.utc).isoformat()
        for model in models:
            for benchmark, (task_name, task_args) in benchmarks.items():
                run_name = f"{model.replace(':', '_')}_{benchmark}"
                run_dir = dry_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                output_path = run_dir / "llm_metrics.json"
                log_dir = run_dir / "inspect_logs"
                benchmark_cmd = [
                    sys.executable, str(self.ROOT_DIR / "llm_benchmark_runner.py"),
                    "--model", model,
                    "--task", f"inspect_evals/{task_name}",
                    "--benchmark", benchmark,
                    "--seed", str(self.sample_seed),
                    "--limit", str(self.dry_run_sample_limit),
                    "--temperature", "0",
                    "--max-tokens", "1024",
                    "--num-ctx", "4096",
                    "--output", str(output_path),
                    "--log-dir", str(log_dir),
                ]
                for task_arg in task_args:
                    benchmark_cmd.extend(["--task-arg", task_arg])
                sampler = subprocess.Popen([
                    sys.executable, str(self.ROOT_DIR / "system_sampler.py"),
                    "--output", str(run_dir / "system_samples.csv"),
                    "--interval-ms", str(self.sampler_interval_ms),
                ])
                with (run_dir / "energibridge.log").open("w", encoding="utf-8") as log:
                    profiler = subprocess.Popen([
                        "energibridge", "--gpu", "--summary",
                        "--output", str(run_dir / "energibridge.csv"),
                        "--command-output", str(run_dir / "inspect_console.txt"),
                        "--", *benchmark_cmd,
                    ], stdout=log, stderr=subprocess.STDOUT)
                    return_code = profiler.wait(timeout=self.run_timeout_seconds)
                if sampler.poll() is None:
                    sampler.terminate()
                    sampler.wait(timeout=10)
                if return_code != 0:
                    raise RuntimeError(f"Dry run failed for {model} / {benchmark}: return code {return_code}")
                subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                time.sleep(self.dry_run_cooldown_seconds)
        dry_run_finished = datetime.now(timezone.utc).isoformat()

        output.console_log(f"Cooling down for {self.post_dry_run_cooldown_seconds}s before final warm-up")
        time.sleep(self.post_dry_run_cooldown_seconds)
        output.console_log(f"Running {self.pre_experiment_warmup_seconds}s final warm-up")
        warmup_model = models[0]
        warmup_started = datetime.now(timezone.utc).isoformat()
        deadline = time.monotonic() + self.pre_experiment_warmup_seconds
        while time.monotonic() < deadline:
            subprocess.run(
                ["ollama", "run", warmup_model, "Read this short prompt and answer briefly: What is the correct option?"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        subprocess.run(["ollama", "stop", warmup_model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        warmup_finished = datetime.now(timezone.utc).isoformat()

        check_record = {
            "idle_baseline_dir": str(idle_dir.relative_to(self.experiment_path)),
            "dry_runs_dir": str(dry_dir.relative_to(self.experiment_path)),
            "idle_started_at": idle_started,
            "idle_finished_at": idle_finished,
            "idle_duration_s": self.idle_baseline_seconds,
            "dry_run_count": len(models) * len(benchmarks),
            "dry_runs_started_at": dry_run_started,
            "dry_runs_finished_at": dry_run_finished,
            "warmup_model": warmup_model,
            "warmup_started_at": warmup_started,
            "warmup_finished_at": warmup_finished,
            "warmup_duration_s": self.pre_experiment_warmup_seconds,
        }
        marker = self.experiment_path / "pre_experiment_checks.json"
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            records = previous if isinstance(previous, list) else [previous]
        else:
            records = []
        records.append(check_record)
        marker.write_text(json.dumps(records, indent=2), encoding="utf-8")
        output.console_log("Pre-experiment idle baseline, dry runs, and final warm-up completed")

    def create_run_table_model(self) -> RunTableModel:
        model = FactorModel(
            "model",
            [
                "gemma2:9b-instruct-q4_K_M",
                "gemma2:9b-instruct-fp16",
                "qwen2.5:7b-instruct-q4_K_M",
                "qwen2.5:7b-instruct-fp16",
                "llama3.1:8b-instruct-q4_K_M",
                "llama3.1:8b-instruct-fp16",
            ],
        )
        benchmark = FactorModel("benchmark", ["mmlu", "mmlu_pro", "race_h"])

        self.run_table_model = RunTableModel(
            factors=[model, benchmark],
            repetitions=20,
            shuffle=True,
            data_columns=[
                "model_family",
                "quantization",

                "cpu_energy_j",
                "gpu_energy_j",
                "total_energy_j",
                "run_time_s",
                "inference_time_s",
                "non_inference_energy_j",
                "inference_gpu_energy_j",
                "average_power_w",
                "average_gpu_power_w",
                "average_inference_power_w",
                "average_inference_gpu_power_w",

                "input_tokens",
                "output_tokens",
                "throughput_tokens_s",

                "cpu_utilization_pct",
                "gpu_utilization_pct",
                "peak_ram_mb",
                "peak_gpu_memory_mb",

                "warmup_time_s",
                "cooldown_time_s",

                "gpu_temp_start_c",
                "gpu_temp_end_c",
                "gpu_temp_mean_c",
                "gpu_temp_max_c",
                "cpu_temp_start_c",
                "cpu_temp_end_c",
                "cpu_temp_mean_c",
                "cpu_temp_max_c",

                "accuracy",
                "token_limit_rate",
            ],
        )

        return self.run_table_model

    def start_run(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]
        output.console_log(f"Loading/warming up model: {model}")
        started = time.monotonic()
        self.phase_metadata = {
            "run_id": context.execute_run["__run_id"],
            "model": model,
            "benchmark": context.execute_run["benchmark"],
            "warmup_started_at": datetime.now(timezone.utc).isoformat(),
        }
        subprocess.run(
            ["ollama", "run", model, "Reply only with OK."],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        self.phase_metadata["warmup_time_s"] = round(time.monotonic() - started, 3)
        self.phase_metadata["measurement_started_at"] = datetime.now(timezone.utc).isoformat()
        self._write_phase_metadata(context)

    def start_measurement(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]
        benchmark = context.execute_run["benchmark"]
        task_names = {
            "mmlu": "mmlu_0_shot",
            "mmlu_pro": "mmlu_pro",
            "race_h": "race_h",
        }
        benchmark_cmd = [
            sys.executable,
            str(self.ROOT_DIR / "llm_benchmark_runner.py"),
            "--model", model,
            "--task", f"inspect_evals/{task_names[benchmark]}",
            "--benchmark", benchmark,
            "--seed", str(self.sample_seed),
            "--limit", str(self.sample_limit),
            "--temperature", "0",
            "--max-tokens", "1024",
            "--num-ctx", "4096",
            "--output", str(context.run_dir / "llm_metrics.json"),
            "--log-dir", str(context.run_dir / "inspect_logs"),
        ]
        if benchmark == "mmlu_pro":
            benchmark_cmd.extend(["--task-arg", "fewshot=0", "--task-arg", "shuffle=false"])
        elif benchmark == "race_h":
            benchmark_cmd.extend(["--task-arg", "shuffle=false"])

        self.sampler = subprocess.Popen(
            [
                sys.executable,
                str(self.ROOT_DIR / "system_sampler.py"),
                "--output", str(context.run_dir / "system_samples.csv"),
                "--interval-ms", str(self.sampler_interval_ms),
            ]
        )

        energibridge_cmd = [
            "energibridge",
            "--gpu",
            "--summary",
            "--output", str(context.run_dir / "energibridge.csv"),
            "--command-output", str(context.run_dir / "inspect_console.txt"),
            "--",
            *benchmark_cmd,
        ]

        output.console_log(f"Measuring {model} | {benchmark} | seed={self.sample_seed}")
        self.energibridge_log = (context.run_dir / "energibridge.log").open("w", encoding="utf-8")

        self.profiler = subprocess.Popen(
            energibridge_cmd,
            stdout=self.energibridge_log,
            stderr=subprocess.STDOUT,
        )

    def interact(self, context: RunnerContext) -> None:
        output.console_log("Waiting for Inspect + EnergyBridge to finish")
        return_code = self.profiler.wait(timeout=self.run_timeout_seconds)
        if return_code != 0:
            raise RuntimeError(f"EnergyBridge/Inspect failed with return code {return_code}")

    def stop_measurement(self, context: RunnerContext) -> None:
        self.phase_metadata["measurement_finished_at"] = datetime.now(timezone.utc).isoformat()
        if self.sampler.poll() is None:
            self.sampler.terminate()
            self.sampler.wait(timeout=10)
        self.energibridge_log.close()
        self._write_phase_metadata(context)

    def stop_run(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]
        output.console_log(f"Unloading model: {model}")
        subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        started = time.monotonic()
        self.phase_metadata["cooldown_started_at"] = datetime.now(timezone.utc).isoformat()
        time.sleep(self.cooldown_seconds)
        self.phase_metadata["cooldown_time_s"] = round(time.monotonic() - started, 3)
        self.phase_metadata["cooldown_finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_phase_metadata(context)

    def _write_phase_metadata(self, context: RunnerContext) -> None:
        (context.run_dir / "phase_metadata.json").write_text(json.dumps(self.phase_metadata, indent=2), encoding="utf-8")

    def populate_run_data(self, context: RunnerContext) -> dict[str, object]:
        energy_df = pd.read_csv(context.run_dir / "energibridge.csv")
        sampler_df = pd.read_csv(context.run_dir / "system_samples.csv")
        llm_metrics = json.loads((context.run_dir / "llm_metrics.json").read_text(encoding="utf-8"))
        phase = json.loads((context.run_dir / "phase_metadata.json").read_text(encoding="utf-8"))

        cpu_energy = counter_energy_j(energy_df["PACKAGE_ENERGY (J)"])
        gpu_energy = counter_energy_j(sampler_df["gpu_energy_counter_mj"], units_per_joule=1000)
        total_energy = cpu_energy + gpu_energy

        eval_stats = read_eval_log(llm_metrics["eval_log"]).stats
        eval_start = pd.to_datetime(eval_stats.started_at, utc=True)
        eval_end = pd.to_datetime(eval_stats.completed_at, utc=True)

        sampler_df["timestamp"] = pd.to_datetime(sampler_df["timestamp_utc"], utc=True)
        energy_df["timestamp"] = pd.to_datetime(energy_df["Time"], unit="ms", utc=True)
        inside_sampler = sampler_df[sampler_df["timestamp"].between(eval_start, eval_end)]
        inside_energy = energy_df[energy_df["timestamp"].between(eval_start, eval_end)]

        cpu_inference_energy = counter_energy_j(inside_energy["PACKAGE_ENERGY (J)"])
        gpu_inference_energy = counter_energy_j(inside_sampler["gpu_energy_counter_mj"], units_per_joule=1000)
        inference_energy = cpu_inference_energy + gpu_inference_energy
        non_inference_energy = total_energy - inference_energy

        inference_time = llm_metrics["inference_time_s"]
        run_time = llm_metrics["run_time_s"]
        gpu_temperatures = sampler_df["gpu_temperature_c"].dropna()
        cpu_temperatures = sampler_df["cpu_temperature_c"].dropna()

        model = context.execute_run["model"]
        return {
            "model_family":                  model.split(":", 1)[0],
            "quantization":                  "q4" if "q4" in model.lower() else "fp16",

            "cpu_energy_j":                  round(cpu_energy, 6),
            "gpu_energy_j":                  round(gpu_energy, 6),
            "total_energy_j":                round(total_energy, 6),
            "run_time_s":                    run_time,
            "inference_time_s":              inference_time,
            "non_inference_energy_j":        round(non_inference_energy, 6),
            "inference_gpu_energy_j":        round(gpu_inference_energy, 6),
            "average_power_w":               round(total_energy / run_time, 6),
            "average_gpu_power_w":           round(gpu_energy / run_time, 6),
            "average_inference_power_w":     round(inference_energy / inference_time, 6),
            "average_inference_gpu_power_w": round(gpu_inference_energy / inference_time, 6),

            "input_tokens":                  llm_metrics["input_tokens"],
            "output_tokens":                 llm_metrics["output_tokens"],
            "throughput_tokens_s":           llm_metrics["throughput_tokens_s"],

            "cpu_utilization_pct":           float(sampler_df["cpu_utilization_pct"].dropna().mean()),
            "gpu_utilization_pct":           float(sampler_df["gpu_utilization_pct"].dropna().mean()),
            "peak_ram_mb":                   float(sampler_df["ollama_rss_mb"].dropna().max()),
            "peak_gpu_memory_mb":            float(sampler_df["gpu_memory_used_mb"].dropna().max()),

            "warmup_time_s":                 phase["warmup_time_s"],
            "cooldown_time_s":               phase["cooldown_time_s"],

            "gpu_temp_start_c":              float(gpu_temperatures.iloc[0]),
            "gpu_temp_end_c":                float(gpu_temperatures.iloc[-1]),
            "gpu_temp_mean_c":               float(gpu_temperatures.mean()),
            "gpu_temp_max_c":                float(gpu_temperatures.max()),
            "cpu_temp_start_c":              float(cpu_temperatures.iloc[0]),
            "cpu_temp_end_c":                float(cpu_temperatures.iloc[-1]),
            "cpu_temp_mean_c":               float(cpu_temperatures.mean()),
            "cpu_temp_max_c":                float(cpu_temperatures.max()),

            "accuracy":                      llm_metrics["accuracy"],
            "token_limit_rate":              llm_metrics["token_limit_rate"],
        }

    experiment_path: Path | None = None
