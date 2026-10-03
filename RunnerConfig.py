import json
import random
import sys
from datetime import datetime, timezone

from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath

import os
import signal
import pandas as pd
import time
import subprocess
import shlex

class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "llm_quantization_main_360"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 0

    # Fixed settings shared by every one of the 360 measured runs.
    sample_seed:                int             = 42
    sample_limit:               int             = 100
    cooldown_seconds:           int             = 120
    sampler_interval_ms:        int             = 100
    run_timeout_seconds:        int             = 7200

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        random.seed(self.sample_seed)

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later
        self.profiler = None
        self.sampler = None
        self.energibridge_log = None
        self.phase_metadata = {}
        output.console_log("Custom config loaded")

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
            ]
        )

        benchmark = FactorModel(
            "benchmark",
            ["mmlu", "mmlu_pro", "race_h"]
        )

        self.run_table_model = RunTableModel(
            factors=[model, benchmark],
            repetitions=20,
            shuffle=True,
            data_columns=[
                "model_family", "quantization", "cpu_energy_j", "gpu_energy_j",
                "total_energy_j", "average_power_w", "inference_time_s", "input_tokens",
                "output_tokens", "throughput_tokens_s", "cpu_utilization_pct",
                "gpu_utilization_pct", "peak_ram_mb", "peak_gpu_memory_mb", "accuracy",
                "warmup_time_s", "cooldown_time_s", "gpu_temp_start_c",
                "gpu_temp_end_c", "gpu_temp_mean_c", "gpu_temp_max_c"
            ]
        )

        return self.run_table_model
        

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        pass

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
            [
                "ollama",
                "run",
                model,
                "Reply only with OK."
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        self.phase_metadata["warmup_time_s"] = round(time.monotonic() - started, 3)
        self.phase_metadata["measurement_started_at"] = datetime.now(timezone.utc).isoformat()
        self._write_phase_metadata(context)

    def start_measurement(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]
        benchmark = context.execute_run["benchmark"]
        task_names = {"mmlu": "mmlu_0_shot", "mmlu_pro": "mmlu_pro", "race_h": "race_h"}
        benchmark_cmd = [
            sys.executable, str(self.ROOT_DIR / "llm_benchmark_runner.py"),
            "--model", model, "--task", f"inspect_evals/{task_names[benchmark]}",
            "--benchmark", benchmark, "--seed", str(self.sample_seed),
            "--limit", str(self.sample_limit), "--temperature", "0", "--max-tokens", "1024",
            "--num-ctx", "4096", "--output", str(context.run_dir / "llm_metrics.json"),
            "--log-dir", str(context.run_dir / "inspect_logs"),
        ]
        if benchmark == "mmlu_pro":
            benchmark_cmd.extend(["--task-arg", "fewshot=0", "--task-arg", "shuffle=false"])

        self.sampler = subprocess.Popen([
            sys.executable, str(self.ROOT_DIR / "system_sampler.py"),
            "--output", str(context.run_dir / "system_samples.csv"),
            "--interval-ms", str(self.sampler_interval_ms),
        ])

        energibridge_cmd = [
            "energibridge",
            "--gpu",
            "--summary",
            "--output",
            str(context.run_dir / "energibridge.csv"),
            "--command-output",
            str(context.run_dir / "inspect_console.txt"),
            "--",
        ] + benchmark_cmd

        output.console_log(
            f"Measuring {model} | {benchmark} | seed={self.sample_seed}"
        )

        self.energibridge_log = open(
            context.run_dir / "energibridge.log",
            "w"
        )

        self.profiler = subprocess.Popen(
            energibridge_cmd,
            stdout=self.energibridge_log,
            stderr=subprocess.STDOUT
        )


    # def interact(self, context: RunnerContext) -> None:
    #     """Perform any interaction with the running target system here, or block here until the target finishes."""

    #     # No interaction. We just run it for XX seconds.
    #     # Another example would be to wait for the target to finish, e.g. via `self.target.wait()`
    #     output.console_log("Running program for 20 seconds")
    #     time.sleep(20)
    
    def interact(self, context: RunnerContext) -> None:
        output.console_log("Waiting for Inspect + EnergyBridge to finish")

        return_code = self.profiler.wait(timeout=self.run_timeout_seconds)

        if return_code != 0:
            raise RuntimeError(
                f"EnergyBridge/Inspect failed with return code {return_code}"
            )

    # def stop_measurement(self, context: RunnerContext) -> None:
    #     """Perform any activity here required for stopping measurements."""
    #     self.profiler.wait()
    def stop_measurement(self, context: RunnerContext) -> None:
        self.phase_metadata["measurement_finished_at"] = datetime.now(timezone.utc).isoformat()
        if self.sampler is not None and self.sampler.poll() is None:
            self.sampler.terminate()
            self.sampler.wait(timeout=10)
        self.energibridge_log.close()
        self._write_phase_metadata(context)


    def stop_run(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]

        output.console_log(f"Unloading model: {model}")

        subprocess.run(
            ["ollama", "stop", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        started = time.monotonic()
        self.phase_metadata["cooldown_started_at"] = datetime.now(timezone.utc).isoformat()
        time.sleep(self.cooldown_seconds)
        self.phase_metadata["cooldown_time_s"] = round(time.monotonic() - started, 3)
        self.phase_metadata["cooldown_finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_phase_metadata(context)

    def _write_phase_metadata(self, context: RunnerContext) -> None:
        (context.run_dir / "phase_metadata.json").write_text(
            json.dumps(self.phase_metadata, indent=2), encoding="utf-8"
        )
    
    # def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
    #     """Parse and process any measurement data here.
    #     You can also store the raw measurement data under `context.run_dir`
    #     Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""

    #     # energibridge.csv - Power consumption of the whole system
    #     df = pd.read_csv(context.run_dir / f"energibridge.csv")
    #     run_data = {
    #         'dram_energy': round(df['DRAM_ENERGY (J)'].iloc[-1] - df['DRAM_ENERGY (J)'].iloc[0], 3),
    #         'package_energy': round(df['PACKAGE_ENERGY (J)'].iloc[-1] - df['PACKAGE_ENERGY (J)'].iloc[0], 3),
    #         'pp0_energy': round(df['PP0_ENERGY (J)'].iloc[-1] - df['PP0_ENERGY (J)'].iloc[0], 3),
    #         'pp1_energy': round(df['PP1_ENERGY (J)'].iloc[-1] - df['PP1_ENERGY (J)'].iloc[0], 3),
    #     }
    #     return run_data

    def populate_run_data(
        self,
        context: RunnerContext
    ) -> Optional[Dict[str, Any]]:

        energy_df = pd.read_csv(context.run_dir / "energibridge.csv")
        package_diff = energy_df["PACKAGE_ENERGY (J)"].diff()
        cpu_energy = float(package_diff[package_diff > 0].sum())
        gpu_energy = None
        if {"GPU0_POWER (mWatts)", "Delta"}.issubset(energy_df.columns):
            gpu_energy = float(((energy_df["GPU0_POWER (mWatts)"] / 1000.0)
                                * (energy_df["Delta"] / 1000.0)).sum())

        sampler_df = pd.read_csv(context.run_dir / "system_samples.csv")
        def mean(column):
            return float(sampler_df[column].dropna().mean()) if column in sampler_df and sampler_df[column].notna().any() else None
        def maximum(column):
            return float(sampler_df[column].dropna().max()) if column in sampler_df and sampler_df[column].notna().any() else None
        gpu_temperatures = sampler_df["gpu_temperature_c"].dropna() if "gpu_temperature_c" in sampler_df else pd.Series(dtype=float)
        if "gpu_energy_counter_mj" in sampler_df and sampler_df["gpu_energy_counter_mj"].notna().sum() > 1:
            counter = sampler_df["gpu_energy_counter_mj"].dropna()
            gpu_energy = float((counter.iloc[-1] - counter.iloc[0]) / 1000.0)

        llm_metrics = json.loads((context.run_dir / "llm_metrics.json").read_text(encoding="utf-8"))
        phase = json.loads((context.run_dir / "phase_metadata.json").read_text(encoding="utf-8"))
        inference_time = llm_metrics.get("inference_time_s")
        total_energy = cpu_energy + (gpu_energy or 0.0)
        model = context.execute_run["model"]
        return {
            "model_family": model.split(":", 1)[0],
            "quantization": "q4" if "q4" in model.lower() else "fp16",
            "cpu_energy_j": round(cpu_energy, 6),
            "gpu_energy_j": round(gpu_energy, 6) if gpu_energy is not None else None,
            "total_energy_j": round(total_energy, 6),
            "average_power_w": round(total_energy / inference_time, 6) if inference_time else None,
            "inference_time_s": inference_time,
            "input_tokens": llm_metrics.get("input_tokens"),
            "output_tokens": llm_metrics.get("output_tokens"),
            "throughput_tokens_s": llm_metrics.get("throughput_tokens_s"),
            "cpu_utilization_pct": mean("cpu_utilization_pct"),
            "gpu_utilization_pct": mean("gpu_utilization_pct"),
            "peak_ram_mb": maximum("ollama_rss_mb"),
            "peak_gpu_memory_mb": maximum("gpu_memory_used_mb"),
            "accuracy": llm_metrics.get("accuracy"),
            "warmup_time_s": phase.get("warmup_time_s"),
            "cooldown_time_s": phase.get("cooldown_time_s"),
            "gpu_temp_start_c": float(gpu_temperatures.iloc[0]) if len(gpu_temperatures) else None,
            "gpu_temp_end_c": float(gpu_temperatures.iloc[-1]) if len(gpu_temperatures) else None,
            "gpu_temp_mean_c": float(gpu_temperatures.mean()) if len(gpu_temperatures) else None,
            "gpu_temp_max_c": float(gpu_temperatures.max()) if len(gpu_temperatures) else None,
        }


    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
