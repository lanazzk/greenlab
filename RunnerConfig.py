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
    name:                       str             = "all_models_mvp"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 1000

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

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
        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:

        model = FactorModel(
            "model",
            [
                "llama3.2:3b-instruct-fp16",
                "llama3.2:3b-instruct-q4_K_M",

                "qwen2.5:3b-instruct-fp16",
                "qwen2.5:3b-instruct-q4_K_M",

                "gemma2:2b-instruct-fp16",
                "gemma2:2b-instruct-q4_K_M",
            ]
        )

        benchmark = FactorModel(
            "benchmark",
            ["mmlu_pro"]
        )

        seed = FactorModel(
            "seed",
            [101, 202]
        )

        self.run_table_model = RunTableModel(
            factors=[benchmark, seed, model],
            repetitions=1,
            data_columns=[
                "package_energy",
                "gpu_energy",
                "runtime",
                "gpu_memory",
                "ram"
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

    def start_measurement(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]
        benchmark = context.execute_run["benchmark"]
        seed = context.execute_run["seed"]

        task = f"inspect_evals/{benchmark}"

        inspect_cmd = [
            "inspect",
            "eval",
            task,
            "--model", f"ollama/{model}",
            "--limit", "2",
            "--temperature", "0",
            "--max-tokens", "512",
            "--max-connections", "1",
            "--sample-shuffle", str(seed),
            "--log-dir", "logs",
        ]

        energibridge_cmd = [
            "energibridge",
            "--gpu",
            "--summary",
            "--output",
            str(context.run_dir / "energibridge.csv"),
            "--command-output",
            str(context.run_dir / "inspect_console.txt"),
            "--",
        ] + inspect_cmd

        output.console_log(
            f"Measuring {model} | {benchmark} | seed={seed}"
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

        return_code = self.profiler.wait()

        if return_code != 0:
            raise RuntimeError(
                f"EnergyBridge/Inspect failed with return code {return_code}"
            )

    # def stop_measurement(self, context: RunnerContext) -> None:
    #     """Perform any activity here required for stopping measurements."""
    #     self.profiler.wait()
    def stop_measurement(self, context: RunnerContext) -> None:
        self.energibridge_log.close()


    def stop_run(self, context: RunnerContext) -> None:
        model = context.execute_run["model"]

        output.console_log(f"Unloading model: {model}")

        subprocess.run(
            ["ollama", "stop", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
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

        df = pd.read_csv(context.run_dir / "energibridge.csv")

        package_diff = df["PACKAGE_ENERGY (J)"].diff()

        package_energy = package_diff[package_diff > 0].sum()

        gpu_energy = (
            (df["GPU0_POWER (mWatts)"] / 1000.0)
            * (df["Delta"] / 1000.0)
        ).sum()

        runtime = df["Delta"].sum() / 1000.0

        gpu_memory = df["GPU0_MEMORY_USED"].max()

        ram = df["USED_MEMORY"].max() / (1024 ** 2)

        return {
            "package_energy": round(package_energy, 3),
            "gpu_energy": round(gpu_energy, 3),
            "runtime": round(runtime, 3),
            "gpu_memory": round(gpu_memory, 3),
            "ram": round(ram, 3),
        }


    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
