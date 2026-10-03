"""Smoke-test configuration: six model variants, four benchmarks, two repetitions."""

import importlib.util
import sys
from pathlib import Path

_base_path = Path(__file__).resolve().with_name("RunnerConfig.py")
_base_spec = importlib.util.spec_from_file_location("base_runner_config", _base_path)
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules["base_runner_config"] = _base_module
_base_spec.loader.exec_module(_base_module)
BaseRunnerConfig = _base_module.RunnerConfig
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunTableModel import RunTableModel


class RunnerConfig(BaseRunnerConfig):
    name = "llm_quantization_smoke_48"
    cooldown_seconds = 30

    SMOKE_MODELS = [
        "gemma2:9b-instruct-q4_K_M",
        "gemma2:9b-instruct-fp16",
        "qwen2.5:7b-instruct-q4_K_M",
        "qwen2.5:7b-instruct-fp16",
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.1:8b-instruct-fp16",
    ]

    def create_run_table_model(self) -> RunTableModel:
        model = FactorModel("model", self.SMOKE_MODELS)
        benchmark = FactorModel("benchmark", ["mmlu", "mmlu_pro", "race_h", "gsm8k"])
        self.run_table_model = RunTableModel(
            factors=[model, benchmark],
            repetitions=2,
            shuffle=True,
            data_columns=[
                "model_family", "quantization", "cpu_energy_j", "gpu_energy_j",
                "total_energy_j", "average_power_w", "inference_time_s", "input_tokens",
                "output_tokens", "throughput_tokens_s", "cpu_utilization_pct",
                "gpu_utilization_pct", "peak_ram_mb", "peak_gpu_memory_mb", "accuracy",
                "warmup_time_s", "cooldown_time_s", "gpu_temp_start_c",
                "gpu_temp_end_c", "gpu_temp_mean_c", "gpu_temp_max_c",
            ],
        )
        return self.run_table_model
