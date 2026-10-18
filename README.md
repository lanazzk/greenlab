# Energy Efficiency after Model Compression

This project compares the energy use, execution time, system resources, and benchmark quality of six Ollama model configurations. Each configuration is evaluated on MMLU, MMLU-Pro, and RACE-H with a fixed seed so that comparable runs use the same sampled questions.

Detailed field definitions, formulas, timing boundaries, and limitations are documented in [REPORT.md](REPORT.md).

## What is measured

Each completed run records:

- CPU package energy, GPU energy, total energy, and average power
- `run_time_s` and Inspect `inference_time_s`
- Energy and power inside and outside the Inspect evaluation window
- Input/output tokens, output throughput, accuracy, and token-limit rate
- CPU/GPU utilization, Ollama RAM, and GPU memory
- CPU/GPU temperature statistics
- Model warm-up and cooldown durations

## Main experiment

The main configuration is [RunnerConfig.py](RunnerConfig.py):

- 6 model configurations × 3 benchmarks × 20 repetitions = 360 runs
- 100 questions per formal run
- Fixed seed: `42`
- Maximum output tokens: `1024`
- System sampler interval: `100 ms`
- 120-second cooldown between formal runs

Before formal data collection, the configuration performs the following pre-experiment sequence:

1. Measure idle CPU package and GPU energy for 120 seconds.
2. Run all 18 model/benchmark configurations once as dry runs, using 10 questions per configuration.
3. Stop each dry-run model and wait 30 seconds after each dry run.
4. Wait an additional 120 seconds after all dry runs.
5. Run a two-minute warm-up workload immediately before the first formal run.

Dry runs and the idle baseline are validation data, not formal benchmark results. Their raw files are stored under timestamped `dry_runs/` and `idle_baseline/` directories. The pre-experiment timestamps and directory names are accumulated in `pre_experiment_checks.json` at the experiment root.

## Requirements

Create the virtual environment and install the project and bundled Experiment Runner dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r experiment-runner/requirements.txt
```

The measurement machine also needs Ollama, Inspect AI, EnergiBridge, and an NVIDIA driver with NVML support.

## Start Ollama

Run one Ollama server with fixed concurrency settings:

```bash
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_CONTEXT_LENGTH=4096 \
ollama serve
```

## Run the main experiment

From the repository root:

```bash
python3 experiment-runner/experiment-runner/ RunnerConfig.py
```

The experiment directory is created under `experiments/<experiment-name>/`. It contains `run_table.csv`, `metadata.json`, `pre_experiment_checks.json`, timestamped idle/dry-run data, and one directory per formal run. A formal run directory normally contains:

```text
energibridge.csv
energibridge.log
system_samples.csv
inspect_console.txt
inspect_logs/
llm_metrics.json
phase_metadata.json
```

## Reproducibility

The benchmark command passes seed `42` to Inspect. MMLU uses seeded sample selection; MMLU-Pro and RACE-H additionally pass `shuffle=false`. Compare sample IDs in the generated `.eval` logs to verify that model runs use the same questions.

## Measurement boundaries

Each formal run loads and warms up its model before EnergiBridge starts the benchmark measurement. The model is stopped and cooldown occurs after measurement. `run_time_s` covers the complete Inspect subprocess, while `inference_time_s` comes from Inspect evaluation timestamps and may have only second-level precision. These boundaries affect energy-window, throughput, and power calculations; see [REPORT.md](REPORT.md) for details.

## Repository layout

```text
RunnerConfig.py              Main experiment configuration
llm_benchmark_runner.py      Runs Inspect and extracts benchmark metrics
system_sampler.py            Samples CPU, GPU, memory, energy, and temperature
experiment-runner/           Bundled Experiment Runner
experiments/                 Raw run data, dry runs, baselines, and tables
REPORT.md                    Field definitions, formulas, and limitations
requirements.txt             Project dependencies
```
