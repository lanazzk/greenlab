# LLM Quantization Experiment

This Green Lab project compares FP16 and Q4 LLM configurations
to study how quantization affects accuracy, energy consumption,
inference performance, and resource usage.

The main experiment includes six model configurations, four
benchmarks, and twenty repetitions, resulting in 480 measurements.

## Tools

- Ollama for model inference
- Inspect AI for benchmark evaluation
- Experiment Runner for orchestration
- EnergiBridge for energy measurement
- NVML for GPU monitoring

## Setup and execution

See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for installation,
Ollama configuration, pilot testing, running and resuming
experiments, and calculating final results.

## Results

Raw measurements and experiment progress are stored in:

`experiments/llm_quantization_main_480_max1024/`

The scoring script creates `result.csv` in that directory.