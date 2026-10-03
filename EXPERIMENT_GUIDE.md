# LLM quantization experiment

`RunnerConfig.py` is run through Experiment Runner and creates 480 measurements:
six model configurations, four benchmarks, and twenty repetitions. Its fixed seed
makes the shuffled run order reproducible. Experiment Runner stores completion in
`experiments/llm_quantization_main_480_max1024/run_table.csv`; rerunning the same command
continues unfinished rows.

Install dependencies on the SSH machine, then run:

```bash
pip install -r experiment-runner/requirements.txt -r requirements-experiment.txt
python3 experiment-runner/experiment-runner/ RunnerConfig.py
```

Start the Ollama server with its concurrency limits before starting the experiment.
For a manually started server:

```bash
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_CONTEXT_LENGTH=4096 ollama serve
```

If Ollama is managed by systemd, set those variables in its service environment and
restart that service instead. Setting them in the Experiment Runner process cannot
change an already-running Ollama server.

Each run warms the model, measures only the Inspect evaluation, stops the model,
and cools down for 120 seconds. Raw data is stored in the corresponding `run_*`
directory. After every completed batch, commit and push the experiment directory.

After the experiment, create a separate final result file. This leaves
`run_table.csv` unchanged for Experiment Runner's progress and resume logic:

```bash
python3 benchmark_scoring.py experiments/llm_quantization_main_480_max1024
```

The command writes `experiments/llm_quantization_main_480_max1024/result.csv`.

Before the full run, change `name` and `repetitions` in `RunnerConfig.py` to run a
separate two-repetition pilot. Restore the main name and twenty repetitions only
after verifying the pilot output.
