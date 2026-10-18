## Execution Workflow for Each Experiment

Each time `RunnerConfig.py` is launched, the experiment proceeds through pre-experiment checks, formal measurements, and cleanup, in that order. The following describes one complete execution.

### 1. Idle Baseline

Before formal measurements begin, the program first records a 120-second idle baseline. No benchmark questions are submitted during this period. Only `system_sampler.py` and EnergiBridge are started to record CPU package energy, GPU energy, utilization, memory usage, and temperatures. The data are saved under `idle_baseline/<timestamp>/`, with a UTC timestamp in the directory name, and are not written to the official `run_table.csv`.

### 2. Dry Runs

After the idle baseline, the program performs 18 dry runs, covering 6 models × 3 benchmarks. Each configuration evaluates only 10 questions and follows the same Inspect, Ollama, EnergiBridge, and system sampler workflow used for formal measurements. These runs verify that the entire pipeline can execute successfully and provide estimates of execution time. After each dry run, the model is stopped and the program waits for 30 seconds. Once all 18 dry runs are complete, the program waits for an additional 120 seconds. Raw dry-run data are stored under `dry_runs/<timestamp>/` and are not written to the official `run_table.csv`.

After each set of pre-experiment procedures is completed, the program writes `pre_experiment_checks.json` to the experiment root directory. This file does not contain benchmark results; it records the pre-experiment procedures, including the timestamps, durations, and data directories for the idle baseline, all 18 dry runs, and the final warm-up. Records from multiple executions are accumulated in this file, making it possible to verify that the preparation steps were completed before each formal experiment.

### 3. Warm-Up Before Formal Measurements

After the dry runs and the final 120-second cooldown, the program continuously sends short Ollama requests to a fixed model for 120 seconds. The generated outputs are discarded and are not included in the benchmark results. The purpose of this stage is to bring the CPU, GPU, and model execution environment to more stable operating temperatures before the first formal run.

### 4. Each Formal Run

Each formal run begins by loading the specified model and sending a single dummy prompt, with the elapsed time recorded as `warmup_time_s`. Only after the model warm-up is complete does the program start EnergiBridge and the system sampler, and then use Inspect to execute the specified benchmark. When measurement is complete, the results are written to `run_table.csv`; see the next section for details.

### 5. Between Formal Runs and at the End of the Experiment

After a formal run is completed, the program stops the current model and waits for a 120-second cooldown before starting the next run. Once the final run is complete, the Experiment Runner writes the completion status to the run table. The raw data for each run are retained in its individual run directory, including `llm_metrics.json`, `system_samples.csv`, `energibridge.csv`, `phase_metadata.json`, and Inspect logs.

## `run_table.csv` Column Definitions and Calculation Details

- `__run_id`
  - Meaning: The run identifier assigned by the Experiment Runner. Each row in `run_table.csv` corresponds to the result of one execution for one model × one benchmark.

- `__done`
  - Meaning: The completion status of the run; `DONE` indicates normal completion.

- `model`
  - Meaning: The full Ollama model name, including the model tag, to identify the actual model being tested.

- `benchmark`
  - Meaning: The benchmark executed in this run, which may be `mmlu`, `mmlu_pro`, or `race_h`.
  - Question sampling configuration: Question selection within the same benchmark is jointly controlled by `sample_seed`, the question limit (`limit`), and the task configuration. MMLU uses `sample_shuffle=42`; MMLU-Pro and RACE-H additionally set `shuffle=false`.

- `model_family`
  - Calculation: Take the substring of `model` before the colon (`:`); for example, `qwen2.5:...` maps to `qwen2.5`.

- `quantization`
  - Calculation: If the `model` name contains `q4`, label it `q4`; otherwise, label it `fp16`.

- `cpu_energy_j`
  - Source: EnergiBridge's `PACKAGE_ENERGY (J)` counter.
  - Calculation: `Σ max(difference between two consecutive PACKAGE_ENERGY readings, 0)`, in J.
  - Note: If the counter resets, wraps around, or decreases, the negative difference is not accumulated. Therefore, this is not equivalent to subtracting the first reading from the last reading. The EnergiBridge sampling frequency is determined by EnergiBridge and defaults to 200 ms; it must not be assumed to match the 100 ms interval of the system sampler (described below).

- `gpu_energy_j`
  - Source: `gpu_energy_counter_mj`, obtained by `system_sampler.py` through `pynvml`.
  - Calculation: `Σ max(difference between two consecutive gpu_energy_counter_mj readings, 0) / 1000`, converting mJ to J.
  - Shared sampling mechanism: `system_sampler.py` is an independent background process that samples every 100 ms by default and writes the data to `system_samples.csv`.
  - At each sampling point, it queries the operating system or driver for the current values. CPU-related information primarily comes from `psutil`, while GPU-related information comes from `pynvml`. Each record includes both UTC and monotonic timestamps.
  - EnergiBridge samples approximately every 200 ms, whereas the system sampler samples every 100 ms. The two processes do not start at exactly the same time, so their data points do not align perfectly. Considering sampler alignment alone, the timing difference at a single endpoint can be up to approximately 200 ms, or approximately 400 ms across the start and end endpoints combined. The actual inference window still uses Inspect timestamps. Because Inspect records timestamps only to second-level precision (see `inference_time_s`), each endpoint may have a quantization error approaching 1 second, making the upper bound on the timing error for the entire window approximately 2 seconds.
  - The sampling interval does not mean that every sensor provides the same temporal resolution. Although the program reads the data every 100 ms, the hardware or driver may update its values less frequently. The driver may also return cached values, so several consecutive readings may be identical, or no new valid data may be available temporarily. Frequent polling does not guarantee a new hardware measurement on every read.
  - Shared impact: The sampling mechanisms and missing data described above affect `inference_gpu_energy_j`, `cpu_utilization_pct`, `gpu_utilization_pct`, `peak_ram_mb`, `peak_gpu_memory_mb`, and all `gpu_temp_` and `cpu_temp_` columns. The restriction to positive GPU counter differences also applies to `inference_gpu_energy_j`.

- `total_energy_j`
  - Formula: `total_energy_j = cpu_energy_j + gpu_energy_j`.
  - Interpretation: The sum of CPU package and GPU energy consumption over the measurement interval, not the complete energy consumption of every hardware component in the entire system.

- `run_time_s`
  - Source/calculation: `llm_benchmark_runner.py` uses a monotonic clock to measure the elapsed time, in seconds, from the start to the end of the entire `inspect eval` subprocess.
  - Coverage: This may include dataset loading, Inspect initialization, model requests, result processing, and file writing. It is not the time spent exclusively on model inference.
  - Measurement boundaries: Model warm-up is performed before EnergiBridge measurement, and cooldown is performed after measurement. Therefore, model loading and warm-up, as well as the subsequent cooldown, are excluded from `run_time_s`, `total_energy_j`, and the inference energy window. These boundaries also affect the interpretation of benchmark energy consumption in `average_power_w`, `average_gpu_power_w`, `non_inference_energy_j`, and related columns below.
  - Cross-source time alignment: Inspect uses UTC timestamps; the system sampler retains both UTC and monotonic timestamps; EnergiBridge's Unix epoch timestamps in milliseconds are converted to UTC when read; and the orchestrator uses UTC metadata and monotonic durations. Currently, cross-source windows are aligned using UTC, while durations are calculated using a monotonic clock. This alignment method also affects `inference_time_s`, `inference_gpu_energy_j`, `non_inference_energy_j`, and all inference power metrics.
  - Variability and outliers: Even with the same model and benchmark, differences in output token counts, CPU/GPU scheduling, background load, Ollama server state, and sampling boundaries can lead to different execution times. When investigating outliers, examine the token counts, Inspect `.eval` logs, timestamps, and `system_samples.csv` together. The same factors may also affect energy consumption and average power columns.

- `inference_time_s`
  - Source: `stats.started_at` and `stats.completed_at` in the Inspect `.eval` log.
  - Formula: `inference_time_s = completed_at - started_at`.
  - Interpretation: This is the duration of the Inspect evaluation window, not the precise request interval from sending the first prompt to receiving the last answer, nor the time spent exclusively on token generation. The current implementation does not record these two high-precision timestamps at the Ollama request layer.
  - Shared limitation: If the `.eval` log provides timestamps only at second-level precision, the start and end boundaries of short benchmarks may introduce substantial errors. This limitation directly affects `throughput_tokens_s`, `average_inference_power_w`, and `average_inference_gpu_power_w`. Errors in window segmentation also affect `inference_gpu_energy_j` and `non_inference_energy_j`.
  - Reason for the precision limitation: Increasing the number of decimal places in `round` only changes the display format; it does not add information to the original timestamps. Higher precision would require recording high-precision monotonic timestamps at the Ollama request layer for when the first prompt is sent and the last answer is received. The current implementation does not contain this layer of event data. Directly substituting the high-precision duration of the entire subprocess would incorrectly count dataset loading, Inspect initialization, and result processing as inference, so this approach is not currently used. Ollama requests are created and managed internally by Inspect, and the existing outer runner has no hook to directly intercept each request. Adding this capability would require modifying the Inspect/Ollama adapter or introducing middleware, with explicit handling of streaming answers, retries, concurrent requests, and request failures. The interception itself could also alter measurement timing and benchmark behavior. Therefore, request boundaries that cannot be verified are not currently included as official columns.

- `non_inference_energy_j`
  - Calculation: First, calculate `inference_energy = cpu_inference_energy + gpu_inference_energy` within the Inspect evaluation window; then calculate `non_inference_energy_j = total_energy_j - inference_energy`.
  - Interpretation: This is the energy remaining after subtracting the energy consumed within the Inspect evaluation window from the total measured benchmark energy. It is not a separate measurement of dataset-loading energy.
  - Note: It may include dataset loading, Inspect initialization, result processing, model startup latency, and errors from cross-source alignment and sampling boundaries. It must not be interpreted directly as pure dataset-loading energy. In short smoke tests, fixed costs such as data loading, initialization, and request setup may account for a relatively large share. Consequently, the energy proportions observed with a small number of questions cannot be directly extrapolated to a full benchmark. This limitation also affects comparisons of `run_time_s`, `average_power_w`, and similar columns across tests.

- `inference_gpu_energy_j`
  - Source/calculation: Accumulate only the positive GPU energy counter differences from GPU sampler records within the Inspect `[started_at, completed_at]` window, then convert the result to J.
  - Interpretation: This represents GPU energy consumption within the evaluation window, not `gpu_energy_j` for the entire run. It is also subject to the sampler resolution and UTC window-alignment limitations described above.

- `average_power_w`
  - Formula: `average_power_w = total_energy_j / run_time_s`, in W.
  - Interpretation: The average combined CPU package and GPU power during the entire `inspect eval` subprocess, not the average power of the entire system.

- `average_gpu_power_w`
  - Formula: `average_gpu_power_w = gpu_energy_j / run_time_s`, in W.
  - Interpretation: Average GPU power over the entire benchmark run, not limited to the Inspect evaluation window.

- `average_inference_power_w`
  - Formula: `average_inference_power_w = inference_energy / inference_time_s`, where `inference_energy = cpu_inference_energy + gpu_inference_energy`.
  - Interpretation: The average combined CPU package and GPU power within the Inspect evaluation window, not power used exclusively for token generation. It is affected by window boundaries and timestamp precision.

- `average_inference_gpu_power_w`
  - Formula: `average_inference_gpu_power_w = inference_gpu_energy_j / inference_time_s`, in W.
  - Interpretation: Average GPU power within the Inspect evaluation window, affected by GPU counter sampling boundaries and the precision of `inference_time_s`.

- `input_tokens`
  - Source/calculation: Sum the input token counts reported for individual samples in the Inspect `.eval` log.
  - Shared output limitation: The current benchmark command sets `max_tokens=1024`. A model may generate explanations before giving an answer, but it must finish within this limit and comply with the scorer's parsing rules. Reaching the token limit does not automatically count as a correct answer. This setting, together with the model's response format, affects `output_tokens`, `accuracy`, and `token_limit_rate` below, as well as time, throughput, and energy metrics indirectly through the output length.

- `output_tokens`
  - Source/calculation: Sum the output token counts reported for individual samples in the Inspect `.eval` log.
  - Note: Output length may vary because a model generates explanations, fails to follow the required answer format, or reaches `max_tokens`. Token counts alone cannot establish whether answers are correct.

- `throughput_tokens_s`
  - Formula: `throughput_tokens_s = output_tokens / inference_time_s`, in output tokens/s.
  - Interpretation: Output token throughput calculated using the Inspect evaluation window; input tokens are not included. This is also not a precise measurement of pure generation speed at the Ollama layer.

- `accuracy`
  - Source/calculation: Read the `accuracy`, `acc`, or exact-match score from the Inspect `.eval` log. If the original value is in `[0, 1]`, convert it to a percentage; for example, `0.5 → 50.0`.
  - Scoring rules: The Inspect scorer parses answers according to the benchmark format. Excessive explanation, failure to provide a parseable option, an answer appearing in an unexpected position, or output reaching the length limit may prevent an otherwise correct answer from being scored successfully. Correctness is determined by the scorer's result.
  - Note: This is the accuracy for the questions sampled in this run and may not represent accuracy on the full benchmark. When the sample is small, a single question can cause a large percentage-point change. The effect of the model's output format on this column may also be reflected in `token_limit_rate`.

- `token_limit_rate`
  - Formula: `token_limit_rate = token_limit_hits / token_limit_total`.
  - Hit criterion: A sample counts as one hit if its stop reason indicates that the length limit was reached, or if its output token count is `>= max_tokens` (currently 1024).
  - Interpretation/limitations: This rate is inferred from Inspect's stop reason and token usage. Incomplete reporting may cause some hits to be missed. A model that fails to follow the requested answer format may also continue generating until it reaches the limit. Therefore, interpret this metric together with `accuracy`, `output_tokens`, and the `.eval` log.

- `cpu_utilization_pct`
  - Source/calculation: `system_sampler.py` obtains CPU utilization samples during the run through `psutil`, then calculates their mean.
  - Shared limitation: System utilization, memory, and temperature metrics are statistics based on discrete sampling points, not continuous monitoring without gaps. The sampling interval, run duration, and missing data affect the mean and maximum values. This also applies to `gpu_utilization_pct`, `peak_ram_mb`, `peak_gpu_memory_mb`, and all `gpu_temp_` and `cpu_temp_` columns below. The run-level sampling statistics in these columns exclude warm-up and cooldown.

- `gpu_utilization_pct`
  - Source/calculation: The system sampler obtains GPU utilization samples during the run through `pynvml`, then calculates their mean.
  - Note: This reflects average utilization at the sampled points, not the fraction of the entire time interval during which the GPU was continuously busy.

- `peak_ram_mb`
  - Source/calculation: The system sampler uses `psutil` to scan for processes whose names or command lines contain `ollama`, sums their resident set sizes (RSS), and takes the maximum observed during the run, in MB.
  - Note: RSS represents system RAM used by Ollama-related processes, not GPU VRAM. The result is affected by process detection and sampling times.

- `peak_gpu_memory_mb`
  - Source/calculation: The system sampler uses `pynvml` to read GPU memory usage and takes the maximum observed during the run, in MB.
  - Note: This is the sampled peak of GPU memory usage and is not equivalent to the RSS of Ollama processes.

- `warmup_time_s`
  - Source/calculation: The time, in seconds, spent warming up the model in `start_run`.
  - Interpretation: This is part of the cost of the overall experimental workflow, but it occurs before EnergiBridge benchmark measurement begins. Therefore, it is not included in `run_time_s`, `total_energy_j`, or the inference energy window.

- `cooldown_time_s`
  - Source/calculation: The time, in seconds, spent waiting for cooldown after stopping the model.
  - Interpretation: This is part of the cost of the overall experimental workflow, but it occurs after measurement ends. Therefore, it is not included in `run_time_s`, `total_energy_j`, or the inference energy window.

- `gpu_temp_start_c`
  - Source/calculation: The first valid GPU temperature reading obtained by the system sampler through `pynvml`, in °C.
  - Shared limitation: Temperatures reflect only valid sensor readings that were obtained. Sampling frequency, missing values, and the measurement window affect the interpretation of `gpu_temp_end_c`, `gpu_temp_mean_c`, and `gpu_temp_max_c` below; these values must not be treated as complete extrema from continuous monitoring.

- `gpu_temp_end_c`
  - Source/calculation: The last valid GPU temperature reading during the run, in °C.

- `gpu_temp_mean_c`
  - Source/calculation: The arithmetic mean of all valid GPU temperature samples during the run, in °C.

- `gpu_temp_max_c`
  - Source/calculation: The maximum of all valid GPU temperature samples during the run, in °C.

- `cpu_temp_start_c`
  - Source/calculation: The system sampler reads available hardware sensors through `psutil.sensors_temperatures()` and takes the first valid CPU temperature reading during the run, in °C.
  - Shared limitation: The names and availability of CPU temperature sensors depend on the operating system and hardware; sensor sources may differ across environments. This also affects `cpu_temp_end_c`, `cpu_temp_mean_c`, and `cpu_temp_max_c` below. Their statistics are likewise limited by the number of valid samples and the sampling boundaries.

- `cpu_temp_end_c`
  - Source/calculation: The last valid CPU temperature reading during the run, in °C.

- `cpu_temp_mean_c`
  - Source/calculation: The arithmetic mean of all valid CPU temperature samples during the run, in °C.

- `cpu_temp_max_c`
  - Source/calculation: The maximum of all valid CPU temperature samples during the run, in °C.
