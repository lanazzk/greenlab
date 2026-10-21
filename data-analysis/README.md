# Data analysis

This folder holds everything needed to reproduce the Results section of the report from the raw data of the final experiment (`experiments/llm_quantization_main_360_v3/`).

## Reproduce the analysis

Run both commands from the repository root:

```bash
# 1. Extract question-level correctness from the Inspect logs (zstd-compressed, so this step is in Python)
pip install pandas zipfile-zstd
python3 data-analysis/prepare_question_level.py

# 2. Run the statistical analysis and draw the figures
Rscript -e 'install.packages(c("dplyr", "tidyr", "readr", "ggplot2"))'   # first time only
Rscript data-analysis/analysis.R
```

Step 1 is optional: `data/question_level.csv` is already committed. Step 2 takes about 20 seconds and prints every number quoted in the report. The same numbers are saved to `output/summary.txt`.

## Folder layout

| Path | Content |
| --- | --- |
| `analysis.R` | Main analysis script: descriptive statistics, IQR outlier counts, Shapiro–Wilk, Levene (median-centred), Student/Welch t or Mann–Whitney U, Cohen's d / Cliff's δ, Benjamini–Hochberg per research question, exact McNemar with paired bootstrap CIs, Spearman with exact 9! permutation p-values, SCI |
| `prepare_question_level.py` | Builds `data/question_level.csv` from the Inspect `.eval` logs |
| `plot_sample.py` | Aggregates the raw `system_samples.csv` files into `data/sampler_100ms_summary.csv` |
| `plot_metrics.R` | Draws the exploratory plots (`01_runtime.png` … `06_metrics_table.png`) into `report_figures/` |
| `data/question_level.csv` | One row per run × question (36,000 rows): model, precision, benchmark, question id, target, answer, correct, output tokens |
| `data/sampler_100ms_summary.csv` | System sampler metrics (temperature, utilization, memory) in 100 ms bins per configuration |
| `output/tables/` | CSV tables written by `analysis.R` |
| `output/figures/` | PNG figures written by `analysis.R` |
| `output/summary.txt` | Numbers quoted in the report |
| `report_appendix/` | The detailed statistics appendix removed from the report: compiled PDF and the LaTeX tables |
| `report_figures/` | Every figure from the report folder as a PNG image, including the output of `plot_metrics.R` |

## Output tables

| File | Research question | Content |
| --- | --- | --- |
| `descriptive_statistics.csv` | all | n, median, mean, SD, min, max, variance per configuration and metric |
| `descriptive_statistics_rq1_table.csv` | RQ1 | Energy (kJ) and power (W) values of the descriptive table in the Results section |
| `rq1_tests.csv` | RQ1 | Energy and power: normality, variance, test, effect size, raw and BH p |
| `rq2_tests.csv` | RQ2 | Duration, throughput, CPU/GPU utilization, RAM, GPU memory |
| `change_ranges_by_benchmark.csv` | RQ1, RQ2 | Min/max relative change per benchmark |
| `rq3_1_accuracy_mcnemar.csv` | RQ3.1 | Accuracy, discordant pairs, bootstrap CI, exact McNemar raw and BH p |
| `rq3_2_3_spearman.csv` | RQ3.2, RQ3.3 | Spearman ρ, exact permutation p, BH p |
| `sci.csv` | SCI bonus | Energy, operational and embodied emissions, SCI per answer |

## Notes

- **Flagged run.** `run_5_repetition_19` (Gemma 2 FP16, RACE-H) recorded only 134 J of GPU energy, against about 6.1–6.2 kJ in the other repetitions, with identical accuracy, tokens and duration. It is a failed GPU energy reading. It is kept in the analysis, as in the report; the script prints it at the start.
- **Restarted runs.** Four run folders contain two Inspect logs because an attempt was interrupted and restarted. `prepare_question_level.py` skips the incomplete log and keeps the one from the measured run.
- **Bootstrap.** The accuracy confidence intervals use 20,000 resamples with seed 42, so they are reproducible.
- **Paths.** All paths are relative to the repository root; no absolute paths are used.
- **Exploratory plots.** `plot_metrics.R` produces the additional plots (`01_runtime.png` … `06_metrics_table.png`). The utilization and temperature plots read `data/sampler_100ms_summary.csv`, which `plot_sample.py` builds:

  ```bash
  python3 data-analysis/plot_sample.py        # optional, the CSV is committed
  Rscript -e 'install.packages(c("tidyverse", "ggh4x"))'   # first time only
  Rscript data-analysis/plot_metrics.R
  ```
