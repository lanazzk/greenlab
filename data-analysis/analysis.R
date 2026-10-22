# =============================================================================
# Energy Efficiency after Model Compression -- main analysis script (Team 5)
#
# Reproduces every number, table and figure of the Results section of the
# report from the raw data in this repository.
#
# Run from the repository root:
#     Rscript data-analysis/analysis.R
#
# Inputs (relative paths, no absolute paths):
#   experiments/llm_quantization_main_360_v3/run_table.csv   run-level metrics
#   data-analysis/data/question_level.csv                    per-question scores
#     (created by data-analysis/prepare_question_level.py from the Inspect logs)
#
# Outputs:
#   data-analysis/output/tables/*.csv     descriptive statistics, tests, ...
#   data-analysis/output/figures/*.png    distribution plots, accuracy plot
#   data-analysis/output/summary.txt      the numbers quoted in the report
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})

set.seed(42)

RUN_TABLE <- "experiments/llm_quantization_main_360_v3/run_table.csv"
QUESTIONS <- "data-analysis/data/question_level.csv"
OUT_DIR   <- "data-analysis/output"
TAB_DIR   <- file.path(OUT_DIR, "tables")
FIG_DIR   <- file.path(OUT_DIR, "figures")
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

ALPHA <- 0.05
N_BOOT <- 20000

MODEL_LABELS <- c(llama3.1 = "Llama 3.1 8B", qwen2.5 = "Qwen 2.5 7B", gemma2 = "Gemma 2 9B")
BENCH_LABELS <- c(mmlu = "MMLU", mmlu_pro = "MMLU-Pro", race_h = "RACE-H")
PREC_LABELS  <- c(fp16 = "FP16", q4 = "Q4_K_M")

RQ1_METRICS <- c("total_energy_j", "average_power_w")
RQ2_METRICS <- c("inference_time_s", "throughput_tokens_s", "cpu_utilization_pct",
                 "gpu_utilization_pct", "peak_ram_mb", "peak_gpu_memory_mb")
METRIC_LABELS <- c(
  total_energy_j = "Total energy (J)", average_power_w = "Average power (W)",
  inference_time_s = "Evaluation duration (s)", throughput_tokens_s = "Output throughput (tokens/s)",
  cpu_utilization_pct = "CPU utilization (%)", gpu_utilization_pct = "GPU utilization (%)",
  peak_ram_mb = "Peak host RAM (MB)", peak_gpu_memory_mb = "Peak GPU memory (MB)"
)
DESCRIPTIVE_METRICS <- c(RQ1_METRICS, RQ2_METRICS, "run_time_s", "output_tokens",
                         "token_limit_rate", "accuracy", "cpu_temp_start_c", "cpu_temp_mean_c",
                         "cpu_temp_max_c", "cpu_temp_end_c", "gpu_temp_start_c",
                         "gpu_temp_mean_c", "gpu_temp_max_c", "gpu_temp_end_c")

summary_lines <- character()
say <- function(...) {
  line <- paste0(...)
  summary_lines <<- c(summary_lines, line)
  cat(line, "\n")
}

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
runs <- read_csv(RUN_TABLE, show_col_types = FALSE) %>%
  rename(run_id = `__run_id`, done = `__done`) %>%
  filter(done == "DONE") %>%
  mutate(
    model_label = factor(MODEL_LABELS[model_family], levels = MODEL_LABELS),
    bench_label = factor(BENCH_LABELS[benchmark], levels = BENCH_LABELS),
    prec_label  = factor(PREC_LABELS[quantization], levels = PREC_LABELS)
  )

questions <- read_csv(QUESTIONS, show_col_types = FALSE)

say("Runs: ", nrow(runs), " (", n_distinct(paste(runs$model, runs$benchmark)),
    " configurations); question records: ", nrow(questions))

# Flagged run: GPU energy reading failed (kept in the analysis, see report Sec. 5)
flagged <- runs %>%
  group_by(model, benchmark) %>%
  mutate(gpu_ratio = gpu_energy_j / median(gpu_energy_j)) %>%
  ungroup() %>%
  filter(gpu_ratio < 0.5)
say("Runs with GPU energy < 50% of their configuration median: ",
    paste(flagged$run_id, flagged$model, flagged$benchmark, round(flagged$gpu_energy_j, 1), "J",
          collapse = "; "))

# -----------------------------------------------------------------------------
# 2. Descriptive statistics (median, mean, SD, min, max, variance; n - 1)
# -----------------------------------------------------------------------------
descriptives <- runs %>%
  select(model_label, bench_label, prec_label, all_of(DESCRIPTIVE_METRICS)) %>%
  pivot_longer(all_of(DESCRIPTIVE_METRICS), names_to = "metric", values_to = "value") %>%
  group_by(metric, model_label, bench_label, prec_label) %>%
  summarise(n = n(), median = median(value), mean = mean(value), sd = sd(value),
            min = min(value), max = max(value), variance = var(value), .groups = "drop") %>%
  arrange(metric, model_label, bench_label, prec_label)
write_csv(descriptives, file.path(TAB_DIR, "descriptive_statistics.csv"))

# Compact table used in the Results section (energy in kJ, power in W)
desc_rq1 <- descriptives %>%
  filter(metric %in% RQ1_METRICS) %>%
  mutate(across(c(median, mean, sd, min, max), ~ ifelse(metric == "total_energy_j", .x / 1000, .x)),
         variance = ifelse(metric == "total_energy_j", variance / 1e6, variance))
write_csv(desc_rq1, file.path(TAB_DIR, "descriptive_statistics_rq1_table.csv"))

# Outliers by the 1.5 x IQR rule (reported, never removed)
iqr_flags <- function(metrics) {
  runs %>%
    select(model, benchmark, quantization, all_of(metrics)) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(metric, model, benchmark) %>%
    mutate(q1 = quantile(value, 0.25), q3 = quantile(value, 0.75), iqr = q3 - q1,
           outlier = value < q1 - 1.5 * iqr | value > q3 + 1.5 * iqr) %>%
    ungroup() %>%
    summarise(n = sum(outlier)) %>%
    pull(n)
}
say("IQR-flagged run-metric observations: RQ1 = ", iqr_flags(RQ1_METRICS),
    ", RQ2 = ", iqr_flags(RQ2_METRICS), " (all retained)")

# -----------------------------------------------------------------------------
# 3. Helper functions for the continuous comparisons
# -----------------------------------------------------------------------------
shapiro_check <- function(x) {
  if (length(unique(x)) == 1) return(list(W = NA, p = NA, status = "invalid (constant)"))
  t <- shapiro.test(x)
  list(W = unname(t$statistic), p = t$p.value,
       status = ifelse(t$p.value < ALPHA, "rejected", "not rejected"))
}

# Levene's test centred on the median (Brown--Forsythe variant)
levene_median <- function(a, b) {
  z <- c(abs(a - median(a)), abs(b - median(b)))
  g <- factor(rep(c("a", "b"), c(length(a), length(b))))
  fit <- anova(lm(z ~ g))
  list(F = fit$`F value`[1], p = fit$`Pr(>F)`[1])
}

cohens_d <- function(q4, fp) {
  n1 <- length(q4); n2 <- length(fp)
  sp <- sqrt(((n1 - 1) * var(q4) + (n2 - 1) * var(fp)) / (n1 + n2 - 2))
  (mean(q4) - mean(fp)) / sp
}

cliffs_delta <- function(q4, fp) {
  cmp <- outer(q4, fp, "-")
  (sum(cmp > 0) - sum(cmp < 0)) / length(cmp)
}

compare_block <- function(fp, q4) {
  s_fp <- shapiro_check(fp); s_q4 <- shapiro_check(q4)
  both_normal <- s_fp$status == "not rejected" && s_q4$status == "not rejected"
  lev <- list(F = NA, p = NA)
  if (both_normal) {
    lev <- levene_median(fp, q4)
    welch <- lev$p < ALPHA
    tt <- t.test(q4, fp, var.equal = !welch)
    test <- ifelse(welch, "Welch t", "Student t")
    p <- tt$p.value; effect_name <- "Cohen's d"; effect <- cohens_d(q4, fp)
  } else {
    wt <- suppressWarnings(wilcox.test(q4, fp, exact = FALSE, correct = TRUE))
    test <- "Mann-Whitney U"
    p <- wt$p.value; effect_name <- "Cliff's delta"; effect <- cliffs_delta(q4, fp)
  }
  tibble(fp16_mean = mean(fp), q4_mean = mean(q4),
         change_pct = (mean(q4) - mean(fp)) / mean(fp) * 100,
         shapiro_fp16_W = s_fp$W, shapiro_fp16_p = s_fp$p, shapiro_fp16 = s_fp$status,
         shapiro_q4_W = s_q4$W, shapiro_q4_p = s_q4$p, shapiro_q4 = s_q4$status,
         levene_F = lev$F, levene_p = lev$p,
         test = test, effect_name = effect_name, effect = effect, p_raw = p)
}

run_family <- function(metrics, family) {
  res <- runs %>%
    select(model_family, model_label, benchmark, bench_label, quantization, all_of(metrics)) %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    group_by(metric, model_label, bench_label) %>%
    group_modify(~ compare_block(.x$value[.x$quantization == "fp16"],
                                 .x$value[.x$quantization == "q4"])) %>%
    ungroup() %>%
    mutate(family = family, p_bh = p.adjust(p_raw, method = "BH"), reject = p_bh < ALPHA) %>%
    arrange(factor(metric, levels = metrics), model_label, bench_label)
  res
}

describe_family <- function(res, label) {
  checks <- c(res$shapiro_fp16, res$shapiro_q4)
  say(label, ": Shapiro-Wilk checks = ", length(checks),
      " (rejected ", sum(checks == "rejected"), ", not rejected ", sum(checks == "not rejected"),
      ", constant ", sum(grepl("invalid", checks)), "); Levene run in ", sum(!is.na(res$levene_p)),
      " comparisons, rejected in ", sum(res$levene_p < ALPHA, na.rm = TRUE))
  say(label, ": tests used -> ", paste(names(table(res$test)), table(res$test), collapse = ", "),
      "; significant after BH: ", sum(res$reject), "/", nrow(res))
  for (m in unique(res$metric)) {
    r <- res %>% filter(metric == m)
    say("  ", m, ": mean change ", sprintf("%.2f", min(r$change_pct)), " to ",
        sprintf("%.2f", max(r$change_pct)), " %; significant ", sum(r$reject), "/9")
  }
}

# -----------------------------------------------------------------------------
# 4. RQ1 and RQ2: FP16 vs Q4 within each model-benchmark block
# -----------------------------------------------------------------------------
rq1 <- run_family(RQ1_METRICS, "RQ1")
rq2 <- run_family(RQ2_METRICS, "RQ2")
write_csv(rq1, file.path(TAB_DIR, "rq1_tests.csv"))
write_csv(rq2, file.path(TAB_DIR, "rq2_tests.csv"))
describe_family(rq1, "RQ1")
describe_family(rq2, "RQ2")

# Per-benchmark ranges quoted in the text
ranges <- bind_rows(rq1, rq2) %>%
  group_by(metric, bench_label) %>%
  summarise(min_change_pct = min(change_pct), max_change_pct = max(change_pct), .groups = "drop")
write_csv(ranges, file.path(TAB_DIR, "change_ranges_by_benchmark.csv"))

# -----------------------------------------------------------------------------
# 5. RQ3.1: accuracy, exact McNemar per block, paired bootstrap CI
# -----------------------------------------------------------------------------
# Answers are identical across the 20 repetitions; verify, then keep one per question.
consistency <- questions %>%
  group_by(model, benchmark, sample_id) %>%
  summarise(k = n_distinct(correct), .groups = "drop")
say("Questions whose correctness differs between repetitions: ", sum(consistency$k > 1))
say("Blank predicted answers: ", sum(is.na(questions$answer)))

per_question <- questions %>%
  group_by(model_family, quantization, benchmark, sample_id) %>%
  summarise(correct = as.integer(mean(correct) >= 0.5), .groups = "drop") %>%  # majority answer
  pivot_wider(names_from = quantization, values_from = correct)

accuracy <- per_question %>%
  group_by(model_family, benchmark) %>%
  group_modify(function(d, key) {
    b <- sum(d$fp16 == 1 & d$q4 == 0)   # FP-only correct
    c <- sum(d$fp16 == 0 & d$q4 == 1)   # Q4-only correct
    p <- if (b + c == 0) 1 else binom.test(b, b + c, 0.5)$p.value
    diff <- d$q4 - d$fp16
    boot <- replicate(N_BOOT, mean(sample(diff, replace = TRUE))) * 100
    tibble(n = nrow(d), fp16_acc = mean(d$fp16) * 100, q4_acc = mean(d$q4) * 100,
           change_pp = mean(diff) * 100, fp_only = b, q4_only = c,
           ci_low = unname(quantile(boot, 0.025)), ci_high = unname(quantile(boot, 0.975)),
           p_raw = p)
  }) %>%
  ungroup() %>%
  mutate(p_bh = p.adjust(p_raw, method = "BH"), reject = p_bh < ALPHA,
         model_label = factor(MODEL_LABELS[model_family], levels = MODEL_LABELS),
         bench_label = factor(BENCH_LABELS[benchmark], levels = BENCH_LABELS)) %>%
  arrange(model_label, bench_label)
write_csv(accuracy, file.path(TAB_DIR, "rq3_1_accuracy_mcnemar.csv"))
say("RQ3.1: accuracy change ", min(accuracy$change_pp), " to ", max(accuracy$change_pp),
    " pp; significant after BH: ", sum(accuracy$reject), "/9; smallest raw p = ",
    signif(min(accuracy$p_raw), 4))

# -----------------------------------------------------------------------------
# 6. RQ3.2 / RQ3.3: Spearman correlations with exact 9! permutation p-values
# -----------------------------------------------------------------------------
all_permutations <- function(n) {
  perms <- matrix(1L, 1, 1)
  for (k in 2:n) {
    m <- nrow(perms)
    perms <- do.call(rbind, lapply(1:k, function(pos) {
      cbind(perms[, seq_len(pos - 1), drop = FALSE], rep(k, m),
            perms[, seq_len(k - 1) >= pos, drop = FALSE])
    }))
  }
  perms
}
PERMS <- all_permutations(9)

spearman_exact <- function(x, y) {
  rx <- rank(x); ry <- rank(y)
  rho <- cor(rx, ry)
  cx <- rx - mean(rx); cy <- ry - mean(ry)
  denom <- sqrt(sum(cx^2) * sum(cy^2))
  rho_perm <- as.vector(matrix(cy[PERMS], nrow(PERMS)) %*% cx) / denom
  list(rho = rho, p = mean(abs(rho_perm) >= abs(rho) - 1e-12))
}

block_changes <- bind_rows(rq1, rq2) %>%
  select(metric, model_label, bench_label, change_pct) %>%
  left_join(accuracy %>% select(model_label, bench_label, change_pp),
            by = c("model_label", "bench_label"))

tradeoffs <- block_changes %>%
  group_by(metric) %>%
  summarise(rho = spearman_exact(change_pct, change_pp)$rho,
            p_raw = spearman_exact(change_pct, change_pp)$p, .groups = "drop") %>%
  mutate(family = ifelse(metric %in% RQ1_METRICS, "RQ3.2", "RQ3.3")) %>%
  group_by(family) %>%
  mutate(p_bh = p.adjust(p_raw, method = "BH")) %>%
  ungroup() %>%
  arrange(family, factor(metric, levels = c(RQ1_METRICS, RQ2_METRICS)))
write_csv(tradeoffs, file.path(TAB_DIR, "rq3_2_3_spearman.csv"))
for (i in seq_len(nrow(tradeoffs))) {
  say(tradeoffs$family[i], " ", tradeoffs$metric[i], ": rho = ", sprintf("%.3f", tradeoffs$rho[i]),
      ", BH p = ", sprintf("%.4f", tradeoffs$p_bh[i]))
}

# -----------------------------------------------------------------------------
# 7. SCI (Software Carbon Intensity) per answered question
# -----------------------------------------------------------------------------
I_GRID   <- 273.8          # gCO2e/kWh, Netherlands 2025 (Electricity Maps)
EMBODIED <- 180000 + 120000 # gCO2e, workstation (Dell PCF) + RTX 4090 (Gupta et al. extrapolation)
LIFETIME_H <- 4 * 365 * 24

sci <- function(d) {
  e_kwh <- sum(d$total_energy_j) / 3.6e6
  hours <- sum(d$run_time_s) / 3600
  answers <- nrow(d) * 100
  tibble(energy_kwh = e_kwh, operational_g = e_kwh * I_GRID,
         embodied_g = EMBODIED * hours / LIFETIME_H, answers = answers,
         sci_mg_per_answer = (e_kwh * I_GRID + EMBODIED * hours / LIFETIME_H) / answers * 1000)
}
sci_table <- bind_rows(
  runs %>% group_by(benchmark, quantization) %>% group_modify(~ sci(.x)) %>% ungroup(),
  runs %>% group_by(quantization) %>% group_modify(~ sci(.x)) %>% ungroup() %>% mutate(benchmark = "all"),
  sci(runs) %>% mutate(benchmark = "all", quantization = "all")
)
write_csv(sci_table, file.path(TAB_DIR, "sci.csv"))
tot <- sci_table %>% filter(benchmark == "all", quantization == "all")
say("SCI: E = ", sprintf("%.3f", tot$energy_kwh), " kWh, O = ", sprintf("%.1f", tot$operational_g),
    " g, M = ", sprintf("%.1f", tot$embodied_g), " g, SCI = ",
    sprintf("%.2f", tot$sci_mg_per_answer), " mg CO2e per answer")

# -----------------------------------------------------------------------------
# 8. Figures
# -----------------------------------------------------------------------------
QUANT_COLORS <- c("FP16" = "#7FB3D5", "Q4_K_M" = "#F5B78C")

for (m in c(RQ1_METRICS, RQ2_METRICS)) {
  p <- ggplot(runs, aes(x = prec_label, y = .data[[m]], fill = prec_label)) +
    geom_boxplot(outlier.shape = NA, width = 0.5) +
    geom_jitter(width = 0.08, size = 0.8, alpha = 0.7) +
    facet_wrap(bench_label ~ model_label, scales = "free_y", ncol = 3) +
    scale_fill_manual(values = QUANT_COLORS) +
    labs(x = NULL, y = METRIC_LABELS[[m]], fill = NULL,
         caption = "20 runs per configuration. Boxes: median and IQR; points: individual runs. Y-axes differ per panel.") +
    theme_bw(base_size = 11) + theme(legend.position = "none")
  for (ext in c("png", "pdf")) {
    ggsave(file.path(FIG_DIR, paste0("distribution_", m, ".", ext)), p, width = 9, height = 7, dpi = 200)
  }
}

p_acc <- accuracy %>%
  mutate(block = factor(paste(model_label, bench_label, sep = " - "),
                        levels = rev(paste(model_label, bench_label, sep = " - ")))) %>%
  ggplot(aes(x = change_pp, y = block)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.25) +
  geom_point(size = 2) +
  labs(x = "Q4 - FP16 accuracy change (percentage points)", y = NULL,
       caption = "Paired bootstrap 95% CIs (20,000 resamples, pointwise). Significance: BH-adjusted exact McNemar.") +
  theme_bw(base_size = 11)
for (ext in c("png", "pdf")) {
  ggsave(file.path(FIG_DIR, paste0("accuracy_change_ci.", ext)), p_acc, width = 8, height = 5, dpi = 200)
}

# Q-Q plots behind the Shapiro-Wilk checks (one page per metric)
qq_dir <- file.path(FIG_DIR, "qq")
dir.create(qq_dir, showWarnings = FALSE)
for (m in c(RQ1_METRICS, RQ2_METRICS)) {
  p_qq <- ggplot(runs, aes(sample = .data[[m]], colour = prec_label)) +
    stat_qq(size = 0.8) + stat_qq_line() +
    facet_wrap(bench_label ~ model_label, scales = "free", ncol = 3) +
    scale_colour_manual(values = c("FP16" = "#2E86C1", "Q4_K_M" = "#E67E22")) +
    labs(x = "Theoretical quantiles", y = METRIC_LABELS[[m]], colour = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")
  ggsave(file.path(qq_dir, paste0("qq_", m, ".png")), p_qq, width = 9, height = 7, dpi = 150)
}

# -----------------------------------------------------------------------------
# 9. LaTeX descriptive tables used in the Results section
# -----------------------------------------------------------------------------
latex_desc_table <- function(metrics, scales, units, digits, label, caption) {
  fmt <- function(x, d) formatC(x, format = "f", digits = d)
  wide <- lapply(seq_along(metrics), function(i) {
    descriptives %>% filter(metric == metrics[i]) %>%
      mutate(across(c(median, mean, sd, min, max), ~ .x / scales[i]),
             variance = variance / scales[i]^2) %>%
      transmute(model_label, bench_label, prec_label,
                cells = paste(fmt(median, digits[i]), fmt(mean, digits[i]), fmt(sd, digits[i]),
                              fmt(min, digits[i]), fmt(max, digits[i]), fmt(variance, digits[i]),
                              sep = " & "))
  })
  rows <- wide[[1]] %>% rename(c1 = cells) %>%
    left_join(wide[[2]] %>% rename(c2 = cells), by = c("model_label", "bench_label", "prec_label")) %>%
    arrange(model_label, bench_label, prec_label)
  body <- character()
  for (i in seq_len(nrow(rows))) {
    r <- rows[i, ]
    first_model <- i == 1 || rows$model_label[i] != rows$model_label[i - 1]
    if (first_model && i > 1) body <- c(body, "\\midrule")
    model <- if (first_model) as.character(r$model_label) else ""
    bench <- if (r$prec_label == "FP16") as.character(r$bench_label) else ""
    prec <- gsub("_", "\\_", as.character(r$prec_label), fixed = TRUE)
    body <- c(body, paste0(model, " & ", bench, " & ", prec, " & ", r$c1, " & ", r$c2, " \\\\"))
  }
  c("\\begin{table*}[t]", "\\centering", "\\small",
    paste0("\\caption{", caption, "}"), paste0("\\label{", label, "}"),
    "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllrrrrrrrrrrrr}", "\\toprule",
    paste0(" & & & \\multicolumn{6}{c}{", units[1], "} & \\multicolumn{6}{c}{", units[2], "} \\\\"),
    "\\cmidrule(lr){4-9}\\cmidrule(lr){10-15}",
    "Model & Benchmark & Precision & Median & Mean & SD & Min & Max & Var & Median & Mean & SD & Min & Max & Var \\\\",
    "\\midrule", body, "\\bottomrule", "\\end{tabular}", "}", "\\end{table*}")
}
latex_tables <- list(
  rq1 = latex_desc_table(RQ1_METRICS, c(1000, 1), c("Total energy (kJ)", "Average power (W)"), c(2, 1),
                         "tab:results-desc-rq1", "Descriptive statistics of total energy (kJ) and average power (W) per configuration, $n=20$ runs each."),
  rq2_time = latex_desc_table(c("inference_time_s", "throughput_tokens_s"), c(1, 1),
                              c("Evaluation duration (s)", "Output throughput (tokens/s)"), c(2, 2),
                              "tab:results-desc-rq2-time", "Descriptive statistics of evaluation duration (s) and output throughput (tokens/s) per configuration, $n=20$ runs each."),
  rq2_util = latex_desc_table(c("cpu_utilization_pct", "gpu_utilization_pct"), c(1, 1),
                              c("CPU utilization (\\%)", "GPU utilization (\\%)"), c(2, 2),
                              "tab:results-desc-rq2-util", "Descriptive statistics of mean CPU and GPU utilization (\\%) per configuration, $n=20$ runs each."),
  rq2_mem = latex_desc_table(c("peak_ram_mb", "peak_gpu_memory_mb"), c(1, 1),
                             c("Peak host RAM (MB)", "Peak GPU memory (MB)"), c(0, 0),
                             "tab:results-desc-rq2-mem", "Descriptive statistics of peak host RAM and peak GPU memory (MB) per configuration, $n=20$ runs each.")
)
latex_dir <- file.path(TAB_DIR, "latex")
dir.create(latex_dir, showWarnings = FALSE)
for (n in names(latex_tables)) writeLines(latex_tables[[n]], file.path(latex_dir, paste0("descriptive_", n, ".tex")))

writeLines(summary_lines, file.path(OUT_DIR, "summary.txt"))
cat("\nDone. Tables in", TAB_DIR, "- figures in", FIG_DIR, "\n")
