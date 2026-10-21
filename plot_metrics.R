library(tidyverse)
library(ggh4x)

# Configuration
CSV_FILE <- "experiments/llm_quantization_main_360_v3/run_table.csv"
OUTPUT_DIR <- "."
BENCHMARK_ORDER <- c("mmlu", "mmlu_pro", "race_h")
BENCHMARK_NAMES <- c("mmlu" = "MMLU", "mmlu_pro" = "MMLU-Pro", "race_h" = "RACE-H")
QUANTIZATION_ORDER <- c("fp16", "q4")
QUANTIZATION_NAMES <- c("fp16" = "FP16", "q4" = "Q4")
MODEL_GAP <- 1.6
BOX_WIDTH <- 0.48
QUANT_COLORS <- c("FP16" = "#0072B2", "Q4" = "#D55E00")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Load and prepare data
df <- read_csv(CSV_FILE, show_col_types = FALSE)
if ("__done" %in% names(df)) df <- df %>% filter(.data[["__done"]] == "DONE")

required_columns <- c(
  "model_family", "benchmark", "quantization", "run_time_s", "inference_time_s",
  "cpu_energy_j", "gpu_energy_j", "total_energy_j", "inference_gpu_energy_j",
  "average_inference_gpu_power_w", "input_tokens", "output_tokens",
  "throughput_tokens_s", "cpu_utilization_pct", "gpu_utilization_pct",
  "warmup_time_s", "peak_ram_mb", "peak_gpu_memory_mb",
  "cpu_temp_mean_c", "cpu_temp_max_c", "gpu_temp_mean_c", "gpu_temp_max_c"
)
missing_columns <- setdiff(required_columns, names(df))
if (length(missing_columns) > 0) {
  stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")))
}

model_order <- sort(unique(df$model_family))
model_positions <- seq_along(model_order) * MODEL_GAP
names(model_positions) <- model_order
df <- df %>%
  filter(benchmark %in% BENCHMARK_ORDER, quantization %in% QUANTIZATION_ORDER) %>%
  mutate(
    benchmark = factor(benchmark, levels = BENCHMARK_ORDER,
                       labels = BENCHMARK_NAMES[BENCHMARK_ORDER]),
    quant_display = factor(quantization, levels = QUANTIZATION_ORDER,
                           labels = QUANTIZATION_NAMES[QUANTIZATION_ORDER]),
    model_family = factor(model_family, levels = model_order),
    model_position = unname(model_positions[as.character(model_family)])
  )

# Shared visual style: same spacing, box width, facet layout, labels and theme
base_theme <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16,
                              margin = margin(b = 4)),
    plot.subtitle = element_text(hjust = 0.5, size = 11, margin = margin(b = 8)),
    strip.text = element_text(face = "bold", size = 12),
    strip.background = element_rect(fill = "grey92", linewidth = 0.5),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10,
                               margin = margin(t = 5)),
    axis.text.y = element_text(size = 9),
    axis.title.x = element_text(size = 11, margin = margin(t = 10)),
    axis.title.y = element_text(size = 11, margin = margin(r = 10)),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = 0.4),
    panel.grid.minor = element_blank(),
    panel.spacing.x = unit(1.8, "lines"),
    panel.spacing.y = unit(1.6, "lines"),
    legend.position = "bottom",
    plot.margin = margin(t = 12, r = 15, b = 12, l = 15)
  )
save_plot <- function(plot, filename, width = 16, height = 11) {
  output_file <- file.path(OUTPUT_DIR, filename)
  ggsave(output_file, plot = plot, width = width, height = height, dpi = 300, bg = "white")
  message("Saved: ", output_file)
}
facet_layout <- function() {
  ggh4x::facet_grid2(rows = vars(benchmark), cols = vars(model_family),
                     scales = "free_y", independent = "y", drop = FALSE)
}
common_x <- scale_x_discrete(limits = QUANTIZATION_NAMES[QUANTIZATION_ORDER])

# 1. Runtime and inference time
runtime_data <- df %>%
  select(model_family, model_position, benchmark, quant_display,
         run_time_s, inference_time_s) %>%
  mutate(non_inference_time_s = run_time_s - inference_time_s) %>%
  pivot_longer(c(inference_time_s, non_inference_time_s),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
                         inference_time_s = "Inference time",
                         non_inference_time_s = "Non-inference time"),
         # Factor order makes inference time the bottom stack; legend order is
         # set separately below.
         metric = factor(metric, levels = c("Non-inference time", "Inference time"))) %>%
  filter(is.finite(value), value >= 0)
runtime_means <- runtime_data %>%
  group_by(model_family, benchmark, quant_display, metric) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
p_runtime <- ggplot(runtime_means, aes(quant_display, value, fill = metric,
                                       group = metric)) +
  geom_col(position = "stack", width = 0.38,
           colour = "white", linewidth = 0.25, alpha = 0.45, na.rm = TRUE) +
  facet_layout() + common_x +
  labs(title = "Total run time",
       x = "Quantization", y = "Time (s)", fill = "Metric") + base_theme
p_runtime <- p_runtime + scale_fill_manual(
  values = c("Inference time" = "#0072B2", "Non-inference time" = "#D55E00"),
  breaks = c("Inference time", "Non-inference time")
)
save_plot(p_runtime, "01_runtime.png")

# 2. Average inference GPU power: one figure with 3 × 3 panels.
energy_data <- df %>%
  select(model_family, benchmark, quant_display, average_inference_gpu_power_w) %>%
  filter(is.finite(average_inference_gpu_power_w), average_inference_gpu_power_w > 0)
p_energy <- ggplot(energy_data, aes(quant_display, average_inference_gpu_power_w,
                                    fill = quant_display)) +
  geom_violin(width = 0.85, alpha = 0.35, linewidth = 0.55,
              trim = FALSE, na.rm = TRUE) +
  geom_jitter(width = 0.08, height = 0, size = 1.5, alpha = 0.55,
              shape = 16, na.rm = TRUE) +
  facet_layout() + common_x +
  labs(title = "Average inference GPU power",
       x = "Quantization", y = "Average inference GPU power (W)",
       fill = "Quantization") +
  scale_fill_manual(values = QUANT_COLORS) + base_theme +
  theme(legend.position = "none")
save_plot(p_energy, "02_energy.png", width = 15, height = 10)

# Total energy comparison, kept as a separate 3 × 3 figure.
total_energy_data <- df %>%
  select(model_family, benchmark, quant_display, total_energy_j) %>%
  filter(is.finite(total_energy_j), total_energy_j > 0)
p_total_energy <- ggplot(total_energy_data,
                         aes(quant_display, total_energy_j, fill = quant_display)) +
  geom_violin(width = 0.85, alpha = 0.35, linewidth = 0.55,
              trim = FALSE, na.rm = TRUE) +
  geom_jitter(width = 0.08, height = 0, size = 1.5, alpha = 0.55,
              shape = 16, na.rm = TRUE) +
  facet_layout() + common_x +
  labs(title = "Total energy (CPU + GPU)",
       x = "Quantization", y = "Total energy (CPU + GPU) (J)",
       fill = "Quantization") +
  scale_fill_manual(values = QUANT_COLORS) + base_theme +
  theme(legend.position = "none")
save_plot(p_total_energy, "02_total_energy.png", width = 15, height = 10)

# 3. Summary table of the main model and system metrics.
fmt_mean_sd <- function(x) {
  if (all(is.na(x))) return("NA")
  sprintf("%.4f ± %.4f", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))
}
fmt_mean_sd_int <- function(x) {
  if (all(is.na(x))) return("NA")
  sprintf("%.0f ± %.0f", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))
}
fmt_mean_sd_pct <- function(x) {
  if (all(is.na(x))) return("NA")
  sprintf("%.2f ± %.2f", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))
}
table_columns <- c(
  "CPU energy (J)", "GPU energy (J)",
  "Run time (s)", "Inference time (s)", "Inference GPU energy (J)",
  "Avg inference GPU power (W)", "Input tokens", "Output tokens",
  "Throughput (tokens/s)", "Warm-up time (s)", "Peak RAM (MB)",
  "Peak GPU memory (MB)", "CPU utilization (%)", "GPU utilization (%)"
)
table_data <- df %>%
  group_by(model, quant_display, benchmark) %>%
  summarise(
    `CPU energy (J)` = fmt_mean_sd(cpu_energy_j),
    `GPU energy (J)` = fmt_mean_sd(gpu_energy_j),
    `Run time (s)` = fmt_mean_sd(run_time_s),
    `Inference time (s)` = fmt_mean_sd(inference_time_s),
    `Inference GPU energy (J)` = fmt_mean_sd(inference_gpu_energy_j),
    `Avg inference GPU power (W)` = fmt_mean_sd(average_inference_gpu_power_w),
    `Input tokens` = fmt_mean_sd_int(input_tokens),
    `Output tokens` = fmt_mean_sd_int(output_tokens),
    `Throughput (tokens/s)` = fmt_mean_sd(throughput_tokens_s),
    `Warm-up time (s)` = fmt_mean_sd(warmup_time_s),
    `Peak RAM (MB)` = fmt_mean_sd(peak_ram_mb),
    `Peak GPU memory (MB)` = fmt_mean_sd(peak_gpu_memory_mb),
    `CPU utilization (%)` = fmt_mean_sd_pct(cpu_utilization_pct),
    `GPU utilization (%)` = fmt_mean_sd_pct(gpu_utilization_pct),
    .groups = "drop"
  ) %>%
  mutate(
    Model = model,
    Quantization = as.character(quant_display),
    Benchmark = as.character(benchmark),
    row_label = paste(Model, Benchmark, sep = "\n")
  ) %>%
  select(row_label, all_of(table_columns)) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-c(row_id, row_label), names_to = "field", values_to = "display")

table_data$field <- factor(table_data$field, levels = table_columns)
n_table_rows <- max(table_data$row_id)
table_row_labels <- table_data %>%
  filter(field == "CPU energy (J)") %>%
  arrange(row_id) %>%
  pull(row_label)
p_table <- ggplot(table_data, aes(x = field, y = row_id, label = display)) +
  geom_text(size = 2.7, family = "sans") +
  scale_x_discrete(limits = table_columns, labels = table_columns,
                   expand = expansion(add = 0.5)) +
  scale_y_continuous(breaks = seq_len(n_table_rows), labels = table_row_labels,
                     expand = expansion(add = 0.5)) +
  labs(title = "Model and system metrics summary", x = NULL, y = NULL) +
  base_theme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(face = "bold", size = 8),
        panel.grid = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "none")
save_plot(p_table, "06_metrics_table.png", width = 20, height = 10)

# 4-5. Utilization and temperature from the 100 ms summary file.
SUMMARY_FILE <- "sampler_100ms_summary.csv"
if (!file.exists(SUMMARY_FILE)) stop("Missing sampler summary: ", SUMMARY_FILE)
sampler_summary <- read_csv(SUMMARY_FILE, show_col_types = FALSE) %>%
  mutate(
    benchmark = factor(benchmark, levels = BENCHMARK_ORDER,
                       labels = BENCHMARK_NAMES[BENCHMARK_ORDER]),
    model_family = factor(model_family, levels = model_order),
    quant_display = factor(QUANTIZATION_NAMES[quantization], levels = c("FP16", "Q4"))
  ) %>%
  group_by(benchmark) %>%
  mutate(time_seconds = time_bin_ms / 1000) %>%
  ungroup()

sampler_util <- sampler_summary %>%
  select(model_family, benchmark, quant_display, time_seconds,
         gpu_utilization_pct_mean) %>%
  transmute(model_family, benchmark, quant_display, time_seconds,
            metric = "GPU utilization (%)", value = gpu_utilization_pct_mean) %>%
  filter(is.finite(value), value >= 0, value <= 100)
p_utilization <- ggplot(sampler_util, aes(value, colour = quant_display,
                                          fill = quant_display,
                                          group = quant_display)) +
  geom_density(alpha = 0.18, linewidth = 0.8, na.rm = TRUE,
               adjust = 1, position = "identity") +
  facet_grid(benchmark ~ model_family, scales = "free_y") +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25)) +
  scale_colour_manual(values = QUANT_COLORS) +
  scale_fill_manual(values = QUANT_COLORS) +
  labs(title = "Utilization density",
       subtitle = "GPU utilization density from 100 ms summary means",
       x = "Utilization (%)", y = "Density", colour = "Quantization",
       fill = "Quantization") + base_theme
save_plot(p_utilization, "04_utilization.png")

sampler_temp <- sampler_summary %>%
  transmute(model_family, benchmark, quant_display, time_seconds,
            `CPU temperature (°C)` = cpu_temperature_c_mean,
            `CPU temperature sd (°C)` = cpu_temperature_c_std,
            `GPU temperature (°C)` = gpu_temperature_c_mean,
            `GPU temperature sd (°C)` = gpu_temperature_c_std) %>%
  pivot_longer(-c(model_family, benchmark, quant_display, time_seconds),
               names_to = "series", values_to = "value") %>%
  mutate(
    metric = if_else(str_detect(series, "^CPU"), "CPU temperature (°C)",
                     "GPU temperature (°C)"),
    statistic = case_when(
      str_detect(series, " sd ") ~ "SD",
      TRUE ~ "Mean"
    ),
    series_label = paste(quant_display, metric, sep = " / ")
  ) %>%
  select(-series) %>%
  pivot_wider(names_from = statistic, values_from = value)
p_temperature <- ggplot(sampler_temp,
                        aes(time_seconds, Mean, colour = series_label,
                            group = series_label)) +
  geom_ribbon(aes(ymin = Mean - SD, ymax = Mean + SD,
                  fill = series_label),
              alpha = 0.16, colour = NA, na.rm = TRUE) +
  geom_line(linewidth = 0.9, na.rm = TRUE) +
  ggh4x::facet_grid2(rows = vars(benchmark), cols = vars(model_family),
                     scales = "free", independent = "x", drop = FALSE) +
  scale_x_continuous(breaks = scales::breaks_pretty(n = 5)) +
  scale_colour_manual(values = c(
    "FP16 / CPU temperature (°C)" = "#0072B2",
    "FP16 / GPU temperature (°C)" = "#56B4E9",
    "Q4 / CPU temperature (°C)" = "#D55E00",
    "Q4 / GPU temperature (°C)" = "#E69F00"
  )) +
  scale_fill_manual(values = c(
    "FP16 / CPU temperature (°C)" = "#0072B2",
    "FP16 / GPU temperature (°C)" = "#56B4E9",
    "Q4 / CPU temperature (°C)" = "#D55E00",
    "Q4 / GPU temperature (°C)" = "#E69F00"
  ), guide = "none") +
  labs(title = "Temperature over time",
       subtitle = "Shaded area shows mean ± standard deviation; line shows the mean",
       x = "Time (s)", y = "Temperature (°C)",
       colour = "Series") + base_theme
save_plot(p_temperature, "05_temperature.png")

message("Finished. Results saved in: ", normalizePath(OUTPUT_DIR))
