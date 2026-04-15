#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  parsed <- list(
    strict_dir = file.path("analysis_outputs", "prism_vs_neuralcd_strict_20260412"),
    hparam_dir = file.path("analysis_outputs", "prism_hparam_sensitivity_20260414")
  )
  if (length(args) == 0) {
    return(parsed)
  }

  if (length(args) %% 2 != 0) {
    stop("Arguments must be provided as --key value pairs.")
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- args[[i + 1]]
    if (key == "--strict_dir") {
      parsed$strict_dir <- value
    } else if (key == "--hparam_dir") {
      parsed$hparam_dir <- value
    } else {
      stop(sprintf("Unsupported argument: %s", key))
    }
    i <- i + 2
  }
  parsed
}

theme_paper <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.major = element_line(color = "#d9d9d9", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "#f4f4f4", colour = "#d0d0d0"),
      legend.title = element_blank(),
      legend.key = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold")
    )
}

format_value_label <- function(value) {
  abs_value <- abs(value)
  if (abs_value >= 1) {
    sprintf("%.2f", value)
  } else if (abs_value >= 0.1) {
    sprintf("%.3f", value)
  } else {
    sprintf("%.4f", value)
  }
}

format_signed_label <- function(value) {
  formatted <- format_value_label(value)
  if (value > 0) {
    paste0("+", formatted)
  } else {
    formatted
  }
}

build_panel_labels <- function(dataset_levels, metric_levels, metric_label_map) {
  panel_levels <- c()
  panel_labels <- character(0)
  for (dataset in dataset_levels) {
    for (metric in metric_levels) {
      panel_key <- paste(dataset, metric, sep = "||")
      panel_levels <- c(panel_levels, panel_key)
      panel_labels[panel_key] <- paste(dataset, metric_label_map[[metric]], sep = "\n")
    }
  }
  list(levels = panel_levels, labels = panel_labels)
}

save_plot_pair <- function(plot, png_path, pdf_path, width, height, dpi = 320) {
  ggsave(filename = png_path, plot = plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(filename = pdf_path, plot = plot, width = width, height = height, device = cairo_pdf, bg = "white")
}

read_csv_checked <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Missing input CSV: %s", path))
  }
  read_csv(path, show_col_types = FALSE)
}

render_slipping_overview <- function(strict_dir) {
  summary_path <- file.path(strict_dir, "slipping_compare_summary.csv")
  df <- read_csv_checked(summary_path)
  dataset_levels <- c("assist_09", "assist_17", "junyi")
  metric_levels <- c(
    "Stress-subset AUC delta",
    "Stress-subset ACC delta",
    "Flipped-sample p75 decoupling gap",
    "Flipped-sample p90 decoupling gap"
  )
  panel_spec <- build_panel_labels(
    dataset_levels = dataset_levels,
    metric_levels = metric_levels,
    metric_label_map = c(
      "Stress-subset AUC delta" = "Stress AUC delta",
      "Stress-subset ACC delta" = "Stress ACC delta",
      "Flipped-sample p75 decoupling gap" = "P75 decoupling gap",
      "Flipped-sample p90 decoupling gap" = "P90 decoupling gap"
    )
  )

  plot_df <- bind_rows(
    transmute(df, dataset, model, ratio, metric = "Stress-subset AUC delta", value = stress_auc_delta_mean, std = stress_auc_delta_std, needs_zero = TRUE),
    transmute(df, dataset, model, ratio, metric = "Stress-subset ACC delta", value = stress_acc_delta_mean, std = stress_acc_delta_std, needs_zero = TRUE),
    transmute(df, dataset, model, ratio, metric = "Flipped-sample p75 decoupling gap", value = flipped_p75_decoupling_gap_mean, std = flipped_p75_decoupling_gap_std, needs_zero = FALSE),
    transmute(df, dataset, model, ratio, metric = "Flipped-sample p90 decoupling gap", value = flipped_p90_decoupling_gap_mean, std = flipped_p90_decoupling_gap_std, needs_zero = FALSE)
  ) %>%
    mutate(
      dataset = factor(dataset, levels = dataset_levels),
      metric = factor(metric, levels = metric_levels),
      panel = factor(paste(dataset, metric, sep = "||"), levels = panel_spec$levels)
    )

  hline_df <- plot_df %>%
    filter(needs_zero) %>%
    distinct(panel) %>%
    mutate(y = 0)

  p <- ggplot(plot_df, aes(x = ratio, y = value, color = model, fill = model, group = model)) +
    geom_hline(
      data = hline_df,
      aes(yintercept = y),
      inherit.aes = FALSE,
      linetype = "dashed",
      color = "#7a7a7a",
      linewidth = 0.35
    ) +
    geom_ribbon(
      aes(ymin = value - std, ymax = value + std),
      alpha = 0.14,
      linewidth = 0
    ) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.8) +
    facet_wrap(
      ~ panel,
      ncol = 4,
      scales = "free_y",
      labeller = labeller(panel = function(values) unname(panel_spec$labels[as.character(values)]))
    ) +
    scale_color_manual(values = c("Prism-CD" = "#1f77b4", "NeuralCD" = "#ff7f0e")) +
    scale_fill_manual(values = c("Prism-CD" = "#1f77b4", "NeuralCD" = "#ff7f0e")) +
    scale_x_continuous(breaks = c(0.1, 0.2, 0.3)) +
    labs(
      title = "Controlled slip simulation overview",
      x = "Flip ratio",
      y = "Value"
    ) +
    theme_paper() +
    theme(
      legend.position = "top",
      strip.text = element_text(size = 10, lineheight = 0.95)
    )

  save_plot_pair(
    p,
    file.path(strict_dir, "slipping_compare_overview.png"),
    file.path(strict_dir, "slipping_compare_overview.pdf"),
    width = 17.2,
    height = 9.0
  )
}

render_case_overview <- function(strict_dir) {
  table_path <- file.path(strict_dir, "case_study_compare_table.csv")
  df <- read_csv_checked(table_path)
  dataset_levels <- c("assist_09", "assist_17", "junyi")
  metric_levels <- c("Stable concept-drop ratio", "Decoupling gap")
  panel_spec <- build_panel_labels(
    dataset_levels = dataset_levels,
    metric_levels = metric_levels,
    metric_label_map = c(
      "Stable concept-drop ratio" = "Stable concept-drop ratio",
      "Decoupling gap" = "Decoupling gap"
    )
  )

  rank_column <- if ("representative_rank" %in% colnames(df)) "representative_rank" else "case_rank"
  case_labels <- paste0("Case ", df[[rank_column]])

  plot_df <- bind_rows(
    tibble(
      dataset = df$dataset,
      case_label = case_labels,
      metric = "Stable concept-drop ratio",
      model = "Prism-CD",
      value = df$prism_stable_concept_drop_ratio
    ),
    tibble(
      dataset = df$dataset,
      case_label = case_labels,
      metric = "Stable concept-drop ratio",
      model = "NeuralCD",
      value = df$baseline_stable_concept_drop_ratio
    ),
    tibble(
      dataset = df$dataset,
      case_label = case_labels,
      metric = "Decoupling gap",
      model = "Prism-CD",
      value = df$prism_decoupling_gap
    ),
    tibble(
      dataset = df$dataset,
      case_label = case_labels,
      metric = "Decoupling gap",
      model = "NeuralCD",
      value = df$baseline_decoupling_gap
    )
  ) %>%
    mutate(
      dataset = factor(dataset, levels = dataset_levels),
      metric = factor(metric, levels = metric_levels),
      panel = factor(paste(dataset, metric, sep = "||"), levels = panel_spec$levels)
    )

  p <- ggplot(plot_df, aes(x = case_label, y = value, fill = model)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64) +
    facet_wrap(
      ~ panel,
      ncol = 2,
      scales = "free_y",
      labeller = labeller(panel = function(values) unname(panel_spec$labels[as.character(values)]))
    ) +
    scale_fill_manual(values = c("Prism-CD" = "#1f77b4", "NeuralCD" = "#ff7f0e")) +
    labs(
      title = "Case-study comparison overview",
      x = NULL,
      y = "Value"
    ) +
    theme_paper() +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 0, vjust = 0.5),
      strip.text = element_text(size = 10, lineheight = 0.95)
    )

  save_plot_pair(
    p,
    file.path(strict_dir, "case_study_compare_overview.png"),
    file.path(strict_dir, "case_study_compare_overview.pdf"),
    width = 13.6,
    height = 9.0
  )
}

render_controlled_gain_summary <- function(strict_dir) {
  df <- read_csv_checked(file.path(strict_dir, "controlled_slip_gain_summary.csv"))

  plot_df <- bind_rows(
    tibble(dataset = df$dataset, metric = "AUC drop gain", value = df$auc_drop_gain),
    tibble(dataset = df$dataset, metric = "ACC drop gain", value = df$acc_drop_gain),
    tibble(dataset = df$dataset, metric = "P75 decoupling gain", value = df$p75_decoupling_gain),
    tibble(dataset = df$dataset, metric = "P90 decoupling gain", value = df$p90_decoupling_gain)
  ) %>%
    mutate(
      dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")),
      metric = factor(metric, levels = c("AUC drop gain", "ACC drop gain", "P75 decoupling gain", "P90 decoupling gain")),
      label = vapply(value, format_value_label, character(1))
    )

  p <- ggplot(plot_df, aes(x = dataset, y = value, fill = metric)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#7a7a7a", linewidth = 0.35) +
    geom_col(width = 0.62, show.legend = FALSE) +
    geom_text(aes(label = label), vjust = -0.45, size = 3.5, show.legend = FALSE) +
    facet_wrap(~ metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = c(
      "AUC drop gain" = "#4c78a8",
      "ACC drop gain" = "#f58518",
      "P75 decoupling gain" = "#54a24b",
      "P90 decoupling gain" = "#b279a2"
    )) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
    labs(
      title = "Controlled slip gains across datasets",
      x = NULL,
      y = "Prism-CD gain over NeuralCD"
    ) +
    theme_paper() +
    theme(legend.position = "none")

  save_plot_pair(
    p,
    file.path(strict_dir, "controlled_slip_gain_summary.png"),
    file.path(strict_dir, "controlled_slip_gain_summary.pdf"),
    width = 15.2,
    height = 4.9
  )
}

render_case_effect_summary <- function(strict_dir) {
  df <- read_csv_checked(file.path(strict_dir, "case_study_effect_summary.csv"))

  plot_df <- bind_rows(
    tibble(dataset = df$dataset, metric = "Median adjustment-ratio reduction", value = df$adjustment_ratio_median_gain),
    tibble(dataset = df$dataset, metric = "P90 adjustment-ratio reduction", value = df$adjustment_ratio_p90_gain),
    tibble(dataset = df$dataset, metric = "Median decoupling-gap gain", value = df$decoupling_gap_median_gain)
  ) %>%
    mutate(
      dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")),
      metric = factor(
        metric,
        levels = c(
          "Median adjustment-ratio reduction",
          "P90 adjustment-ratio reduction",
          "Median decoupling-gap gain"
        )
      ),
      label = vapply(value, format_value_label, character(1))
    )

  p <- ggplot(plot_df, aes(x = dataset, y = value, fill = metric)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#7a7a7a", linewidth = 0.35) +
    geom_col(width = 0.62, show.legend = FALSE) +
    geom_text(aes(label = label), vjust = -0.45, size = 3.5, show.legend = FALSE) +
    facet_wrap(~ metric, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = c(
      "Median adjustment-ratio reduction" = "#4c78a8",
      "P90 adjustment-ratio reduction" = "#f58518",
      "Median decoupling-gap gain" = "#54a24b"
    )) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
    labs(
      title = "Case-study effect sizes across datasets",
      x = NULL,
      y = "Prism-CD gain over NeuralCD"
    ) +
    theme_paper() +
    theme(legend.position = "none")

  save_plot_pair(
    p,
    file.path(strict_dir, "case_study_effect_summary.png"),
    file.path(strict_dir, "case_study_effect_summary.pdf"),
    width = 15.0,
    height = 4.8
  )
}

render_hparam_sensitivity <- function(hparam_dir, hparam_name, display_name) {
  summary_df <- read_csv_checked(file.path(hparam_dir, "prism_hparam_sensitivity_summary.csv"))
  best_df <- read_csv_checked(file.path(hparam_dir, "prism_hparam_sensitivity_best.csv"))

  sub_summary <- summary_df %>% filter(hparam == hparam_name)
  sub_best <- best_df %>% filter(hparam == hparam_name)
  range_df <- sub_summary %>%
    group_by(dataset) %>%
    summarise(x_min = min(value), x_max = max(value), .groups = "drop")

  plot_df <- bind_rows(
    tibble(dataset = sub_summary$dataset, value = sub_summary$value, metric = "AUC", score = sub_summary$test_auc),
    tibble(dataset = sub_summary$dataset, value = sub_summary$value, metric = "ACC", score = sub_summary$test_acc)
  ) %>%
    mutate(dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")))

  vline_df <- sub_best %>%
    transmute(dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")), xintercept = default_value)

  star_df <- sub_best %>%
    left_join(range_df, by = "dataset") %>%
    transmute(
      dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")),
      value = best_value_by_auc,
      score = best_auc,
      label = sprintf("best=%.4f", best_auc),
      hjust = ifelse(abs(best_value_by_auc - x_min) < 1e-9, 0, ifelse(abs(best_value_by_auc - x_max) < 1e-9, 1, 0.5))
    )

  p <- ggplot(plot_df, aes(x = value, y = score, color = metric, group = metric)) +
    geom_vline(
      data = vline_df,
      aes(xintercept = xintercept),
      inherit.aes = FALSE,
      linetype = "dotted",
      color = "#6a6a6a",
      linewidth = 0.45
    ) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.0) +
    geom_point(
      data = star_df,
      aes(x = value, y = score),
      inherit.aes = FALSE,
      shape = 8,
      size = 4,
      color = "#1f77b4"
    ) +
    geom_text(
      data = star_df,
      aes(x = value, y = score, label = label),
      inherit.aes = FALSE,
      nudge_y = 0.003,
      color = "#1f77b4",
      size = 3.5,
      hjust = star_df$hjust
    ) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_x") +
    scale_color_manual(values = c("AUC" = "#1f77b4", "ACC" = "#ff7f0e")) +
    labs(
      title = sprintf("Prism-CD sensitivity to %s", display_name),
      x = display_name,
      y = "Metric value"
    ) +
    theme_paper() +
    theme(legend.position = "top")

  save_plot_pair(
    p,
    file.path(hparam_dir, sprintf("%s_sensitivity.png", hparam_name)),
    file.path(hparam_dir, sprintf("%s_sensitivity.pdf", hparam_name)),
    width = 15.8,
    height = 4.9
  )
}

render_hparam_gain_summary <- function(hparam_dir) {
  gain_df <- read_csv_checked(file.path(hparam_dir, "prism_hparam_gain_summary.csv"))

  x_label <- function(hparam, best_value_label) {
    display <- c(
      hyper_weight = "Hyper Weight",
      dropout = "Dropout",
      hidden_dim = "Hidden Dimension"
    )[[hparam]]
    sprintf("%s\n(best=%s)", display, best_value_label)
  }

  hparam_levels <- c("hyper_weight", "dropout", "hidden_dim")
  gain_df <- gain_df %>%
    mutate(
      dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")),
      hparam = factor(hparam, levels = hparam_levels),
      label = mapply(x_label, as.character(hparam), best_value_label)
    ) %>%
    arrange(dataset, hparam)

  plot_df <- bind_rows(
    tibble(
      dataset = gain_df$dataset,
      hparam = gain_df$hparam,
      label = gain_df$label,
      metric = "AUC gain",
      value = gain_df$auc_gain_vs_default
    ),
    tibble(
      dataset = gain_df$dataset,
      hparam = gain_df$hparam,
      label = gain_df$label,
      metric = "ACC gain",
      value = gain_df$acc_gain_vs_default
    )
  ) %>%
    mutate(
      dataset = factor(dataset, levels = c("assist_09", "assist_17", "junyi")),
      hparam = factor(hparam, levels = hparam_levels),
      x_key = paste(dataset, hparam, sep = "__")
    )

  level_df <- gain_df %>% mutate(x_key = paste(dataset, hparam, sep = "__"))
  plot_df$x_key <- factor(plot_df$x_key, levels = level_df$x_key)
  plot_df <- plot_df %>%
    mutate(
      label = vapply(value, format_signed_label, character(1)),
      label_vjust = ifelse(value >= 0, -0.35, 1.15)
    )

  p <- ggplot(plot_df, aes(x = x_key, y = value, fill = metric)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#7a7a7a", linewidth = 0.35) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64) +
    geom_text(
      aes(label = label, group = metric, vjust = label_vjust),
      position = position_dodge(width = 0.72),
      size = 3.1
    ) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_x") +
    scale_fill_manual(values = c("AUC gain" = "#1f77b4", "ACC gain" = "#ff7f0e")) +
    scale_x_discrete(labels = function(x) {
      label_map <- setNames(level_df$label, level_df$x_key)
      unname(label_map[x])
    }) +
    scale_y_continuous(expand = expansion(mult = c(0.15, 0.20))) +
    labs(
      title = "Prism-CD best-point gain over default",
      x = NULL,
      y = "Gain vs default"
    ) +
    theme_paper() +
    theme(
      legend.position = "top",
      axis.text.x = element_text(size = 9, lineheight = 0.95)
    )

  save_plot_pair(
    p,
    file.path(hparam_dir, "best_vs_default_gain_summary.png"),
    file.path(hparam_dir, "best_vs_default_gain_summary.pdf"),
    width = 15.8,
    height = 5.4
  )
}

main <- function() {
  args <- parse_args()

  render_slipping_overview(args$strict_dir)
  render_case_overview(args$strict_dir)
  render_controlled_gain_summary(args$strict_dir)
  render_case_effect_summary(args$strict_dir)

  render_hparam_sensitivity(args$hparam_dir, "hyper_weight", "Hyper Weight")
  render_hparam_sensitivity(args$hparam_dir, "dropout", "Dropout")
  render_hparam_sensitivity(args$hparam_dir, "hidden_dim", "Hidden Dimension")
  render_hparam_gain_summary(args$hparam_dir)

  cat("Rendered strict and hyperparameter figures with ggplot2.\n")
}

main()
