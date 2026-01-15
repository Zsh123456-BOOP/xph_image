#!/usr/bin/env Rscript
# Experiment 2: Gating Consistency Plots (R version)
# Generate all plots for exp_m2_out/assist_09

library(ggplot2)
library(dplyr)
library(scales)

# Set working directory
setwd("/home/zsh/xph_image/exp_m2_out/assist_09")

# Publication-quality theme
theme_pub <- theme_minimal(base_size = 12) +
  theme(
    text = element_text(family = "sans"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "#E6E6E6", linetype = "dashed", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "#333333", linewidth = 0.6),
    axis.text = element_text(color = "#333333"),
    axis.title = element_text(color = "#333333"),
    legend.position = "bottom",
    legend.background = element_rect(fill = "white", color = NA)
  )

# Color palette
colors <- c(
  blue = "#1f77b4",
  orange = "#ff7f0e",
  green = "#2ca02c",
  red = "#d62728",
  purple = "#9467bd",
  gray = "#7f7f7f"
)

# ============================================================
# Plot 1: Robust Curve (AUC + Accuracy vs Dropout Rate)
# ============================================================
cat("Generating robust_curve.png...\n")
robust <- read.csv("robust_curve.csv")

p1 <- ggplot(robust, aes(x = drop_rate)) +
  geom_line(aes(y = auc, color = "AUC"), size = 1.2) +
  geom_point(aes(y = auc, color = "AUC"), size = 3) +
  geom_line(aes(y = accuracy, color = "Accuracy"), size = 1.2, linetype = "dashed") +
  geom_point(aes(y = accuracy, color = "Accuracy"), size = 3, shape = 15) +
  scale_color_manual(values = c("AUC" = colors["blue"], "Accuracy" = colors["orange"])) +
  labs(
    title = "Robustness to Graph Edge Dropout",
    x = "Graph Edge Dropout Rate",
    y = "Metric Value",
    color = "Metric"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub

ggsave("R_robust_curve.png", p1, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 2: Pareto Trade-off (D_view vs Error)
# ============================================================
cat("Generating pareto_tradeoff.png...\n")
pareto <- read.csv("pareto.csv")
pareto$error <- 1 - pareto$test_auc

# Find lambda star (minimum normalized score)
normalize01 <- function(x) { (x - min(x)) / (max(x) - min(x) + 1e-12) }
pareto$score <- normalize01(pareto$D_view_mean) + normalize01(pareto$error)
lambda_star <- pareto$lambda_contrastive[which.min(pareto$score)]

p2 <- ggplot(pareto, aes(x = D_view_mean, y = error)) +
  geom_path(color = colors["gray"], linetype = "dashed", size = 1) +
  geom_point(aes(color = lambda_contrastive), size = 4) +
  scale_color_viridis_c(option = "viridis", name = expression(lambda)) +
  geom_point(
    data = pareto %>% filter(lambda_contrastive == lambda_star),
    aes(x = D_view_mean, y = error),
    shape = 8, size = 6, color = colors["red"], stroke = 2
  ) +
  labs(
    title = "Consistency-Accuracy Trade-off",
    x = expression(D[view] ~ "(↓ better consistency)"),
    y = "Error Rate (1 - AUC) (↓ better accuracy)"
  ) +
  annotate("text", x = pareto$D_view_mean[which.min(pareto$score)] + 0.01,
           y = pareto$error[which.min(pareto$score)],
           label = paste0("λ* = ", round(lambda_star, 3)), hjust = 0, color = colors["red"]) +
  theme_pub

ggsave("R_pareto_tradeoff.png", p2, width = 8, height = 6, dpi = 300)

# ============================================================
# Plot 3: Lambda vs AUC
# ============================================================
cat("Generating lambda_auc.png...\n")
p3 <- ggplot(pareto, aes(x = lambda_contrastive, y = test_auc)) +
  geom_line(color = colors["blue"], size = 1.2) +
  geom_point(color = colors["blue"], size = 3) +
  geom_vline(xintercept = lambda_star, linetype = "dashed", color = colors["red"]) +
  labs(
    title = "Test AUC vs Contrastive Lambda",
    x = expression(lambda[contrastive]),
    y = "Test AUC"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  annotate("text", x = lambda_star + 0.05, y = max(pareto$test_auc),
           label = paste0("λ* = ", round(lambda_star, 3)), color = colors["red"]) +
  theme_pub

ggsave("R_lambda_auc.png", p3, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 4: D_exer vs D_cpt (if columns exist)
# ============================================================
if ("D_exer" %in% colnames(pareto) && "D_cpt" %in% colnames(pareto)) {
  cat("Generating d_exer_vs_d_cpt.png...\n")
  p4 <- ggplot(pareto, aes(x = D_exer, y = D_cpt, color = lambda_contrastive)) +
    geom_point(size = 4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = colors["gray"]) +
    scale_color_viridis_c(option = "viridis", name = expression(lambda)) +
    labs(
      title = "View Distances: Exercise vs Concept",
      x = expression(D[exer] ~ "(exercise view)"),
      y = expression(D[cpt] ~ "(concept view)")
    ) +
    theme_pub
  
  ggsave("R_d_exer_vs_d_cpt.png", p4, width = 7, height = 6, dpi = 300)
}

cat("All Experiment 2 plots generated!\n")
