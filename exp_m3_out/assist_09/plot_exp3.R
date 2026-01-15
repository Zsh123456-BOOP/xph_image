#!/usr/bin/env Rscript
# Experiment 3: Interaction & Q-Noise Plots (R version)
# Generate all plots for exp_m3_out/assist_09

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(reshape2)

# Set working directory
setwd("/home/zsh/xph_image/exp_m3_out/assist_09")

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
# Plot 1: Q-Noise Robustness Curve (missing vs false)
# ============================================================
cat("Generating qnoise_curve.png...\n")
qnoise <- read.csv("qnoise_curve.csv")

p1 <- ggplot(qnoise, aes(x = rho, y = auc, color = mode, shape = mode)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("missing" = colors["orange"], "false" = colors["red"])) +
  scale_shape_manual(values = c("missing" = 16, "false" = 17)) +
  labs(
    title = "Q-Noise Robustness (Inference-time cpt_seq Corruption)",
    x = expression(rho ~ "(Q-noise rate)"),
    y = "Test AUC",
    color = "Mode",
    shape = "Mode"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub

ggsave("R_qnoise_curve.png", p1, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 2: Hard Q-Noise Curve
# ============================================================
cat("Generating qnoise_hard_curve.png...\n")
qnoise_hard <- read.csv("qnoise_hard_curve.csv")

p2 <- ggplot(qnoise_hard, aes(x = rho, y = auc)) +
  geom_line(color = colors["purple"], size = 1.2) +
  geom_point(color = colors["purple"], size = 3) +
  labs(
    title = "Hard Q-Noise Robustness (Semantic Hard False Concepts)",
    x = expression(rho ~ "(Q-noise rate)"),
    y = "Test AUC"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub

ggsave("R_qnoise_hard_curve.png", p2, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 3: Q-Noise Combo (2 panels)
# ============================================================
cat("Generating qnoise_combo.png...\n")
library(patchwork)

# Panel a: missing vs false
pa <- ggplot(qnoise, aes(x = rho, y = auc, color = mode)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("missing" = colors["orange"], "false" = colors["red"])) +
  labs(title = "(a) Missing vs False", x = expression(rho), y = "Test AUC") +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub + theme(legend.position = "right")

# Panel b: hard false
pb <- ggplot(qnoise_hard, aes(x = rho, y = auc)) +
  geom_line(color = colors["purple"], size = 1.2) +
  geom_point(color = colors["purple"], size = 3) +
  labs(title = "(b) Hard False", x = expression(rho), y = "Test AUC") +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub

p3 <- pa + pb + plot_layout(ncol = 2) +
  plot_annotation(title = "Q-Noise Robustness")

ggsave("R_qnoise_combo.png", p3, width = 12, height = 5, dpi = 300)

# ============================================================
# Plot 4: Attribution Violin (by concept count bucket)
# ============================================================
cat("Generating attribution_violin.png...\n")
attribution <- read.csv("attribution_table.csv")

# Bucket concept count
attribution$bucket <- cut(attribution$concept_count,
                          breaks = c(0, 2, 3, 4, Inf),
                          labels = c("2", "3", "4", "5+"),
                          right = TRUE)

p4 <- ggplot(attribution, aes(x = bucket, y = attr_logit_mean, fill = bucket)) +
  geom_violin(alpha = 0.7, draw_quantiles = c(0.5)) +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Concept Attribution vs Concept Count (Violin)",
    x = "|Kq| Bucket",
    y = "Attribution (mean |Δ logit|)"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub +
  theme(legend.position = "none")

ggsave("R_attribution_violin.png", p4, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 5: Interaction Synergy Heatmap
# ============================================================
cat("Generating interaction_heatmap.png...\n")
interaction_matrix <- read.csv("interaction_matrix.csv", row.names = 1)
interaction_long <- melt(as.matrix(interaction_matrix))
colnames(interaction_long) <- c("Concept_i", "Concept_j", "Synergy")

p5 <- ggplot(interaction_long, aes(x = Concept_j, y = Concept_i, fill = Synergy)) +
  geom_tile() +
  scale_fill_gradient2(low = colors["blue"], mid = "white", high = colors["red"], midpoint = 0) +
  labs(
    title = "Interaction Synergy Heatmap",
    x = "Concept j",
    y = "Concept i",
    fill = "Syn(i,j)"
  ) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("R_interaction_heatmap.png", p5, width = 7, height = 6, dpi = 300)

# ============================================================
# Plot 6: Attribution Scatter (concept_count vs attr_logit_mean)
# ============================================================
cat("Generating attribution_scatter.png...\n")
p6 <- ggplot(attribution, aes(x = concept_count, y = attr_logit_mean)) +
  geom_point(color = colors["blue"], alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", color = colors["red"], linetype = "dashed", se = TRUE) +
  labs(
    title = "Attribution vs Concept Count",
    x = "Concept Count per Exercise",
    y = "Attribution (mean |Δ logit|)"
  ) +
  scale_y_continuous(labels = label_number(accuracy = 0.0001)) +
  theme_pub

ggsave("R_attribution_scatter.png", p6, width = 7, height = 5, dpi = 300)

cat("All Experiment 3 plots generated!\n")
