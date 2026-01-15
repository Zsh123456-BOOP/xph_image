#!/usr/bin/env Rscript
# Experiment 1: Disentanglement Plots (R version)
# Generate all plots for exp_m1_out/assist_09

library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(reshape2)

# Set working directory
setwd("/home/zsh/xph_image/exp_m1_out/assist_09")

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
  purple = "#9467bd"
)

# ============================================================
# Plot 1: Alignment Matrix Heatmap
# ============================================================
cat("Generating alignment_heatmap.png...\n")
alignment_matrix <- read.csv("alignment_matrix.csv", row.names = 1)
alignment_long <- melt(as.matrix(alignment_matrix))
colnames(alignment_long) <- c("LatentDim", "Concept", "Correlation")

p1 <- ggplot(alignment_long, aes(x = Concept, y = LatentDim, fill = Correlation)) +
  geom_tile() +
  scale_fill_viridis_c(option = "viridis", limits = c(0, NA), na.value = "gray90") +
  labs(
    title = "Alignment Matrix (|Spearman Corr|)",
    x = "Concept Index",
    y = "Latent Dimension"
  ) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6),
        axis.text.y = element_text(size = 6))

ggsave("R_alignment_heatmap.png", p1, width = 10, height = 8, dpi = 300)

# ============================================================
# Plot 2: Leakage Distribution (Histogram)
# ============================================================
cat("Generating leakage_histogram.png...\n")
leakage_thr <- 0.15
alignment_matrix_vals <- as.matrix(alignment_matrix)
leakage_per_dim <- rowSums(abs(alignment_matrix_vals) > leakage_thr, na.rm = TRUE)

df_leakage <- data.frame(leakage = leakage_per_dim)

p2 <- ggplot(df_leakage, aes(x = leakage)) +
  geom_histogram(bins = 30, fill = colors["blue"], alpha = 0.85, color = "#333333") +
  labs(
    title = "Alignment Leakage Distribution",
    x = paste0("Leakage = #Concepts with |corr| > ", leakage_thr),
    y = "Count of Latent Dimensions"
  ) +
  theme_pub

ggsave("R_leakage_histogram.png", p2, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 3: Specialists Bar Chart
# ============================================================
cat("Generating specialists_bar.png...\n")
specialists <- read.csv("alignment_specialists.csv")

p3 <- ggplot(specialists, aes(x = reorder(paste0("dim_", dim), -max_corr), y = max_corr)) +
  geom_bar(stat = "identity", fill = colors["blue"], alpha = 0.85) +
  labs(
    title = "Specialist Dimensions (Top Concepts per Dimension)",
    x = "Latent Dimension",
    y = "|Spearman Correlation|"
  ) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 8))

ggsave("R_specialists_bar.png", p3, width = 12, height = 5, dpi = 300)

# ============================================================
# Plot 4: MI Matrix (from cmig_pairs.csv)
# ============================================================
cat("Generating mi_scatter.png...\n")
cmig_pairs <- read.csv("cmig_pairs.csv")

p4 <- ggplot(cmig_pairs, aes(x = MI, y = NMI)) +
  geom_point(alpha = 0.5, color = colors["blue"]) +
  geom_smooth(method = "lm", se = FALSE, color = colors["red"], linetype = "dashed") +
  labs(
    title = "Mutual Information vs Normalized MI (Latent Pairs)",
    x = "MI",
    y = "NMI"
  ) +
  theme_pub

ggsave("R_mi_scatter.png", p4, width = 7, height = 5, dpi = 300)

# ============================================================
# Plot 5: MI Histogram
# ============================================================
cat("Generating mi_histogram.png...\n")
p5 <- ggplot(cmig_pairs, aes(x = MI)) +
  geom_histogram(bins = 40, fill = colors["purple"], alpha = 0.85, color = "#333333") +
  labs(
    title = "Mutual Information Distribution (Latent Dim Pairs)",
    x = "MI",
    y = "Count"
  ) +
  theme_pub

ggsave("R_mi_histogram.png", p5, width = 7, height = 5, dpi = 300)

cat("All Experiment 1 plots generated!\n")
