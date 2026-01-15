#!/usr/bin/env Rscript
# ============================================================
# Experiment 3: Interaction & Q-Noise (Enhanced Visuals)
# Output: /home/zsh/xph_image/R_image/exp3/assist_09/
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(reshape2)
  library(gridExtra)
  library(cowplot)
  library(scales)
})

base_dir <- "/home/zsh/xph_image"
r_image_dir <- file.path(base_dir, "R_image")

# -----------------------------
# Load unified style
# -----------------------------
style_path <- file.path(r_image_dir, "_style_cd.R")
if (!file.exists(style_path)) {
  stop(sprintf("[Exp3][ERR] style file not found: %s", style_path))
}
source(style_path, encoding = "UTF-8")

dataset <- "assist_09"
cat(sprintf("\n[Exp3] Processing %s\n", dataset))

data_dir <- file.path(base_dir, "exp_m3_out", dataset)
out_dir <- file.path(r_image_dir, "exp3", dataset)
single_dir <- file.path(out_dir, "single")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)

if (!dir.exists(data_dir)) stop("Data dir not found")

# ============================================================
# 1) Q-Noise Curve - Enhanced (legend moved up)
# ============================================================
if (file.exists(file.path(data_dir, "qnoise_curve.csv"))) {
  qnoise <- read.csv(file.path(data_dir, "qnoise_curve.csv"))

  p <- ggplot(qnoise, aes(x = rho, y = auc, color = mode, shape = mode)) +
    geom_line(linewidth = CD_STYLE$lw_main, lineend = "round") +
    geom_point(size = CD_STYLE$pt_big) +
    scale_color_manual(
      values = c("missing" = CD_PAL$orange, "false" = CD_PAL$red),
      labels = c("missing" = "Missing Edges", "false" = "False Edges")
    ) +
    scale_shape_manual(
      values = c("missing" = 17, "false" = 15),
      labels = c("missing" = "Missing Edges", "false" = "False Edges")
    ) +
    scale_y_continuous(labels = lab_num(0.001)) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      x = expression(bold(rho ~ "(Noise Rate)")), 
      y = "Test AUC", 
      color = "Mode", 
      shape = "Mode"
    ) +
    theme_cd_pub() + 
    theme(
      # Move legend to top-right corner (inside plot)
      legend.position = c(0.80, 0.65),
      legend.background = element_rect(fill = alpha("white", 0.9), color = NA),
      legend.key.size = unit(0.6, "cm")
    )

  save_pdf(file.path(out_dir, "qnoise_curve.pdf"), p, 
           CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
}

# ============================================================
# 2) Q-Noise Combo (Split + Combined) - No A/B labels
# ============================================================
if (file.exists(file.path(data_dir, "qnoise_curve.csv")) &&
    file.exists(file.path(data_dir, "qnoise_hard_curve.csv"))) {

  qnoise <- read.csv(file.path(data_dir, "qnoise_curve.csv"))
  qhard <- read.csv(file.path(data_dir, "qnoise_hard_curve.csv"))

  all_vals <- c(qnoise$auc, qhard$auc)
  ylim <- c(min(all_vals) * 0.998, max(all_vals) * 1.002)

  # Left: Missing vs False (has its own legend)
  p1 <- ggplot(qnoise, aes(x = rho, y = auc, color = mode, shape = mode)) +
    geom_line(linewidth = CD_STYLE$lw_main, lineend = "round") +
    geom_point(size = CD_STYLE$pt_big) +
    scale_color_manual(
      values = c("missing" = CD_PAL$orange, "false" = CD_PAL$red),
      name = "Noise Type"
    ) +
    scale_shape_manual(
      values = c("missing" = 17, "false" = 15),
      name = "Noise Type"
    ) +
    scale_y_continuous(limits = ylim, labels = lab_num(0.001)) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(x = expression(rho ~ "(Noise Rate)"), y = "Test AUC") +
    theme_cd_pub() + 
    theme(
      legend.position = "bottom",
      legend.box.spacing = unit(0.8, "cm"),
      legend.spacing.x = unit(1.0, "cm"),
      legend.key.width = unit(1.5, "cm"),
      legend.text = element_text(margin = margin(l = 8, r = 15))
    )

  # Right: Hard False
  p2 <- ggplot(qhard, aes(x = rho, y = auc)) +
    geom_line(
      color = CD_PAL$purple, 
      linewidth = CD_STYLE$lw_main, 
      lineend = "round"
    ) +
    geom_point(
      color = CD_PAL$purple, 
      size = CD_STYLE$pt_big,
      shape = 18
    ) +
    geom_ribbon(
      aes(ymin = ylim[1], ymax = auc),
      fill = CD_PAL$purple,
      alpha = 0.15
    ) +
    scale_y_continuous(limits = ylim, labels = lab_num(0.001)) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(x = expression(rho ~ "(Noise Rate)"), y = "Test AUC") +
    ggtitle("Hard False Noise") +
    theme_cd_pub() +
    theme(plot.title = element_text(
      size = 12, face = "bold", hjust = 0.5, 
      margin = margin(b = 10),
      family = CD_STYLE$font_family
    ))

  save_pdf(file.path(single_dir, "qnoise_missing_false.pdf"), p1, 
           CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
  save_pdf(file.path(single_dir, "qnoise_hard_false.pdf"), p2, 
           CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)

  # Combo - NO A/B labels (legend belongs to left panel)
  p_combo <- cowplot::plot_grid(
    p1, p2, 
    ncol = 2, 
    align = "h"
  )
  save_pdf(file.path(out_dir, "fig_m3_qnoise_combo.pdf"), p_combo, 
           CD_STYLE$fig_combo_w, CD_STYLE$fig_combo_h)
}

# ============================================================
# 3) Interaction Heatmap - Matching reference style (coolwarm)
# ============================================================
if (file.exists(file.path(data_dir, "interaction_matrix.csv"))) {
  syn_mat <- as.matrix(read.csv(file.path(data_dir, "interaction_matrix.csv"), row.names = 1))
  syn_long <- melt(syn_mat)
  colnames(syn_long) <- c("C1", "C2", "Val")

  # Determine color limits symmetrically
  max_abs <- max(abs(syn_long$Val), na.rm = TRUE)
  
  # Use coolwarm-style colors matching reference image
  # Get exercise count for title (use placeholder if not available)
  exercise_count <- nrow(syn_mat)
  
  p <- ggplot(syn_long, aes(x = C1, y = C2, fill = Val)) +
    geom_tile(color = NA) +  # No white borders for cleaner look
    scale_fill_gradient2(
      low = "#3B4CC0",      # Coolwarm blue
      mid = "#F0F0F0",      # Light gray mid
      high = "#B40426",     # Coolwarm red
      midpoint = 0,
      limits = c(-max_abs, max_abs),
      name = "Synergy",
      guide = guide_colorbar(barwidth = 1.5, barheight = 12, title.position = "top")
    ) +
    scale_x_discrete(labels = function(x) gsub("cpt_", "", x), expand = c(0, 0)) +
    scale_y_discrete(labels = function(x) gsub("cpt_", "", x), expand = c(0, 0)) +
    labs(
      x = NULL, y = NULL,
      title = sprintf("Interaction Heatmap via Synergy (concepts=%d)", exercise_count)
    ) +
    theme_cd_pub() + 
    coord_fixed() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 10)),
      plot.title.position = "plot",
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 11, face = "bold"),
      axis.text.y = element_text(size = 11, face = "bold"),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 12, face = "bold", hjust = 0.5)
    )

  save_pdf(file.path(out_dir, "interaction_heatmap.pdf"), p, 
           CD_STYLE$fig_square_w, CD_STYLE$fig_square_h)
}

cat("\n[Exp3] Completed\n")
