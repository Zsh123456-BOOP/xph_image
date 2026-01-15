#!/usr/bin/env Rscript
# ============================================================
# Experiment 3: Interaction & Q-Noise (Publication Style)
# Output: /home/zsh/xph_image/R_image/exp3/assist_09/
# NOTE: Fix layout_with_fr() weight positivity requirement:
#       use weights = abs(weight) + eps
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(reshape2)
  library(gridExtra)
  library(igraph)
  library(scales)
})

# ------------------------------------------------------------
# 0) One-place style controls (adjust ONLY here)
# ------------------------------------------------------------
CD_STYLE <- list(
  font_family = "sans",
  base_size = 12,

  axis_title_size = 12,
  axis_text_size  = 10,
  legend_title_size = 10,
  legend_text_size  = 9,

  lw_main = 1.15,
  lw_grid = 0.35,
  pt_main = 3.0,

  num_acc = 0.001,

  # figure size (inch)
  fig_main_w   = 6.0,
  fig_main_h   = 4.5,
  fig_single_w = 5.0,
  fig_single_h = 4.5,
  fig_combo_w  = 10.0,
  fig_combo_h  = 4.5,
  fig_heat_w   = 5.5,
  fig_heat_h   = 5.0
)

CD_PAL <- list(
  blue   = "#2563EB",
  orange = "#F59E0B",
  red    = "#EF4444",
  purple = "#8B5CF6",
  teal   = "#0EA5A4",
  gray   = "#6B7280",
  grid   = "#E5E7EB",
  ink    = "#111827"
)

theme_pub <- function() {
  theme_classic(base_size = CD_STYLE$base_size, base_family = CD_STYLE$font_family) +
    theme(
      text = element_text(color = CD_PAL$ink, face = "bold"),
      plot.title = element_blank(),
      axis.title = element_text(size = CD_STYLE$axis_title_size, face = "bold"),
      axis.text  = element_text(size = CD_STYLE$axis_text_size),
      legend.title = element_text(size = CD_STYLE$legend_title_size, face = "bold"),
      legend.text  = element_text(size = CD_STYLE$legend_text_size, face = "bold"),
      legend.position = "right",
      panel.grid.major = element_line(color = CD_PAL$grid, linewidth = CD_STYLE$lw_grid),
      panel.grid.minor = element_blank(),
      axis.line = element_line(linewidth = 0.5),
      strip.background = element_blank()
    )
}

lab_num <- function() label_number(accuracy = CD_STYLE$num_acc)

save_pdf <- function(path, plot, w, h) {
  ggsave(
    filename = path,
    plot = plot,
    width = w, height = h, units = "in",
    device = grDevices::cairo_pdf, bg = "white"
  )
}

# ------------------------------------------------------------
# 1) Run
# ------------------------------------------------------------
base_dir <- "/home/zsh/xph_image"
r_image_dir <- file.path(base_dir, "R_image")
dataset <- "assist_09"

cat(sprintf("\n[Exp3] Processing %s\n", dataset))

data_dir <- file.path(base_dir, "exp_m3_out", dataset)
out_dir <- file.path(r_image_dir, "exp3", dataset)
single_dir <- file.path(out_dir, "single")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)

if (!dir.exists(data_dir)) stop("Data dir not found")

# ============================================================
# 1) Q-Noise Curve
# ============================================================
if (file.exists(file.path(data_dir, "qnoise_curve.csv"))) {
  qnoise <- read.csv(file.path(data_dir, "qnoise_curve.csv"))

  p <- ggplot(qnoise, aes(x = rho, y = auc, color = mode, shape = mode)) +
    geom_line(linewidth = CD_STYLE$lw_main, lineend = "round") +
    geom_point(size = CD_STYLE$pt_main) +
    scale_color_manual(values = c("missing" = CD_PAL$orange, "false" = CD_PAL$red)) +
    scale_y_continuous(labels = lab_num()) +
    labs(x = expression(rho ~ "(Noise Rate)"), y = "Test AUC", color = "Mode", shape = "Mode") +
    theme_pub() + theme(legend.position = c(0.8, 0.8))

  save_pdf(file.path(out_dir, "qnoise_curve.pdf"), p, CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
}

# ============================================================
# 2) Q-Noise Combo (Split + Combined)
# ============================================================
if (file.exists(file.path(data_dir, "qnoise_curve.csv")) &&
    file.exists(file.path(data_dir, "qnoise_hard_curve.csv"))) {

  qnoise <- read.csv(file.path(data_dir, "qnoise_curve.csv"))
  qhard <- read.csv(file.path(data_dir, "qnoise_hard_curve.csv"))

  all_vals <- c(qnoise$auc, qhard$auc)
  ylim <- c(min(all_vals) * 0.999, max(all_vals) * 1.001)

  p1 <- ggplot(qnoise, aes(x = rho, y = auc, color = mode)) +
    geom_line(linewidth = CD_STYLE$lw_main, lineend = "round") +
    geom_point(size = CD_STYLE$pt_main) +
    scale_color_manual(values = c("missing" = CD_PAL$orange, "false" = CD_PAL$red)) +
    scale_y_continuous(limits = ylim, labels = lab_num()) +
    labs(x = expression(rho), y = "Test AUC") +
    theme_pub() + theme(legend.position = "bottom")

  p2 <- ggplot(qhard, aes(x = rho, y = auc)) +
    geom_line(color = CD_PAL$purple, linewidth = CD_STYLE$lw_main, lineend = "round") +
    geom_point(color = CD_PAL$purple, size = CD_STYLE$pt_main) +
    scale_y_continuous(limits = ylim, labels = lab_num()) +
    labs(x = expression(rho), y = "Test AUC") +
    theme_pub()

  save_pdf(file.path(single_dir, "qnoise_missing_false.pdf"), p1, CD_STYLE$fig_single_w, CD_STYLE$fig_single_h)
  save_pdf(file.path(single_dir, "qnoise_hard_false.pdf"), p2, CD_STYLE$fig_single_w, CD_STYLE$fig_single_h)

  p_combo <- gridExtra::arrangeGrob(p1, p2, ncol = 2)
  save_pdf(file.path(out_dir, "fig_m3_qnoise_combo.pdf"), p_combo, CD_STYLE$fig_combo_w, CD_STYLE$fig_combo_h)
}

# ============================================================
# 3) Attribution (Violin + Ridge)
# ============================================================
if (file.exists(file.path(data_dir, "attribution_table.csv"))) {
  attr <- read.csv(file.path(data_dir, "attribution_table.csv"))
  attr$bucket <- cut(attr$concept_count, breaks = c(0, 2, 3, 4, Inf),
                     labels = c("2", "3", "4", "5+"), right = TRUE)

  # Violin
  p <- ggplot(attr, aes(x = bucket, y = attr_logit_mean, fill = bucket)) +
    geom_violin(alpha = 0.6, color = NA, trim = FALSE) +
    stat_summary(fun = median, geom = "point", shape = 18, size = 3, color = CD_PAL$ink) +
    scale_fill_brewer(palette = "Pastel1") +
    scale_y_continuous(labels = lab_num()) +
    labs(x = "Concept Count (|Kq|)", y = "Mean Attribution (Logit)") +
    theme_pub() + theme(legend.position = "none")

  save_pdf(file.path(out_dir, "attribution_violin.pdf"), p, CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)

  # Ridge (Density)
  p2 <- ggplot(attr, aes(x = attr_logit_mean, color = bucket)) +
    geom_density(linewidth = CD_STYLE$lw_main) +
    scale_color_brewer(palette = "Set1") +
    labs(x = "Attribution Score", y = "Density", color = "|Kq|") +
    theme_pub() + theme(legend.position = c(0.85, 0.8))

  save_pdf(file.path(out_dir, "attribution_ridge.pdf"), p2, CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
}

# ============================================================
# 4) Interaction Heatmap
# ============================================================
if (file.exists(file.path(data_dir, "interaction_matrix.csv"))) {
  syn_mat <- as.matrix(read.csv(file.path(data_dir, "interaction_matrix.csv"), row.names = 1))
  syn_long <- melt(syn_mat); colnames(syn_long) <- c("C1", "C2", "Val")

  p <- ggplot(syn_long, aes(x = C1, y = C2, fill = Val)) +
    geom_tile(color = "white", linewidth = 0.15) +
    scale_fill_gradient2(low = CD_PAL$blue, mid = "white", high = CD_PAL$red, midpoint = 0) +
    scale_x_discrete(labels = function(x) gsub("cpt_", "", x)) +
    scale_y_discrete(labels = function(x) gsub("cpt_", "", x)) +
    labs(x = "", y = "", fill = "Synergy") +
    theme_pub() + coord_fixed() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  save_pdf(file.path(out_dir, "interaction_heatmap.pdf"), p, CD_STYLE$fig_heat_w, CD_STYLE$fig_heat_h)
}

# ============================================================
# 5) Interaction Network (FIXED: positive layout weights)
# ============================================================
if (file.exists(file.path(data_dir, "interaction_matrix.csv"))) {
  syn_mat <- as.matrix(read.csv(file.path(data_dir, "interaction_matrix.csv"), row.names = 1))
  concepts <- gsub("cpt_", "", rownames(syn_mat))
  n <- nrow(syn_mat)

  edges <- data.frame()
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      w <- syn_mat[i, j]
      if (abs(w) > 1e-6) {
        edges <- rbind(edges, data.frame(from = concepts[i], to = concepts[j], weight = w))
      }
    }
  }

  if (nrow(edges) > 0) {
    g <- graph_from_data_frame(edges, directed = FALSE, vertices = data.frame(name = concepts))

    # --- FIX: layout weights must be positive
    eps <- 1e-6
    w_layout <- pmax(abs(E(g)$weight), eps)

    set.seed(42)
    lay <- layout_with_fr(g, weights = w_layout)  # always positive now

    # plot attrs (keep sign for color)
    edge_cols <- ifelse(E(g)$weight > 0,
                        adjustcolor(CD_PAL$red, 0.70),
                        adjustcolor(CD_PAL$blue, 0.70))
    edge_w <- pmax(1.0, abs(E(g)$weight) * 5 + 1)

    grDevices::cairo_pdf(file.path(out_dir, "interaction_network.pdf"), width = 6, height = 6, bg = "white")
    par(mar = c(1, 1, 1, 1), bg = "white", family = CD_STYLE$font_family)

    plot(
      g, layout = lay,
      vertex.size = 30,
      vertex.color = "white",
      vertex.frame.color = "black",
      vertex.label.color = "black",
      vertex.label.font = 2,      # bold
      edge.width = edge_w,
      edge.color = edge_cols
    )

    invisible(dev.off())
  }
}
