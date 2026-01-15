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
  library(igraph)
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
    theme(legend.position = "bottom")

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
# 3) Attribution (Violin + Ridge) - Enhanced
# ============================================================
if (file.exists(file.path(data_dir, "attribution_table.csv"))) {
  attr <- read.csv(file.path(data_dir, "attribution_table.csv"))
  attr$bucket <- cut(attr$concept_count, breaks = c(0, 2, 3, 4, Inf),
                     labels = c("2", "3", "4", "5+"), right = TRUE)

  # Enhanced color palette for buckets
  bucket_colors <- c(
    "2" = CD_PAL$blue, 
    "3" = CD_PAL$teal, 
    "4" = CD_PAL$orange, 
    "5+" = CD_PAL$red
  )

  # Violin plot
  p <- ggplot(attr, aes(x = bucket, y = attr_logit_mean, fill = bucket)) +
    geom_violin(
      alpha = 0.7, 
      color = CD_PAL$dark, 
      linewidth = 0.5,
      trim = FALSE,
      scale = "width"
    ) +
    geom_boxplot(
      width = 0.15, 
      fill = "white", 
      color = CD_PAL$dark,
      outlier.shape = NA,
      alpha = 0.8
    ) +
    stat_summary(
      fun = median, 
      geom = "point", 
      shape = 18, 
      size = 4, 
      color = CD_PAL$dark
    ) +
    scale_fill_manual(values = bucket_colors) +
    scale_y_continuous(labels = lab_num(0.01)) +
    labs(
      x = expression(bold("Concept Count (|K"[q]*"|)")), 
      y = "Mean Attribution (Logit)"
    ) +
    theme_cd_pub() + 
    theme(legend.position = "none")

  save_pdf(file.path(out_dir, "attribution_violin.pdf"), p, 
           CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)

  # Ridge (Density) plot
  p2 <- ggplot(attr, aes(x = attr_logit_mean, fill = bucket, color = bucket)) +
    geom_density(alpha = 0.5, linewidth = CD_STYLE$lw_main) +
    scale_fill_manual(values = bucket_colors, name = "|Kq|") +
    scale_color_manual(values = bucket_colors, name = "|Kq|") +
    labs(x = "Attribution Score", y = "Density") +
    theme_cd_pub() + 
    theme(
      legend.position = c(0.85, 0.75),
      legend.background = element_rect(fill = alpha("white", 0.9), color = NA)
    )

  save_pdf(file.path(out_dir, "attribution_ridge.pdf"), p2, 
           CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
}

# ============================================================
# 4) Interaction Heatmap - Matching reference style (coolwarm)
# ============================================================
if (file.exists(file.path(data_dir, "interaction_matrix.csv"))) {
  syn_mat <- as.matrix(read.csv(file.path(data_dir, "interaction_matrix.csv"), row.names = 1))
  syn_long <- melt(syn_mat)
  colnames(syn_long) <- c("C1", "C2", "Val")

  # Determine color limits symmetrically
  max_abs <- max(abs(syn_long$Val), na.rm = TRUE)
  
  # Use coolwarm-style colors matching reference image
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
    labs(x = NULL, y = NULL) +
    theme_cd_pub() + 
    coord_fixed() +
    theme(
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

# ============================================================
# 5) Interaction Network - Enhanced
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

    # Layout weights must be positive
    eps <- 1e-6
    w_layout <- pmax(abs(E(g)$weight), eps)

    set.seed(42)
    lay <- layout_with_fr(g, weights = w_layout)

    # Enhanced visual attributes
    edge_cols <- ifelse(
      E(g)$weight > 0,
      adjustcolor("#B40426", 0.80),  # Coolwarm red
      adjustcolor("#3B4CC0", 0.80)   # Coolwarm blue
    )
    edge_w <- pmax(1.5, abs(E(g)$weight) * 6 + 1.5)

    # Calculate node degree for sizing
    node_degree <- degree(g)
    node_sizes <- 25 + (node_degree / max(node_degree)) * 15

    grDevices::cairo_pdf(
      file.path(out_dir, "interaction_network.pdf"), 
      width = 7, height = 7, bg = "white"
    )
    par(mar = c(1, 1, 1, 1), bg = "white", family = CD_STYLE$font_family)

    plot(
      g, layout = lay,
      vertex.size = node_sizes,
      vertex.color = adjustcolor(CD_PAL$teal, 0.3),
      vertex.frame.color = CD_PAL$dark,
      vertex.frame.width = 1.5,
      vertex.label.color = CD_PAL$dark,
      vertex.label.font = 2,
      vertex.label.cex = 1.0,
      vertex.label.family = CD_STYLE$font_family,
      edge.width = edge_w,
      edge.color = edge_cols,
      edge.curved = 0.15
    )

    # Add legend
    legend(
      "bottomleft",
      legend = c("Positive (Synergy)", "Negative (Antagonism)"),
      col = c("#B40426", "#3B4CC0"),
      lwd = 3,
      bty = "n",
      cex = 0.9
    )

    invisible(dev.off())
    cat("[SAVED] interaction_network.pdf\n")
  }
}

cat("\n[Exp3] Completed\n")
