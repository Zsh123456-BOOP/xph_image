#!/usr/bin/env Rscript
# ============================================================
# Experiment 2: Robustness & Consistency (Publication Style)
# Output: /home/zsh/xph_image/R_image/exp2/{dataset}/
# NOTE: Fix encoding warning for "λ* = ..." via plotmath (parse=TRUE)
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(gridExtra)
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
  fig_single_w = 5.0,
  fig_single_h = 4.0,
  fig_main_w   = 6.0,
  fig_main_h   = 4.5,
  fig_combo_w  = 11.0,
  fig_combo_h  = 4.5
)

CD_PAL <- list(
  blue   = "#2563EB",
  orange = "#F59E0B",
  red    = "#EF4444",
  purple = "#8B5CF6",
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
datasets <- c("assist_09", "assist_17", "junyi")

for (dataset in datasets) {
  cat(sprintf("\n[Exp2] Processing %s\n", dataset))

  data_dir <- file.path(base_dir, "exp_m2_out", dataset)
  out_dir <- file.path(r_image_dir, "exp2", dataset)
  single_dir <- file.path(out_dir, "single")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)

  if (!dir.exists(data_dir)) next

  # ============================================================
  # 1) Robustness Curve (dual axis)
  # ============================================================
  if (file.exists(file.path(data_dir, "robust_curve.csv"))) {
    robust <- read.csv(file.path(data_dir, "robust_curve.csv"))

    auc_range <- range(robust$auc)
    acc_range <- range(robust$accuracy)
    robust$acc_scaled <- (robust$accuracy - acc_range[1]) / diff(acc_range) * diff(auc_range) + auc_range[1]

    p <- ggplot(robust, aes(x = drop_rate)) +
      geom_line(aes(y = auc, color = "AUC"), linewidth = CD_STYLE$lw_main, lineend = "round") +
      geom_point(aes(y = auc, color = "AUC"), size = CD_STYLE$pt_main) +
      geom_line(aes(y = acc_scaled, color = "Accuracy"),
                linewidth = CD_STYLE$lw_main, linetype = "dashed", lineend = "round") +
      geom_point(aes(y = acc_scaled, color = "Accuracy"), size = CD_STYLE$pt_main, shape = 15) +
      scale_color_manual(values = c("AUC" = CD_PAL$blue, "Accuracy" = CD_PAL$orange)) +
      scale_y_continuous(
        name = "Test AUC",
        labels = lab_num(),
        sec.axis = sec_axis(
          ~ (. - auc_range[1]) / diff(auc_range) * diff(acc_range) + acc_range[1],
          name = "Test Accuracy",
          labels = lab_num()
        )
      ) +
      labs(x = "Graph Edge Dropout Rate", color = "") +
      theme_pub() +
      theme(legend.position = "bottom")

    save_pdf(file.path(out_dir, "robust_curve.pdf"), p, CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
  }

  # ============================================================
  # 2) Pareto Trajectory (Combo)
  # ============================================================
  if (file.exists(file.path(data_dir, "pareto.csv"))) {
    pareto <- read.csv(file.path(data_dir, "pareto.csv"))
    pareto <- pareto[order(pareto$lambda_contrastive), ]
    pareto$error <- 1 - pareto$test_auc

    normalize01 <- function(x) {
      if (max(x) == min(x)) return(rep(0, length(x)))
      (x - min(x)) / (max(x) - min(x))
    }

    pareto$score <- normalize01(pareto$D_view_mean) + normalize01(pareto$error)
    idx_star <- which.min(pareto$score)
    lambda_star <- pareto$lambda_contrastive[idx_star]

    # ---- Fix: use plotmath (ASCII) to avoid mbcsToSbcs warning
    # label_expr renders λ^"*" via plotmath, not Unicode "λ"
    label_expr <- sprintf('lambda^"*"%s%.3f', "==", lambda_star)

    # Left: Trajectory
    p1 <- ggplot(pareto, aes(x = D_view_mean, y = error)) +
      geom_path(linetype = "dashed", color = CD_PAL$gray, linewidth = 0.85) +
      geom_point(aes(color = lambda_contrastive), size = CD_STYLE$pt_main + 0.5, alpha = 0.92) +
      scale_color_viridis_c(name = expression(lambda)) +
      geom_point(
        data = pareto[idx_star, ],
        aes(x = D_view_mean, y = error),
        shape = 8, size = 5, color = CD_PAL$red, stroke = 1.4
      ) +
      labs(
        x = expression(D[view] ~ "(Consistency Loss)"),
        y = "Error Rate (1 - AUC)"
      ) +
      annotate(
        "text",
        x = pareto$D_view_mean[idx_star],
        y = pareto$error[idx_star],
        label = label_expr,
        parse = TRUE,
        color = CD_PAL$red,
        size = 3.6,
        hjust = -0.1,
        vjust = -0.6,
        fontface = "bold"
      ) +
      theme_pub()

    save_pdf(file.path(single_dir, "pareto_trajectory.pdf"), p1, CD_STYLE$fig_single_w, CD_STYLE$fig_single_h)

    # Right: Evidence
    if (all(c("D_exer", "D_cpt") %in% names(pareto))) {
      p2 <- ggplot(pareto, aes(x = D_exer, y = D_cpt, color = lambda_contrastive)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = CD_PAL$grid, linewidth = 0.9) +
        geom_point(size = CD_STYLE$pt_main + 0.5, alpha = 0.92) +
        scale_color_viridis_c(name = expression(lambda)) +
        geom_point(
          data = pareto[idx_star, ],
          aes(x = D_exer, y = D_cpt),
          shape = 8, size = 5, color = CD_PAL$red, stroke = 1.4
        ) +
        labs(
          x = expression(D[exer] ~ "(Exercise View)"),
          y = expression(D[cpt] ~ "(Concept View)")
        ) +
        theme_pub()
    } else {
      p2 <- ggplot(pareto, aes(x = lambda_contrastive, y = test_auc)) +
        geom_line(color = CD_PAL$blue, linewidth = CD_STYLE$lw_main) +
        geom_point(color = CD_PAL$blue, size = CD_STYLE$pt_main) +
        labs(x = expression(lambda), y = "Test AUC") +
        theme_pub()
    }

    save_pdf(file.path(single_dir, "pareto_evidence.pdf"), p2, CD_STYLE$fig_single_w, CD_STYLE$fig_single_h)

    # Combo output (do NOT delete any plot)
    p_combo <- gridExtra::arrangeGrob(p1, p2, ncol = 2)
    save_pdf(file.path(out_dir, "pareto.pdf"), p_combo, CD_STYLE$fig_combo_w, CD_STYLE$fig_combo_h)
  }
}
