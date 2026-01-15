#!/usr/bin/env Rscript
# ============================================================
# Experiment 2: Robustness & Consistency (Enhanced Visuals)
# Output: /home/zsh/xph_image/R_image/exp2/{dataset}/
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(gridExtra)
  library(cowplot)
})

base_dir <- "/home/zsh/xph_image"
r_image_dir <- file.path(base_dir, "R_image")

# -----------------------------
# Load unified style (replaces local definitions)
# -----------------------------
style_path <- file.path(r_image_dir, "_style_cd.R")
if (!file.exists(style_path)) {
  stop(sprintf("[Exp2][ERR] style file not found: %s", style_path))
}
source(style_path, encoding = "UTF-8")

datasets <- c("assist_09", "assist_17", "junyi")

for (dataset in datasets) {
  cat(sprintf("\n[Exp2] Processing %s\n", dataset))

  data_dir <- file.path(base_dir, "exp_m2_out", dataset)
  out_dir <- file.path(r_image_dir, "exp2", dataset)
  single_dir <- file.path(out_dir, "single")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)

  if (!dir.exists(data_dir)) next

  # Storage for combo plots
  p_robust <- NULL
  p_pareto <- NULL

  # ============================================================
  # 1) Robustness Curve (Dual Axis)
  # ============================================================
  if (file.exists(file.path(data_dir, "robust_curve.csv"))) {
    robust <- read.csv(file.path(data_dir, "robust_curve.csv"))

    auc_range <- range(robust$auc)
    acc_range <- range(robust$accuracy)
    robust$acc_scaled <- (robust$accuracy - acc_range[1]) / diff(acc_range) * diff(auc_range) + auc_range[1]

    p_robust <- ggplot(robust, aes(x = drop_rate)) +
      # AUC line
      geom_line(
        aes(y = auc, color = "AUC"), 
        linewidth = CD_STYLE$lw_main, 
        lineend = "round"
      ) +
      geom_point(
        aes(y = auc, color = "AUC"), 
        size = CD_STYLE$pt_main,
        shape = 19
      ) +
      # Accuracy line (dashed)
      geom_line(
        aes(y = acc_scaled, color = "Accuracy"),
        linewidth = CD_STYLE$lw_main, 
        linetype = "dashed", 
        lineend = "round"
      ) +
      geom_point(
        aes(y = acc_scaled, color = "Accuracy"), 
        size = CD_STYLE$pt_main, 
        shape = 17
      ) +
      scale_color_manual(
        values = c("AUC" = CD_PAL$blue, "Accuracy" = CD_PAL$orange),
        name = "Metric"
      ) +
      scale_y_continuous(
        name = "AUC",
        labels = lab_num(0.001),
        sec.axis = sec_axis(
          ~ (. - auc_range[1]) / diff(auc_range) * diff(acc_range) + acc_range[1],
          name = "ACC",
          labels = lab_num(0.001)
        )
      ) +
      scale_x_continuous(
        labels = percent_format(accuracy = 1),
        breaks = pretty_breaks(n = 6)
      ) +
      labs(x = "Graph Edge Dropout Rate") +
      theme_cd_pub() +
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal",
        axis.title.y.right = element_text(
          color = CD_PAL$orange, 
          face = "bold",
          margin = margin(l = 10),
          family = CD_STYLE$font_family
        ),
        axis.text.y.right = element_text(
          color = CD_PAL$orange,
          family = CD_STYLE$font_family
        ),
        axis.title.y.left = element_text(
          color = CD_PAL$blue,
          margin = margin(r = 10),
          family = CD_STYLE$font_family
        ),
        axis.text.y.left = element_text(
          color = CD_PAL$blue,
          family = CD_STYLE$font_family
        )
      )

    save_pdf(file.path(out_dir, "robust_curve.pdf"), p_robust, 
             CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)
  }

  # ============================================================
  # 2) Pareto Trajectory
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

    # Use plotmath for lambda expression
    label_expr <- sprintf('lambda^"*" == %.3f', lambda_star)

    # Trajectory plot
    p1 <- ggplot(pareto, aes(x = D_view_mean, y = error)) +
      geom_path(
        linetype = "dashed", 
        color = CD_PAL$gray, 
        linewidth = 1.0,
        alpha = 0.7
      ) +
      geom_point(
        aes(color = lambda_contrastive), 
        size = CD_STYLE$pt_big, 
        alpha = 0.9
      ) +
      scale_color_viridis_c(
        name = expression(lambda), 
        option = "plasma",
        guide = guide_colorbar(barwidth = 0.8, barheight = 3)
      ) +
      # Highlight optimal point - simple filled circle
      geom_point(
        data = pareto[idx_star, ],
        aes(x = D_view_mean, y = error),
        shape = 21,
        size = 6, 
        color = CD_PAL$dark,
        fill = CD_PAL$red,
        stroke = 1.5
      ) +
      labs(
        x = expression(bold(D[view] ~ "(Consistency Loss)")),
        y = "Error Rate (1 - AUC)"
      ) +
      annotate(
        "text",
        x = pareto$D_view_mean[idx_star],
        y = pareto$error[idx_star],
        label = label_expr,
        parse = TRUE,
        color = CD_PAL$red,
        size = 4.2,
        hjust = -0.15,
        vjust = -0.8,
        fontface = "bold",
        family = CD_STYLE$font_family
      ) +
      theme_cd_pub() +
      theme(legend.position = "right")

    save_pdf(file.path(single_dir, "pareto_trajectory.pdf"), p1, 
             CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)

    # Evidence plot
    if (all(c("D_exer", "D_cpt") %in% names(pareto))) {
      p2 <- ggplot(pareto, aes(x = D_exer, y = D_cpt, color = lambda_contrastive)) +
        geom_abline(
          slope = 1, intercept = 0, 
          linetype = "dashed", 
          color = CD_PAL$grid, 
          linewidth = 1.0
        ) +
        geom_point(size = CD_STYLE$pt_big, alpha = 0.9) +
        scale_color_viridis_c(
          name = expression(lambda),
          option = "plasma",
          guide = guide_colorbar(barwidth = 0.8, barheight = 3)
        ) +
        geom_point(
          data = pareto[idx_star, ],
          aes(x = D_exer, y = D_cpt),
          shape = 21,
          size = 6, 
          color = CD_PAL$dark,
          fill = CD_PAL$red,
          stroke = 1.5
        ) +
        labs(
          x = expression(bold(D[exer] ~ "(Exercise View)")),
          y = expression(bold(D[cpt] ~ "(Concept View)"))
        ) +
        theme_cd_pub() +
        theme(legend.position = "right")
    } else {
      p2 <- ggplot(pareto, aes(x = lambda_contrastive, y = test_auc)) +
        geom_line(color = CD_PAL$blue, linewidth = CD_STYLE$lw_main) +
        geom_point(color = CD_PAL$blue, size = CD_STYLE$pt_main) +
        labs(x = expression(lambda), y = "Test AUC") +
        theme_cd_pub()
    }

    save_pdf(file.path(single_dir, "pareto_evidence.pdf"), p2, 
             CD_STYLE$fig_main_w, CD_STYLE$fig_main_h)

    # Pareto combo (horizontal)
    p_pareto <- cowplot::plot_grid(
      p1, p2, 
      ncol = 2, 
      align = "h"
    )
    save_pdf(file.path(out_dir, "pareto.pdf"), p_pareto, 
             CD_STYLE$fig_combo_w + 2, CD_STYLE$fig_combo_h)
  }

  # ============================================================
  # 3) Combined: Robust + Pareto side by side (horizontal)
  # ============================================================
  if (!is.null(p_robust) && !is.null(p_pareto)) {
    # Modify robust plot for combo (larger x-axis font)
    p_robust_combo <- p_robust + 
      theme(
        axis.title.x = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(size = 12)
      )
    
    # Create horizontal combo (side by side)
    p_full_combo <- cowplot::plot_grid(
      p_robust_combo,
      p_pareto,
      ncol = 2,
      rel_widths = c(1, 1.8),
      align = "h"
    )
    save_pdf(file.path(out_dir, "combo_robust_pareto.pdf"), p_full_combo, 
             w = 16, h = 4.5)
  }
}

cat("\n[Exp2] Completed all datasets\n")
