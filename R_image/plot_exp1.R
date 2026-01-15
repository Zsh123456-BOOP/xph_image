#!/usr/bin/env Rscript
# ============================================================
# Experiment 1: Disentanglement Analysis (Enhanced Visuals)
# Output: /home/zsh/xph_image/R_image/exp1/{dataset}/
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(reshape2)
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
  stop(sprintf("[Exp1][ERR] style file not found: %s", style_path))
}
source(style_path, encoding = "UTF-8")

datasets <- c("assist_09", "assist_17", "junyi")

# Helper functions
.safe_read_csv <- function(path, row.names = NULL) {
  tryCatch({
    read.csv(path, row.names = row.names, check.names = FALSE)
  }, error = function(e) {
    stop(sprintf("[Exp1][ERR] Failed to read csv: %s | %s", path, e$message))
  })
}

.as_int <- function(x) as.integer(as.character(x))
.as_num <- function(x) as.numeric(as.character(x))

for (dataset in datasets) {
  cat(sprintf("\n[Exp1] Processing %s\n", dataset))

  data_dir <- file.path(base_dir, "exp_m1_out", dataset)
  out_dir  <- file.path(r_image_dir, "exp1", dataset)
  single_dir <- file.path(out_dir, "single")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(single_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(data_dir)) next

  leakage_thr <- 0.15

  # ------------------------------------------------------------
  # 1) Alignment Leakage (Histogram) - Enhanced
  # ------------------------------------------------------------
  if (file.exists(file.path(data_dir, "alignment_matrix.csv"))) {
    R_df <- .safe_read_csv(file.path(data_dir, "alignment_matrix.csv"), row.names = 1)
    R_mat <- as.matrix(R_df)
    storage.mode(R_mat) <- "numeric"

    leakage <- rowSums(abs(R_mat) > leakage_thr, na.rm = TRUE)
    df_leak <- data.frame(leakage = leakage)

    p <- ggplot(df_leak, aes(x = leakage)) +
      geom_histogram(
        binwidth = 1,
        fill = CD_PAL$teal, 
        color = CD_PAL$dark,
        alpha = CD_STYLE$alpha_fill, 
        linewidth = 0.5
      ) +
      geom_vline(
        xintercept = mean(leakage), 
        linetype = "dashed", 
        color = CD_PAL$red, 
        linewidth = 0.8
      ) +
      scale_x_continuous(breaks = pretty_breaks(n = 8)) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
      labs(
        x = bquote(bold("Leakage Count (|corr| > " ~ .(leakage_thr) ~ ")")),
        y = "Count (Latent Dims)"
      ) +
      theme_cd_pub()

    save_pdf(file.path(out_dir, "alignment_leakage.pdf"), p, w = 5.0, h = 3.8)
  }

  # ------------------------------------------------------------
  # 2) Specialist Dimensions (Bar Chart) - Simple style
  # ------------------------------------------------------------
  if (file.exists(file.path(data_dir, "alignment_specialists.csv"))) {
    spec <- .safe_read_csv(file.path(data_dir, "alignment_specialists.csv"))

    bar_data <- data.frame()
    for (i in seq_len(nrow(spec))) {
      if ("top1_concept" %in% names(spec) && !is.na(spec$top1_concept[i])) {
        bar_data <- rbind(
          bar_data,
          data.frame(
            dim = .as_int(spec$dim[i]),
            label = sprintf("d%d:c%d", .as_int(spec$dim[i]), .as_int(spec$top1_concept[i])),
            corr = .as_num(spec$top1_corr[i])
          )
        )
      }
    }

    if (nrow(bar_data) > 0) {
      bar_data <- bar_data[order(bar_data$dim), ]
      bar_data$label <- factor(bar_data$label, levels = bar_data$label)

      # Simple bar chart matching reference
      p <- ggplot(bar_data, aes(x = label, y = corr)) +
        geom_bar(
          stat = "identity",
          fill = "#4472C4",  # Blue color matching reference
          color = NA,
          width = 0.75
        ) +
        scale_y_continuous(
          limits = c(0, max(bar_data$corr) * 1.1), 
          breaks = pretty_breaks(n = 5),
          expand = c(0, 0)
        ) +
        labs(
          x = "Dimension:Concept Mapping", 
          y = "Spearman Corr"
        ) +
        theme_cd_pub() +
        theme(
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size = 10),
          panel.grid.major.x = element_blank()
        )

      save_pdf(file.path(out_dir, "alignment_specialist_dims.pdf"), p, w = 7.5, h = 4.0)
    }
  }


  # ------------------------------------------------------------
  # 4) Combo: Leakage (Bubble + ECDF) - Enhanced
  # ------------------------------------------------------------
  if (file.exists(file.path(data_dir, "alignment_matrix.csv"))) {
    R_df <- .safe_read_csv(file.path(data_dir, "alignment_matrix.csv"), row.names = 1)
    R_mat <- as.matrix(R_df)
    storage.mode(R_mat) <- "numeric"

    leakage <- rowSums(abs(R_mat) > leakage_thr, na.rm = TRUE)
    max_corr <- apply(abs(R_mat), 1, max, na.rm = TRUE)
    mean_corr <- rowMeans(abs(R_mat), na.rm = TRUE)
    df_leak <- data.frame(leakage = leakage, max_corr = max_corr, mean_corr = mean_corr)

    # Bubble plot
    p1 <- ggplot(df_leak, aes(x = leakage, y = max_corr)) +
      geom_vline(
        xintercept = 2.5, 
        linetype = "dashed", 
        color = CD_PAL$gray, 
        linewidth = 0.8
      ) +
      geom_point(
        aes(size = max_corr, fill = mean_corr),
        shape = 21, 
        color = "white", 
        stroke = 0.5, 
        alpha = 0.85
      ) +
      scale_fill_viridis_c(
        option = "viridis", 
        name = "Mean\nCorr",
        guide = guide_colorbar(barwidth = 1.0, barheight = 5)
      ) +
      scale_size_continuous(range = c(3, 8), guide = "none") +
      labs(x = "Leakage Count", y = "Max Correlation") +
      theme_cd_pub() +
      theme(legend.position = "right")

    # ECDF plot
    p2 <- ggplot(df_leak, aes(x = leakage)) +
      stat_ecdf(
        geom = "step", 
        color = CD_PAL$blue, 
        linewidth = CD_STYLE$lw_main
      ) +
      geom_hline(
        yintercept = 0.5, 
        linetype = "dotted", 
        color = CD_PAL$gray
      ) +
      scale_y_continuous(labels = percent_format(accuracy = 1)) +
      labs(x = "Leakage Count", y = "Cumulative Proportion") +
      theme_cd_pub()

    # Save individual plots
    save_pdf(file.path(single_dir, "alignment_leakage_bubble.pdf"), p1, w = 5.0, h = 3.8)
    save_pdf(file.path(single_dir, "alignment_leakage_ecdf.pdf"), p2, w = 4.5, h = 3.5)

    # Combo
    p_combo <- cowplot::plot_grid(
      p1, p2, 
      ncol = 2, 
      rel_widths = c(1.2, 1), 
      align = "h",
      labels = c("A", "B"),
      label_size = 14,
      label_fontface = "bold"
    )
    save_pdf(file.path(out_dir, "combo_alignment_leakage.pdf"), p_combo, w = 10.0, h = 4.0)
  }
}

cat("\n[Exp1] Completed all datasets\n")
