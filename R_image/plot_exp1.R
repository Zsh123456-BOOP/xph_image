#!/usr/bin/env Rscript
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
# Load style (required)
# -----------------------------
style_path <- file.path(r_image_dir, "_style_cd.R")
if (!file.exists(style_path)) {
  stop(sprintf("[Exp1][ERR] style file not found: %s", style_path))
}
source(style_path, encoding = "UTF-8")

datasets <- c("assist_09", "assist_17", "junyi")

# small helpers
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
  # 1) Alignment Leakage (Histogram)
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
        fill = CD_PAL$teal, color = CD_PAL$navy,
        alpha = CD_STYLE$alpha_fill, linewidth = 0.35
      ) +
      scale_x_continuous(breaks = pretty_breaks()) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
      labs(
        x = bquote("Leakage Count (|corr| > " ~ .(leakage_thr) ~ ")"),
        y = "Count (Latent Dims)"
      ) +
      theme_cd_pub() +
      theme(
        axis.title.x = element_text(face = "bold")
      )

    save_pdf(file.path(out_dir, "alignment_leakage.pdf"), p, w = 4.2, h = 3.2)

  }

  # ------------------------------------------------------------
  # 2) Specialist Dimensions (Lollipop)
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

      p <- ggplot(bar_data, aes(x = label, y = corr)) +
        geom_segment(aes(xend = label, y = 0, yend = corr),
                     color = CD_PAL$gray, linewidth = 0.9) +
        geom_point(
          size = CD_STYLE$pt_main + 0.6,
          shape = 21, stroke = 0.9,
          color = CD_PAL$navy, fill = CD_PAL$blue
        ) +
        scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0, 0.04)) +
        labs(x = "Dimension:Concept Mapping", y = "|Spearman Correlation|") +
        theme_cd_pub() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

      save_pdf(file.path(out_dir, "alignment_specialist_dims.pdf"), p, w = 6.8, h = 3.2)
    }
  }

  # ------------------------------------------------------------
  # 3) MI Matrix (Heatmap)
  # ------------------------------------------------------------
  if (file.exists(file.path(data_dir, "cmig_pairs.csv"))) {
    pairs <- .safe_read_csv(file.path(data_dir, "cmig_pairs.csv"))

    if (nrow(pairs) > 0 && all(c("i", "j", "MI") %in% names(pairs))) {
      pairs$i <- .as_int(pairs$i)
      pairs$j <- .as_int(pairs$j)
      pairs$MI <- .as_num(pairs$MI)

      K <- min(max(c(pairs$i, pairs$j), na.rm = TRUE) + 1L, 64L)
      K <- max(K, 1L)

      M <- matrix(0, nrow = K, ncol = K)
      for (row_idx in seq_len(nrow(pairs))) {
        i <- pairs$i[row_idx] + 1L
        j <- pairs$j[row_idx] + 1L
        if (is.na(i) || is.na(j)) next
        if (i <= K && j <= K && i >= 1L && j >= 1L) {
          M[i, j] <- pairs$MI[row_idx]
          M[j, i] <- pairs$MI[row_idx]
        }
      }

      M_long <- reshape2::melt(M)
      colnames(M_long) <- c("Dim_i", "Dim_j", "MI")

      p <- ggplot(M_long, aes(x = Dim_j, y = Dim_i, fill = MI)) +
        geom_tile(color = "white", linewidth = 0.18) +
        scale_fill_viridis_c(
          option = "magma", direction = -1,
          limits = c(0, 0.2), oob = squish, name = "MI"
        ) +
        scale_x_continuous(expand = c(0,0)) +
        scale_y_continuous(expand = c(0,0)) +
        labs(x = "Latent Dimension", y = "Latent Dimension") +
        theme_cd_pub() +
        theme(
          legend.position = "bottom",
          axis.text = element_blank(),
          axis.ticks = element_blank()
        ) +
        coord_fixed()

      save_pdf(file.path(out_dir, "mi_matrix_sorted.pdf"), p, w = 4.6, h = 4.2)
    }
  }

  # ------------------------------------------------------------
  # 4) Combo: Leakage (Bubble + ECDF)  [kept]
  # ------------------------------------------------------------
  if (file.exists(file.path(data_dir, "alignment_matrix.csv"))) {
    R_df <- .safe_read_csv(file.path(data_dir, "alignment_matrix.csv"), row.names = 1)
    R_mat <- as.matrix(R_df)
    storage.mode(R_mat) <- "numeric"

    leakage <- rowSums(abs(R_mat) > leakage_thr, na.rm = TRUE)
    max_corr <- apply(abs(R_mat), 1, max, na.rm = TRUE)
    mean_corr <- rowMeans(abs(R_mat), na.rm = TRUE)
    df_leak <- data.frame(leakage = leakage, max_corr = max_corr, mean_corr = mean_corr)

    p1 <- ggplot(df_leak, aes(x = leakage, y = max_corr)) +
      geom_vline(xintercept = 2.5, linetype = "dashed", color = CD_PAL$gray, linewidth = 0.7) +
      geom_point(
        aes(size = max_corr, fill = mean_corr),
        shape = 21, color = "white", stroke = 0.35, alpha = 0.9
      ) +
      scale_fill_viridis_c(option = "viridis", name = "Mean |Corr|") +
      scale_size_continuous(range = c(2, 6), guide = "none") +
      labs(x = "Leakage Count", y = "Max |Correlation|") +
      theme_cd_pub() +
      theme(legend.position = "right")

    p2 <- ggplot(df_leak, aes(x = leakage)) +
      stat_ecdf(geom = "step", color = CD_PAL$blue, linewidth = CD_STYLE$lw_main) +
      scale_y_continuous(labels = percent) +
      labs(x = "Leakage Count", y = "Cumulative Proportion") +
      theme_cd_pub()

    # keep the two supporting single figs
    save_pdf(file.path(single_dir, "alignment_leakage_bubble.pdf"), p1, w = 4.3, h = 3.2)
    save_pdf(file.path(single_dir, "alignment_leakage_ecdf.pdf"),   p2, w = 4.0, h = 3.2)

    # combo main fig
    p_combo <- cowplot::plot_grid(p1, p2, ncol = 2, rel_widths = c(1.25, 1), align = "h")
    save_pdf(file.path(out_dir, "combo_alignment_leakage.pdf"), p_combo, w = 8.8, h = 3.2)
  }
}
