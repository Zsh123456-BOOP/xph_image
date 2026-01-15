# /home/zsh/xph_image/R_image/_style_cd.R
# ============================================================
# Scientific Style Definition (Nature/Science Standard)
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(grid)
  library(ggsci) # Recommended for journal palettes
})

# 1. Color Palette (Nature Publishing Group Inspired)
# ------------------------------------------------------------
CD_PAL <- list(
  # Deep, saturated colors suitable for print
  blue   = "#3C5488FF", # NPG Blue
  red    = "#E64B35FF", # NPG Red
  teal   = "#00A087FF", # NPG Teal
  orange = "#F39B7FFF", # NPG Orange
  purple = "#8491B4FF", # NPG Light Blue/Purple
  brown  = "#7E6148FF", 
  grey   = "#7B7B7B",   # Neutral Grey
  dark   = "#000000",   # Pure Black for text
  grid   = "#E0E0E0"    # Very light grey for grids
)

# 2. Geometry Defaults
# ------------------------------------------------------------
CD_STYLE <- list(
  base_size       = 12,
  font_family     = "sans", # Safe for PDF (Helvetica/Arial)
  
  # Line widths (in mm/pt logic)
  lw_thin         = 0.3, # Grids
  lw_frame        = 0.6, # Axes
  lw_thick        = 1.0, # Main Curves
  
  # Point sizes
  pt_small        = 1.5,
  pt_norm         = 2.5,
  pt_big          = 4.0,
  
  # Text sizes (relative to base_size in theme)
  txt_title       = 12,
  txt_axis        = 10,
  txt_tick        = 9
)

# 3. Publication Theme
# ------------------------------------------------------------
theme_cd_pub <- function(base_size = CD_STYLE$base_size) {
  theme_classic(base_size = base_size, base_family = CD_STYLE$font_family) +
    theme(
      # Text - Strict Black
      text = element_text(color = CD_STYLE$dark),
      plot.title = element_text(face = "bold", size = CD_STYLE$txt_title, hjust = 0, margin = margin(b = 10)),
      plot.subtitle = element_text(size = CD_STYLE$txt_axis, color = "grey30", margin = margin(b = 10)),
      
      # Axes - Clean L-shape
      axis.line = element_line(color = "black", linewidth = CD_STYLE$lw_frame),
      axis.title = element_text(face = "bold", size = CD_STYLE$txt_axis),
      axis.text = element_text(size = CD_STYLE$txt_tick, color = "black"),
      axis.ticks = element_line(color = "black", linewidth = CD_STYLE$lw_frame),
      axis.ticks.length = unit(0.15, "cm"),
      
      # Grid - Minimalist
      panel.grid.major = element_line(color = CD_STYLE$grid, linewidth = CD_STYLE$lw_thin, linetype = "dashed"),
      panel.grid.minor = element_blank(),
      
      # Legend - Clean
      legend.position = "top",
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.title = element_text(face = "bold", size = CD_STYLE$txt_tick),
      legend.text = element_text(size = CD_STYLE$txt_tick),
      
      # Margins
      plot.margin = margin(15, 15, 15, 15),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = CD_STYLE$txt_axis)
    )
}

# 4. Helpers
# ------------------------------------------------------------
lab_num <- function(acc = 0.01) label_number(accuracy = acc)

save_pdf <- function(path, plot, w, h) {
  # Use cairo_pdf for high-quality vector output with font support
  ggsave(
    filename = path,
    plot = plot,
    width = w, height = h, units = "in",
    device = grDevices::cairo_pdf,
    bg = "white" # Ensure white background
  )
}