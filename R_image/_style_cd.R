# /home/zsh/xph_image/R_image/_style_cd.R
# ============================================================
# Enhanced Scientific Style Definition (Modern Publication Quality)
# ============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(grid)
  library(ggsci)
})

# 1. Modern Color Palette (Vibrant & Professional)
# ------------------------------------------------------------
CD_PAL <- list(
  # Primary colors - more vibrant
  blue   = "#2563EB",   # Vibrant Blue
  red    = "#DC2626",   # Vibrant Red  
  teal   = "#0D9488",   # Rich Teal
  orange = "#EA580C",   # Warm Orange
  purple = "#7C3AED",   # Vibrant Purple
  green  = "#16A34A",   # Fresh Green
  pink   = "#DB2777",   # Modern Pink
  
  # NPG inspired (for compatibility)
  navy   = "#1E3A5F",   # Deep Navy
  brown  = "#92400E",   # Warm Brown
  
  # Neutrals
  gray   = "#6B7280",   # Medium Gray
  dark   = "#111827",   # Near Black
  ink    = "#1F2937",   # Ink for text
  
  # Backgrounds & Grids
  grid   = "#E5E7EB",   # Light grid
  bg     = "#FAFAFA",   # Subtle background
  white  = "#FFFFFF"
)

# Secondary palette for multi-series
CD_COLORS_MULTI <- c(
  CD_PAL$blue, CD_PAL$red, CD_PAL$teal, CD_PAL$orange, 
  CD_PAL$purple, CD_PAL$green, CD_PAL$pink, CD_PAL$navy
)

# 2. Enhanced Geometry Defaults
# ------------------------------------------------------------
CD_STYLE <- list(
  # Base typography - Serif (Times-like)
  base_size       = 14,
  font_family     = "serif",  # Cross-platform serif (Times-like)
  
  # Text sizes (bold hierarchy)
  txt_title       = 16,
  txt_subtitle    = 13,
  txt_axis_title  = 13,
  txt_axis_text   = 11,
  txt_legend_title = 11,
  txt_legend_text  = 10,
  txt_strip       = 12,
  txt_annotation  = 10,
  
  # Line widths
  lw_thin         = 0.4,        # Grids, minor elements
  lw_frame        = 0.7,        # Axes
  lw_main         = 1.3,        # Main curves
  lw_thick        = 1.8,        # Emphasized lines
  
  # Point sizes
  pt_small        = 2.0,
  pt_main         = 3.5,
  pt_big          = 5.0,
  
  # Alpha/transparency
  alpha_fill      = 0.75,
  alpha_point     = 0.85,
  alpha_line      = 0.9,
  
  # Figure dimensions (inches)
  fig_small_w     = 4.5,
  fig_small_h     = 3.5,
  fig_main_w      = 6.0,
  fig_main_h      = 4.5,
  fig_wide_w      = 8.0,
  fig_wide_h      = 4.0,
  fig_combo_w     = 10.0,
  fig_combo_h     = 4.5,
  fig_square_w    = 5.5,
  fig_square_h    = 5.0
)

# 3. Modern Publication Theme
# ------------------------------------------------------------
theme_cd_pub <- function(base_size = CD_STYLE$base_size) {
  theme_classic(base_size = base_size, base_family = CD_STYLE$font_family) +
    theme(
      # Text styling - bold and clear
      text = element_text(color = CD_PAL$ink, family = CD_STYLE$font_family),
      
      # No titles (for paper figures)
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      
      # Axes - clean and bold
      axis.line = element_line(color = CD_PAL$dark, linewidth = CD_STYLE$lw_frame),
      axis.title = element_text(
        face = "bold", 
        size = CD_STYLE$txt_axis_title,
        color = CD_PAL$dark,
        family = CD_STYLE$font_family
      ),
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.text = element_text(
        size = CD_STYLE$txt_axis_text, 
        color = CD_PAL$dark,
        face = "bold",
        family = CD_STYLE$font_family
      ),
      axis.ticks = element_line(color = CD_PAL$dark, linewidth = CD_STYLE$lw_frame),
      axis.ticks.length = unit(0.2, "cm"),
      
      # Grid - subtle dashed lines
      panel.grid.major = element_line(
        color = CD_PAL$grid, 
        linewidth = CD_STYLE$lw_thin, 
        linetype = "dashed"
      ),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = CD_PAL$white, color = NA),
      plot.background = element_rect(fill = CD_PAL$white, color = NA),
      
      # Legend - modern styling
      legend.position = "right",
      legend.background = element_rect(fill = alpha(CD_PAL$white, 0.9), color = NA),
      legend.key = element_rect(fill = "transparent", color = NA),
      legend.key.size = unit(0.8, "cm"),
      legend.title = element_text(
        face = "bold", 
        size = CD_STYLE$txt_legend_title,
        color = CD_PAL$dark,
        family = CD_STYLE$font_family
      ),
      legend.text = element_text(
        size = CD_STYLE$txt_legend_text,
        color = CD_PAL$ink,
        face = "bold",
        family = CD_STYLE$font_family
      ),
      legend.spacing.x = unit(0.3, "cm"),
      
      # Facets
      strip.background = element_rect(fill = CD_PAL$grid, color = NA),
      strip.text = element_text(
        face = "bold", 
        size = CD_STYLE$txt_strip,
        color = CD_PAL$dark,
        margin = margin(5, 5, 5, 5),
        family = CD_STYLE$font_family
      ),
      
      # Margins
      plot.margin = margin(15, 20, 15, 15)
    )
}

# 4. Helper Functions
# ------------------------------------------------------------

# Number formatting
lab_num <- function(acc = 0.01) label_number(accuracy = acc)
lab_pct <- function(acc = 1) label_percent(accuracy = acc)

# Save PDF with high quality
save_pdf <- function(path, plot, w, h) {
  ggsave(
    filename = path,
    plot = plot,
    width = w, height = h, units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
  cat(sprintf("[SAVED] %s\n", basename(path)))
}

# Color scales for continuous data
scale_fill_cd <- function(...) {
  scale_fill_gradient2(
    low = CD_PAL$blue, 
    mid = "white", 
    high = CD_PAL$red, 
    midpoint = 0,
    ...
  )
}

scale_color_cd <- function(...) {
  scale_color_manual(values = CD_COLORS_MULTI, ...)
}

# Enhanced viridis options
scale_fill_cd_viridis <- function(option = "viridis", ...) {
  scale_fill_viridis_c(option = option, alpha = CD_STYLE$alpha_fill, ...)
}

scale_color_cd_viridis <- function(option = "viridis", ...) {
  scale_color_viridis_c(option = option, ...)
}

# Panel labels (A, B, C...)
add_panel_label <- function(label, x = 0.02, y = 0.98) {
  annotate(
    "text",
    x = x, y = y,
    label = label,
    hjust = 0, vjust = 1,
    fontface = "bold",
    size = 5,
    color = CD_PAL$dark,
    family = CD_STYLE$font_family
  )
}

cat("[STYLE] Loaded enhanced _style_cd.R (Serif/Times)\n")