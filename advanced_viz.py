#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

# -----------------------------
# 0. Publication-Quality Style Setup (White Background)
# -----------------------------
def set_pub_style():
    """配置极简、高对比度的出版级风格 (White Background)"""
    # 基础 Seaborn 风格
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks", {
        "axes.grid": True,
        "grid.color": "#E0E0E0",  # 非常淡的网格
        "grid.linestyle": "--",
        "grid.linewidth": 0.8
    })
    
    # Matplotlib 深度定制
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 300,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path)
    print(f"  [Saved] {os.path.basename(path)}")
    plt.close()

# -----------------------------
# 1. MI/NMI: 2D Heatmap + Contour (Replaces 3D)
# -----------------------------
def plot_mi_heatmap(pairs_df: pd.DataFrame, K: int, out_path: str, value_col: str = "MI"):
    print(f"... Plotting 2D Heatmap for {value_col}")
    
    # Reconstruct Matrix
    M = np.zeros((K, K), dtype=float)
    for _, r in pairs_df.iterrows():
        i, j = int(r["i"]), int(r["j"])
        v = float(r[value_col])
        M[i, j] = M[j, i] = v
    # 对角线置零或设为NaN以避免干扰色彩映射
    np.fill_diagonal(M, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 1. Heatmap
    # 使用 'mako_r' (倒序) 或 'viridis'，浅色底，深色表示高相关
    heatmap = ax.imshow(M, cmap="Blues", interpolation="nearest", aspect="auto", origin="lower")
    
    # 2. Contour (等高线) - 增加地形学解释感
    # 处理NaN以便contour正常工作
    M_filled = np.nan_to_num(M, nan=0.0)
    # 只在高值区域画线
    levels = np.linspace(np.nanmax(M)*0.3, np.nanmax(M), 4)
    if len(levels) > 1 and levels[-1] > levels[0]:
        contours = ax.contour(M_filled, levels=levels, colors="#2C3E50", linewidths=0.8, alpha=0.6, origin="lower")
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    # 3. 标记 Top-k 极值点 (可选)
    # 找到前 5 个最大值点
    flat_indices = np.argsort(M_filled.ravel())[-5:][::-1]
    for idx in flat_indices:
        r, c = np.unravel_index(idx, M.shape)
        if M[r, c] > 0:
            ax.scatter(c, r, s=30, facecolors='none', edgecolors='#E74C3C', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label(value_col, rotation=270, labelpad=15, fontsize=12)
    cbar.outline.set_visible(False)

    ax.set_title(f"{value_col} Matrix (Heatmap + Contours)", fontsize=14, pad=10)
    ax.set_xlabel("Latent Dimension Index", fontsize=11)
    ax.set_ylabel("Latent Dimension Index", fontsize=11)
    
    # 保持正方形比例
    ax.set_aspect('equal')
    savefig(out_path)

# -----------------------------
# 2. Bubble Chart: Clean & Colorbar
# -----------------------------
def plot_leakage_bubble(R: np.ndarray, thr: float, out_path: str):
    print("... Plotting Leakage Bubble Chart (Clean)")
    leakage = (R > thr).sum(axis=1)
    max_corr = R.max(axis=1)
    mean_corr = R.mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 映射设置
    cmap = cm.get_cmap("Spectral_r") # 红-黄-蓝, 倒序让红色表示高相关
    norm = Normalize(vmin=mean_corr.min(), vmax=mean_corr.max())
    
    # 散点图: Size=Max Corr, Color=Mean Corr
    # 调整 size 的缩放比例，使其不过大
    sizes = (max_corr - max_corr.min()) / (max_corr.max() - max_corr.min() + 1e-9)
    sizes = 50 + 300 * sizes 

    sc = ax.scatter(
        leakage, max_corr, 
        s=sizes, 
        c=mean_corr, 
        cmap=cmap, 
        norm=norm,
        alpha=0.8, 
        edgecolor="#555555", 
        linewidth=0.8,
        zorder=10
    )

    # 辅助线
    ax.axvline(x=2, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=1.5)
    ax.text(2.2, max_corr.min(), "Ideal Threshold (≤2)", color='#E74C3C', fontsize=10, fontweight='bold')

    # 1. Colorbar (表示 Mean Correlation)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mean Correlation (Global)", rotation=270, labelpad=15)
    cbar.outline.set_visible(False)

    # 2. Size Legend (手动构建，仅显示3个档位)
    # 创建 dummy points 用于 legend
    handles = []
    labels = []
    q_vals = np.quantile(max_corr, [0.1, 0.5, 0.9])
    for q in q_vals:
        # 反算 size
        s_norm = (q - max_corr.min()) / (max_corr.max() - max_corr.min() + 1e-9)
        s_real = 50 + 300 * s_norm
        h = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=np.sqrt(s_real), alpha=0.6, label=f"{q:.2f}")
        handles.append(h)
        labels.append(f"{q:.2f}")

    legend = ax.legend(handles, labels, title="Max Corr (Focus)", 
                       loc="lower right", bbox_to_anchor=(1.0, 0.05),
                       frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc')
    legend.get_title().set_fontsize('9')

    ax.set_title(f"Disentanglement Performance (Thr={thr})", fontsize=14, weight='bold')
    ax.set_xlabel(f"Leakage Count (Concepts with corr > {thr})", fontsize=12)
    ax.set_ylabel("Max Correlation (Specialization Strength)", fontsize=12)
    
    # 稍微扩展Y轴留出空间
    ax.set_ylim(bottom=max_corr.min()*0.9, top=max_corr.max()*1.05)
    savefig(out_path)

# -----------------------------
# 3. CDF: Standard Step Plot + Percentiles
# -----------------------------
def plot_leakage_ecdf(R: np.ndarray, thr: float, out_path: str):
    print("... Plotting Standard ECDF")
    leakage = (R > thr).sum(axis=1)
    
    plt.figure(figsize=(7, 5))
    
    # 标准 ECDF (Step plot)
    sns.ecdfplot(data=leakage, linewidth=2.5, color="#2980B9", label="Model Dimensions")
    
    # 计算分位点
    p50 = np.percentile(leakage, 50)
    p80 = np.percentile(leakage, 80)
    
    # 添加辅助线
    # 50%
    plt.axvline(x=p50, ymin=0, ymax=0.5, color="#7F8C8D", linestyle=":", linewidth=1.5)
    plt.axhline(y=0.5, xmin=0, xmax=p50/leakage.max(), color="#7F8C8D", linestyle=":", linewidth=1.5)
    plt.text(p50 + 0.5, 0.45, f"Median: {int(p50)}", color="#555", fontsize=10)
    
    # 80%
    plt.axvline(x=p80, ymin=0, ymax=0.8, color="#7F8C8D", linestyle=":", linewidth=1.5)
    plt.axhline(y=0.8, xmin=0, xmax=p80/leakage.max(), color="#7F8C8D", linestyle=":", linewidth=1.5)
    plt.text(p80 + 0.5, 0.75, f"80%: {int(p80)}", color="#555", fontsize=10)

    plt.xlabel("Leakage Count", fontsize=12)
    plt.ylabel("Cumulative Proportion", fontsize=12)
    plt.title(f"Leakage Distribution (CDF)", fontsize=13)
    
    # 去除多余边框
    sns.despine(offset=10, trim=True)
    savefig(out_path)

# -----------------------------
# 4. Sankey: Neutral Nodes + Single Color Links
# -----------------------------
def draw_bezier_link(ax, p1, p2, width, color, alpha):
    x1, y1 = p1
    x2, y2 = p2
    dist = (x2 - x1) * 0.5
    verts = [(x1, y1), (x1 + dist, y1), (x2 - dist, y2), (x2, y2)]
    codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, lw=width, alpha=alpha, capstyle='round')
    ax.add_patch(patch)

def plot_sankey_clean(spec_df: pd.DataFrame, out_path: str, topk: int = 2):
    print("... Plotting Clean Sankey Diagram")
    
    # Helper to extract edges
    edges = []
    for _, row in spec_df.iterrows():
        d = int(row["dim"])
        for t in range(1, topk + 1):
            c_col, v_col = f"top{t}_concept", f"top{t}_corr"
            if c_col in spec_df.columns and not pd.isna(row.get(c_col)):
                edges.append((d, int(row[c_col]), float(row[v_col])))
    if not edges: return

    # Unique Nodes
    dims = sorted(list(set(e[0] for e in edges)))
    concepts = sorted(list(set(e[1] for e in edges)))
    
    # Calculate Weights for stack height
    dim_w = {d: 0.0 for d in dims}
    cpt_w = {c: 0.0 for c in concepts}
    for d, c, v in edges:
        dim_w[d] += v
        cpt_w[c] += v
        
    def get_pos(weights):
        items = sorted(weights.items(), key=lambda x: x[0])
        total = sum(weights.values()) + 1e-9
        pos = {}
        curr = 0.05
        # Allocate 90% of height
        for k, w in items:
            h = (w / total) * 0.9
            pos[k] = (curr, curr + h)
            curr += h + (0.05 / len(items)) # small gap
        return pos

    dim_pos = get_pos(dim_w)
    cpt_pos = get_pos(cpt_w)
    
    # Track current Y for stacking links
    dim_curr = {k: v[0] for k, v in dim_pos.items()}
    cpt_curr = {k: v[0] for k, v in cpt_pos.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color Map: Single Hue (e.g., Greys or Blues) based on correlation strength
    # 避免彩虹色，使用单一色调的深浅表示强度
    cmap = cm.get_cmap("Blues") 
    v_values = [e[2] for e in edges]
    norm = Normalize(vmin=min(v_values)*0.5, vmax=max(v_values)) # 稍微调整vmin让弱连接不完全消失

    # Draw Links
    for (d, c, v) in edges:
        h_d = v / (dim_w[d] + 1e-9) * (dim_pos[d][1] - dim_pos[d][0])
        h_c = v / (cpt_w[c] + 1e-9) * (cpt_pos[c][1] - cpt_pos[c][0])
        
        y1 = dim_curr[d] + h_d/2
        y2 = cpt_curr[c] + h_c/2
        
        dim_curr[d] += h_d
        cpt_curr[c] += h_c
        
        # Width proportional to value
        lw = 1 + 10 * (v / max(v_values))
        color = cmap(norm(v))
        
        draw_bezier_link(ax, (0.15, y1), (0.85, y2), width=lw, color=color, alpha=0.7)

    # Draw Nodes (Neutral Grey)
    for d in dims:
        y0, y1 = dim_pos[d]
        rect = mpatches.Rectangle((0.12, y0), 0.03, y1-y0, color="#555555", alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.10, (y0+y1)/2, f"D{d}", ha="right", va="center", fontsize=9, fontweight='bold', color="#333")

    for c in concepts:
        y0, y1 = cpt_pos[c]
        rect = mpatches.Rectangle((0.85, y0), 0.03, y1-y0, color="#555555", alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.90, (y0+y1)/2, f"C{c}", ha="left", va="center", fontsize=9, fontweight='bold', color="#333")

    # Add Scale for Line Width (Legend)
    # 创建几个 dummy lines 放在图外或角落
    sample_corrs = [0.2, 0.5, 0.8]
    legend_lines = []
    legend_labels = []
    for val in sample_corrs:
        w = 1 + 10 * (val / max(v_values))
        c = cmap(norm(val))
        line = mlines.Line2D([], [], color=c, linewidth=w, alpha=0.7, label=f"Corr={val}")
        legend_lines.append(line)
        legend_labels.append(f"Corr={val}")
        
    ax.legend(legend_lines, legend_labels, title="Connection Strength", 
              loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3, frameon=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Specialist Dimension $\\to$ Concept Flow", fontsize=14)
    savefig(out_path)

# -----------------------------
# 5. Separation: Violin + Box Plot (Academic Style)
# -----------------------------
def plot_separation_violin(pairs_df: pd.DataFrame, out_path: str, col: str = "NMI"):
    print(f"... Plotting Violin Plot for {col}")
    if col not in pairs_df.columns or "same_concept" not in pairs_df.columns: return

    # Prepare Data
    data = pairs_df.copy()
    data["Group"] = data["same_concept"].map({0: "Inter-Group\n(Noise)", 1: "Intra-Group\n(Signal)"})
    
    plt.figure(figsize=(6, 6))
    
    # Violin Plot
    # inner="box" 会直接在小提琴里画出箱线图，非常节省空间且信息量大
    ax = sns.violinplot(
        data=data, x="Group", y=col, 
        palette={"Inter-Group\n(Noise)": "#95A5A6", "Intra-Group\n(Signal)": "#27AE60"},
        inner="box", 
        linewidth=1.2,
        saturation=0.8,
        cut=0 # 不推断超过数据范围的部分
    )
    
    # 调整 Box 的样式 (通过 artwork 访问，或者简单依赖默认)
    # 为了更清晰，可以加一点透明度
    for poly in ax.collections:
        poly.set_alpha(0.6)

    plt.title(f"Distribution of {col}: Signal vs Noise", fontsize=14)
    plt.xlabel("", fontsize=12) # Group 名字已经说明了
    plt.ylabel(f"{col} Value", fontsize=12)
    
    sns.despine(trim=True)
    savefig(out_path)

# -----------------------------
# Main
# -----------------------------
def main():
    set_pub_style()
    
    p = argparse.ArgumentParser()
    p.add_argument("--alignment_matrix", type=str, required=True)
    p.add_argument("--alignment_specialists", type=str, required=True)
    p.add_argument("--cmig_pairs", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./paper_figs_revised")
    p.add_argument("--leakage_thr", type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        R_df = pd.read_csv(args.alignment_matrix, index_col=0)
        R = R_df.values.astype(float)
        K = R.shape[0]
        spec_df = pd.read_csv(args.alignment_specialists)
        pairs_df = pd.read_csv(args.cmig_pairs)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    # 1. MI Heatmap (2D)
    plot_mi_heatmap(pairs_df, K, os.path.join(args.out_dir, "fig_1a_mi_heatmap.png"), "MI")
    if "NMI" in pairs_df.columns:
        plot_mi_heatmap(pairs_df, K, os.path.join(args.out_dir, "fig_1b_nmi_heatmap.png"), "NMI")

    # 2. Bubble (Clean)
    plot_leakage_bubble(R, args.leakage_thr, os.path.join(args.out_dir, "fig_2_alignment_bubble.png"))

    # 3. CDF (Step)
    plot_leakage_ecdf(R, args.leakage_thr, os.path.join(args.out_dir, "fig_3_leakage_cdf.png"))

    # 4. Sankey (Clean)
    plot_sankey_clean(spec_df, os.path.join(args.out_dir, "fig_4_specialist_flow.png"))

    # 5. Violin (Academic)
    plot_separation_violin(pairs_df, os.path.join(args.out_dir, "fig_5_separation_violin.png"), col="NMI")

    print(f"\n[Success] Revised figures saved to: {args.out_dir}")

if __name__ == "__main__":
    main()