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
# Publication Style (white background, paper-friendly)
# -----------------------------
def set_pub_style():
    sns.set_context("paper", font_scale=1.15)
    sns.set_style("ticks", {
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8
    })
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# -----------------------------
# Core computations
# -----------------------------
def compute_leakage_stats(R: np.ndarray, thr: float):
    leakage = (R > thr).sum(axis=1).astype(int)
    max_corr = R.max(axis=1)
    mean_corr = R.mean(axis=1)
    return leakage, max_corr, mean_corr

def build_pair_matrix(pairs_df: pd.DataFrame, K: int, col: str):
    M = np.zeros((K, K), dtype=float)
    for _, r in pairs_df.iterrows():
        i, j = int(r["i"]), int(r["j"])
        v = float(r[col])
        M[i, j] = v
        M[j, i] = v
    np.fill_diagonal(M, np.nan)  # avoid dominating diagonal
    return M

# -----------------------------
# Subplots (draw on given axes)
# -----------------------------
def ax_heatmap_contour(ax, M: np.ndarray, title: str, cmap: str = "Blues", mark_topk: int = 5):
    # robust vmax (avoid single outlier)
    vals = M[~np.isnan(M)]
    vmax = np.percentile(vals, 99) if len(vals) else 1.0
    vmax = max(vmax, 1e-12)

    im = ax.imshow(M, cmap=cmap, interpolation="nearest", origin="lower", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("latent dim")
    ax.set_ylabel("latent dim")

    # contour (sparse levels to avoid clutter)
    Mf = np.nan_to_num(M, nan=0.0)
    levels = np.linspace(vmax * 0.6, vmax, 3)
    if levels[-1] > levels[0]:
        cs = ax.contour(Mf, levels=levels, colors="#2C3E50", linewidths=0.8, alpha=0.55, origin="lower")
        # no numeric labels (cleaner)
        # ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    # mark top-k peaks (optional)
    if mark_topk > 0:
        flat = Mf.ravel()
        idx = np.argsort(flat)[-mark_topk:][::-1]
        for t in idx:
            r, c = np.unravel_index(t, Mf.shape)
            if Mf[r, c] > 0:
                ax.scatter(c, r, s=26, facecolors="none", edgecolors="#C0392B", linewidth=1.2)

    return im

def ax_nmi_violin(ax, pairs_df: pd.DataFrame, col: str = "NMI"):
    if col not in pairs_df.columns or "same_concept" not in pairs_df.columns:
        ax.axis("off")
        ax.set_title("N/A")
        return

    data = pairs_df.copy()
    data["Group"] = data["same_concept"].map({0: "Inter (noise)", 1: "Intra (signal)"})
    # paper-friendly palette (muted)
    palette = {"Inter (noise)": "#A6A6A6", "Intra (signal)": "#4C9A6A"}

    sns.violinplot(
        data=data, x="Group", y=col, ax=ax,
        palette=palette, inner="box", linewidth=1.0, cut=0, saturation=0.85
    )
    ax.set_title(f"{col} separation (signal vs noise)")
    ax.set_xlabel("")
    ax.set_ylabel(col)

def ax_leakage_bubble(ax, R: np.ndarray, thr: float):
    leakage, max_corr, mean_corr = compute_leakage_stats(R, thr)

    # use single-hue colormap for paper (avoid rainbow)
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=float(mean_corr.min()), vmax=float(mean_corr.max()))

    # size mapped to max_corr (moderate scaling)
    s_norm = (max_corr - max_corr.min()) / (max_corr.max() - max_corr.min() + 1e-12)
    sizes = 30 + 240 * s_norm

    sc = ax.scatter(
        leakage, max_corr,
        s=sizes, c=mean_corr, cmap=cmap, norm=norm,
        alpha=0.85, edgecolors="#444444", linewidths=0.6
    )

    ax.axvline(x=2, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(2.2, float(max_corr.min()), "target leakage ≤ 2", color="#C0392B", fontsize=9, va="bottom")

    ax.set_title(f"Leakage vs focus (thr={thr})")
    ax.set_xlabel(f"Leakage count (|corr| > {thr})")
    ax.set_ylabel("max |corr|")

    return sc

def ax_leakage_ecdf(ax, R: np.ndarray, thr: float):
    leakage, _, _ = compute_leakage_stats(R, thr)
    xs = np.sort(leakage)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    ax.step(xs, ys, where="post", linewidth=2.0)

    # annotate quantiles
    for q in [0.5, 0.8]:
        xq = xs[int(np.ceil(q * len(xs))) - 1]
        ax.axvline(xq, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(xq, q, f"{int(q*100)}% @ {xq}", fontsize=9, va="bottom", ha="left")

    ax.set_title("Leakage ECDF")
    ax.set_xlabel(f"Leakage count (|corr| > {thr})")
    ax.set_ylabel("ECDF")

def draw_bezier_link(ax, p1, p2, width, color, alpha):
    x1, y1 = p1
    x2, y2 = p2
    dist = (x2 - x1) * 0.5
    verts = [(x1, y1), (x1 + dist, y1), (x2 - dist, y2), (x2, y2)]
    codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor="none", edgecolor=color, lw=width, alpha=alpha, capstyle="round")
    ax.add_patch(patch)

def ax_specialist_flow(ax, spec_df: pd.DataFrame, topk: int = 2):
    # build edges: (dim, concept, corr)
    edges = []
    for _, row in spec_df.iterrows():
        d = int(row["dim"])
        for t in range(1, topk + 1):
            c_col, v_col = f"top{t}_concept", f"top{t}_corr"
            if c_col in spec_df.columns and not pd.isna(row.get(c_col)):
                edges.append((d, int(row[c_col]), float(row[v_col])))

    if not edges:
        ax.axis("off")
        ax.set_title("No specialists")
        return

    dims = sorted(set(d for d, _, _ in edges))
    cpts = sorted(set(c for _, c, _ in edges))

    # weight sums
    dim_w = {d: 0.0 for d in dims}
    cpt_w = {c: 0.0 for c in cpts}
    for d, c, v in edges:
        dim_w[d] += v
        cpt_w[c] += v

    def alloc_pos(weights):
        items = sorted(weights.items(), key=lambda x: x[0])
        total = sum(weights.values()) + 1e-12
        pos = {}
        cur = 0.06
        gap = 0.03
        usable = 1.0 - 2 * cur - gap * (len(items) - 1)
        y = cur
        for k, w in items:
            h = usable * (w / total)
            pos[k] = (y, y + h)
            y += h + gap
        return pos

    dim_pos = alloc_pos(dim_w)
    cpt_pos = alloc_pos(cpt_w)
    dim_cur = {k: v[0] for k, v in dim_pos.items()}
    cpt_cur = {k: v[0] for k, v in cpt_pos.items()}

    v_values = [v for _, _, v in edges]
    vmax = max(v_values) + 1e-12
    cmap = cm.get_cmap("Blues")
    norm = Normalize(vmin=np.percentile(v_values, 10), vmax=np.percentile(v_values, 95))

    # links
    for d, c, v in edges:
        h_d = (dim_pos[d][1] - dim_pos[d][0]) * (v / (dim_w[d] + 1e-12))
        h_c = (cpt_pos[c][1] - cpt_pos[c][0]) * (v / (cpt_w[c] + 1e-12))

        y1 = dim_cur[d] + h_d / 2
        y2 = cpt_cur[c] + h_c / 2
        dim_cur[d] += h_d
        cpt_cur[c] += h_c

        lw = 0.8 + 5.5 * (v / vmax)  # capped width
        color = cmap(norm(v))
        draw_bezier_link(ax, (0.18, y1), (0.82, y2), width=lw, color=color, alpha=0.75)

    # nodes (neutral)
    for d in dims:
        y0, y1 = dim_pos[d]
        ax.add_patch(mpatches.Rectangle((0.12, y0), 0.05, y1 - y0, color="#4D4D4D", alpha=0.9))
        ax.text(0.10, (y0 + y1) / 2, f"d{d}", ha="right", va="center", fontsize=9)

    for c in cpts:
        y0, y1 = cpt_pos[c]
        ax.add_patch(mpatches.Rectangle((0.82, y0), 0.05, y1 - y0, color="#4D4D4D", alpha=0.9))
        ax.text(0.89, (y0 + y1) / 2, f"c{c}", ha="left", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Specialist flow (top-{topk})")

def ax_specialist_table(ax, spec_df: pd.DataFrame, max_rows: int = 8):
    """
    Right panel: compact specialists summary table.
    Requirement: floats keep 4 decimals (e.g., 0.2196).
    """
    # build rows
    rows = []
    for _, r in spec_df.iterrows():
        dim = int(r["dim"])
        leak = int(r["leakage"]) if "leakage" in spec_df.columns else -1
        max_corr = float(r["max_corr"]) if "max_corr" in spec_df.columns else np.nan

        # top1 mapping
        top1_cpt = int(r["top1_concept"]) if "top1_concept" in spec_df.columns else int(r.get("top1_concept", -1))
        top1_corr = float(r["top1_corr"]) if "top1_corr" in spec_df.columns else np.nan

        rows.append([dim, leak, max_corr, top1_cpt, top1_corr])

    df = pd.DataFrame(rows, columns=["dim", "leak", "max_corr", "top1_cpt", "top1_corr"])

    # sort for readability: low leak first, then high corr
    df = df.sort_values(by=["leak", "top1_corr"], ascending=[True, False]).head(max_rows)

    # format: integers as int, floats as 4-dec strings
    df_fmt = df.copy()
    df_fmt["dim"] = df_fmt["dim"].astype(int).astype(str)
    df_fmt["leak"] = df_fmt["leak"].astype(int).astype(str)
    df_fmt["top1_cpt"] = df_fmt["top1_cpt"].astype(int).astype(str)
    df_fmt["max_corr"] = df_fmt["max_corr"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "NA")
    df_fmt["top1_corr"] = df_fmt["top1_corr"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "NA")

    ax.axis("off")
    ax.set_title("Specialists summary", pad=8)

    tbl = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.35)


# -----------------------------
# Combo figures
# -----------------------------
def plot_combo_mi(pairs_df: pd.DataFrame, K: int, out_path: str):
    fig = plt.figure(figsize=(15.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    M_mi = build_pair_matrix(pairs_df, K, "MI")
    im1 = ax_heatmap_contour(ax1, M_mi, "MI heatmap + contours", cmap="Blues")

    # NMI panel optional; if missing, show MI again
    if "NMI" in pairs_df.columns:
        M_nmi = build_pair_matrix(pairs_df, K, "NMI")
        im2 = ax_heatmap_contour(ax2, M_nmi, "NMI heatmap + contours", cmap="Blues")
        ax_nmi_violin(ax3, pairs_df, col="NMI")
    else:
        im2 = ax_heatmap_contour(ax2, M_mi, "MI heatmap (proxy)", cmap="Blues")
        ax3.axis("off")
        ax3.set_title("NMI not found")

    # one shared colorbar per heatmap (clean)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("value", rotation=270, labelpad=12)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("value", rotation=270, labelpad=12)

    fig.suptitle("MI Independence (main + supporting evidence)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=320)
    plt.close()
    print(f"  [Saved] {os.path.basename(out_path)}")

def plot_combo_leakage(R: np.ndarray, thr: float, out_path: str):
    fig = plt.figure(figsize=(12.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sc = ax_leakage_bubble(ax1, R, thr)
    ax_leakage_ecdf(ax2, R, thr)

    # shared colorbar for bubble (mean_corr)
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("mean |corr|", rotation=270, labelpad=12)

    fig.suptitle("Alignment leakage (main + supporting evidence)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=320)
    plt.close()
    print(f"  [Saved] {os.path.basename(out_path)}")

def plot_combo_specialists(spec_df: pd.DataFrame, out_path: str, topk_flow: int = 2):
    fig = plt.figure(figsize=(12.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax_specialist_flow(ax1, spec_df, topk=topk_flow)
    ax_specialist_table(ax2, spec_df, max_rows=8)

    fig.suptitle("Specialist dimensions (flow + summary)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=320)
    plt.close()
    print(f"  [Saved] {os.path.basename(out_path)}")

# -----------------------------
# Main
# -----------------------------
def main():
    set_pub_style()

    p = argparse.ArgumentParser()
    p.add_argument("--alignment_matrix", type=str, required=True, default="./exp_m1_out/assist_09/alignment_matrix.csv")
    p.add_argument("--alignment_specialists", type=str, required=True, default="./exp_m1_out/assist_09/alignment_specialists.csv")
    p.add_argument("--cmig_pairs", type=str, required=True, default="./exp_m1_out/assist_09/cmig_pairs.csv")
    p.add_argument("--out_dir", type=str, default="./advanced_viz_out_1")
    p.add_argument("--leakage_thr", type=float, default=0.15)
    args = p.parse_args()

    _safe_mkdir(args.out_dir)

    R = pd.read_csv(args.alignment_matrix, index_col=0).values.astype(float)
    K = R.shape[0]
    spec_df = pd.read_csv(args.alignment_specialists)
    pairs_df = pd.read_csv(args.cmig_pairs)

    # 3 combo figures (replace your 3 original evidence figures in the paper)
    plot_combo_mi(pairs_df, K, os.path.join(args.out_dir, "combo_mi_independence.png"))
    plot_combo_leakage(R, args.leakage_thr, os.path.join(args.out_dir, "combo_alignment_leakage.png"))
    plot_combo_specialists(spec_df, os.path.join(args.out_dir, "combo_specialists.png"), topk_flow=2)

    print("\n[OK] Combo figures generated:")
    print("  - combo_mi_independence.png    (replaces mi_matrix_sorted.png)")
    print("  - combo_alignment_leakage.png  (replaces alignment_leakage.png)")
    print("  - combo_specialists.png        (replaces alignment_specialist_dims.png)")

if __name__ == "__main__":
    main()
