#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-1 Experiments: 特征解耦实验 (Feature Disentanglement) - FINAL SELECTED (+COMBO)
================================================================================
保留原始三张核心单图及其数据（不删除），并可选生成“组合主图（combo）”用于主文展示。

原始保留（单图）：
1. [Exp-1A] MI Independence
   - mi_matrix_sorted.png
   - cmig_pairs.csv, mi_metrics.json
2. [Exp-1B] Alignment Leakage
   - alignment_leakage.png
   - alignment_matrix.csv
3. [Exp-1B] Specialist Dimensions
   - alignment_specialist_dims.png
   - alignment_specialists.csv, alignment_metrics.json

新增（默认开启）：
- --make_combo: 从上述 CSV 直接生成三张 combo 图（白底、审稿友好）
- --no_combo: 禁用 combo 图生成

实现策略：
- 将 make_combo_from_csv_1.py 的绘图逻辑内嵌进本脚本，避免额外脚本管理与 seaborn 依赖
- 输出目录默认与 args.out_dir 一致，并放到 out_dir/combo/ 下（可配置）
================================================================================
"""

import os
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from torch.utils.data import DataLoader

from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def spearman_corr_masked(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Spearman corr on masked samples (no scipy)."""
    mask = mask.astype(bool)
    if mask.sum() < 3:
        return 0.0
    xs = x[mask]
    ys = y[mask]

    xr = pd.Series(xs).rank(method="average").to_numpy()
    yr = pd.Series(ys).rank(method="average").to_numpy()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = (np.sqrt((xr**2).mean()) * np.sqrt((yr**2).mean()) + 1e-12)
    return float((xr * yr).mean() / denom)


def hist_entropy_1d(x: np.ndarray, bins: int = 20) -> float:
    x = np.asarray(x)
    hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def hist_mi_2d(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    hxy, _, _ = np.histogram2d(
        x, y,
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        density=False
    )
    pxy = hxy.astype(np.float64)
    pxy = pxy / (pxy.sum() + 1e-12)

    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                mi += pxy[i, j] * (
                    math.log(pxy[i, j] + 1e-12)
                    - math.log(px[i, 0] + 1e-12)
                    - math.log(py[0, j] + 1e-12)
                )
    return float(mi)


def fmt4(x: float) -> str:
    return f"{float(x):.4f}"


# ============================================================
# Data Bundle
# ============================================================
@dataclass
class DataBundle:
    num_students: int
    num_exercises: int
    num_concepts: int
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    loaders: Tuple[DataLoader, DataLoader, DataLoader]
    graphs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _parse_cpt_seq(cpt_seq) -> List[int]:
    s = str(cpt_seq)
    if s.strip() == "" or s.lower() == "nan":
        return []
    return [int(c) for c in s.split(",") if str(c).strip() != ""]


def load_and_map_data(train_file: str, valid_file: str, test_file: str):
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    student_ids = sorted(all_df["stu_id"].unique())
    exercise_ids = sorted(all_df["exer_id"].unique())
    all_cpts: Set[int] = set()
    for cpt_seq in all_df["cpt_seq"]:
        all_cpts.update(_parse_cpt_seq(cpt_seq))
    concept_ids = sorted(all_cpts)

    student_id_map = {old: new for new, old in enumerate(student_ids)}
    exercise_id_map = {old: new for new, old in enumerate(exercise_ids)}
    concept_id_map = {old: new for new, old in enumerate(concept_ids)}

    def map_concepts(cpt_seq):
        cpts = _parse_cpt_seq(cpt_seq)
        cpts = [concept_id_map[int(c)] for c in cpts]
        return ",".join(str(c) for c in cpts)

    for df in [train_df, valid_df, test_df]:
        df["stu_id"] = df["stu_id"].map(student_id_map)
        df["exer_id"] = df["exer_id"].map(exercise_id_map)
        df["cpt_seq"] = df["cpt_seq"].apply(map_concepts)

    return train_df, valid_df, test_df


def build_all_graphs(train_df: pd.DataFrame, num_students: int, num_exercises: int, num_concepts: int,
                     graph_dir: str, device: torch.device):
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_wrong_se   = build_graph(train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_correct_sc = build_graph(train_df, num_students, num_concepts, correct=True, item_type="concept", graph_dir=graph_dir).to(device)
    adj_wrong_sc   = build_graph(train_df, num_students, num_concepts, correct=False, item_type="concept", graph_dir=graph_dir).to(device)
    return adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc


def make_bundle(args) -> DataBundle:
    train_df, valid_df, test_df = load_and_map_data(args.train_file, args.valid_file, args.test_file)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    num_students = int(all_df["stu_id"].nunique())
    num_exercises = int(all_df["exer_id"].nunique())
    all_cpts: Set[int] = set()
    for cpt_seq in all_df["cpt_seq"]:
        all_cpts.update(_parse_cpt_seq(cpt_seq))
    num_concepts = len(all_cpts)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    train_loader = DataLoader(CDDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(CDDataset(valid_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(CDDataset(test_df),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return DataBundle(num_students, num_exercises, num_concepts, train_df, valid_df, test_df,
                      (train_loader, valid_loader, test_loader), graphs)


def load_model(args, bundle: DataBundle, device: torch.device) -> CognitiveDiagnosisModel:
    model = CognitiveDiagnosisModel(
        num_students=bundle.num_students,
        num_exercises=bundle.num_exercises,
        num_concepts=bundle.num_concepts,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        fusion_type=args.fusion_type,
        temperature=args.temperature,
        num_heads=args.num_heads,
        use_supervised_contrastive=args.use_supervised_contrastive,
        gated_num_gates=args.gated_num_gates,
        ortho_weight=args.ortho_weight,
        dropout=args.dropout
    ).to(device)

    if args.model_path and os.path.exists(args.model_path):
        sd = torch.load(args.model_path, map_location=device)
        model.load_state_dict(sd)
    else:
        raise FileNotFoundError(f"--model_path not found: {args.model_path}")
    model.eval()
    return model


@torch.no_grad()
def collect_student_knowledge_state(model: CognitiveDiagnosisModel, loader: DataLoader, graphs, device,
                                    num_students: int, num_concepts: int) -> np.ndarray:
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs
    sum_z = torch.zeros(num_students, num_concepts, device=device)
    cnt = torch.zeros(num_students, device=device)

    for batch in loader:
        stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels = batch
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        cpt_ids_padded = cpt_ids_padded.to(device)
        cpt_mask = cpt_mask.to(device)
        labels = labels.to(device)

        _, _, ks = model(stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels,
                         adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc)
        for i in range(stu_ids.size(0)):
            u = int(stu_ids[i].item())
            sum_z[u] += ks[i]
            cnt[u] += 1

    cnt = torch.clamp(cnt, min=1.0).unsqueeze(1)
    z = (sum_z / cnt).detach().cpu().numpy()
    return z


def empirical_concept_accuracy_and_count(df: pd.DataFrame, num_students: int, num_concepts: int) -> Tuple[np.ndarray, np.ndarray]:
    num = np.zeros((num_students, num_concepts), dtype=np.float64)
    den = np.zeros((num_students, num_concepts), dtype=np.float64)
    for _, r in df.iterrows():
        u = int(r["stu_id"])
        y = int(r["label"])
        cpts = _parse_cpt_seq(r["cpt_seq"])
        for k in cpts:
            num[u, k] += y
            den[u, k] += 1.0
    acc = num / (den + 1e-12)
    return acc, den


# ============================================================
# Exp-1B (Alignment): Leakage + Specialists
# ============================================================
def compute_alignment_matrix(
    z: np.ndarray,
    acc: np.ndarray,
    den: np.ndarray,
    out_dir: str,
    align_min_attempts: int,
    align_min_students: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R[i,k] = abs(Spearman(z_dim_i, acc_concept_k)) on valid students/concepts.
    Save full matrix as alignment_matrix.csv
    Return: R, concept_valid_mask
    """
    K = z.shape[1]
    R = np.zeros((K, K), dtype=np.float64)

    valid_students_per_k = []
    for k in range(K):
        mask_k = den[:, k] >= float(align_min_attempts)
        valid_students_per_k.append(int(mask_k.sum()))
    valid_students_per_k = np.asarray(valid_students_per_k, dtype=np.int32)
    concept_valid_mask = valid_students_per_k >= int(align_min_students)

    for i in range(K):
        zi = z[:, i].astype(np.float64)
        for k in range(K):
            if not concept_valid_mask[k]:
                R[i, k] = 0.0
                continue
            mask = den[:, k] >= float(align_min_attempts)
            R[i, k] = abs(spearman_corr_masked(zi, acc[:, k].astype(np.float64), mask))

    df_matrix = pd.DataFrame(R, index=[f"z_{i}" for i in range(K)], columns=[f"acc_{k}" for k in range(K)])
    df_matrix.to_csv(os.path.join(out_dir, "alignment_matrix.csv"))
    return R, concept_valid_mask


def plot_alignment_leakage_and_specialists(
    R: np.ndarray,
    concept_valid_mask: np.ndarray,
    out_dir: str,
    leakage_thr: float = 0.15,
    specialist_max_leakage: int = 2,
    specialist_topk: int = 8,
    max_specialists_to_plot: int = 5
) -> Dict:
    """
    1. Plot leakage distribution (alignment_leakage.png)
    2. Plot specialist dimensions top-k concepts (alignment_specialist_dims.png)
    """
    K = R.shape[0]
    valid_cols = np.where(concept_valid_mask)[0]
    Rv = R[:, valid_cols] if len(valid_cols) > 0 else R

    leakage = (Rv > leakage_thr).sum(axis=1).astype(int)
    max_corr = Rv.max(axis=1)
    mean_corr = Rv.mean(axis=1)

    # ---- 1. Leakage plot (alignment_leakage.png) ----
    plt.figure(figsize=(7.0, 4.8))
    plt.hist(leakage, bins=min(30, max(5, int(np.sqrt(K)))), alpha=0.85)
    plt.xlabel(f"Leakage = #Concepts with |corr| > {leakage_thr}")
    plt.ylabel("Count of Latent Dimensions")
    plt.title("Alignment Leakage Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_leakage.png"), dpi=220)
    plt.close()

    # ---- 2. Select specialist dims ----
    candidates = np.where((leakage <= specialist_max_leakage) & (max_corr > leakage_thr))[0]
    if len(candidates) == 0:
        order = np.lexsort((-max_corr, leakage))
        candidates = order[:max_specialists_to_plot]

    cand_order = np.lexsort((-max_corr[candidates], leakage[candidates]))
    specialists = candidates[cand_order][:max_specialists_to_plot]

    specialist_rows = []
    for d in specialists:
        row = {
            "dim": int(d),
            "leakage": int(leakage[d]),
            "max_corr": float(max_corr[d]),
            "mean_corr": float(mean_corr[d]),
        }
        topk_idx = np.argsort(Rv[d])[::-1][:specialist_topk]
        topk_concepts_abs = [int(valid_cols[i]) if len(valid_cols) > 0 else int(i) for i in topk_idx]
        topk_vals = [float(Rv[d, i]) for i in topk_idx]
        for t, (ck, cv) in enumerate(zip(topk_concepts_abs, topk_vals), start=1):
            row[f"top{t}_concept"] = ck
            row[f"top{t}_corr"] = cv
        specialist_rows.append(row)

    df_spec = pd.DataFrame(specialist_rows)
    df_spec.to_csv(os.path.join(out_dir, "alignment_specialists.csv"), index=False)

    # ---- 3. Plot specialists (alignment_specialist_dims.png) ----
    labels = []
    vals = []
    for r in specialist_rows:
        d = r["dim"]
        for t in range(1, specialist_topk + 1):
            ck = r.get(f"top{t}_concept", None)
            cv = r.get(f"top{t}_corr", None)
            if ck is None:
                continue
            labels.append(f"d{d}:c{ck}")
            vals.append(float(cv))

    plt.figure(figsize=(12.0, 4.8))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("|Spearman corr|")
    plt.title(f"Specialist Dimensions (Top-{specialist_topk} Concepts)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_specialist_dims.png"), dpi=220)
    plt.close()

    metrics = {
        "leakage_thr": float(leakage_thr),
        "specialist_max_leakage": int(specialist_max_leakage),
        "n_valid_concepts": int(concept_valid_mask.sum()),
        "leakage_mean": float(leakage.mean()),
        "leakage_median": float(np.median(leakage)),
        "n_specialists_plotted": int(len(specialists)),
        "specialist_dims": [int(x) for x in specialists.tolist()],
        "note": "Top specialists extracted."
    }
    with open(os.path.join(out_dir, "alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


# ============================================================
# Exp-1A (MI Independence): Matrix Only
# ============================================================
def _sample_pairs(rng: np.random.Generator, K: int, pairs: int) -> List[Tuple[int, int]]:
    pair_list = set()
    max_pairs = K * (K - 1) // 2
    target = min(pairs, max_pairs)
    while len(pair_list) < target:
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, K))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pair_list.add((i, j))
    return sorted(list(pair_list))


def plot_mi_matrix(z: np.ndarray, out_dir: str, bins: int, max_dims: int = 64):
    """
    Produce mi_matrix_sorted.png
    """
    K = z.shape[1]
    order = np.arange(K)
    if K > max_dims:
        order = order[:max_dims]
    Kp = len(order)

    z2 = np.clip(z[:, order], 0.0, 1.0)
    M = np.zeros((Kp, Kp), dtype=np.float64)
    for i in range(Kp):
        xi = z2[:, i]
        for j in range(i, Kp):
            xj = z2[:, j]
            mi = hist_mi_2d(xi, xj, bins=bins)
            M[i, j] = mi
            M[j, i] = mi

    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(M, aspect="auto", vmin=0.0, vmax=0.2)
    plt.colorbar(label="MI")
    plt.title("MI Matrix (first dims)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_matrix_sorted.png"), dpi=220)
    plt.close()

    return M


def exp_mi_independence_clean(
    z: np.ndarray,
    out_dir: str,
    seed: int,
    bins: int,
    pairs: int,
    top1_concept_per_factor: np.ndarray
):
    """
    Produce: cmig_pairs.csv, mi_metrics.json, mi_matrix_sorted.png
    """
    rng = np.random.default_rng(seed)
    K = z.shape[1]

    H = np.array([hist_entropy_1d(np.clip(z[:, k], 0.0, 1.0), bins=bins) for k in range(K)], dtype=np.float64)
    pair_list = _sample_pairs(rng, K, pairs)
    rows = []

    for (i, j) in pair_list:
        xi = np.clip(z[:, i], 0.0, 1.0)
        xj = np.clip(z[:, j], 0.0, 1.0)
        mi = hist_mi_2d(xi, xj, bins=bins)
        denom = math.sqrt(max(H[i] * H[j], 1e-12))
        nmi = mi / denom

        ci = int(top1_concept_per_factor[i]) if top1_concept_per_factor is not None else -1
        cj = int(top1_concept_per_factor[j]) if top1_concept_per_factor is not None else -1
        same = int((ci == cj) and (ci != -1))

        rows.append({"i": i, "j": j, "same_concept": same, "MI": float(mi), "NMI": float(nmi)})

    df_pairs = pd.DataFrame(rows)
    df_pairs.to_csv(os.path.join(out_dir, "cmig_pairs.csv"), index=False)

    plot_mi_matrix(z, out_dir, bins)

    metrics = {
        "MI_mean": float(df_pairs["MI"].mean()),
        "MI_median": float(df_pairs["MI"].median()),
        "NMI_mean": float(df_pairs["NMI"].mean()),
        "IndependenceScore": float(1.0 - df_pairs["NMI"].mean()),
        "note": "Clean version: MI matrix + pairs data."
    }
    with open(os.path.join(out_dir, "mi_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def top1_concept_per_factor_from_R(R: np.ndarray) -> np.ndarray:
    return np.argmax(R, axis=1).astype(int)


# ============================================================
# COMBO VIS (inlined make_combo_from_csv_1.py, NO seaborn)
# ============================================================
def set_pub_style_combo():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "grid.color": "#E6E6E6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


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
        if 0 <= i < K and 0 <= j < K:
            M[i, j] = v
            M[j, i] = v
    np.fill_diagonal(M, np.nan)
    return M


def ax_heatmap_contour(ax, M: np.ndarray, title: str, cmap_name: str = "Blues", mark_topk: int = 5):
    vals = M[~np.isnan(M)]
    vmax = np.percentile(vals, 99) if len(vals) else 1.0
    vmax = max(vmax, 1e-12)

    im = ax.imshow(M, cmap=cmap_name, interpolation="nearest", origin="lower", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("latent dim")
    ax.set_ylabel("latent dim")

    Mf = np.nan_to_num(M, nan=0.0)
    levels = np.linspace(vmax * 0.6, vmax, 3)
    if levels[-1] > levels[0]:
        ax.contour(Mf, levels=levels, colors="#2C3E50", linewidths=0.8, alpha=0.55, origin="lower")

    if mark_topk > 0:
        flat = Mf.ravel()
        idx = np.argsort(flat)[-mark_topk:][::-1]
        for t in idx:
            r, c = np.unravel_index(t, Mf.shape)
            if Mf[r, c] > 0:
                ax.scatter(c, r, s=26, facecolors="none", edgecolors="#C0392B", linewidth=1.2)

    ax.grid(False)
    return im


def ax_nmi_violin_matplotlib(ax, pairs_df: pd.DataFrame, col: str = "NMI"):
    """
    seaborn-free violin with box-like center:
    - groups: same_concept=0 -> Inter (noise), 1 -> Intra (signal)
    """
    if col not in pairs_df.columns or "same_concept" not in pairs_df.columns:
        ax.axis("off")
        ax.set_title("N/A")
        return

    inter = pairs_df[pairs_df["same_concept"] == 0][col].to_numpy(dtype=float)
    intra = pairs_df[pairs_df["same_concept"] == 1][col].to_numpy(dtype=float)
    data = [inter, intra]
    labels = ["Inter (noise)", "Intra (signal)"]

    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    # subtle fills
    for i, pc in enumerate(parts["bodies"]):
        pc.set_alpha(0.70)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.8)

    # median line format
    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_title(f"{col} separation (signal vs noise)")
    ax.set_xlabel("")
    ax.set_ylabel(col)
    ax.grid(True, axis="y", alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    # annotate medians
    meds = [np.median(inter) if len(inter) else np.nan, np.median(intra) if len(intra) else np.nan]
    for i, m in enumerate(meds, start=1):
        if np.isfinite(m):
            ax.text(i, m, f"med={fmt4(m)}", ha="center", va="bottom", fontsize=9)


def ax_leakage_bubble(ax, R: np.ndarray, thr: float):
    leakage, max_corr, mean_corr = compute_leakage_stats(R, thr)

    cmap_obj = cm.get_cmap("viridis")
    norm = Normalize(vmin=float(mean_corr.min()), vmax=float(mean_corr.max()))

    s_norm = (max_corr - max_corr.min()) / (max_corr.max() - max_corr.min() + 1e-12)
    sizes = 30 + 240 * s_norm

    sc = ax.scatter(
        leakage, max_corr,
        s=sizes, c=mean_corr, cmap=cmap_obj, norm=norm,
        alpha=0.85, edgecolors="#444444", linewidths=0.6
    )

    ax.axvline(x=2, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(2.2, float(max_corr.min()), "target leakage ≤ 2", color="#C0392B", fontsize=9, va="bottom")

    ax.set_title(f"Leakage vs focus (thr={thr})")
    ax.set_xlabel(f"Leakage count (|corr| > {thr})")
    ax.set_ylabel("max |corr|")
    ax.grid(True, alpha=0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    return sc


def ax_leakage_ecdf(ax, R: np.ndarray, thr: float):
    leakage, _, _ = compute_leakage_stats(R, thr)
    xs = np.sort(leakage)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    ax.step(xs, ys, where="post", linewidth=2.0)

    for q in [0.5, 0.8]:
        idx = int(np.ceil(q * len(xs))) - 1
        idx = max(0, min(idx, len(xs) - 1))
        xq = xs[idx]
        ax.axvline(xq, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(xq, q, f"{int(q*100)}% @ {xq}", fontsize=9, va="bottom", ha="left")

    ax.set_title("Leakage ECDF")
    ax.set_xlabel(f"Leakage count (|corr| > {thr})")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.55)


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
    cmap_obj = cm.get_cmap("Blues")
    norm = Normalize(vmin=np.percentile(v_values, 10), vmax=np.percentile(v_values, 95))

    for d, c, v in edges:
        h_d = (dim_pos[d][1] - dim_pos[d][0]) * (v / (dim_w[d] + 1e-12))
        h_c = (cpt_pos[c][1] - cpt_pos[c][0]) * (v / (cpt_w[c] + 1e-12))

        y1 = dim_cur[d] + h_d / 2
        y2 = cpt_cur[c] + h_c / 2
        dim_cur[d] += h_d
        cpt_cur[c] += h_c

        lw = 0.8 + 5.5 * (v / vmax)
        color = cmap_obj(norm(v))
        draw_bezier_link(ax, (0.18, y1), (0.82, y2), width=lw, color=color, alpha=0.75)

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
    rows = []
    for _, r in spec_df.iterrows():
        dim = int(r["dim"])
        leak = int(r["leakage"]) if "leakage" in spec_df.columns else -1
        max_corr = float(r["max_corr"]) if "max_corr" in spec_df.columns else float("nan")

        top1_cpt = int(r["top1_concept"]) if "top1_concept" in spec_df.columns else int(r.get("top1_concept", -1))
        top1_corr = float(r["top1_corr"]) if "top1_corr" in spec_df.columns else float("nan")
        rows.append([dim, leak, max_corr, top1_cpt, top1_corr])

    df = pd.DataFrame(rows, columns=["dim", "leak", "max_corr", "top1_cpt", "top1_corr"])
    df = df.sort_values(by=["leak", "top1_corr"], ascending=[True, False]).head(max_rows)

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


def plot_combo_mi(pairs_df: pd.DataFrame, K: int, out_path: str):
    fig = plt.figure(figsize=(15.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.9])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    M_mi = build_pair_matrix(pairs_df, K, "MI")
    im1 = ax_heatmap_contour(ax1, M_mi, "MI heatmap + contours", cmap_name="Blues")

    if "NMI" in pairs_df.columns:
        M_nmi = build_pair_matrix(pairs_df, K, "NMI")
        im2 = ax_heatmap_contour(ax2, M_nmi, "NMI heatmap + contours", cmap_name="Blues")
        ax_nmi_violin_matplotlib(ax3, pairs_df, col="NMI")
    else:
        im2 = ax_heatmap_contour(ax2, M_mi, "MI heatmap (proxy)", cmap_name="Blues")
        ax3.axis("off")
        ax3.set_title("NMI not found")

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label("value", rotation=270, labelpad=12)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("value", rotation=270, labelpad=12)

    fig.suptitle("MI Independence (main + supporting evidence)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=320)
    plt.close()


def plot_combo_leakage(R: np.ndarray, thr: float, out_path: str):
    fig = plt.figure(figsize=(12.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sc = ax_leakage_bubble(ax1, R, thr)
    ax_leakage_ecdf(ax2, R, thr)

    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("mean |corr|", rotation=270, labelpad=12)

    fig.suptitle("Alignment leakage (main + supporting evidence)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=320)
    plt.close()


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


def make_combo_from_csvs(out_dir: str, leakage_thr: float):
    """
    从 exp_module1_disentangle.py 生成的 CSV 中读取数据并输出 combo 图。
    """
    set_pub_style_combo()

    path_R = os.path.join(out_dir, "alignment_matrix.csv")
    path_spec = os.path.join(out_dir, "alignment_specialists.csv")
    path_pairs = os.path.join(out_dir, "cmig_pairs.csv")

    if not (os.path.exists(path_R) and os.path.exists(path_spec) and os.path.exists(path_pairs)):
        raise FileNotFoundError(
            "Combo needs CSVs: alignment_matrix.csv, alignment_specialists.csv, cmig_pairs.csv "
            f"under out_dir={out_dir}"
        )

    R = pd.read_csv(path_R, index_col=0).values.astype(float)
    K = R.shape[0]
    spec_df = pd.read_csv(path_spec)
    pairs_df = pd.read_csv(path_pairs)

    combo_dir = os.path.join(out_dir, "combo")
    safe_mkdir(combo_dir)

    plot_combo_mi(pairs_df, K, os.path.join(combo_dir, "combo_mi_independence.png"))
    plot_combo_leakage(R, leakage_thr, os.path.join(combo_dir, "combo_alignment_leakage.png"))
    plot_combo_specialists(spec_df, os.path.join(combo_dir, "combo_specialists.png"), topk_flow=2)

    return {
        "combo_dir": combo_dir,
        "generated": [
            "combo_mi_independence.png",
            "combo_alignment_leakage.png",
            "combo_specialists.png"
        ]
    }


# ============================================================
# Args / Main
# ============================================================
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()

    # Dataset selection
    p.add_argument("--dataset", type=str, default="assist_09",
                   choices=["assist_09", "assist_17", "junyi"],
                   help="选择数据集：assist_09, assist_17, junyi")

    # Paths
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--test_file",  type=str, default=None)
    p.add_argument("--graph_dir",  type=str, default=None)
    p.add_argument("--out_dir",    type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)

    # Model Config
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--fusion_type", type=str, default="enhanced_gated")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_supervised_contrastive", action="store_true", default=True)
    p.add_argument("--gated_num_gates", type=int, default=3)
    p.add_argument("--ortho_weight", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=1024)

    # Exp Params - MI
    p.add_argument("--mi_pairs", type=int, default=512)
    p.add_argument("--mi_bins", type=int, default=20)

    # Exp Params - Alignment
    p.add_argument("--align_min_attempts", type=int, default=5)
    p.add_argument("--align_min_students", type=int, default=200)
    p.add_argument("--leakage_thr", type=float, default=0.15)
    p.add_argument("--specialist_max_leakage", type=int, default=2)
    p.add_argument("--specialist_topk", type=int, default=8)
    p.add_argument("--max_specialists_to_plot", type=int, default=5)

    # NEW: combo
    p.add_argument("--make_combo", action="store_true", default=True, help="生成 combo 图（默认开启）")
    p.add_argument("--no_combo", dest="make_combo", action="store_false", help="禁用 combo 图生成")
    p.add_argument("--combo_dir", type=str, default=None, help="Optional override. Default: out_dir/combo")

    args = p.parse_args()

    # Auto-configure paths based on dataset
    dataset = args.dataset
    data_dir = os.path.join(root, "data", dataset)

    if args.train_file is None:
        args.train_file = os.path.join(data_dir, "train.csv")
    if args.valid_file is None:
        args.valid_file = os.path.join(data_dir, "valid.csv")
    if args.test_file is None:
        args.test_file = os.path.join(data_dir, "test.csv")
    if args.graph_dir is None:
        args.graph_dir = os.path.join(root, "graphs", dataset)
    if args.out_dir is None:
        args.out_dir = os.path.join(root, "exp_m1_out", dataset)
    if args.model_path is None:
        model_path_dataset = os.path.join(root, "saved_models", dataset, "best_model.pth")
        model_path_default = os.path.join(root, "saved_models", "best_model.pth")
        args.model_path = model_path_dataset if os.path.exists(model_path_dataset) else model_path_default

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"[ERR] model_path not found: {args.model_path}")

    # combo_dir (optional override)
    if args.combo_dir is None:
        args.combo_dir = os.path.join(args.out_dir, "combo")

    return args


def main():
    args = get_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[Info] Loading Data & Model...")
    bundle = make_bundle(args)
    model = load_model(args, bundle, device)

    full_df = pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True)
    full_loader = DataLoader(CDDataset(full_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    z = collect_student_knowledge_state(model, full_loader, bundle.graphs, device,
                                        bundle.num_students, bundle.num_concepts)
    acc, den = empirical_concept_accuracy_and_count(full_df, bundle.num_students, bundle.num_concepts)

    # -----------------------------
    # Exp-1B (Alignment): Leakage + Specialists
    # -----------------------------
    print(">>> Running Exp-1B: Alignment (Leakage + Specialists)...")
    R, concept_valid_mask = compute_alignment_matrix(
        z=z, acc=acc, den=den, out_dir=args.out_dir,
        align_min_attempts=args.align_min_attempts,
        align_min_students=args.align_min_students
    )
    align_metrics = plot_alignment_leakage_and_specialists(
        R=R,
        concept_valid_mask=concept_valid_mask,
        out_dir=args.out_dir,
        leakage_thr=args.leakage_thr,
        specialist_max_leakage=args.specialist_max_leakage,
        specialist_topk=args.specialist_topk,
        max_specialists_to_plot=args.max_specialists_to_plot
    )
    top1_concept_per_factor = top1_concept_per_factor_from_R(R)

    # -----------------------------
    # Exp-1A (MI Independence): Matrix Only
    # -----------------------------
    print(">>> Running Exp-1A: MI Independence (Matrix + Pairs Data)...")
    mi_metrics = exp_mi_independence_clean(
        z=z,
        out_dir=args.out_dir,
        seed=args.seed,
        bins=args.mi_bins,
        pairs=args.mi_pairs,
        top1_concept_per_factor=top1_concept_per_factor
    )

    # -----------------------------
    # Optional: Combo figures from CSVs
    # -----------------------------
    combo_info = None
    if args.make_combo:
        print(">>> Generating COMBO figures from CSVs...")
        combo_info = make_combo_from_csvs(args.out_dir, leakage_thr=args.leakage_thr)
        # if user passed --combo_dir, move files accordingly
        if args.combo_dir != os.path.join(args.out_dir, "combo"):
            safe_mkdir(args.combo_dir)
            src_dir = combo_info["combo_dir"]
            for fn in combo_info["generated"]:
                src = os.path.join(src_dir, fn)
                dst = os.path.join(args.combo_dir, fn)
                if os.path.exists(src):
                    try:
                        os.replace(src, dst)
                    except Exception:
                        # fallback copy
                        import shutil
                        shutil.copy2(src, dst)
            combo_info["combo_dir"] = args.combo_dir

    # -----------------------------
    # Final Summary
    # -----------------------------
    summary = {
        "Exp-1A_MI": mi_metrics,
        "Exp-1B_Alignment": align_metrics,
        "Combo": combo_info
    }
    with open(os.path.join(args.out_dir, "summary_final.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Experiments Finished. Evidence saved in {args.out_dir}:")
    print(f"    Single figs:")
    print(f"      1) mi_matrix_sorted.png")
    print(f"      2) alignment_leakage.png")
    print(f"      3) alignment_specialist_dims.png")
    print(f"    Data:")
    print(f"      - cmig_pairs.csv, alignment_matrix.csv, alignment_specialists.csv")
    if args.make_combo:
        print(f"    Combo figs (paper-friendly) in {summary['Combo']['combo_dir']}:")
        for fn in summary["Combo"]["generated"]:
            print(f"      - {fn}")


if __name__ == "__main__":
    main()
