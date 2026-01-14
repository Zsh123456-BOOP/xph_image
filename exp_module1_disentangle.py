#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-1 Experiments: 特征解耦实验 (Feature Disentanglement)
================================================================================
(内容同你原注释，略)
"""

import os
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # rank correlation without scipy dependency
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
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


# -----------------------------
# Hungarian (maximize) - pure python, O(n^3)
# -----------------------------
def hungarian_maximize(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve assignment maximizing cost[i,j].
    Returns row_ind, col_ind like scipy.linear_sum_assignment.
    Implementation: convert to minimization by (max - cost), then run Hungarian.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    assert cost.shape[0] == cost.shape[1], "Hungarian requires square matrix"
    maxv = np.max(cost)
    a = (maxv - cost)

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = np.zeros(n, dtype=int)
    col_ind = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        row_ind[p[j] - 1] = p[j] - 1
        col_ind[p[j] - 1] = j - 1
    return row_ind, col_ind


# -----------------------------
# Data + Graphs
# -----------------------------
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


def load_and_map_data(train_file: str, valid_file: str, test_file: str):
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    student_ids = sorted(all_df["stu_id"].unique())
    exercise_ids = sorted(all_df["exer_id"].unique())
    all_cpts = set()
    for cpt_seq in all_df["cpt_seq"]:
        all_cpts.update(int(c) for c in str(cpt_seq).split(","))
    concept_ids = sorted(all_cpts)

    student_id_map = {old: new for new, old in enumerate(student_ids)}
    exercise_id_map = {old: new for new, old in enumerate(exercise_ids)}
    concept_id_map = {old: new for new, old in enumerate(concept_ids)}

    def map_concepts(cpt_seq):
        cpts = [concept_id_map[int(c)] for c in str(cpt_seq).split(",")]
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

    all_cpts = set()
    for cpt_seq in all_df["cpt_seq"]:
        all_cpts.update(int(c) for c in str(cpt_seq).split(","))
    num_concepts = len(all_cpts)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    train_loader = DataLoader(CDDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(CDDataset(valid_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(CDDataset(test_df),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return DataBundle(
        num_students=num_students,
        num_exercises=num_exercises,
        num_concepts=num_concepts,
        train_df=train_df, valid_df=valid_df, test_df=test_df,
        loaders=(train_loader, valid_loader, test_loader),
        graphs=graphs
    )


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


# -----------------------------
# Collect representations
# -----------------------------
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


def empirical_concept_accuracy(df: pd.DataFrame, num_students: int, num_concepts: int) -> np.ndarray:
    num = np.zeros((num_students, num_concepts), dtype=np.float64)
    den = np.zeros((num_students, num_concepts), dtype=np.float64)

    for _, r in df.iterrows():
        u = int(r["stu_id"])
        y = int(r["label"])
        cpts = [int(c) for c in str(r["cpt_seq"]).split(",")]
        for k in cpts:
            num[u, k] += y
            den[u, k] += 1.0

    acc = num / (den + 1e-12)
    return acc


# -----------------------------
# Exp-1A: MI independence
# -----------------------------
def exp_mi_independence(z: np.ndarray, out_dir: str, seed: int, pairs: int = 512, bins: int = 20):
    rng = np.random.default_rng(seed)
    K = z.shape[1]
    H = np.array([hist_entropy_1d(np.clip(z[:, k], 0.0, 1.0), bins=bins) for k in range(K)], dtype=np.float64)

    pair_list = set()
    while len(pair_list) < min(pairs, K * (K - 1) // 2):
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, K))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pair_list.add((i, j))
    pair_list = sorted(list(pair_list))

    rows = []
    mi_vals = []
    nmi_vals = []
    for (i, j) in pair_list:
        xi = np.clip(z[:, i], 0.0, 1.0)
        xj = np.clip(z[:, j], 0.0, 1.0)
        mi = hist_mi_2d(xi, xj, bins=bins)
        denom = math.sqrt(max(H[i] * H[j], 1e-12))
        nmi = mi / denom
        rows.append({"i": i, "j": j, "H_i": float(H[i]), "H_j": float(H[j]), "MI": float(mi), "NMI": float(nmi)})
        mi_vals.append(mi)
        nmi_vals.append(nmi)

    df_pairs = pd.DataFrame(rows)
    df_pairs.to_csv(os.path.join(out_dir, "cmig_pairs.csv"), index=False)

    plt.figure(figsize=(7, 4.5))
    plt.hist(mi_vals, bins=40)
    plt.title("Histogram MI over sampled (i,j) pairs (knowledge_state dims)")
    plt.xlabel("MI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mi_hist.png"), dpi=220)
    plt.close()

    metrics = {
        "K": int(K),
        "pairs": int(len(pair_list)),
        "MI_mean": float(np.mean(mi_vals)),
        "MI_std": float(np.std(mi_vals)),
        "NMI_mean": float(np.mean(nmi_vals)),
        "NMI_std": float(np.std(nmi_vals)),
        "IndependenceScore_(1-NMI_mean)": float(1.0 - np.mean(nmi_vals)),
    }
    with open(os.path.join(out_dir, "mi_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


# -----------------------------
# 1B plotting (3 figures)
# -----------------------------
def plot_alignment_raw(R: np.ndarray, out_dir: str):
    plt.figure(figsize=(9, 7))
    plt.imshow(R, aspect="auto")
    plt.colorbar()
    plt.title("Semantic Anchor Alignment: Spearman corr(z_i, acc_k)")
    plt.xlabel("Concept k (empirical accuracy)")
    plt.ylabel("Factor i (knowledge_state dim)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_heatmap_raw.png"), dpi=220)
    plt.close()


def plot_alignment_permuted(R: np.ndarray, col_ind: np.ndarray, out_dir: str):
    """
    Use Hungarian assignment to permute columns, and reorder rows by assigned column.
    This removes permutation ambiguity and makes the alignment visually interpretable.
    """
    K = R.shape[0]
    absR = np.abs(R)
    assigned_col = np.asarray(col_ind).astype(int)

    # row order: sort by which concept it matches to
    row_order = np.argsort(assigned_col)
    # column order: sort by concept index for readability
    col_order = np.argsort(assigned_col[row_order])

    R_perm = R[row_order][:, assigned_col[row_order][col_order]]

    plt.figure(figsize=(9, 7))
    plt.imshow(R_perm, aspect="auto")
    plt.colorbar()
    plt.title("Alignment Heatmap after Permutation Matching (Hungarian)")
    plt.xlabel("Matched concept index (permuted)")
    plt.ylabel("Factors reordered by matched concept")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_heatmap_perm.png"), dpi=220)
    plt.close()


def plot_alignment_biembed_3d(R: np.ndarray, out_dir: str, rank: int = 3):
    """
    Innovative view: embed factors and concepts into a shared 3D space via SVD on |R|.
    factor_coords = U * sqrt(S), concept_coords = V * sqrt(S)
    """
    absR = np.abs(R).astype(np.float64)
    U, S, Vt = np.linalg.svd(absR, full_matrices=False)
    r = min(rank, U.shape[1])
    U = U[:, :r]
    V = Vt[:r, :].T
    S = S[:r]
    W = np.sqrt(S + 1e-12)

    factor_xyz = U * W
    concept_xyz = V * W

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(factor_xyz[:, 0], factor_xyz[:, 1], factor_xyz[:, 2], s=10, label="Factors (z_i)")
    ax.scatter(concept_xyz[:, 0], concept_xyz[:, 1], concept_xyz[:, 2], s=10, label="Concept anchors (acc_k)")

    ax.set_title("3D Bi-Embedding of Factors & Concepts (SVD on |corr|)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_zlabel("dim-3")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_biembed_3d.png"), dpi=220)
    plt.close()


# -----------------------------
# Exp-1B: Semantic Anchor Alignment
# -----------------------------
def exp_anchor_alignment(z: np.ndarray, acc: np.ndarray, out_dir: str, tau_list=(0.10, 0.15)):
    K = z.shape[1]
    R = np.zeros((K, K), dtype=np.float64)

    for i in range(K):
        zi = z[:, i]
        for k in range(K):
            R[i, k] = spearman_corr(zi, acc[:, k])

    df_matrix = pd.DataFrame(R, index=[f"z_{i}" for i in range(K)],
                             columns=[f"acc_{k}" for k in range(K)])
    df_matrix.to_csv(os.path.join(out_dir, "alignment_matrix.csv"))

    # --- metrics (permutation-invariant) ---
    absR = np.abs(R)

    row_mean = absR.mean(axis=1) + 1e-12
    row_top1 = absR.max(axis=1)
    row_concentration = row_top1 / row_mean

    row_entropy = []
    for i in range(K):
        p = absR[i] / (absR[i].sum() + 1e-12)
        p = p[p > 0]
        row_entropy.append(float(-(p * np.log(p + 1e-12)).sum()))
    row_entropy = np.array(row_entropy, dtype=np.float64)

    _, col_ind = hungarian_maximize(absR)
    assigned = absR[np.arange(K), col_ind]
    assignment_mean = float(np.mean(assigned))
    assignment_median = float(np.median(assigned))

    match_rates = {}
    for tau in tau_list:
        match_rates[f"MatchRate@{tau:.2f}"] = float(np.mean(assigned >= tau))

    metrics = {
        "RowConcentrationScore_mean(top1/mean_abs)": float(np.mean(row_concentration)),
        "RowEntropy_mean(lower_better)": float(np.mean(row_entropy)),
        "AssignmentMeanAbsCorr": assignment_mean,
        "AssignmentMedianAbsCorr": assignment_median,
        **match_rates,
        "note": "No diagonal requirement. Use row concentration + permutation-invariant assignment score."
    }

    with open(os.path.join(out_dir, "alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- 1B: generate 3 figures ---
    plot_alignment_raw(R, out_dir)
    plot_alignment_permuted(R, col_ind, out_dir)
    plot_alignment_biembed_3d(R, out_dir, rank=3)

    return metrics


# -----------------------------
# Exp-1C: Counterfactual traversal
# -----------------------------
def pick_median_student(train_df: pd.DataFrame) -> int:
    g = train_df.groupby("stu_id")["label"].mean()
    u = int(g.sort_values().index[len(g) // 2])
    return u


def build_exer_concepts_map(df: pd.DataFrame) -> Dict[int, List[int]]:
    m = {}
    for exer_id, sub in df.groupby("exer_id"):
        cpts = [int(c) for c in str(sub.iloc[0]["cpt_seq"]).split(",")]
        m[int(exer_id)] = cpts
    return m


@torch.no_grad()
def precompute_global_embeddings(model: CognitiveDiagnosisModel, graphs):
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs

    stu_c_se, exer_c = model.gcn_correct_se(adj_correct_se)
    stu_w_se, exer_w = model.gcn_wrong_se(adj_wrong_se)
    stu_c_sc, cpt_c = model.gcn_correct_sc(adj_correct_sc)
    stu_w_sc, cpt_w = model.gcn_wrong_sc(adj_wrong_sc)

    stu_se, _ = model.fusion_se(stu_c_se, stu_w_se)
    stu_sc, _ = model.fusion_sc(stu_c_sc, stu_w_sc)

    stu_final, _ = model.gated_fusion_student(stu_se, stu_sc)
    exer_final, _ = model.gated_fusion_exercise(exer_c, exer_w)
    cpt_final, _ = model.gated_fusion_concept(cpt_c, cpt_w)

    return stu_final, exer_final, cpt_final


def _orthogonalize_direction(gk: torch.Tensor, other_grads: List[torch.Tensor]) -> torch.Tensor:
    d = gk.clone()
    eps = 1e-12
    for gj in other_grads:
        denom = torch.dot(gj, gj) + eps
        d = d - (torch.dot(d, gj) / denom) * gj
    dn = torch.norm(d) + eps
    return d / dn


@torch.no_grad()
def _ks_from_stu_emb(model: CognitiveDiagnosisModel, stu_emb: torch.Tensor) -> torch.Tensor:
    if stu_emb.dim() == 1:
        stu_emb = stu_emb.unsqueeze(0)
    ks = model.knowledge_diagnosis(stu_emb).squeeze(0)
    return ks


def _line_search_range(
    model: CognitiveDiagnosisModel,
    stu_base: torch.Tensor,
    direction: torch.Tensor,
    k: int,
    target_delta: float = 0.25,
    max_iter: int = 25,
) -> Tuple[float, float, Dict[str, float]]:
    base_ks = _ks_from_stu_emb(model, stu_base)
    base = float(base_ks[k].item())

    def eval_t(t: float) -> float:
        ks = _ks_from_stu_emb(model, stu_base + t * direction)
        return float(ks[k].item())

    t_hi = 0.0
    val_hi = base
    step = 1.0
    for _ in range(max_iter):
        t_try = t_hi + step
        v = eval_t(t_try)
        if v >= base + target_delta or v >= 0.99:
            t_hi = t_try
            val_hi = v
            break
        t_hi = t_try
        val_hi = v
        step *= 1.3

    t_lo = 0.0
    val_lo = base
    step = 1.0
    for _ in range(max_iter):
        t_try = t_lo - step
        v = eval_t(t_try)
        if v <= base - target_delta or v <= 0.01:
            t_lo = t_try
            val_lo = v
            break
        t_lo = t_try
        val_lo = v
        step *= 1.3

    info = {"base_ks_k": base, "ks_k_at_low": val_lo, "ks_k_at_high": val_hi}
    return float(t_lo), float(t_hi), info


# -----------------------------
# 1C plotting (3 figures)
# -----------------------------
def plot_traversal_heatmap_sorted(heat: np.ndarray, row_labels: List[str], ts: np.ndarray, out_dir: str):
    # parse group prefix: "A:xxxx"
    groups = [lbl.split(":", 1)[0] for lbl in row_labels]

    # sensitivity by range
    ranges = np.max(heat, axis=1) - np.min(heat, axis=1)

    # group-wise sort: A then B then C; within group by sensitivity desc
    order = []
    for g in ["A", "B", "C"]:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        idx_sorted = sorted(idx, key=lambda i: float(ranges[i]), reverse=True)
        order.extend(idx_sorted)

    heat2 = heat[order]
    labels2 = [row_labels[i] for i in order]

    plt.figure(figsize=(11, 7))
    plt.imshow(heat2, aspect="auto")
    plt.colorbar(label="P(correct)")
    plt.title("Counterfactual Traversal (sorted by group and sensitivity)")
    plt.xlabel("Traversal step (t from low -> high)")
    plt.ylabel("Items (sorted)")
    plt.yticks(range(len(labels2)), labels2, fontsize=7)

    # mark t=0 column (closest)
    j0 = int(np.argmin(np.abs(ts - 0.0)))
    plt.axvline(j0, linewidth=1)

    # group separators
    # find last index of each group in sorted list
    for g in ["A", "B"]:
        last = max([i for i, lbl in enumerate(labels2) if lbl.startswith(g + ":")], default=-1)
        if last >= 0:
            plt.axhline(last + 0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traversal_heatmap_sorted.png"), dpi=220)
    plt.close()


def plot_traversal_group_curves(heat: np.ndarray, row_labels: List[str], ts: np.ndarray, out_dir: str):
    groups = np.array([lbl.split(":", 1)[0] for lbl in row_labels])
    xs = np.arange(heat.shape[1], dtype=np.float32)

    plt.figure(figsize=(8.5, 5.2))
    for g in ["A", "B", "C"]:
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        m = heat[idx].mean(axis=0)
        s = heat[idx].std(axis=0)
        plt.plot(xs, m, label=f"{g} mean")
        plt.fill_between(xs, m - s, m + s, alpha=0.2)

    plt.title("Traversal Group Response Curves (mean ± std)")
    plt.xlabel("Traversal step index (low -> high)")
    plt.ylabel("P(correct)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traversal_group_curves.png"), dpi=220)
    plt.close()


def plot_traversal_group_3d(heat: np.ndarray, row_labels: List[str], out_dir: str):
    groups = np.array([lbl.split(":", 1)[0] for lbl in row_labels])
    x = np.arange(heat.shape[1], dtype=np.float32)

    group_means = []
    for g in ["A", "B", "C"]:
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            group_means.append(None)
        else:
            group_means.append(heat[idx].mean(axis=0))

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for yi, g in enumerate(["A", "B", "C"]):
        m = group_means[yi]
        if m is None:
            continue
        y = np.full_like(x, yi, dtype=np.float32)
        ax.plot(x, y, m, label=f"{g} mean")

    ax.set_title("3D Group Trajectories (Traversal Response)")
    ax.set_xlabel("step")
    ax.set_ylabel("group (A=0,B=1,C=2)")
    ax.set_zlabel("P(correct)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traversal_group_3d.png"), dpi=220)
    plt.close()


def exp_counterfactual_traversal(
    model: CognitiveDiagnosisModel,
    bundle: DataBundle,
    out_dir: str,
    device: torch.device,
    concept_k: Optional[int] = None,
    student_u: Optional[int] = None,
    steps: int = 21,
    n_items_each: int = 12,
    gs_other_dims: int = 12,
    target_delta_ks: float = 0.25,
):
    model.eval()
    if student_u is None:
        student_u = pick_median_student(bundle.train_df)
    if concept_k is None:
        concept_k = int(np.random.randint(0, bundle.num_concepts))

    full_df = pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True)
    exer_map = build_exer_concepts_map(full_df)

    A_only = [q for q, cpts in exer_map.items() if (len(cpts) == 1 and cpts[0] == concept_k)]
    B_mix  = [q for q, cpts in exer_map.items() if (concept_k in cpts and len(cpts) >= 2)]
    C_not  = [q for q, cpts in exer_map.items() if (concept_k not in cpts)]

    rng = np.random.default_rng(0)
    rng.shuffle(A_only); rng.shuffle(B_mix); rng.shuffle(C_not)

    A = A_only[:n_items_each]
    B = B_mix[:n_items_each]
    C = C_not[:n_items_each]
    items = A + B + C
    item_groups = (["A"] * len(A)) + (["B"] * len(B)) + (["C"] * len(C))

    if len(A) == 0 or len(C) == 0:
        print(f"[WARN] Exp-1C: A_only={len(A)}, C_not={len(C)}. "
              f"若 A_only 为空，说明该 concept_k 几乎不作为单概念题出现，建议换 k 或放宽 A 定义。")

    stu_final, exer_final, cpt_final = precompute_global_embeddings(model, bundle.graphs)

    stu_base = stu_final[student_u].detach().clone().to(device)

    # gradient of ks[k] w.r.t stu_emb
    stu_var = stu_base.detach().clone().requires_grad_(True)
    ks = model.knowledge_diagnosis(stu_var.unsqueeze(0)).squeeze(0)
    ks[concept_k].backward()
    gk = stu_var.grad.detach().clone()

    all_dims = list(range(bundle.num_concepts))
    if concept_k in all_dims:
        all_dims.remove(concept_k)
    rng2 = np.random.default_rng(123)
    rng2.shuffle(all_dims)
    other_dims = all_dims[: min(gs_other_dims, len(all_dims))]

    other_grads = []
    for j in other_dims:
        stu_tmp = stu_base.detach().clone().requires_grad_(True)
        ksj = model.knowledge_diagnosis(stu_tmp.unsqueeze(0)).squeeze(0)[j]
        ksj.backward()
        other_grads.append(stu_tmp.grad.detach().clone())

    direction = _orthogonalize_direction(gk, other_grads).detach()

    t_lo, t_hi, range_info = _line_search_range(
        model=model,
        stu_base=stu_base,
        direction=direction,
        k=concept_k,
        target_delta=target_delta_ks,
        max_iter=25,
    )

    ts = np.linspace(t_lo, t_hi, steps).astype(np.float32)

    # leakage (simplified & correct)
    with torch.no_grad():
        ks0 = _ks_from_stu_emb(model, stu_base)
        ksL = _ks_from_stu_emb(model, stu_base + float(t_lo) * direction)
        ksH = _ks_from_stu_emb(model, stu_base + float(t_hi) * direction)
        leakage_all = float((torch.mean(torch.abs(ksL - ks0)) + torch.mean(torch.abs(ksH - ks0))).item() / 2.0)
        if bundle.num_concepts > 1:
            mask = torch.ones(bundle.num_concepts, dtype=torch.bool, device=device)
            mask[concept_k] = False
            leakage_other = float(((torch.mean(torch.abs((ksL - ks0)[mask])) + torch.mean(torch.abs((ksH - ks0)[mask]))).item()) / 2.0)
        else:
            leakage_other = 0.0

    heat = np.zeros((len(items), steps), dtype=np.float32)

    for ti, t in enumerate(ts):
        stu_emb_t = (stu_base + (float(t) * direction)).unsqueeze(0)
        for ri, q in enumerate(items):
            exer_emb = exer_final[q].unsqueeze(0)
            cpts = exer_map[q]
            cpt_embs = cpt_final[cpts].unsqueeze(0)
            mask = torch.ones(1, len(cpts), dtype=torch.bool, device=device)
            pred = model.diagnosis_head(stu_emb_t, exer_emb, cpt_embs, mask).item()
            heat[ri, ti] = float(pred)

    row_labels = [f"{item_groups[i]}:{items[i]}" for i in range(len(items))]
    col_labels = [f"t={ts[i]:+.3f}" for i in range(len(ts))]
    df_heat = pd.DataFrame(heat, index=row_labels, columns=col_labels)
    df_heat.to_csv(os.path.join(out_dir, "traversal_matrix.csv"))

    # keep your original heatmap name (optional)
    plt.figure(figsize=(10, 6))
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="P(correct)")
    plt.title(f"Counterfactual Traversal Heatmap (student={student_u}, concept_k={concept_k})")
    plt.xlabel("Traversal step (t from low -> high)")
    plt.ylabel("Items (A: only k, B: k+others, C: not k)")
    plt.yticks(range(len(items)), row_labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traversal_heatmap.png"), dpi=220)
    plt.close()

    # --- 1C: generate 3 figures ---
    plot_traversal_heatmap_sorted(heat, row_labels, ts, out_dir)
    plot_traversal_group_curves(heat, row_labels, ts, out_dir)
    plot_traversal_group_3d(heat, row_labels, out_dir)

    # metrics
    def row_spearman(vals: np.ndarray) -> float:
        return spearman_corr(np.arange(len(vals), dtype=np.float32), vals.astype(np.float32))

    A_rows = [i for i, g in enumerate(item_groups) if g == "A"]
    C_rows = [i for i, g in enumerate(item_groups) if g == "C"]

    A_s = [row_spearman(heat[i]) for i in A_rows] if len(A_rows) else []
    C_s = [row_spearman(heat[i]) for i in C_rows] if len(C_rows) else []

    def row_range(vals: np.ndarray) -> float:
        return float(np.max(vals) - np.min(vals))

    A_range = [row_range(heat[i]) for i in A_rows] if len(A_rows) else []
    C_range = [row_range(heat[i]) for i in C_rows] if len(C_rows) else []
    C_std   = [float(np.std(heat[i])) for i in C_rows] if len(C_rows) else []

    A_s_mean = float(np.mean(A_s)) if A_s else 0.0
    A_pos_rate = float(np.mean(np.array(A_s) > 0.0)) if A_s else 0.0
    C_std_mean = float(np.mean(C_std)) if C_std else 0.0

    A_range_mean = float(np.mean(A_range)) if A_range else 0.0
    C_range_mean = float(np.mean(C_range)) if C_range else 0.0
    effect_ratio = float(A_range_mean / (C_range_mean + 1e-6)) if (A_range or C_range) else 0.0

    metrics = {
        "student_u": int(student_u),
        "concept_k": int(concept_k),
        "steps": int(steps),
        "t_low": float(t_lo),
        "t_high": float(t_hi),
        **range_info,
        "A_spearman_mean": A_s_mean,
        "A_spearman_pos_rate": A_pos_rate,
        "C_std_mean": C_std_mean,
        "A_range_mean": A_range_mean,
        "C_range_mean": C_range_mean,
        "EffectRatio(A_range/C_range)": effect_ratio,
        "leakage_mean_all_dims": leakage_all,
        "leakage_mean_other_dims": leakage_other,
        "note": "Good: A monotonic↑ (high spearman, pos_rate), C stable (low std/range), effect_ratio>>1, low leakage."
    }
    with open(os.path.join(out_dir, "traversal_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


# -----------------------------
# Main
# -----------------------------
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default=os.path.join(root, "assist_09", "train.csv"))
    p.add_argument("--valid_file", type=str, default=os.path.join(root, "assist_09", "valid.csv"))
    p.add_argument("--test_file",  type=str, default=os.path.join(root, "assist_09", "test.csv"))
    p.add_argument("--graph_dir",  type=str, default=os.path.join(root, "graphs"))
    p.add_argument("--out_dir",    type=str, default=os.path.join(root, "exp_m1_out"))

    p.add_argument("--model_path", type=str, default=os.path.join(root, "saved_models", "best_model.pth"))

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--fusion_type", type=str, default="enhanced_gated", choices=["enhanced_gated", "concat_gate"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_supervised_contrastive", action="store_true", default=True)
    p.add_argument("--gated_num_gates", type=int, default=3)
    p.add_argument("--ortho_weight", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--batch_size", type=int, default=1024)

    p.add_argument("--mi_pairs", type=int, default=512)
    p.add_argument("--mi_bins", type=int, default=20)

    p.add_argument("--traversal_steps", type=int, default=21)
    p.add_argument("--traversal_items_each", type=int, default=12)
    p.add_argument("--traversal_student", type=int, default=-1)
    p.add_argument("--traversal_concept", type=int, default=-1)

    p.add_argument("--traversal_other_dims", type=int, default=12)
    p.add_argument("--traversal_target_delta_ks", type=float, default=0.25)

    args = p.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"[ERR] model_path not found: {args.model_path}\n"
            f"请先运行: python main.py 训练生成 saved_models/best_model.pth，或手动指定 --model_path"
        )
    return args


def main():
    args = get_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = make_bundle(args)
    model = load_model(args, bundle, device)

    full_df = pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True)
    full_loader = DataLoader(CDDataset(full_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    z = collect_student_knowledge_state(model, full_loader, bundle.graphs, device, bundle.num_students, bundle.num_concepts)
    acc = empirical_concept_accuracy(full_df, bundle.num_students, bundle.num_concepts)

    mi_metrics = exp_mi_independence(z, args.out_dir, seed=args.seed, pairs=args.mi_pairs, bins=args.mi_bins)
    align_metrics = exp_anchor_alignment(z, acc, args.out_dir)

    student_u = None if args.traversal_student < 0 else args.traversal_student
    concept_k = None if args.traversal_concept < 0 else args.traversal_concept
    trav_metrics = exp_counterfactual_traversal(
        model=model,
        bundle=bundle,
        out_dir=args.out_dir,
        device=device,
        concept_k=concept_k,
        student_u=student_u,
        steps=args.traversal_steps,
        n_items_each=args.traversal_items_each,
        gs_other_dims=args.traversal_other_dims,
        target_delta_ks=args.traversal_target_delta_ks,
    )

    summary = {"Exp-1A_MI": mi_metrics, "Exp-1B_Align": align_metrics, "Exp-1C_Traversal": trav_metrics}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] Module-1 experiments finished:", args.out_dir)


if __name__ == "__main__":
    main()
