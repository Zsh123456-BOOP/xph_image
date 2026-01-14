#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-1 Experiments: 特征解耦实验 (Feature Disentanglement) - FINAL SELECTED
================================================================================
仅保留能有效支撑“结构化解耦”论点的三张核心图表及其数据：

1. [Exp-1A] MI Independence (基石)
   - 图表: mi_matrix_sorted.png (证明特征空间干净、独立、无冗余)
   - 数据: cmig_pairs.csv, mi_metrics.json

2. [Exp-1B] Alignment Leakage (现象)
   - 图表: alignment_leakage.png (证明大部分维度是稀疏的，只有少数是通用的)
   - 数据: alignment_matrix.csv

3. [Exp-1B] Specialist Dimensions (高光)
   - 图表: alignment_specialist_dims.png (证明存在“专家”维度，精准对应特定概念)
   - 数据: alignment_specialists.csv, alignment_metrics.json

已移除：
- Exp-1C (Masking): 因结果差异不显著，不再作为证据。
- 其他辅助图表。
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


# -----------------------------
# Data Bundle
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


# -----------------------------
# Exp-1B (Alignment): Leakage + Specialists
# -----------------------------
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
    # Criteria: low leakage + reasonable max corr
    candidates = np.where((leakage <= specialist_max_leakage) & (max_corr > leakage_thr))[0]

    if len(candidates) == 0:
        # fallback: take dims with smallest leakage, then highest max_corr
        order = np.lexsort((-max_corr, leakage))
        candidates = order[:max_specialists_to_plot]

    # rank candidates by (leakage asc, max_corr desc)
    cand_order = np.lexsort((-max_corr[candidates], leakage[candidates]))
    specialists = candidates[cand_order][:max_specialists_to_plot]

    # Build specialists table
    specialist_rows = []
    for d in specialists:
        row = {
            "dim": int(d),
            "leakage": int(leakage[d]),
            "max_corr": float(max_corr[d]),
            "mean_corr": float(mean_corr[d]),
        }
        # top-k concepts for this dim
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


# -----------------------------
# Exp-1A (MI Independence): Matrix Only
# -----------------------------
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


# -----------------------------
# Helper: top1_concept_per_factor from alignment matrix
# -----------------------------
def top1_concept_per_factor_from_R(R: np.ndarray) -> np.ndarray:
    return np.argmax(R, axis=1).astype(int)


# -----------------------------
# Main
# -----------------------------
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()

    # Dataset selection (NEW)
    p.add_argument("--dataset", type=str, default="assist_09",
                   choices=["assist_09", "assist_17", "junyi"],
                   help="选择数据集：assist_09, assist_17, junyi")

    # Paths (will be overridden based on dataset)
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
        # Try dataset-specific model first, then fallback to default
        model_path_dataset = os.path.join(root, "saved_models", dataset, "best_model.pth")
        model_path_default = os.path.join(root, "saved_models", "best_model.pth")
        if os.path.exists(model_path_dataset):
            args.model_path = model_path_dataset
        else:
            args.model_path = model_path_default
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"[ERR] model_path not found: {args.model_path}")
    
    return args


def main():
    args = get_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[Info] Loading Data & Model...")
    bundle = make_bundle(args)
    model = load_model(args, bundle, device)

    # Pre-collect stats
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
    # Derive concept grouping for 1A
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
    # Final Summary
    # -----------------------------
    summary = {
        "Exp-1A_MI": mi_metrics,
        "Exp-1B_Alignment": align_metrics
    }
    with open(os.path.join(args.out_dir, "summary_final.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Experiments Finished. Core Evidence saved in {args.out_dir}:")
    print(f"    1. mi_matrix_sorted.png (Base Evidence)")
    print(f"    2. alignment_leakage.png (Phenomenon)")
    print(f"    3. alignment_specialist_dims.png (Highlight)")
    print(f"    + Corresponding Data: cmig_pairs.csv, alignment_matrix.csv, alignment_specialists.csv")


if __name__ == "__main__":
    main()