#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-2 Experiments: 鲁棒性与机制验证实验 (Robustness & Mechanism Verification)
================================================================================
核心目标: 证明门控融合与一致性约束在稀疏数据下的鲁棒性。

保留实验:
1. Exp-2A: 基础图鲁棒性 (Robust Curve) - 宏观证明模型抗噪能力强。
2. Exp-2B: 模型扫描 (Pareto) - 展示 Trade-off 轨迹，并获取最佳一致性模型 λ* (带缓存机制)。

本版本改动（面向论文作图）：
- 不改变训练逻辑，保留 model_lambda_*.pth 缓存：有缓存就不训练。
- pareto.csv 补充 D_exer / D_cpt / error 等列，便于做“主图+支撑证据”组合图。
- 绘图风格统一为白底出版级；pareto.png 输出为组合图（trajectory + supporting scatter）。
"""

import os
import json
import argparse
import random
import multiprocessing as mp
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from torch.utils.data import DataLoader

from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph, train_epoch, evaluate, EarlyStopping


# -----------------------------
# Style (Publication-quality, white background)
# -----------------------------
def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "figure.dpi": 300,
    })


def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {os.path.basename(path)}")


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


def sparse_edge_dropout(adj: torch.Tensor, drop_rate: float, seed: int) -> torch.Tensor:
    """Drop a fraction of non-zero edges in a sparse adjacency matrix."""
    if drop_rate <= 0:
        return adj
    g = torch.Generator(device=adj.device)
    g.manual_seed(seed)

    adj = adj.coalesce()
    idx = adj.indices()
    val = adj.values()
    nnz = val.numel()
    keep = torch.rand(nnz, generator=g, device=adj.device) > drop_rate
    idx2 = idx[:, keep]
    val2 = val[keep] / (1.0 - drop_rate + 1e-12)
    out = torch.sparse_coo_tensor(idx2, val2, adj.shape, device=adj.device).coalesce()
    return out


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if str(x).strip()]


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _pareto_front_indices_min2(df: pd.DataFrame, x_col: str, y_col: str) -> List[int]:
    """Minimize both x and y. Return indices of non-dominated points."""
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
    idxs = df.index.to_list()
    front = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        dominated = False
        for j, (x2, y2) in enumerate(zip(xs, ys)):
            if j == i:
                continue
            if (x2 <= x and y2 <= y) and (x2 < x or y2 < y):
                dominated = True
                break
        if not dominated:
            front.append(idxs[i])
    return front


def _select_lambda_star(df_pareto: pd.DataFrame) -> float:
    """
    从 Pareto 结果中自动选择最佳 lambda (权衡 D_view 和 Error)
    score = norm(D_view) + norm(error)
    """
    df = df_pareto.copy()
    if "error" not in df.columns:
        df["error"] = 1.0 - df["test_auc"]

    x = df["D_view_mean"].to_numpy()
    y = df["error"].to_numpy()
    score = _normalize01(x) + _normalize01(y)
    best_i = int(np.argmin(score))
    return float(df.iloc[best_i]["lambda_contrastive"])


# -----------------------------
# Data & Graph Loading
# -----------------------------
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

    return train_df, valid_df, test_df, len(student_id_map), len(exercise_id_map), len(concept_id_map)


def build_all_graphs(train_df, num_students, num_exercises, num_concepts, graph_dir, device):
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_wrong_se = build_graph(train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_correct_sc = build_graph(train_df, num_students, num_concepts, correct=True, item_type="concept", graph_dir=graph_dir).to(device)
    adj_wrong_sc = build_graph(train_df, num_students, num_concepts, correct=False, item_type="concept", graph_dir=graph_dir).to(device)
    return adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc


def init_model(args, num_students, num_exercises, num_concepts, device):
    model = CognitiveDiagnosisModel(
        num_students=num_students,
        num_exercises=num_exercises,
        num_concepts=num_concepts,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        fusion_type=args.fusion_type,
        temperature=args.temperature,
        num_heads=args.num_heads,
        use_supervised_contrastive=args.use_supervised_contrastive,
        gated_num_gates=args.gated_num_gates,
        ortho_weight=args.ortho_weight,
        dropout=args.dropout,
    ).to(device)
    return model


# -----------------------------
# Metric Helpers
# -----------------------------
@torch.no_grad()
def compute_view_distances_basic(model, graphs):
    """计算视图距离：exercise-view, concept-view, mean"""
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs

    _, exer_c = model.gcn_correct_se(adj_correct_se)
    _, exer_w = model.gcn_wrong_se(adj_wrong_se)
    d_exer = float((1.0 - F.cosine_similarity(exer_c, exer_w, dim=-1)).mean().item())

    _, cpt_c = model.gcn_correct_sc(adj_correct_sc)
    _, cpt_w = model.gcn_wrong_sc(adj_wrong_sc)
    d_cpt = float((1.0 - F.cosine_similarity(cpt_c, cpt_w, dim=-1)).mean().item())

    return d_exer, d_cpt, (d_exer + d_cpt) / 2.0


# -----------------------------
# Exp-2A: Graph Robustness
# -----------------------------
def plot_robust_curve_combo(df: pd.DataFrame, out_path: str):
    """AUC 主轴 + ACC 次轴，论文更友好（同图支撑）"""
    fig, ax1 = plt.subplots(figsize=(6.6, 4.4))

    x = df["drop_rate"].to_numpy()
    auc = df["auc"].to_numpy()
    acc = df["accuracy"].to_numpy()

    ax1.plot(x, auc, marker="o", linewidth=2.0, label="AUC")
    ax1.set_xlabel("Graph edge dropout rate")
    ax1.set_ylabel("Test AUC")
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    ax2 = ax1.twinx()
    ax2.plot(x, acc, marker="s", linewidth=2.0, linestyle="--", alpha=0.85, label="Accuracy")
    ax2.set_ylabel("Test Accuracy")
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    # 合并 legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left")

    ax1.set_title("Robustness to graph edge dropout")
    savefig(out_path)


def exp_graph_robustness(args, model_path, train_df, valid_df, test_df,
                        num_students, num_exercises, num_concepts, device):
    print("[Exp-2A] Running Graph Robustness (Dropout vs AUC/ACC)...")
    if not os.path.exists(model_path):
        print(f"[Warn] Baseline model {model_path} not found for Exp-2A. Skipping.")
        return None

    graphs_clean = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(CDDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    rows = []
    for dr in args.graph_drop_rates:
        graphs_noisy = tuple(sparse_edge_dropout(g, dr, seed=args.seed + int(dr * 1000)) for g in graphs_clean)
        _, metrics, _ = evaluate(model, test_loader, device, *graphs_noisy)
        rows.append({
            "drop_rate": float(dr),
            "auc": float(metrics["auc"]),
            "accuracy": float(metrics["accuracy"])
        })

    df = pd.DataFrame(rows).sort_values("drop_rate")
    df.to_csv(os.path.join(args.out_dir, "robust_curve.csv"), index=False)

    plot_robust_curve_combo(df, os.path.join(args.out_dir, "robust_curve.png"))
    return df


# -----------------------------
# Exp-2B: Pareto (Caching + Better CSV + Combo Figure)
# -----------------------------
def train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device):
    best_path = os.path.join(args.out_dir, f"model_lambda_{args.lambda_contrastive:.4f}.pth")
    if os.path.exists(best_path):
        return best_path

    model = init_model(args, num_students, num_exercises, num_concepts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, verbose=False)

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, device, *graphs, args, epoch, verbose=False)
        _, v_metrics, _ = evaluate(model, valid_loader, device, *graphs)
        stopper(v_metrics["auc"], model)
        if stopper.early_stop:
            break

    torch.save(stopper.best_model_state, best_path)
    return best_path


def _pareto_worker(gpu_id, lambdas, args_dict, train_df, valid_df, ns, ne, nc, return_dict):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    set_seed(args.seed + gpu_id * 1000)

    graphs = build_all_graphs(train_df, ns, ne, nc, args.graph_dir, device)
    train_loader = DataLoader(CDDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(CDDataset(valid_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    res = {}
    for lam in lambdas:
        args.lambda_contrastive = float(lam)
        args.contrastive_decay_epochs = int(1e9)
        args.contrastive_min_weight = 0.0
        path = train_one(args, train_loader, valid_loader, graphs, ns, ne, nc, device)
        res[float(lam)] = path
    return_dict[gpu_id] = res


def plot_pareto_combo(df: pd.DataFrame, out_path: str):
    """
    组合图（替换单一 pareto scatter）：
    - 左：D_view vs error trajectory，高亮 Pareto front，标出 λ* 与 λ=0
    - 右：supporting evidence：D_exer vs D_cpt（看一致性来源）
    """
    if df.empty:
        return

    df_plot = df.sort_values("lambda_contrastive").reset_index(drop=True).copy()
    df_plot["error"] = 1.0 - df_plot["test_auc"]

    lam_star = _select_lambda_star(df_plot)
    front_idx = _pareto_front_indices_min2(df_plot, "D_view_mean", "error")

    # 找 λ=0 行（可能没有）
    idx_l0 = None
    if np.any(np.isclose(df_plot["lambda_contrastive"].to_numpy(), 0.0)):
        idx_l0 = int(np.where(np.isclose(df_plot["lambda_contrastive"].to_numpy(), 0.0))[0][0])

    idx_star = int(np.argmin(np.abs(df_plot["lambda_contrastive"].to_numpy() - lam_star)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"wspace": 0.28})

    # ---- Left: trajectory + Pareto front ----
    x = df_plot["D_view_mean"].to_numpy()
    y = df_plot["error"].to_numpy()

    ax1.plot(x, y, linestyle="--", color="#777777", linewidth=1.6, alpha=0.8, label="Trajectory")

    # scatter with lambda colormap
    sc = ax1.scatter(
        x, y,
        c=df_plot["lambda_contrastive"].to_numpy(),
        cmap="viridis",
        s=70,
        edgecolors="#222222",
        linewidths=0.6,
        zorder=3
    )

    # Pareto front highlight
    df_front = df_plot.loc[front_idx].sort_values("D_view_mean")
    ax1.plot(
        df_front["D_view_mean"].to_numpy(),
        df_front["error"].to_numpy(),
        color="#1f77b4",
        linewidth=2.6,
        label="Pareto front",
        zorder=4
    )

    # mark lambda-star
    ax1.scatter([x[idx_star]], [y[idx_star]], s=160, marker="*", color="#d62728", edgecolors="#222222",
                linewidths=0.8, zorder=6, label=r"$\lambda^\star$")

    # mark lambda=0
    if idx_l0 is not None:
        ax1.scatter([x[idx_l0]], [y[idx_l0]], s=120, marker="D", color="#ff7f0e", edgecolors="#222222",
                    linewidths=0.8, zorder=6, label=r"$\lambda=0$")

    ax1.set_xlabel(r"View distance $D_{\mathrm{view}}$ (↓ better consistency)")
    ax1.set_ylabel(r"Error rate $(1-\mathrm{AUC})$ (↓ better accuracy)")
    ax1.set_title("Consistency–accuracy trade-off")
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\lambda$")

    ax1.legend(loc="upper right")

    # ---- Right: supporting evidence (D_exer vs D_cpt) ----
    # 若缺列就降级画 D_view vs AUC
    if ("D_exer" in df_plot.columns) and ("D_cpt" in df_plot.columns):
        ax2.scatter(
            df_plot["D_exer"].to_numpy(),
            df_plot["D_cpt"].to_numpy(),
            c=df_plot["lambda_contrastive"].to_numpy(),
            cmap="viridis",
            s=70,
            edgecolors="#222222",
            linewidths=0.6,
            zorder=3
        )
        ax2.scatter([df_plot.loc[idx_star, "D_exer"]], [df_plot.loc[idx_star, "D_cpt"]],
                    s=160, marker="*", color="#d62728", edgecolors="#222222", linewidths=0.8, zorder=6)

        ax2.set_xlabel(r"$D_{\mathrm{exer}}$ (exercise-view)")
        ax2.set_ylabel(r"$D_{\mathrm{cpt}}$ (concept-view)")
        ax2.set_title("Supporting evidence: where inconsistency comes from")

        # 参考线 y=x，帮助读“哪个视图更不一致”
        mn = float(min(df_plot["D_exer"].min(), df_plot["D_cpt"].min()))
        mx = float(max(df_plot["D_exer"].max(), df_plot["D_cpt"].max()))
        ax2.plot([mn, mx], [mn, mx], linestyle="--", color="#777777", linewidth=1.2, alpha=0.8)
    else:
        ax2.plot(df_plot["lambda_contrastive"], df_plot["test_auc"], marker="o", linewidth=2.0)
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("Test AUC")
        ax2.set_title("Fallback: AUC vs lambda (missing D_exer/D_cpt)")

    savefig(out_path)


def exp_pareto_with_cache(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device):
    print("[Exp-2B] Running Pareto Scan (Caching + Eval + Combo Plot)...")
    results = {}

    # 1) Identify which lambdas need training
    lambdas_to_train = []
    for lam in args.contrastive_lambdas:
        path = os.path.join(args.out_dir, f"model_lambda_{lam:.4f}.pth")
        if os.path.exists(path):
            results[float(lam)] = path
            print(f"  -> Found cached model for lambda={lam:.4f}, skipping training.")
        else:
            lambdas_to_train.append(float(lam))

    # 2) Train missing models (unless skip flag)
    if (not args.skip_pareto_training) and len(lambdas_to_train) > 0:
        print(f"  -> Training missing models: {lambdas_to_train}")
        use_cuda = torch.cuda.is_available()
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if use_cuda else []

        if use_cuda and (not args.pareto_serial) and len(gpus) > 0:
            manager = mp.Manager()
            return_dict = manager.dict()
            buckets = {g: [] for g in gpus}
            for i, l in enumerate(lambdas_to_train):
                buckets[gpus[i % len(gpus)]].append(l)

            procs = []
            for gid, lams in buckets.items():
                if not lams:
                    continue
                p = mp.Process(
                    target=_pareto_worker,
                    args=(gid, lams, vars(args), train_df, valid_df,
                          num_students, num_exercises, num_concepts, return_dict)
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            for m in return_dict.values():
                results.update(m)
        else:
            graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
            tl = DataLoader(CDDataset(train_df), args.batch_size, True, collate_fn=collate_fn)
            vl = DataLoader(CDDataset(valid_df), args.batch_size, False, collate_fn=collate_fn)
            for lam in lambdas_to_train:
                args.lambda_contrastive = float(lam)
                args.contrastive_decay_epochs = int(1e9)
                args.contrastive_min_weight = 0.0
                results[float(lam)] = train_one(args, tl, vl, graphs, num_students, num_exercises, num_concepts, device)

    # 若 skip_pareto_training=True 且缺模型，则直接提示但仍继续评估已有缓存
    if args.skip_pareto_training and len(lambdas_to_train) > 0:
        print(f"[Info] skip_pareto_training=True, missing models not trained: {lambdas_to_train}")

    # 3) Evaluate ALL available models (cached + newly trained)
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    test_loader = DataLoader(CDDataset(test_df), args.batch_size, False, collate_fn=collate_fn)

    rows = []
    for lam in args.contrastive_lambdas:
        path = results.get(float(lam))
        if (not path) or (not os.path.exists(path)):
            print(f"[Warn] Model for lambda={lam} missing, skipping evaluation.")
            continue

        args.lambda_contrastive = float(lam)
        model = init_model(args, num_students, num_exercises, num_concepts, device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        _, tm, _ = evaluate(model, test_loader, device, *graphs)
        d_ex, d_cp, d_mean = compute_view_distances_basic(model, graphs)

        rows.append({
            "lambda_contrastive": float(lam),
            "test_auc": float(tm["auc"]),
            "error": float(1.0 - tm["auc"]),
            "D_exer": float(d_ex),
            "D_cpt": float(d_cp),
            "D_view_mean": float(d_mean),
            "model_path": path
        })

    df = pd.DataFrame(rows).sort_values("lambda_contrastive")
    df.to_csv(os.path.join(args.out_dir, "pareto.csv"), index=False)

    # 4) Plot combo figure
    if not df.empty:
        plot_pareto_combo(df, os.path.join(args.out_dir, "pareto.png"))

        lam_star = _select_lambda_star(df)
        with open(os.path.join(args.out_dir, "lambda_star.json"), "w") as f:
            json.dump({"lambda_star": lam_star}, f, indent=2)
        print(f"  -> lambda_star = {lam_star}")

    return df


# -----------------------------
# Main & Args
# -----------------------------
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()

    # Dataset selection
    p.add_argument("--dataset", type=str, default="assist_09",
                   choices=["assist_09", "assist_17", "junyi"],
                   help="选择数据集：assist_09, assist_17, junyi")

    # Paths (auto overridden)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--graph_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")

    # Model Args
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--fusion_type", type=str, default="enhanced_gated")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_supervised_contrastive", action="store_true", default=True)
    p.add_argument("--gated_num_gates", type=int, default=3)
    p.add_argument("--ortho_weight", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.3)

    # Training Args
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=7)

    # Training schedule params
    p.add_argument("--fusion_warmup_epochs", type=int, default=1)
    p.add_argument("--contrastive_decay_epochs", type=int, default=18)
    p.add_argument("--contrastive_min_weight", type=float, default=0.12)
    p.add_argument("--grad_clip", type=float, default=0.1)

    # Hyperparams for exp
    p.add_argument("--lambda_fusion", type=float, default=0.7)
    p.add_argument("--lambda_contrastive", type=float, default=0.4)
    p.add_argument("--graph_drop_rates", type=str, default="0,0.1,0.2,0.3,0.4")
    p.add_argument("--contrastive_lambdas", type=str, default="0,0.05,0.1,0.2,0.4,0.8,1.2")

    # Control flags
    p.add_argument("--skip_pareto_training", action="store_true",
                   help="不训练 Pareto 缺失模型，仅评估已缓存模型并画图")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--pareto_serial", action="store_true")

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
        args.out_dir = os.path.join(root, "exp_m2_out", dataset)
    if args.model_path is None:
        model_path_dataset = os.path.join(root, "saved_models", dataset, "best_model.pth")
        model_path_default = os.path.join(root, "saved_models", "best_model.pth")
        args.model_path = model_path_dataset if os.path.exists(model_path_dataset) else model_path_default

    return args


def main():
    set_pub_style()
    args = get_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    args.graph_drop_rates = parse_list_floats(args.graph_drop_rates)
    args.contrastive_lambdas = parse_list_floats(args.contrastive_lambdas)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Data
    train_df, valid_df, test_df, num_students, num_exercises, num_concepts = load_and_map_data(
        args.train_file, args.valid_file, args.test_file
    )

    # 1) Exp-2A Robustness
    df_robust = exp_graph_robustness(
        args, args.model_path, train_df, valid_df, test_df,
        num_students, num_exercises, num_concepts, device
    )

    # 2) Exp-2B Pareto
    df_pareto = exp_pareto_with_cache(
        args, train_df, valid_df, test_df,
        num_students, num_exercises, num_concepts, device
    )

    # Summary
    summary = {
        "status": "Success",
        "out_dir": args.out_dir,
        "experiments": ["Robustness Curve", "Pareto (cached training + eval + combo plot)"],
        "has_robust": df_robust is not None,
        "has_pareto": (df_pareto is not None and (not df_pareto.empty)),
    }
    # 写入 lambda_star（若存在）
    lam_star_path = os.path.join(args.out_dir, "lambda_star.json")
    if os.path.exists(lam_star_path):
        try:
            with open(lam_star_path, "r") as f:
                summary["lambda_star"] = json.load(f).get("lambda_star", None)
        except Exception:
            pass

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Module-2 experiments finished. Check outputs in: {args.out_dir}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
