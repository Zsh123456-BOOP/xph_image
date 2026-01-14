#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-2 Experiments: 鲁棒性与机制验证实验 (Robustness & Mechanism Verification)
================================================================================
核心目标: 证明门控融合与一致性约束在稀疏数据下的鲁棒性。

保留实验:
1. Exp-2A: 基础图鲁棒性 (Robust Curve) - 宏观证明模型抗噪能力强。
2. Exp-2B: 模型扫描 (Pareto) - 用于展示 Trade-off 轨迹，并获取最佳一致性模型 λ* (带缓存机制)。
3. Exp-3D: 多层级敏感度分析 (Multi-level Sensitivity) - 微观证明门控过滤了不稳定的概念视图。
"""

import os
import json
import argparse
import random
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph, train_epoch, evaluate, EarlyStopping


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
    """
    Drop a fraction of non-zero edges in a sparse adjacency matrix.
    """
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


def _pareto_front_indices(df: pd.DataFrame, x_col: str, y_col: str) -> List[int]:
    """
    Minimize both x and y. Return indices of non-dominated points.
    """
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


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _select_lambda_star(df_pareto: pd.DataFrame) -> float:
    """
    从 Pareto 结果中自动选择最佳 lambda (权衡 D_view 和 Error)
    """
    df = df_pareto.copy()
    df["error"] = 1.0 - df["test_auc"]
    
    x = df["D_view_mean"].to_numpy()
    y = df["error"].to_numpy()
    score = _normalize01(x) + _normalize01(y)

    best_i = int(np.argmin(score))
    lam_star = float(df.iloc[best_i]["lambda_contrastive"])
    return lam_star


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


def warmup_graph_cache(train_df, num_students, num_exercises, num_concepts, graph_dir):
    safe_mkdir(graph_dir)
    cpu = torch.device("cpu")
    _ = build_all_graphs(train_df, num_students, num_exercises, num_concepts, graph_dir, cpu)


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
    """
    仅计算基本的视图距离，用于 Exp-2B Pareto 绘图
    """
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs
    _, exer_c = model.gcn_correct_se(adj_correct_se)
    _, exer_w = model.gcn_wrong_se(adj_wrong_se)
    d_exer = float((1.0 - F.cosine_similarity(exer_c, exer_w, dim=-1)).mean().item())
    
    # 即使 Concept 视图可能没被充分利用，我们依然计算它的距离，用于观察
    _, cpt_c = model.gcn_correct_sc(adj_correct_sc)
    _, cpt_w = model.gcn_wrong_sc(adj_wrong_sc)
    d_cpt = float((1.0 - F.cosine_similarity(cpt_c, cpt_w, dim=-1)).mean().item())
    
    return d_exer, d_cpt, (d_exer + d_cpt) / 2.0


# -----------------------------
# Exp-2A: Graph Robustness (Baseline)
# -----------------------------
def exp_graph_robustness(args, model_path, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device):
    print("[Exp-2A] Running Graph Robustness (Dropout vs AUC)...")
    if not os.path.exists(model_path):
        print(f"[Warn] Baseline model {model_path} not found for Exp-2A. Skipping.")
        return None

    graphs_clean = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(
        CDDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

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

    plt.figure()
    plt.plot(df["drop_rate"], df["auc"], marker="o")
    plt.xlabel("Graph Edge Dropout Rate")
    plt.ylabel("Test AUC")
    plt.title("Robustness to Graph Noise")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "robust_curve.png"), dpi=220)
    plt.close()
    return df


# -----------------------------
# Exp-2B: Pareto (With Caching and New Plotting)
# -----------------------------
def train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device):
    best_path = os.path.join(args.out_dir, f"model_lambda_{args.lambda_contrastive:.4f}.pth")
    # Caching check
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
    if torch.cuda.is_available(): torch.cuda.set_device(gpu_id)
    set_seed(args.seed + gpu_id * 1000)
    
    graphs = build_all_graphs(train_df, ns, ne, nc, args.graph_dir, device)
    train_loader = DataLoader(CDDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(CDDataset(valid_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    res = {}
    for lam in lambdas:
        args.lambda_contrastive = float(lam)
        # Restore necessary training args
        args.contrastive_decay_epochs = int(1e9) 
        args.contrastive_min_weight = 0.0
        path = train_one(args, train_loader, valid_loader, graphs, ns, ne, nc, device)
        res[float(lam)] = path
    return_dict[gpu_id] = res

def exp_pareto_with_cache(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device):
    print("[Exp-2B] Running Pareto Scan (Finding Lambda_Star) with Cache Check...")
    results = {}
    
    # Identify which lambdas need training
    lambdas_to_train = []
    for lam in args.contrastive_lambdas:
        path = os.path.join(args.out_dir, f"model_lambda_{lam:.4f}.pth")
        if os.path.exists(path):
            results[float(lam)] = path
            print(f"  -> Found cached model for lambda={lam:.4f}, skipping training.")
        else:
            lambdas_to_train.append(float(lam))

    # Train missing models
    if len(lambdas_to_train) > 0:
        print(f"  -> Training missing models: {lambdas_to_train}")
        use_cuda = torch.cuda.is_available()
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()] if use_cuda else []
        
        if use_cuda and not args.pareto_serial and len(gpus) > 0:
            manager = mp.Manager()
            return_dict = manager.dict()
            buckets = {g: [] for g in gpus}
            for i, l in enumerate(lambdas_to_train): buckets[gpus[i%len(gpus)]].append(l)
            procs = []
            for gid, lams in buckets.items():
                if not lams: continue
                p = mp.Process(target=_pareto_worker, args=(gid, lams, vars(args), train_df, valid_df, num_students, num_exercises, num_concepts, return_dict))
                p.start(); procs.append(p)
            for p in procs: p.join()
            for m in return_dict.values(): results.update(m)
        else:
            # Serial
            graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
            tl = DataLoader(CDDataset(train_df), args.batch_size, True, collate_fn=collate_fn)
            vl = DataLoader(CDDataset(valid_df), args.batch_size, False, collate_fn=collate_fn)
            for lam in lambdas_to_train:
                args.lambda_contrastive = float(lam)
                args.contrastive_decay_epochs = int(1e9) 
                args.contrastive_min_weight = 0.0
                results[float(lam)] = train_one(args, tl, vl, graphs, num_students, num_exercises, num_concepts, device)

    # Evaluate ALL models (cached + newly trained)
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    test_loader = DataLoader(CDDataset(test_df), args.batch_size, False, collate_fn=collate_fn)
    
    rows = []
    for lam in args.contrastive_lambdas:
        path = results.get(float(lam))
        if not path or not os.path.exists(path):
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
            "D_view_mean": d_mean,
            "model_path": path
        })
        
    df = pd.DataFrame(rows).sort_values("lambda_contrastive")
    df.to_csv(os.path.join(args.out_dir, "pareto.csv"), index=False)
    
    # -----------------------------
    # [优化后的绘图代码: Trade-off Trajectory]
    # -----------------------------
    plt.figure(figsize=(6, 5))
    
    # 1. 准备数据
    df_plot = df.sort_values("lambda_contrastive")
    x = df_plot["D_view_mean"]
    y = 1.0 - df_plot["test_auc"]  # Error Rate
    lams = df_plot["lambda_contrastive"]

    # 2. 画轨迹线
    plt.plot(x, y, linestyle='--', color='gray', alpha=0.6, label='Trade-off Trajectory')
    
    # 3. 画散点
    sc = plt.scatter(x, y, c=range(len(x)), cmap='viridis', s=80, zorder=5, edgecolors='k')
    
    # 4. 标注 Lambda 值
    for i, txt in enumerate(lams):
        plt.annotate(f"$\lambda={txt:.2g}$", (x.iloc[i], y.iloc[i]), 
                     xytext=(0, 8), textcoords='offset points', 
                     ha='center', fontsize=9, fontweight='bold')

    # 5. 标注坐标轴和标题
    plt.xlabel(r"View Distance $D_{view}$ ($\downarrow$ Better Consistency)", fontsize=11)
    plt.ylabel(r"Error Rate ($1 - AUC$) ($\downarrow$ Better Accuracy)", fontsize=11)
    plt.title("Consistency-Accuracy Trade-off", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pareto.png"), dpi=300)
    plt.close()
    
    return df


# -----------------------------
# [NEW] Exp-3D: Multi-level Sensitivity Analysis
# -----------------------------
def exp_multilevel_sensitivity(args, df_pareto, train_df, num_students, num_exercises, num_concepts, device):
    print("[Exp-3D] Running Multi-level Sensitivity Analysis...")
    
    # 1. Load Best Model (Lambda Star)
    lam_star = _select_lambda_star(df_pareto)
    try:
        pathS = df_pareto.loc[np.isclose(df_pareto["lambda_contrastive"], lam_star), "model_path"].iloc[0]
        print(f"  -> Using lambda_star={lam_star:.4f}, model={pathS}")
    except: 
        print("[Error] Model path for lambda_star not found.")
        return

    model = init_model(args, num_students, num_exercises, num_concepts, device)
    model.load_state_dict(torch.load(pathS, map_location=device))
    model.eval()

    # 2. Prepare Graphs
    graphs_clean = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    # Use a significant dropout to show the difference
    dropout_rate = 0.3
    graphs_noisy = tuple(sparse_edge_dropout(g, dropout_rate, seed=12345) for g in graphs_clean)

    with torch.no_grad():
        # Helper to get intermediate embeddings
        def get_intermediates(gs):
            adj_cse, adj_wse, adj_csc, adj_wsc = gs
            # 1. GCN Outputs
            s_c_se, _ = model.gcn_correct_se(adj_cse)
            s_w_se, _ = model.gcn_wrong_se(adj_wse)
            s_c_sc, _ = model.gcn_correct_sc(adj_csc)
            s_w_sc, _ = model.gcn_wrong_sc(adj_wsc)
            
            # 2. First Fusion (Intra-view)
            s_se, _ = model.fusion_se(s_c_se, s_w_se)
            s_sc, _ = model.fusion_sc(s_c_sc, s_w_sc)
            
            # 3. Final Gated Fusion
            s_final, _ = model.gated_fusion_student(s_se, s_sc)
            
            return s_se, s_sc, s_final

        # Get embeddings
        se_clean, sc_clean, final_clean = get_intermediates(graphs_clean)
        se_noisy, sc_noisy, final_noisy = get_intermediates(graphs_noisy)

        # Calculate Relative Shift for each component
        def calc_shift(clean, noisy):
            diff = clean - noisy
            dist = torch.norm(diff, p=2, dim=1)
            base = torch.norm(clean, p=2, dim=1) + 1e-9
            return (dist / base).mean().item()

        shift_se = calc_shift(se_clean, se_noisy)
        shift_sc = calc_shift(sc_clean, sc_noisy)
        shift_final = calc_shift(final_clean, final_noisy)

    # 3. Save & Plot
    results = [
        {"Component": "Concept View (SC)", "Shift": shift_sc, "Type": "Input View"},
        {"Component": "Exercise View (SE)", "Shift": shift_se, "Type": "Input View"},
        {"Component": "Fused Representation", "Shift": shift_final, "Type": "Output"},
    ]
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "multilevel_sensitivity.csv"), index=False)
    
    print("\n[Exp-3D Results] Relative Shift under Noise:")
    print(df)

    plt.figure(figsize=(6, 5))
    # Colors: Unstable (orange), Stable (blue), Final (green)
    colors = ['tab:orange', 'tab:blue', 'tab:green']
    bars = plt.bar(df["Component"], df["Shift"], color=colors, width=0.6)
    
    plt.ylabel(f"Relative Shift under Noise (Dropout={dropout_rate})")
    plt.title("Stability Filter Mechanism Analysis")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "multilevel_sensitivity.png"), dpi=220)
    plt.close()


# -----------------------------
# Main & Args
# -----------------------------
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default=os.path.join(root, "assist_09", "train.csv"))
    p.add_argument("--valid_file", type=str, default=os.path.join(root, "assist_09", "valid.csv"))
    p.add_argument("--test_file", type=str, default=os.path.join(root, "assist_09", "test.csv"))
    p.add_argument("--graph_dir", type=str, default=os.path.join(root, "graphs"))
    p.add_argument("--out_dir", type=str, default=os.path.join(root, "exp_m2_out"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    
    # Model Args
    p.add_argument("--model_path", type=str, default=os.path.join(root, "saved_models", "best_model.pth"))
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--fusion_type", type=str, default="enhanced_gated")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--use_supervised_contrastive", action="store_true", default=True)
    p.add_argument("--gated_num_gates", type=int, default=3)
    p.add_argument("--ortho_weight", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.3)
    
    # Training Args (RESTORED MISSING ARGS HERE)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=7)
    
    # Missing params that caused AttributeError
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
    p.add_argument("--skip_pareto_training", action="store_true")
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--pareto_serial", action="store_true")
    p.add_argument("--gain_lambda0", type=float, default=0.0) # Kept for compat
    p.add_argument("--gain_lambda_star", type=float, default=None)

    return p.parse_args()


def main():
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

    # 1. Exp-2A: Robustness Curve
    df_robust = exp_graph_robustness(
        args, args.model_path, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device
    )

    # 2. Exp-2B: Pareto (Finding Models with Caching)
    # 这步是必须的，因为它负责准备 model_lambda_*.pth 文件供 Exp-3D 使用
    df_pareto = exp_pareto_with_cache(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device)

    # 3. [NEW] Exp-3D: Multi-level Sensitivity
    exp_multilevel_sensitivity(args, df_pareto, train_df, num_students, num_exercises, num_concepts, device)

    # Summary
    summary = {
        "status": "Success", 
        "out_dir": args.out_dir,
        "experiments": ["Robustness Curve", "Pareto (Model Prep)", "Multilevel Sensitivity"]
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Module-2 experiments finished. Check outputs in: {args.out_dir}")

if __name__ == "__main__":
    try: mp.set_start_method("spawn", force=True)
    except: pass
    main()