#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-2 Experiments: 门控一致性实验 (Gating Consistency)
================================================================================

实验目的: 验证多视图图神经网络的门控融合机制是否有效，以及模型对图结构扰动的鲁棒性

实验列表:
--------------------------------------------------------------------------------
Exp-2A: 图鲁棒性 (Graph Robustness)
        测试模型对图边丢弃噪声的鲁棒性
        
        输出文件: robust_curve.png, robust_curve.csv
        
        如何判断结果好坏:
        ✓ 好: drop_rate=0.4 时 AUC 下降 < 2%
              曲线应该平缓下降，说明模型对图噪声鲁棒
        △ 中: drop_rate=0.4 时 AUC 下降 2-5%
              模型有一定鲁棒性但仍受图质量影响
        ✗ 差: drop_rate=0.4 时 AUC 下降 > 5%
              模型过度依赖图结构，缺乏鲁棒性
        
        图像解读: robust_curve.png 曲线越平缓越好
                  理想情况是接近水平线

--------------------------------------------------------------------------------
Exp-2B: 一致性-准确性 Pareto (Consistency-Accuracy Pareto)
        扫描 lambda_contrastive 参数，观察视图距离与性能的权衡
        
        输出文件: pareto.png, pareto.csv
        
        如何判断结果好坏（修订版，更贴近你当前模型的可解释口径）:
        ✓ 好: λ 增大能显著降低 D_view_mean（更一致），且 AUC 下降幅度很小（例如 < 0.5% 相对下降）
              说明对比一致性约束“确实在起作用”，并且不会明显伤害预测能力
        △ 中: D_view_mean 下降不明显，但 AUC 也基本不变
              说明一致性损失可能权重不足或被门控/主任务掩盖
        ✗ 差: D_view_mean 不降反升，或 AUC 大幅下降
              说明对比学习没有形成有效一致性约束，甚至破坏了表征
        
        图像解读: pareto.png 中
                  - X轴: D_view_mean（视图距离，越小表示两个视图越一致）
                  - Y轴: 1-AUC（误差，越小越好）
                  - 点图（scatter）更合理；并额外标出 Pareto front（非支配点下包络）
                  - λ=0（无对比学习）通常在右侧（更不一致），性能作为参照点

--------------------------------------------------------------------------------
Exp-2C: 信息流分组 (Information Flow Grouping)
        按概念数量分组分析“题目视图距离”和“相关概念视图距离”
        
        输出文件: flow_group.png, flow_group.csv
        
        如何判断结果好坏（修订：避免原实现“全局距离复用导致各组完全一样”的无效口径）:
        ✓ 好: 不同概念数量组的题目视图距离接近（或 c>=3 略高但可控）
              说明门控融合 + 一致性约束在不同复杂度题目上都能保持稳定对齐
        △ 中: c>=3 组略大于 c=1/c=2
              可接受，多概念题本身更难建模
        ✗ 差: 组间差异很大（尤其 c>=3 显著更高）
              说明复杂题的多视图一致性较差，门控融合未能平衡难度差异
        
        图像解读: flow_group.png 堆叠柱状图
                  - 底部：Exercise view distance（按 bucket 内题目求均值）
                  - 顶部：Concept view distance（按 bucket 涉及概念集合求均值）
                  - 若 c>=3 明显更高，可作为“复杂度导致一致性变差”的诊断信号

汇总输出: summary.json
"""

import os
import json
import argparse
import random
import multiprocessing as mp
from typing import Dict, Tuple, List, Set

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
    Drop a fraction of non-zero edges in a sparse adjacency matrix (no renorm; rescale kept edges).
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


def cosine_reliance(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> Tuple[float, float]:
    ca = F.cosine_similarity(out, a, dim=-1).mean().item()
    cb = F.cosine_similarity(out, b, dim=-1).mean().item()
    s = ca + cb + 1e-12
    return ca / s, cb / s


def binary_entropy_from_preds(preds: np.ndarray, eps: float = 1e-12) -> float:
    """
    preds: shape (N,) or (N,1) or (N,...) -> we flatten to 1-D.
    """
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _bucket_c(cnt: int) -> str:
    if cnt <= 1:
        return "c=1"
    if cnt == 2:
        return "c=2"
    return "c>=3"


def _pareto_front_indices(df: pd.DataFrame, x_col: str, y_col: str) -> List[int]:
    """
    Minimize both x and y. Return indices of non-dominated points (Pareto front).
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


# -----------------------------
# Data mapping (mirror main.py behavior)
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

    num_students = len(student_id_map)
    num_exercises = len(exercise_id_map)
    num_concepts = len(concept_id_map)
    return train_df, valid_df, test_df, num_students, num_exercises, num_concepts


def build_all_graphs(train_df, num_students, num_exercises, num_concepts, graph_dir, device):
    adj_correct_se = build_graph(
        train_df, num_students, num_exercises, correct=True, item_type="exercise", graph_dir=graph_dir
    ).to(device)
    adj_wrong_se = build_graph(
        train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir
    ).to(device)
    adj_correct_sc = build_graph(
        train_df, num_students, num_concepts, correct=True, item_type="concept", graph_dir=graph_dir
    ).to(device)
    adj_wrong_sc = build_graph(
        train_df, num_students, num_concepts, correct=False, item_type="concept", graph_dir=graph_dir
    ).to(device)
    return adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc


def warmup_graph_cache(train_df, num_students, num_exercises, num_concepts, graph_dir):
    """
    关键修复点：
    - 多进程并行训练时，若 build_graph() 会落盘缓存（同名文件），并发写容易卡住/死锁。
    - 这里在主进程先“预热构图并落盘”，子进程只做读取与搬运到各自 GPU。
    """
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


@torch.no_grad()
def compute_view_distances(model: CognitiveDiagnosisModel, graphs) -> Dict[str, float]:
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs

    # exercise views (correct vs wrong)
    _, exer_c = model.gcn_correct_se(adj_correct_se)
    _, exer_w = model.gcn_wrong_se(adj_wrong_se)
    d_exer = float((1.0 - F.cosine_similarity(exer_c, exer_w, dim=-1)).mean().item())

    # concept views (correct vs wrong)
    _, cpt_c = model.gcn_correct_sc(adj_correct_sc)
    _, cpt_w = model.gcn_wrong_sc(adj_wrong_sc)
    d_cpt = float((1.0 - F.cosine_similarity(cpt_c, cpt_w, dim=-1)).mean().item())

    return {"D_exercise": d_exer, "D_concept": d_cpt, "D_view_mean": float((d_exer + d_cpt) / 2.0)}


@torch.no_grad()
def estimate_reliance_proxy_student(model: CognitiveDiagnosisModel, graphs) -> Dict[str, float]:
    """
    Proxy for gating behavior (student view fusion se vs sc):
    reliance_se = cos(stu_final, stu_se) / (cos(stu_final, stu_se)+cos(stu_final, stu_sc))
    """
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs

    stu_c_se, _ = model.gcn_correct_se(adj_correct_se)
    stu_w_se, _ = model.gcn_wrong_se(adj_wrong_se)
    stu_c_sc, _ = model.gcn_correct_sc(adj_correct_sc)
    stu_w_sc, _ = model.gcn_wrong_sc(adj_wrong_sc)

    stu_se, _ = model.fusion_se(stu_c_se, stu_w_se)
    stu_sc, _ = model.fusion_sc(stu_c_sc, stu_w_sc)
    stu_final, _ = model.gated_fusion_student(stu_se, stu_sc)

    r_se, r_sc = cosine_reliance(stu_se, stu_sc, stu_final)
    return {"student_reliance_se": float(r_se), "student_reliance_sc": float(r_sc)}


@torch.no_grad()
def per_node_view_distances(model: CognitiveDiagnosisModel, graphs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      d_exer_each: shape [num_exercises], per-exercise view distance
      d_cpt_each : shape [num_concepts], per-concept  view distance
    """
    adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = graphs

    _, exer_c = model.gcn_correct_se(adj_correct_se)
    _, exer_w = model.gcn_wrong_se(adj_wrong_se)
    d_exer_each = (1.0 - F.cosine_similarity(exer_c, exer_w, dim=-1)).detach().cpu().numpy()

    _, cpt_c = model.gcn_correct_sc(adj_correct_sc)
    _, cpt_w = model.gcn_wrong_sc(adj_wrong_sc)
    d_cpt_each = (1.0 - F.cosine_similarity(cpt_c, cpt_w, dim=-1)).detach().cpu().numpy()

    return d_exer_each, d_cpt_each


# -----------------------------
# Exp-2A: Robustness to graph noise
# -----------------------------
def exp_graph_robustness(
    args, model_path: str, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device
):
    graphs_clean = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    model = init_model(args, num_students, num_exercises, num_concepts, device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    test_loader = DataLoader(
        CDDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )

    rows = []
    for lam in args.graph_drop_rates:
        graphs_noisy = tuple(sparse_edge_dropout(g, lam, seed=args.seed + int(lam * 1000)) for g in graphs_clean)
        _, metrics, _ = evaluate(model, test_loader, device, *graphs_noisy)
        rel = estimate_reliance_proxy_student(model, graphs_noisy)

        rows.append(
            {
                "drop_rate": lam,
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "rmse": metrics["rmse"],
                **rel,
            }
        )

    df = pd.DataFrame(rows).sort_values("drop_rate")
    df.to_csv(os.path.join(args.out_dir, "robust_curve.csv"), index=False)

    plt.figure()
    plt.plot(df["drop_rate"], df["auc"], marker="o")
    plt.xlabel("Graph edge dropout rate")
    plt.ylabel("Test AUC")
    plt.title("Robustness to Graph Noise (edge dropout)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "robust_curve.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Exp-2B: Pareto scan over lambda_contrastive
# -----------------------------
def train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device) -> str:
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, verbose=False)

    best_path = os.path.join(args.out_dir, f"model_lambda_{args.lambda_contrastive:.4f}.pth")

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, device, *graphs, args, epoch, verbose=False)
        _, v_metrics, _ = evaluate(model, valid_loader, device, *graphs)
        stopper(v_metrics["auc"], model)
        if stopper.early_stop:
            break

    torch.save(stopper.best_model_state, best_path)
    return best_path


def _pareto_worker(
    gpu_id: int,
    lambdas: List[float],
    args_dict: Dict,
    train_df,
    valid_df,
    num_students: int,
    num_exercises: int,
    num_concepts: int,
    return_dict,
):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    args = argparse.Namespace(**args_dict)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)

    set_seed(int(args.seed) + int(gpu_id) * 1000)

    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    train_loader = DataLoader(
        CDDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )
    valid_loader = DataLoader(
        CDDataset(valid_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )

    local_results = {}
    for lam in lambdas:
        args.lambda_contrastive = float(lam)
        args.contrastive_decay_epochs = int(10**9)
        args.contrastive_min_weight = 0.0

        best_path = train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device)
        local_results[float(lam)] = best_path

    return_dict[gpu_id] = local_results


def exp_pareto(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device):
    results = {}

    # 1) 跳过训练（用预训练）
    if getattr(args, "skip_pareto_training", False):
        print("[Exp-2B] 跳过训练，加载预训练模型...")
        for lam in args.contrastive_lambdas:
            model_path = os.path.join(args.out_dir, f"model_lambda_{lam:.4f}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"找不到预训练模型: {model_path}\n请先运行本脚本生成模型，或关闭 --skip_pareto_training"
                )
            results[float(lam)] = model_path
            print(f"  λ={lam:.4f} -> {model_path}")

    # 2) 训练（默认并行；可用 --pareto_serial 强制串行）
    else:
        use_cuda = torch.cuda.is_available()
        gpus = [int(x) for x in str(args.gpus).split(",") if str(x).strip() != ""]
        if not use_cuda:
            gpus = []

        do_parallel = use_cuda and (not args.pareto_serial) and (len(gpus) >= 1) and (len(args.contrastive_lambdas) > 1)

        if do_parallel:
            print(f"[Exp-2B] 并行训练 {len(args.contrastive_lambdas)} 个模型，GPU={gpus}（每 GPU 单进程，进程内串行跑多个 λ）")
            warmup_graph_cache(train_df, num_students, num_exercises, num_concepts, args.graph_dir)

            buckets = {gpu: [] for gpu in gpus}
            for i, lam in enumerate(args.contrastive_lambdas):
                buckets[gpus[i % len(gpus)]].append(float(lam))

            manager = mp.Manager()
            return_dict = manager.dict()
            args_dict = dict(vars(args))

            procs = []
            for gpu_id, lam_list in buckets.items():
                if len(lam_list) == 0:
                    continue
                p = mp.Process(
                    target=_pareto_worker,
                    args=(int(gpu_id), lam_list, args_dict, train_df, valid_df, num_students, num_exercises, num_concepts, return_dict),
                )
                p.daemon = False
                p.start()
                procs.append(p)

            for p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"[Exp-2B] 子进程异常退出（exitcode={p.exitcode}）。请检查日志输出。")

            for _, m in dict(return_dict).items():
                for lam, path in m.items():
                    results[float(lam)] = path

            missing = [float(l) for l in args.contrastive_lambdas if float(l) not in results]
            if len(missing) > 0:
                raise RuntimeError(f"[Exp-2B] 并行训练后仍缺少模型: {missing}")

        else:
            print(f"[Exp-2B] 串行训练 {len(args.contrastive_lambdas)} 个模型...")
            graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

            train_loader = DataLoader(
                CDDataset(train_df),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available() and device.type == "cuda",
            )
            valid_loader = DataLoader(
                CDDataset(valid_df),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available() and device.type == "cuda",
            )

            for i, lam in enumerate(args.contrastive_lambdas):
                print(f"[{i+1}/{len(args.contrastive_lambdas)}] 训练 λ={lam:.4f}...")
                args.lambda_contrastive = float(lam)
                args.contrastive_decay_epochs = int(10**9)
                args.contrastive_min_weight = 0.0

                best_path = train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device)
                results[float(lam)] = best_path
                print(f"[{i+1}/{len(args.contrastive_lambdas)}] λ={lam:.4f} 完成 -> {best_path}")

    # 3) 统一评估
    print("[Exp-2B] 开始评估所有模型...")

    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    test_loader = DataLoader(
        CDDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )

    rows = []
    for lam in args.contrastive_lambdas:
        lam = float(lam)
        best_path = results[lam]

        args.lambda_contrastive = float(lam)
        model = init_model(args, num_students, num_exercises, num_concepts, device)
        sd = torch.load(best_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()

        _, t_metrics, _ = evaluate(model, test_loader, device, *graphs)
        d = compute_view_distances(model, graphs)

        preds = []
        with torch.no_grad():
            for batch in test_loader:
                stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels = batch
                stu_ids = stu_ids.to(device)
                exer_ids = exer_ids.to(device)
                cpt_ids_padded = cpt_ids_padded.to(device)
                cpt_mask = cpt_mask.to(device)
                labels = labels.to(device)
                p, _, _ = model(stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels, *graphs)
                preds.append(p.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        pred_ent = binary_entropy_from_preds(preds)  # FIX: no NaN

        rows.append(
            {
                "lambda_contrastive": float(lam),
                "test_auc": float(t_metrics["auc"]),
                "test_acc": float(t_metrics["accuracy"]),
                "D_view_mean": d["D_view_mean"],
                "D_exercise": d["D_exercise"],
                "D_concept": d["D_concept"],
                "pred_entropy": pred_ent,
                "model_path": best_path,
            }
        )

    df = pd.DataFrame(rows).sort_values("lambda_contrastive")
    df.to_csv(os.path.join(args.out_dir, "pareto.csv"), index=False)

    # ---- plot (FIX): scatter + Pareto front
    df_plot = df.copy()
    df_plot["error"] = 1.0 - df_plot["test_auc"]

    front_idx = _pareto_front_indices(df_plot, x_col="D_view_mean", y_col="error")
    front = df_plot.loc[front_idx].sort_values("D_view_mean")

    plt.figure()
    plt.scatter(df_plot["D_view_mean"], df_plot["error"])
    for _, r in df_plot.iterrows():
        plt.text(r["D_view_mean"], r["error"], f"{r['lambda_contrastive']:.2g}", fontsize=8)

    # Pareto front line (only non-dominated points)
    if len(front) >= 2:
        plt.plot(front["D_view_mean"], front["error"])

    plt.xlabel("View distance D_view_mean (↓ better consistency)")
    plt.ylabel("1 - AUC (↓ better performance)")
    plt.title("Consistency–Accuracy Pareto (scan lambda_contrastive)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pareto.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Exp-2C: Information flow proxy (group items) - FIXED
# -----------------------------
def exp_flow_group(
    args, model_path: str, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device
):
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    model = init_model(args, num_students, num_exercises, num_concepts, device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # ---- build per-exercise metadata
    exer_to_cpts = full_df.groupby("exer_id")["cpt_seq"].first()
    exer_cpt_cnt = exer_to_cpts.apply(lambda s: len(str(s).split(",")))
    exer_diff = full_df.groupby("exer_id")["label"].mean()

    meta = pd.DataFrame(
        {
            "exer_id": exer_to_cpts.index.astype(int),
            "cpt_seq": exer_to_cpts.values,
            "cpt_cnt": exer_cpt_cnt.values,
            "difficulty": exer_diff.reindex(exer_to_cpts.index).values,
        }
    )
    meta["bucket"] = meta["cpt_cnt"].apply(lambda x: _bucket_c(int(x)))

    # ---- FIX: compute per-node view distances, then aggregate by bucket
    d_exer_each, d_cpt_each = per_node_view_distances(model, graphs)

    rows = []
    for b, sub in meta.groupby("bucket"):
        exer_ids = sub["exer_id"].astype(int).to_numpy()
        d_exer_bucket = float(np.mean(d_exer_each[exer_ids])) if len(exer_ids) > 0 else float("nan")

        # concepts involved in this bucket
        cpt_set: Set[int] = set()
        for s in sub["cpt_seq"].tolist():
            for c in str(s).split(","):
                cpt_set.add(int(c))
        cpt_ids = np.array(sorted(list(cpt_set)), dtype=np.int64)
        d_cpt_bucket = float(np.mean(d_cpt_each[cpt_ids])) if len(cpt_ids) > 0 else float("nan")

        rows.append(
            {
                "bucket": b,
                "n_items": int(len(sub)),
                "avg_cpt_cnt": float(sub["cpt_cnt"].mean()),
                "avg_difficulty": float(sub["difficulty"].mean()),
                "D_exercise_bucket": d_exer_bucket,
                "D_concept_bucket": d_cpt_bucket,
                "D_view_mean_bucket": float((d_exer_bucket + d_cpt_bucket) / 2.0),
                "n_unique_concepts_in_bucket": int(len(cpt_ids)),
            }
        )

    df = pd.DataFrame(rows).sort_values("bucket")
    df.to_csv(os.path.join(args.out_dir, "flow_group.csv"), index=False)

    # plot stacked bars (exercise + concept)
    plt.figure()
    x = np.arange(len(df))
    plt.bar(x, df["D_exercise_bucket"], label="Exercise view distance (bucket-mean)")
    plt.bar(x, df["D_concept_bucket"], bottom=df["D_exercise_bucket"], label="Concept view distance (bucket-mean)")
    plt.xticks(x, df["bucket"])
    plt.ylabel("Distance (stacked)")
    plt.title("Information Flow Grouping by Item Complexity (FIXED)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "flow_group.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Args
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

    # 用于鲁棒性/信息流的基准模型（同样来自 main.py 训练产物）
    p.add_argument("--model_path", type=str, default=os.path.join(root, "saved_models", "best_model.pth"))

    # base model/training args
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=7)

    # loss weights
    p.add_argument("--lambda_fusion", type=float, default=0.7)
    p.add_argument("--lambda_contrastive", type=float, default=0.4)
    p.add_argument("--fusion_warmup_epochs", type=int, default=1)
    p.add_argument("--contrastive_decay_epochs", type=int, default=18)
    p.add_argument("--contrastive_min_weight", type=float, default=0.12)
    p.add_argument("--grad_clip", type=float, default=0.1)

    # exp settings
    p.add_argument("--graph_drop_rates", type=str, default="0,0.1,0.2,0.3,0.4")
    p.add_argument("--contrastive_lambdas", type=str, default="0,0.05,0.1,0.2,0.4,0.8,1.2")
    p.add_argument(
        "--skip_pareto_training",
        action="store_true",
        help="跳过 Exp-2B Pareto 训练，直接使用 out_dir 下已存在的 model_lambda_*.pth",
    )

    # 并行控制
    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--pareto_serial", action="store_true")

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

    args.graph_drop_rates = parse_list_floats(args.graph_drop_rates)
    args.contrastive_lambdas = parse_list_floats(args.contrastive_lambdas)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df, valid_df, test_df, num_students, num_exercises, num_concepts = load_and_map_data(
        args.train_file, args.valid_file, args.test_file
    )

    # Exp-2A
    df_robust = exp_graph_robustness(
        args, args.model_path, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device
    )

    # Exp-2B
    df_pareto = exp_pareto(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device)

    # Exp-2C
    df_flow = exp_flow_group(
        args, args.model_path, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device
    )

    summary = {
        "robust_curve_rows": int(len(df_robust)),
        "pareto_rows": int(len(df_pareto)),
        "flow_rows": int(len(df_flow)),
        "out_dir": args.out_dir,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Module-2 experiments finished:", args.out_dir)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
