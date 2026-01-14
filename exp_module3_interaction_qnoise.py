#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-3 Experiments: 交互建模实验 (Interaction Modeling)
================================================================================

最优组合（审稿人友好闭环）：
- Exp-3A: Q-noise 鲁棒性（含 stratified 3D surface + hard false noise）
- Exp-3B: mask-in-place 归因（logit attribution） + 分布图（violin + ridge）
- Exp-3C: 真正的概念交互可视化（Synergy Syn(i,j) heatmap + network）

注意：
- 不改模型结构，仅改实验脚本逻辑与可视化
- 若 diagnosis_head 对 mask 生效，则本脚本的归因/交互均有效；
  若 mask 在 head 内被忽略，你会看到 attribution 接近 0 —— 这是模型实现问题，不是实验问题。
"""

import os
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from torch.utils.data import DataLoader

from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph, evaluate


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

def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))

def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    # 不依赖 scipy，使用 rank-corr
    if len(x) < 3:
        return 0.0
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


# -----------------------------
# Data mapping (mirror main.py)
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
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_wrong_se   = build_graph(train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_correct_sc = build_graph(train_df, num_students, num_concepts, correct=True, item_type="concept", graph_dir=graph_dir).to(device)
    adj_wrong_sc   = build_graph(train_df, num_students, num_concepts, correct=False, item_type="concept", graph_dir=graph_dir).to(device)
    return adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc


def init_model(args, num_students, num_exercises, num_concepts, device):
    model = CognitiveDiagnosisModel(
        num_students=num_students, num_exercises=num_exercises, num_concepts=num_concepts,
        embedding_dim=args.embedding_dim, num_layers=args.num_layers,
        fusion_type=args.fusion_type, temperature=args.temperature,
        num_heads=args.num_heads, use_supervised_contrastive=args.use_supervised_contrastive,
        gated_num_gates=args.gated_num_gates, ortho_weight=args.ortho_weight, dropout=args.dropout
    ).to(device)
    return model


def exercise_concept_map(df: pd.DataFrame) -> Dict[int, List[int]]:
    m = {}
    for exer_id, sub in df.groupby("exer_id"):
        cpts = [int(c) for c in str(sub.iloc[0]["cpt_seq"]).split(",")]
        m[int(exer_id)] = cpts
    return m


# -----------------------------
# Q-noise corruption
# -----------------------------
def corrupt_cpt_list(cpts: List[int], mode: str, rho: float, num_concepts: int, rng: np.random.Generator) -> List[int]:
    """
    修复点：
    - false 在 rho=0 时不应强行加错概念
    """
    cpts = list(dict.fromkeys(cpts))  # unique keep order

    if rho <= 0:
        return cpts

    if mode == "missing":
        if len(cpts) <= 1:
            return cpts
        keep = [c for c in cpts if rng.random() > rho]
        if len(keep) == 0:
            keep = [cpts[rng.integers(0, len(cpts))]]
        return keep

    elif mode == "false":
        add_n = int(round(rho * max(1, len(cpts))))  # 允许为 0
        if add_n <= 0:
            return cpts
        pool = [k for k in range(num_concepts) if k not in set(cpts)]
        if len(pool) == 0:
            return cpts
        add = rng.choice(pool, size=min(add_n, len(pool)), replace=False).tolist()
        return cpts + add

    else:
        raise ValueError("mode must be 'missing' or 'false'")


def apply_q_noise(df: pd.DataFrame, mode: str, rho: float, num_concepts: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = exercise_concept_map(df)

    m2 = {}
    for q, cpts in m.items():
        m2[q] = corrupt_cpt_list(cpts, mode, rho, num_concepts, rng)

    df2 = df.copy()
    df2["cpt_seq"] = df2["exer_id"].apply(lambda q: ",".join(str(c) for c in m2[int(q)]))
    return df2


# -----------------------------
# Hard false noise (semantic hard negatives)
# -----------------------------
def build_hard_false_map(
    m: Dict[int, List[int]],
    cpt_emb: torch.Tensor,
    num_concepts: int,
    topk_pool: int = 50,
) -> Dict[int, List[int]]:
    """
    对每个题目，从概念嵌入空间中选择 hard negatives（与真实概念最相似但不在集合内）
    返回：每题的 hard negative list（候选池）
    """
    cpt_emb = torch.nn.functional.normalize(cpt_emb, dim=-1)
    hard_map = {}

    # 预先计算概念两两相似度可能过大；这里按题目做局部计算
    for q, cpts in m.items():
        cpts_u = list(dict.fromkeys(cpts))
        if len(cpts_u) == 0:
            hard_map[q] = []
            continue
        K = torch.tensor(cpts_u, dtype=torch.long, device=cpt_emb.device)
        anchor = cpt_emb[K]  # [L,D]
        # 用均值向量表示该题概念“语义中心”
        center = torch.mean(anchor, dim=0, keepdim=True)  # [1,D]

        sims = torch.matmul(cpt_emb, center.t()).squeeze(-1)  # [C]
        # 排除真实概念
        mask = torch.ones(num_concepts, dtype=torch.bool, device=cpt_emb.device)
        mask[K] = False
        sims = sims.masked_fill(~mask, -1e9)

        idx = torch.topk(sims, k=min(topk_pool, int(mask.sum().item()))).indices.detach().cpu().tolist()
        hard_map[q] = idx
    return hard_map


def apply_hard_false_noise(
    df: pd.DataFrame,
    rho: float,
    seed: int,
    hard_map: Dict[int, List[int]],
    base_map: Dict[int, List[int]],
) -> pd.DataFrame:
    """
    在 false-noise 下，优先添加 hard negatives。
    add_n = round(rho*|Kq|)，rho=0 时返回原始
    """
    if rho <= 0:
        return df.copy()

    rng = np.random.default_rng(seed)
    df2 = df.copy()

    def _corrupt(q: int) -> str:
        cpts = list(dict.fromkeys(base_map[int(q)]))
        add_n = int(round(rho * max(1, len(cpts))))
        if add_n <= 0:
            return ",".join(str(c) for c in cpts)
        pool = hard_map.get(int(q), [])
        if len(pool) == 0:
            return ",".join(str(c) for c in cpts)
        add = rng.choice(pool, size=min(add_n, len(pool)), replace=False).tolist()
        return ",".join(str(c) for c in (cpts + add))

    df2["cpt_seq"] = df2["exer_id"].apply(_corrupt)
    return df2


# -----------------------------
# Precompute embeddings
# -----------------------------
@torch.no_grad()
def precompute_embeddings(model: CognitiveDiagnosisModel, graphs):
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


# -----------------------------
# Prediction helper (mask-in-place)
# -----------------------------
@torch.no_grad()
def predict_p(
    model: CognitiveDiagnosisModel,
    stu_final: torch.Tensor,
    exer_final: torch.Tensor,
    cpt_final: torch.Tensor,
    u: int,
    q: int,
    cpts: List[int],
    mask_bool: Optional[np.ndarray],
    device: torch.device,
) -> float:
    """
    固定长度输入：
    - cpt_embs shape: [1,L,D]
    - mask shape: [1,L]，True=keep, False=mask
    """
    L = len(cpts)
    stu_emb = stu_final[u].unsqueeze(0)      # [1,D]
    exer_emb = exer_final[q].unsqueeze(0)    # [1,D]
    cpt_embs = cpt_final[cpts].unsqueeze(0)  # [1,L,D]
    if mask_bool is None:
        mask = torch.ones(1, L, dtype=torch.bool, device=device)
    else:
        mask = torch.tensor(mask_bool, dtype=torch.bool, device=device).view(1, L)
    p = model.diagnosis_head(stu_emb, exer_emb, cpt_embs, mask).item()
    return float(p)


# -----------------------------
# Exp-3A: Q-noise robustness (baseline curve)
# -----------------------------
def exp_qnoise_curve(args, model, graphs, test_df, device):
    rows = []
    for mode in ["missing", "false"]:
        for rho in args.qnoise_rates:
            df_noisy = apply_q_noise(
                test_df,
                mode=mode,
                rho=rho,
                num_concepts=args.num_concepts,
                seed=args.seed + int(rho * 1000) + (0 if mode == "missing" else 999),
            )
            loader = DataLoader(CDDataset(df_noisy), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            _, metrics, _ = evaluate(model, loader, device, *graphs)
            rows.append({
                "mode": mode, "rho": float(rho),
                "auc": float(metrics["auc"]),
                "accuracy": float(metrics["accuracy"]),
                "rmse": float(metrics["rmse"])
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "qnoise_curve.csv"), index=False)

    plt.figure()
    for mode in ["missing", "false"]:
        sub = df[df["mode"] == mode].sort_values("rho")
        plt.plot(sub["rho"], sub["auc"], marker="o", label=mode)
    plt.xlabel("Q-noise rate rho")
    plt.ylabel("Test AUC")
    plt.title("Q-noise Robustness (inference-time cpt_seq corruption)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_curve.png"), dpi=240)
    plt.close()
    return df


# -----------------------------
# Exp-3A+: Stratified surface (rho x |Kq| -> delta AUC)
# -----------------------------
def exp_qnoise_stratified_surface(args, model, graphs, test_df, device):
    """
    产出：
      - qnoise_stratified.csv
      - qnoise_surface_3d.png
      - qnoise_surface_contour.png
      - qnoise_surface_heatmap.png
    """
    m = exercise_concept_map(test_df)
    # 题目 concept_count
    q2k = {q: len(list(dict.fromkeys(cpts))) for q, cpts in m.items()}

    # 分桶：2/3/4/5+
    def bucket(k: int) -> str:
        if k <= 2: return "2"
        if k == 3: return "3"
        if k == 4: return "4"
        return "5+"

    # 先算基线 AUC（rho=0）按桶
    base_rows = []
    for b in ["2", "3", "4", "5+"]:
        qs = [q for q, k in q2k.items() if bucket(k) == b]
        if len(qs) == 0:
            continue
        df_sub = test_df[test_df["exer_id"].isin(qs)].copy()
        loader = DataLoader(CDDataset(df_sub), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        _, metrics, _ = evaluate(model, loader, device, *graphs)
        base_rows.append({"bucket": b, "rho": 0.0, "mode": "base", "auc": float(metrics["auc"])})

    base_df = pd.DataFrame(base_rows)
    base_auc = {r["bucket"]: r["auc"] for r in base_rows}

    rows = []
    for mode in ["missing", "false"]:
        for rho in args.qnoise_rates:
            for b in ["2", "3", "4", "5+"]:
                qs = [q for q, k in q2k.items() if bucket(k) == b]
                if len(qs) == 0:
                    continue
                df_sub = test_df[test_df["exer_id"].isin(qs)].copy()
                df_noisy = apply_q_noise(
                    df_sub, mode=mode, rho=rho, num_concepts=args.num_concepts,
                    seed=args.seed + int(rho * 1000) + (0 if mode == "missing" else 999) + (123 if b == "3" else 0)
                )
                loader = DataLoader(CDDataset(df_noisy), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
                _, metrics, _ = evaluate(model, loader, device, *graphs)
                auc = float(metrics["auc"])
                delta = auc - float(base_auc.get(b, auc))
                rows.append({"mode": mode, "rho": float(rho), "bucket": b, "auc": auc, "delta_auc": float(delta)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "qnoise_stratified.csv"), index=False)

    # 为了画 surface：固定 mode= false（更苛刻）为主图；missing 可作为补充
    df_false = df[df["mode"] == "false"].copy()
    buckets = ["2", "3", "4", "5+"]
    rhos = sorted(df_false["rho"].unique().tolist())

    Z = np.zeros((len(buckets), len(rhos)), dtype=float)
    for i, b in enumerate(buckets):
        for j, rho in enumerate(rhos):
            sub = df_false[(df_false["bucket"] == b) & (df_false["rho"] == rho)]
            Z[i, j] = float(sub["delta_auc"].mean()) if len(sub) else 0.0

    X, Y = np.meshgrid(rhos, np.arange(len(buckets)))

    # 3D surface
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("rho")
    ax.set_ylabel("|Kq| bucket")
    ax.set_zlabel("ΔAUC (false - base)")
    ax.set_yticks(np.arange(len(buckets)))
    ax.set_yticklabels(buckets)
    ax.set_title("Stratified Q-noise Robustness Surface (false mode)")
    fig.colorbar(surf, shrink=0.6, pad=0.08)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_surface_3d.png"), dpi=260)
    plt.close()

    # contour
    plt.figure(figsize=(7.2, 4.8))
    cs = plt.contourf(X, Y, Z, levels=12, cmap=cm.viridis)
    plt.colorbar(cs, label="ΔAUC")
    plt.yticks(np.arange(len(buckets)), buckets)
    plt.xlabel("rho")
    plt.ylabel("|Kq| bucket")
    plt.title("Stratified Q-noise Robustness (false) - Contour")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_surface_contour.png"), dpi=260)
    plt.close()

    # heatmap (2D)
    plt.figure(figsize=(7.2, 4.6))
    plt.imshow(Z, aspect="auto", cmap=cm.viridis)
    plt.colorbar(label="ΔAUC")
    plt.xticks(np.arange(len(rhos)), [f"{r:.1f}" for r in rhos])
    plt.yticks(np.arange(len(buckets)), buckets)
    plt.xlabel("rho")
    plt.ylabel("|Kq| bucket")
    plt.title("Stratified Q-noise Robustness (false) - Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_surface_heatmap.png"), dpi=260)
    plt.close()

    return df


# -----------------------------
# Exp-3A++: Hard false-noise curve
# -----------------------------
def exp_qnoise_hard_false_curve(args, model, graphs, test_df, device, cpt_final: torch.Tensor):
    """
    产出：
      - qnoise_hard_curve.csv
      - qnoise_hard_curve.png
    """
    base_map = exercise_concept_map(test_df)
    hard_map = build_hard_false_map(base_map, cpt_final, args.num_concepts, topk_pool=args.hard_pool)

    rows = []
    # 只做 false-hard（missing-hard 没意义）
    for rho in args.qnoise_rates:
        df_noisy = apply_hard_false_noise(
            test_df, rho=rho,
            seed=args.seed + int(rho * 1000) + 2027,
            hard_map=hard_map,
            base_map=base_map,
        )
        loader = DataLoader(CDDataset(df_noisy), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        _, metrics, _ = evaluate(model, loader, device, *graphs)
        rows.append({"mode": "false_hard", "rho": float(rho), "auc": float(metrics["auc"]), "accuracy": float(metrics["accuracy"]), "rmse": float(metrics["rmse"])})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "qnoise_hard_curve.csv"), index=False)

    plt.figure()
    plt.plot(df["rho"], df["auc"], marker="o", label="false_hard")
    plt.xlabel("rho")
    plt.ylabel("Test AUC")
    plt.title("Hard Q-noise Robustness (semantic hard false concepts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_hard_curve.png"), dpi=240)
    plt.close()

    return df


# -----------------------------
# Exp-3B: Mask-in-place attribution (logit delta)
# -----------------------------
def exp_attribution_mask_in_place(args, model, graphs, test_df, device, stu_final, exer_final, cpt_final):
    """
    输出：
      - attribution_table.csv
      - attribution_violin.png
      - attribution_ridge.png
    """
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)

    # 分层抽样：避免几乎全是 |Kq|=2
    q_by_k = {}
    for q, cpts in m.items():
        k = len(list(dict.fromkeys(cpts)))
        if k < 2:
            continue
        q_by_k.setdefault(k, []).append(q)

    # 每个 k 等量抽
    selected = []
    per_k = max(5, args.max_exercises // max(1, len(q_by_k)))
    for k, qs in q_by_k.items():
        rng.shuffle(qs)
        selected.extend(qs[: min(per_k, len(qs))])
    rng.shuffle(selected)
    selected = selected[: min(args.max_exercises, len(selected))]

    rows = []
    for q in selected:
        sub = test_df[test_df["exer_id"] == q]
        if len(sub) == 0:
            continue
        sub = sub.sample(n=min(args.max_users_per_exer, len(sub)), random_state=args.seed)

        cpts = list(dict.fromkeys(m[q]))[: args.max_concepts_per_exer]
        L = len(cpts)
        if L < 2:
            continue

        deltas = []
        for _, r in sub.iterrows():
            u = int(r["stu_id"])

            mask_full = np.ones((L,), dtype=bool)
            p_full = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask_full, device)
            lf = logit(p_full)

            for j in range(L):
                mask = np.ones((L,), dtype=bool)
                mask[j] = False  # mask-in-place
                p2 = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
                l2 = logit(p2)
                deltas.append(abs(lf - l2))

        if len(deltas) == 0:
            continue

        rows.append({
            "exer_id": int(q),
            "concept_count": int(len(list(dict.fromkeys(m[q])))),
            "concept_count_used": int(L),
            "attr_logit_mean": float(np.mean(deltas)),
            "attr_logit_std": float(np.std(deltas)),
            "difficulty": float(sub["label"].mean()),
            "n_pairs": int(len(deltas))
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "attribution_table.csv"), index=False)

    # Violin by bucket
    def bucket(k: int) -> str:
        if k <= 2: return "2"
        if k == 3: return "3"
        if k == 4: return "4"
        return "5+"

    df["bucket"] = df["concept_count"].apply(bucket)
    order = ["2", "3", "4", "5+"]

    data = [df[df["bucket"] == b]["attr_logit_mean"].to_numpy() for b in order]
    data = [d for d in data if len(d) > 0]
    labels = [b for b in order if len(df[df["bucket"] == b]) > 0]

    plt.figure(figsize=(7.2, 4.8))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.xlabel("|Kq| bucket")
    plt.ylabel("Attribution (mean |Δ logit|, mask-in-place)")
    plt.title("Concept Attribution vs Concept Count (Violin)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attribution_violin.png"), dpi=260)
    plt.close()

    # Ridge-like plot（用多条密度曲线近似 ridge，不依赖 seaborn）
    plt.figure(figsize=(7.4, 5.2))
    y0 = 0.0
    for idx, b in enumerate(order):
        vals = df[df["bucket"] == b]["attr_logit_mean"].to_numpy()
        if len(vals) < 5:
            continue
        # 简单 KDE：用直方+平滑替代
        hist, bin_edges = np.histogram(vals, bins=30, density=True)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        # 平滑
        kernel = np.ones(5) / 5.0
        dens = np.convolve(hist, kernel, mode="same")
        y = dens / (dens.max() + 1e-12) * 0.9 + y0
        plt.plot(x, y, linewidth=2)
        plt.fill_between(x, y0, y, alpha=0.25)
        plt.text(x.min(), y0 + 0.05, f"|Kq|={b}", fontsize=9)
        y0 += 1.0

    plt.xlabel("Attribution (mean |Δ logit|)")
    plt.yticks([])
    plt.title("Attribution Distribution (Ridge-style)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attribution_ridge.png"), dpi=260)
    plt.close()

    # 统计写入 summary（可选）
    spearman = corr_spearman(df["concept_count"].to_numpy(), df["attr_logit_mean"].to_numpy()) if len(df) > 2 else 0.0
    with open(os.path.join(args.out_dir, "attribution_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman(concept_count, attr_logit_mean)": float(spearman), "n_exercises": int(len(df))},
                  f, ensure_ascii=False, indent=2)

    return df


# -----------------------------
# Exp-3C: True interaction map via Synergy Syn(i,j)
# -----------------------------
def exp_interaction_synergy(args, model, graphs, test_df, device, stu_final, exer_final, cpt_final):
    """
    修复后的 3C：
    - 不再构造 outer(att,att)
    - 直接计算二阶协同 Syn(i,j) = Δ(i,j) - Δ(i) - Δ(j)
    - 输出 interaction heatmap + circular network

    输出：
      - interaction_matrix.csv
      - interaction_heatmap.png
      - interaction_network.png
      - interaction_meta.json
    """
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)

    # 选一个概念数 >=3 的题（更可解释）
    cand = [q for q, cpts in m.items() if len(list(dict.fromkeys(cpts))) >= 3]
    if len(cand) == 0:
        return None

    q = args.focus_exercise if args.focus_exercise >= 0 else int(rng.choice(cand))
    cpts = list(dict.fromkeys(m[q]))[: args.max_concepts_per_exer]
    L = len(cpts)
    if L < 3:
        return None

    sub = test_df[test_df["exer_id"] == q]
    sub = sub.sample(n=min(args.max_users_per_exer, len(sub)), random_state=args.seed)

    # 计算平均 Δ(i), Δ(i,j)
    delta1 = np.zeros((L,), dtype=float)
    delta2 = np.zeros((L, L), dtype=float)
    cnt = 0

    for _, r in sub.iterrows():
        u = int(r["stu_id"])
        mask_full = np.ones((L,), dtype=bool)

        p_full = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask_full, device)
        lf = logit(p_full)

        # Δ(i)
        li = np.zeros((L,), dtype=float)
        for i in range(L):
            mask = np.ones((L,), dtype=bool)
            mask[i] = False
            p_i = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
            li[i] = (lf - logit(p_i))
        delta1 += li

        # Δ(i,j)
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                mask = np.ones((L,), dtype=bool)
                mask[i] = False
                mask[j] = False
                p_ij = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
                delta2[i, j] += (lf - logit(p_ij))

        cnt += 1

    if cnt == 0:
        return None

    delta1 /= cnt
    delta2 /= cnt

    # Syn(i,j)
    syn = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            if i == j:
                syn[i, j] = 0.0
            else:
                syn[i, j] = delta2[i, j] - delta1[i] - delta1[j]

    # 保存矩阵
    idx = [f"cpt_{c}" for c in cpts]
    dfM = pd.DataFrame(syn, index=idx, columns=idx)
    dfM.to_csv(os.path.join(args.out_dir, "interaction_matrix.csv"))

    # Heatmap
    plt.figure(figsize=(6.8, 5.6))
    im = plt.imshow(syn, aspect="auto", cmap=cm.coolwarm)
    plt.colorbar(im, label="Synergy Syn(i,j)")
    plt.xticks(range(L), [str(c) for c in cpts], rotation=45, fontsize=8)
    plt.yticks(range(L), [str(c) for c in cpts], fontsize=8)
    plt.title(f"Interaction Heatmap via Synergy (exercise={q})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "interaction_heatmap.png"), dpi=260)
    plt.close()

    # Circular network plot（不用 networkx）
    # 节点均匀分布在圆周上，边宽与 |syn| 成正比
    plt.figure(figsize=(7.0, 7.0))
    theta = np.linspace(0, 2 * np.pi, L, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    # 画节点
    plt.scatter(xs, ys, s=220)
    for i in range(L):
        plt.text(xs[i] * 1.08, ys[i] * 1.08, str(cpts[i]), ha="center", va="center", fontsize=10)

    # 画边（只画 strongest 的若干条，避免太乱）
    edges = []
    for i in range(L):
        for j in range(i + 1, L):
            w = syn[i, j] + syn[j, i]  # 近似对称合并
            edges.append((i, j, float(w)))
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    edges = edges[: min(args.max_edges, len(edges))]

    maxw = max([abs(w) for _, _, w in edges] + [1e-12])
    for i, j, w in edges:
        lw = 0.5 + 4.0 * abs(w) / maxw
        # 正协同 / 负协同 用颜色区分（coolwarm）
        col = cm.coolwarm((w / (maxw + 1e-12) + 1) / 2)
        plt.plot([xs[i], xs[j]], [ys[i], ys[j]], linewidth=lw, color=col, alpha=0.8)

    plt.axis("off")
    plt.title(f"Interaction Network (Synergy) exercise={q}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "interaction_network.png"), dpi=260)
    plt.close()

    meta = {
        "focus_exercise": int(q),
        "concepts": [int(c) for c in cpts],
        "n_users": int(cnt),
        "delta1_mean_abs": float(np.mean(np.abs(delta1))),
        "syn_mean_abs": float(np.mean(np.abs(syn))),
        "top_edges": [{"i": int(cpts[i]), "j": int(cpts[j]), "w": float(w)} for i, j, w in edges[:10]]
    }
    with open(os.path.join(args.out_dir, "interaction_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return q


# -----------------------------
# Args / Main
# -----------------------------
def get_args():
    root = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser()

    p.add_argument("--train_file", type=str, default=os.path.join(root, "assist_09", "train.csv"))
    p.add_argument("--valid_file", type=str, default=os.path.join(root, "assist_09", "valid.csv"))
    p.add_argument("--test_file",  type=str, default=os.path.join(root, "assist_09", "test.csv"))
    p.add_argument("--graph_dir",  type=str, default=os.path.join(root, "graphs"))
    p.add_argument("--out_dir",    type=str, default=os.path.join(root, "exp_m3_out"))

    p.add_argument("--model_path", type=str, default=os.path.join(root, "saved_models", "best_model.pth"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # model args
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

    # exp args
    p.add_argument("--qnoise_rates", type=str, default="0,0.1,0.2,0.3,0.4")
    p.add_argument("--max_exercises", type=int, default=200)
    p.add_argument("--max_users_per_exer", type=int, default=128)
    p.add_argument("--max_concepts_per_exer", type=int, default=6)
    p.add_argument("--focus_exercise", type=int, default=-1)

    # hard noise / network
    p.add_argument("--hard_pool", type=int, default=50, help="hard negative candidate pool size per exercise")
    p.add_argument("--max_edges", type=int, default=20, help="max edges drawn in interaction network")

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
    args.qnoise_rates = parse_list_floats(args.qnoise_rates)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df, valid_df, test_df, num_students, num_exercises, num_concepts = load_and_map_data(
        args.train_file, args.valid_file, args.test_file
    )
    args.num_concepts = num_concepts

    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    model = init_model(args, num_students, num_exercises, num_concepts, device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # 预计算 embedding（用于 hard noise / attribution / interaction）
    stu_final, exer_final, cpt_final = precompute_embeddings(model, graphs)

    # Exp-3A (baseline curve, fixed bug)
    df_curve = exp_qnoise_curve(args, model, graphs, test_df, device)

    # Exp-3A+ (stratified 3D surface)
    df_strat = exp_qnoise_stratified_surface(args, model, graphs, test_df, device)

    # Exp-3A++ (hard false noise)
    df_hard = exp_qnoise_hard_false_curve(args, model, graphs, test_df, device, cpt_final)

    # Exp-3B (mask-in-place attribution)
    df_attr = exp_attribution_mask_in_place(args, model, graphs, test_df, device, stu_final, exer_final, cpt_final)

    # Exp-3C (fixed: synergy interaction)
    focus = exp_interaction_synergy(args, model, graphs, test_df, device, stu_final, exer_final, cpt_final)

    summary = {
        "out_dir": args.out_dir,
        "focus_exercise": focus,
        "qnoise_rows": int(len(df_curve)),
        "strat_rows": int(len(df_strat)),
        "hard_rows": int(len(df_hard)),
        "attr_rows": int(len(df_attr)),
        "note": "Exp-3C uses synergy Syn(i,j) instead of outer(att,att)."
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Module-3 experiments finished:", args.out_dir)


if __name__ == "__main__":
    main()
