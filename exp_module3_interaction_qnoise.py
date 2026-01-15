#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-3 Experiments: 交互建模实验 (Interaction Modeling)
================================================================================

最优组合（审稿人友好闭环）：
- Exp-3A: Q-noise 鲁棒性（baseline + hard false）
- Exp-3B: mask-in-place 归因（logit attribution） + 分布图（violin + ridge）
- Exp-3C: 概念交互可视化（Synergy Syn(i,j) heatmap + network）
  * 若单题概念数太小，则自动切换为 global interaction 聚合图

要求：
- 不改模型结构，仅改实验脚本逻辑与可视化
- 若 diagnosis_head 对 mask 生效，则归因/交互均有效；
  若 mask 在 head 内被忽略，则 attribution 接近 0 —— 这是模型实现问题（自动跳过 3B/3C）

本版增强（不删单图，额外新增组合主图）：
- 保留全部单图：
  qnoise_curve.png, qnoise_hard_curve.png,
  attribution_violin.png, attribution_ridge.png,
  interaction_heatmap.png, interaction_network.png,
  interaction_heatmap_global.png
- 新增“组合主图”（更适合放主文）：
  fig_m3_qnoise_combo.png: (a) missing/false  (b) hard-false
  fig_m3_interaction_global_combo.png: (a) global heatmap (sym) (b) top edges bar (pos/neg)
- 统一四位小数显示与白底出版风格
"""

import os
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from torch.utils.data import DataLoader

from dataset import CDDataset, collate_fn
from model import CognitiveDiagnosisModel
from utils import build_graph, evaluate


# -----------------------------
# Plot style (white, publication)
# -----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08
    })


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def fmt4(x: float) -> str:
    return f"{float(x):.4f}"


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


def logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def write_full_excel(out_path: str, sheet_map: Dict[str, pd.DataFrame], meta: Dict):
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for name, df in sheet_map.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
        pd.DataFrame([meta]).to_excel(w, sheet_name="summary", index=False)


# -----------------------------
# Data mapping (mirror main.py)
# -----------------------------
def load_and_map_data(train_file: str, valid_file: str, test_file: str):
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df  = pd.read_csv(test_file)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    student_ids = sorted(all_df["stu_id"].unique())
    exercise_ids = sorted(all_df["exer_id"].unique())
    all_cpts = set()
    for cpt_seq in all_df["cpt_seq"]:
        all_cpts.update(int(c) for c in str(cpt_seq).split(","))
    concept_ids = sorted(all_cpts)

    student_id_map = {old: new for new, old in enumerate(student_ids)}
    exercise_id_map = {old: new for new, old in enumerate(exercise_ids)}
    concept_id_map  = {old: new for new, old in enumerate(concept_ids)}

    def map_concepts(cpt_seq):
        cpts = [concept_id_map[int(c)] for c in str(cpt_seq).split(",")]
        return ",".join(str(c) for c in cpts)

    for df in [train_df, valid_df, test_df]:
        df["stu_id"]  = df["stu_id"].map(student_id_map)
        df["exer_id"] = df["exer_id"].map(exercise_id_map)
        df["cpt_seq"] = df["cpt_seq"].apply(map_concepts)

    num_students = len(student_id_map)
    num_exercises = len(exercise_id_map)
    num_concepts = len(concept_id_map)
    return train_df, valid_df, test_df, num_students, num_exercises, num_concepts


def build_all_graphs(train_df, num_students, num_exercises, num_concepts, graph_dir, device):
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True,  item_type="exercise", graph_dir=graph_dir).to(device)
    adj_wrong_se   = build_graph(train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_correct_sc = build_graph(train_df, num_students, num_concepts,  correct=True,  item_type="concept",  graph_dir=graph_dir).to(device)
    adj_wrong_sc   = build_graph(train_df, num_students, num_concepts,  correct=False, item_type="concept",  graph_dir=graph_dir).to(device)
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
# Q-noise corruption (FIXED)
# -----------------------------
def corrupt_cpt_list(cpts: List[int], mode: str, rho: float, num_concepts: int, rng: np.random.Generator) -> List[int]:
    """
    修复点：
    - rho>0 时，false 至少添加 1 个概念（否则在 |Kq|=2 上 rho=0.1/0.2 基本不生效）
    - rho>0 时，missing 若随机未删除任何概念，则强制删除 1 个（确保扰动发生）
    """
    cpts = list(dict.fromkeys(cpts))  # unique keep order
    if rho <= 0:
        return cpts

    if mode == "missing":
        if len(cpts) <= 1:
            return cpts
        keep = [c for c in cpts if rng.random() > rho]
        # 若没删掉任何概念，强制删一个，避免扰动“空转”
        if len(keep) == len(cpts):
            drop_idx = int(rng.integers(0, len(cpts)))
            keep = [c for i, c in enumerate(cpts) if i != drop_idx]
        if len(keep) == 0:
            keep = [cpts[int(rng.integers(0, len(cpts)))]]
        return keep

    elif mode == "false":
        add_n = int(np.ceil(rho * max(1, len(cpts))))
        add_n = max(1, add_n)
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
    m2 = {q: corrupt_cpt_list(cpts, mode, rho, num_concepts, rng) for q, cpts in m.items()}

    df2 = df.copy()
    df2["cpt_seq"] = df2["exer_id"].apply(lambda q: ",".join(str(c) for c in m2[int(q)]))
    return df2


# -----------------------------
# Hard false noise
# -----------------------------
def build_hard_false_map(m: Dict[int, List[int]], cpt_emb: torch.Tensor, num_concepts: int, topk_pool: int = 50) -> Dict[int, List[int]]:
    cpt_emb = F.normalize(cpt_emb, dim=-1)
    hard_map = {}
    for q, cpts in m.items():
        cpts_u = list(dict.fromkeys(cpts))
        if len(cpts_u) == 0:
            hard_map[q] = []
            continue
        K = torch.tensor(cpts_u, dtype=torch.long, device=cpt_emb.device)
        center = torch.mean(cpt_emb[K], dim=0, keepdim=True)  # [1,D]
        sims = torch.matmul(cpt_emb, center.t()).squeeze(-1)  # [C]

        mask = torch.ones(num_concepts, dtype=torch.bool, device=cpt_emb.device)
        mask[K] = False
        sims = sims.masked_fill(~mask, -1e9)
        idx = torch.topk(sims, k=min(topk_pool, int(mask.sum().item()))).indices.detach().cpu().tolist()
        hard_map[q] = idx
    return hard_map


def apply_hard_false_noise(df: pd.DataFrame, rho: float, seed: int, hard_map: Dict[int, List[int]], base_map: Dict[int, List[int]]) -> pd.DataFrame:
    if rho <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    df2 = df.copy()

    def _corrupt(q: int) -> str:
        cpts = list(dict.fromkeys(base_map[int(q)]))
        add_n = int(np.ceil(rho * max(1, len(cpts))))
        add_n = max(1, add_n)
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
    stu_c_sc, cpt_c  = model.gcn_correct_sc(adj_correct_sc)
    stu_w_sc, cpt_w  = model.gcn_wrong_sc(adj_wrong_sc)

    stu_se, _ = model.fusion_se(stu_c_se, stu_w_se)
    stu_sc, _ = model.fusion_sc(stu_c_sc, stu_w_sc)
    stu_final, _  = model.gated_fusion_student(stu_se, stu_sc)
    exer_final, _ = model.gated_fusion_exercise(exer_c, exer_w)
    cpt_final, _  = model.gated_fusion_concept(cpt_c, cpt_w)
    return stu_final, exer_final, cpt_final


# -----------------------------
# Prediction helper
# -----------------------------
@torch.no_grad()
def predict_p(model, stu_final, exer_final, cpt_final, u: int, q: int, cpts: List[int], mask_bool: Optional[np.ndarray], device):
    L = len(cpts)
    stu_emb  = stu_final[u].unsqueeze(0)
    exer_emb = exer_final[q].unsqueeze(0)
    cpt_embs = cpt_final[cpts].unsqueeze(0)  # [1,L,D]
    if mask_bool is None:
        mask = torch.ones(1, L, dtype=torch.bool, device=device)
    else:
        mask = torch.tensor(mask_bool, dtype=torch.bool, device=device).view(1, L)
    p = model.diagnosis_head(stu_emb, exer_emb, cpt_embs, mask).item()
    return float(p)


def quick_mask_sanity_check(args, model, test_df, device, stu_final, exer_final, cpt_final) -> Dict:
    """
    判定是否“模型缺陷”：mask 是否在 diagnosis_head 内被忽略。
    返回中位数 attribution（Δlogit），若过小则认为缺陷。
    """
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)
    cand_q = [q for q, cpts in m.items() if len(list(dict.fromkeys(cpts))) >= 2]
    if len(cand_q) == 0:
        return {"ok": False, "reason": "no exercise with >=2 concepts"}

    deltas = []
    tries = 0
    while tries < args.sanity_trials and len(deltas) < args.sanity_keep:
        q = int(rng.choice(cand_q))
        cpts = list(dict.fromkeys(m[q]))[: min(6, len(m[q]))]
        L = len(cpts)
        sub = test_df[test_df["exer_id"] == q]
        if len(sub) == 0:
            tries += 1
            continue
        r = sub.sample(n=1, random_state=int(args.seed + tries)).iloc[0]
        u = int(r["stu_id"])
        p_full = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, np.ones((L,), dtype=bool), device)
        lf = logit(p_full)
        # mask one concept
        j = int(rng.integers(0, L))
        mask = np.ones((L,), dtype=bool)
        mask[j] = False
        p2 = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
        d = abs(lf - logit(p2))
        deltas.append(float(d))
        tries += 1

    if len(deltas) == 0:
        return {"ok": False, "reason": "sanity check failed (no samples)"}

    med = float(np.median(deltas))
    return {
        "ok": True,
        "median_abs_delta_logit": med,
        "mean_abs_delta_logit": float(np.mean(deltas)),
        "n": int(len(deltas)),
        "mask_effective": bool(med >= args.mask_effect_eps)
    }


# -----------------------------
# Exp-3A: Q-noise robustness (curve)
# -----------------------------
def exp_qnoise_curve(args, model, graphs, test_df, device) -> pd.DataFrame:
    rows = []
    for mode in ["missing", "false"]:
        for rho in args.qnoise_rates:
            df_noisy = apply_q_noise(
                test_df, mode=mode, rho=rho, num_concepts=args.num_concepts,
                seed=args.seed + int(rho * 1000) + (0 if mode == "missing" else 999),
            )
            loader = DataLoader(CDDataset(df_noisy), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            _, metrics, _ = evaluate(model, loader, device, *graphs)
            rows.append({"mode": mode, "rho": float(rho),
                         "auc": float(metrics["auc"]),
                         "accuracy": float(metrics["accuracy"]),
                         "rmse": float(metrics["rmse"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "qnoise_curve.csv"), index=False)

    # 单图（保留）
    plt.figure(figsize=(7.0, 4.8))
    ax = plt.gca()
    for mode in ["missing", "false"]:
        sub = df[df["mode"] == mode].sort_values("rho")
        ax.plot(sub["rho"], sub["auc"], marker="o", label=mode)
    ax.set_xlabel("Q-noise rate ρ")
    ax.set_ylabel("Test AUC")
    ax.set_title("Q-noise Robustness (inference-time cpt_seq corruption)")
    ax.grid(True, alpha=0.6)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ymin, ymax = df["auc"].min(), df["auc"].max()
    pad = max(1e-4, (ymax - ymin) * 0.25)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend()
    savefig(os.path.join(args.out_dir, "qnoise_curve.png"))
    return df


# -----------------------------
# Exp-3A++: Hard false-noise curve
# -----------------------------
def exp_qnoise_hard_false_curve(args, model, graphs, test_df, device, cpt_final) -> pd.DataFrame:
    base_map = exercise_concept_map(test_df)
    hard_map = build_hard_false_map(base_map, cpt_final, args.num_concepts, topk_pool=args.hard_pool)

    rows = []
    for rho in args.qnoise_rates:
        df_noisy = apply_hard_false_noise(
            test_df, rho=rho, seed=args.seed + int(rho * 1000) + 2027,
            hard_map=hard_map, base_map=base_map
        )
        loader = DataLoader(CDDataset(df_noisy), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        _, metrics, _ = evaluate(model, loader, device, *graphs)
        rows.append({"mode": "false_hard", "rho": float(rho),
                     "auc": float(metrics["auc"]),
                     "accuracy": float(metrics["accuracy"]),
                     "rmse": float(metrics["rmse"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "qnoise_hard_curve.csv"), index=False)

    # 单图（保留）
    plt.figure(figsize=(7.0, 4.8))
    ax = plt.gca()
    ax.plot(df["rho"], df["auc"], marker="o", label="false_hard")
    ax.set_xlabel("Q-noise rate ρ")
    ax.set_ylabel("Test AUC")
    ax.set_title("Hard Q-noise Robustness (semantic hard false concepts)")
    ax.grid(True, alpha=0.6)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ymin, ymax = df["auc"].min(), df["auc"].max()
    pad = max(1e-4, (ymax - ymin) * 0.25)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend()
    savefig(os.path.join(args.out_dir, "qnoise_hard_curve.png"))
    return df


# -----------------------------
# New: Combo figure for Exp-3A (main paper friendly)
# -----------------------------
def plot_qnoise_combo(args, df_q: pd.DataFrame, df_hard: pd.DataFrame):
    out_path = os.path.join(args.out_dir, "fig_m3_qnoise_combo.png")

    # Build common y-limits for comparability
    all_auc = np.concatenate([df_q["auc"].to_numpy(), df_hard["auc"].to_numpy()])
    ymin, ymax = float(all_auc.min()), float(all_auc.max())
    pad = max(1e-4, (ymax - ymin) * 0.25)
    ylo, yhi = ymin - pad, ymax + pad

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ax1, ax2 = axes

    # (a) missing/false
    for mode in ["missing", "false"]:
        sub = df_q[df_q["mode"] == mode].sort_values("rho")
        ax1.plot(sub["rho"], sub["auc"], marker="o", label=mode)
        # annotate endpoints
        if len(sub) > 0:
            a0 = float(sub.iloc[0]["auc"])
            aT = float(sub.iloc[-1]["auc"])
            ax1.annotate(fmt4(a0), (float(sub.iloc[0]["rho"]), a0), xytext=(0, 8),
                         textcoords='offset points', ha='center', fontsize=9)
            ax1.annotate(fmt4(aT), (float(sub.iloc[-1]["rho"]), aT), xytext=(0, 8),
                         textcoords='offset points', ha='center', fontsize=9)

    ax1.set_title("(a) Missing vs False")
    ax1.set_xlabel("ρ")
    ax1.set_ylabel("Test AUC")
    ax1.set_ylim(ylo, yhi)
    ax1.grid(True, alpha=0.6)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.legend()

    # (b) hard false
    sub = df_hard.sort_values("rho")
    ax2.plot(sub["rho"], sub["auc"], marker="o", label="false_hard")
    if len(sub) > 0:
        a0 = float(sub.iloc[0]["auc"])
        aT = float(sub.iloc[-1]["auc"])
        ax2.annotate(fmt4(a0), (float(sub.iloc[0]["rho"]), a0), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)
        ax2.annotate(fmt4(aT), (float(sub.iloc[-1]["rho"]), aT), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)

        # delta AUC at last point
        d = aT - a0
        ax2.text(0.02, 0.02, f"ΔAUC@ρ_max={fmt4(d)}", transform=ax2.transAxes, fontsize=9)

    ax2.set_title("(b) Hard False")
    ax2.set_xlabel("ρ")
    ax2.set_ylabel("Test AUC")
    ax2.set_ylim(ylo, yhi)
    ax2.grid(True, alpha=0.6)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.legend()

    plt.suptitle("Q-noise Robustness (Single Model, Inference-time cpt_seq Corruption)", y=1.02, fontsize=12)
    savefig(out_path)


# -----------------------------
# Exp-3B: Mask-in-place attribution
# -----------------------------
def exp_attribution_mask_in_place(args, model, test_df, device, stu_final, exer_final, cpt_final) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)

    # 分层抽样：按 concept_count 取题
    q_by_k = {}
    for q, cpts in m.items():
        k = len(list(dict.fromkeys(cpts)))
        if k < 2:
            continue
        q_by_k.setdefault(k, []).append(q)

    selected = []
    per_k = max(10, args.max_exercises // max(1, len(q_by_k)))
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
            p_full = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, np.ones((L,), dtype=bool), device)
            lf = logit(p_full)
            for j in range(L):
                mask = np.ones((L,), dtype=bool)
                mask[j] = False
                p2 = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
                deltas.append(abs(lf - logit(p2)))

        if len(deltas) == 0:
            continue

        rows.append({
            "exer_id": int(q),
            "concept_count": int(len(list(dict.fromkeys(m[q])))),
            "concept_count_used": int(L),
            "attr_logit_mean": float(np.mean(deltas)),
            "attr_logit_std": float(np.std(deltas)),
            "difficulty": float(sub["label"].mean()),
            "n_pairs": int(len(deltas)),
            "n_interactions": int(len(sub))
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "attribution_table.csv"), index=False)

    # 画图（violin + ridge）
    def bucket(k: int) -> str:
        if k <= 2: return "2"
        if k == 3: return "3"
        if k == 4: return "4"
        return "5+"

    df["bucket"] = df["concept_count"].apply(bucket)
    order = ["2", "3", "4", "5+"]

    data = []
    labels = []
    ns = []
    meds = []
    for b in order:
        vals = df[df["bucket"] == b]["attr_logit_mean"].to_numpy()
        if len(vals) > 0:
            data.append(vals)
            labels.append(b)
            ns.append(int(len(vals)))
            meds.append(float(np.median(vals)))

    # Violin (single figure) - keep
    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()
    ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(labels, ns)])
    ax.set_xlabel("|Kq| bucket")
    ax.set_ylabel("Attribution (mean |Δ logit|, mask-in-place)")
    ax.set_title("Concept Attribution vs Concept Count (Violin)")
    ax.grid(True, alpha=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    # annotate medians (4 decimals)
    for i, med in enumerate(meds, start=1):
        ax.text(i, med, f"med={fmt4(med)}", ha="center", va="bottom", fontsize=9)

    savefig(os.path.join(args.out_dir, "attribution_violin.png"))

    # Ridge-style (single figure) - keep
    plt.figure(figsize=(7.4, 5.2))
    ax = plt.gca()
    y0 = 0.0
    for b in order:
        vals = df[df["bucket"] == b]["attr_logit_mean"].to_numpy()
        if len(vals) < 8:
            continue
        hist, bin_edges = np.histogram(vals, bins=30, density=True)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        kernel = np.ones(5) / 5.0
        dens = np.convolve(hist, kernel, mode="same")
        y = dens / (dens.max() + 1e-12) * 0.9 + y0
        ax.plot(x, y, linewidth=2)
        ax.fill_between(x, y0, y, alpha=0.25)
        ax.text(float(x.min()), y0 + 0.05, f"|Kq|={b}", fontsize=9)
        y0 += 1.0
    ax.set_xlabel("Attribution (mean |Δ logit|)")
    ax.set_yticks([])
    ax.set_title("Attribution Distribution (Ridge-style, smoothed histogram density)")
    ax.grid(True, axis="x", alpha=0.35)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    savefig(os.path.join(args.out_dir, "attribution_ridge.png"))

    spearman = corr_spearman(df["concept_count"].to_numpy(), df["attr_logit_mean"].to_numpy()) if len(df) > 2 else 0.0
    with open(os.path.join(args.out_dir, "attribution_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman(concept_count, attr_logit_mean)": float(spearman),
                   "n_exercises": int(len(df))},
                  f, ensure_ascii=False, indent=2)

    return df


# -----------------------------
# Exp-3C: Interaction Synergy (single exercise OR global aggregate)
# -----------------------------
def choose_best_focus_exercise(test_df: pd.DataFrame) -> int:
    """
    选择 concept_count 最大，且交互次数最多的题，避免随机挑到 3 点玩具图。
    """
    m = exercise_concept_map(test_df)
    rows = []
    for q, cpts in m.items():
        k = len(list(dict.fromkeys(cpts)))
        n = int((test_df["exer_id"] == q).sum())
        rows.append((q, k, n))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return int(rows[0][0]) if len(rows) else -1


def _synergy_from_samples(args, model, sub: pd.DataFrame, device,
                          stu_final, exer_final, cpt_final,
                          q: int, cpts: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    返回：delta1[L], delta2[L,L], syn[L,L], cnt_users
    """
    L = len(cpts)
    delta1 = np.zeros((L,), dtype=float)
    delta2 = np.zeros((L, L), dtype=float)
    cnt = 0

    for _, r in sub.iterrows():
        u = int(r["stu_id"])
        p_full = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, np.ones((L,), dtype=bool), device)
        lf = logit(p_full)

        li = np.zeros((L,), dtype=float)
        for i in range(L):
            mask = np.ones((L,), dtype=bool); mask[i] = False
            p_i = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
            li[i] = (lf - logit(p_i))
        delta1 += li

        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                mask = np.ones((L,), dtype=bool); mask[i] = False; mask[j] = False
                p_ij = predict_p(model, stu_final, exer_final, cpt_final, u, q, cpts, mask, device)
                delta2[i, j] += (lf - logit(p_ij))

        cnt += 1

    if cnt > 0:
        delta1 /= cnt
        delta2 /= cnt

    syn = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            syn[i, j] = 0.0 if i == j else (delta2[i, j] - delta1[i] - delta1[j])

    return delta1, delta2, syn, cnt


def exp_interaction_synergy_single(args, model, test_df, device, stu_final, exer_final, cpt_final) -> Optional[int]:
    m = exercise_concept_map(test_df)

    q = args.focus_exercise if args.focus_exercise >= 0 else choose_best_focus_exercise(test_df)
    if q < 0 or q not in m:
        return None

    cpts = list(dict.fromkeys(m[q]))[: args.max_concepts_per_exer]
    L = len(cpts)
    if L < args.min_concepts_for_single:
        return None

    sub_all = test_df[test_df["exer_id"] == q]
    if len(sub_all) == 0:
        return None
    sub = sub_all.sample(n=min(args.max_users_per_exer, len(sub_all)), random_state=args.seed)

    delta1, delta2, syn, cnt = _synergy_from_samples(args, model, sub, device, stu_final, exer_final, cpt_final, int(q), cpts)
    if cnt == 0:
        return None

    idx = [f"cpt_{c}" for c in cpts]
    pd.DataFrame(syn, index=idx, columns=idx).to_csv(os.path.join(args.out_dir, "interaction_matrix.csv"))

    # Heatmap (single) - keep
    plt.figure(figsize=(7.0, 5.8))
    ax = plt.gca()
    im = ax.imshow(syn, aspect="auto", cmap=cm.coolwarm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Synergy Syn(i,j)")
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(c) for c in cpts], rotation=45, fontsize=9)
    ax.set_yticks(range(L))
    ax.set_yticklabels([str(c) for c in cpts], fontsize=9)
    ax.set_title(f"Interaction Heatmap via Synergy (exercise={q})")
    savefig(os.path.join(args.out_dir, "interaction_heatmap.png"))

    # Network (single) - keep (but this is appendix-friendly)
    plt.figure(figsize=(7.0, 7.0))
    ax = plt.gca()
    theta = np.linspace(0, 2*np.pi, L, endpoint=False)
    xs, ys = np.cos(theta), np.sin(theta)
    ax.scatter(xs, ys, s=260)
    for i in range(L):
        ax.text(xs[i]*1.10, ys[i]*1.10, str(cpts[i]), ha="center", va="center", fontsize=11)

    edges = []
    for i in range(L):
        for j in range(i+1, L):
            w = float(syn[i, j] + syn[j, i])  # 合并近似对称
            edges.append((i, j, w))
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    edges = edges[: min(args.max_edges, len(edges))]

    maxw = max([abs(w) for _, _, w in edges] + [1e-12])
    for i, j, w in edges:
        lw = 0.7 + 4.5 * abs(w) / maxw
        col = cm.coolwarm((w / (maxw + 1e-12) + 1) / 2)
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], linewidth=lw, color=col, alpha=0.85)

    ax.axis("off")
    ax.set_title(f"Interaction Network (Synergy) exercise={q}")
    savefig(os.path.join(args.out_dir, "interaction_network.png"))

    meta = {
        "mode": "single_exercise",
        "focus_exercise": int(q),
        "concepts": [int(c) for c in cpts],
        "n_users": int(cnt),
        "n_interactions_total": int(len(sub_all)),
        "concept_count_used": int(L),
        "delta1_mean_abs": float(np.mean(np.abs(delta1))),
        "syn_mean_abs": float(np.mean(np.abs(syn))),
    }
    with open(os.path.join(args.out_dir, "interaction_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return int(q)


def exp_interaction_synergy_global(args, model, test_df, device, stu_final, exer_final, cpt_final) -> bool:
    """
    Global 聚合：跨多题抽样，把 synergy 叠加到“概念对”层面，得到更像论文图的矩阵。
    """
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)

    # 选出 concept>=3 的题做抽样，否则 synergy 不成立
    cand = [q for q, cpts in m.items() if len(list(dict.fromkeys(cpts))) >= 3]
    if len(cand) == 0:
        return False

    # 统计概念频次，选 topM 概念（让图更集中可读）
    all_cpts = []
    for cpt_seq in test_df["cpt_seq"]:
        all_cpts.extend([int(x) for x in str(cpt_seq).split(",")])
    vc = pd.Series(all_cpts).value_counts()
    top_concepts = vc.head(args.global_top_concepts).index.tolist()
    top_set = set(int(x) for x in top_concepts)

    # global 累加器
    idx_map = {c: i for i, c in enumerate(top_concepts)}
    M = len(top_concepts)
    syn_sum = np.zeros((M, M), dtype=float)
    syn_cnt = np.zeros((M, M), dtype=float)

    rng.shuffle(cand)
    cand = cand[: min(args.global_max_exercises, len(cand))]

    used_exercises = 0
    used_pairs = 0

    for q in cand:
        cpts_raw = list(dict.fromkeys(m[int(q)]))
        cpts = [c for c in cpts_raw if c in top_set]
        cpts = cpts[: args.max_concepts_per_exer]
        L = len(cpts)
        if L < 3:
            continue

        sub_all = test_df[test_df["exer_id"] == q]
        if len(sub_all) == 0:
            continue
        sub = sub_all.sample(n=min(args.global_users_per_exer, len(sub_all)), random_state=args.seed + used_exercises)

        delta1, delta2, syn, cntu = _synergy_from_samples(args, model, sub, device, stu_final, exer_final, cpt_final, int(q), cpts)
        if cntu == 0:
            continue

        # 汇总到 global 概念对矩阵
        for i in range(L):
            for j in range(L):
                if i == j:
                    continue
                ci, cj = cpts[i], cpts[j]
                gi, gj = idx_map[ci], idx_map[cj]
                syn_sum[gi, gj] += float(syn[i, j])
                syn_cnt[gi, gj] += 1.0
                used_pairs += 1

        used_exercises += 1

    if used_exercises == 0:
        return False

    syn_avg = syn_sum / np.maximum(syn_cnt, 1.0)

    # 保存
    dfM = pd.DataFrame(
        syn_avg,
        index=[f"cpt_{c}" for c in top_concepts],
        columns=[f"cpt_{c}" for c in top_concepts]
    )
    dfM.to_csv(os.path.join(args.out_dir, "interaction_matrix_global.csv"))

    # 单图：global heatmap（保留）
    plt.figure(figsize=(9.0, 7.6))
    ax = plt.gca()
    im = ax.imshow(syn_avg, aspect="auto", cmap=cm.coolwarm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Synergy Syn(i,j) (global avg)")
    step = max(1, M // 15)
    ticks = list(range(0, M, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(top_concepts[t]) for t in ticks], rotation=45, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(top_concepts[t]) for t in ticks], fontsize=8)
    ax.set_title(f"Global Concept Interaction Heatmap (top{M}, sampled_exer={used_exercises})")
    savefig(os.path.join(args.out_dir, "interaction_heatmap_global.png"))

    meta = {
        "mode": "global_aggregate",
        "top_concepts": [int(c) for c in top_concepts],
        "sampled_exercises": int(used_exercises),
        "used_pairs": int(used_pairs),
        "avg_abs_syn": float(np.mean(np.abs(syn_avg[syn_cnt > 0]))),
    }
    with open(os.path.join(args.out_dir, "interaction_meta_global.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


# -----------------------------
# New: global combo fig (heatmap + top-edges bar)
# -----------------------------
def _top_edges_from_global(syn_avg: np.ndarray, syn_cnt: np.ndarray, top_concepts: List[int], topk_each: int = 10):
    """
    计算对称 synergy：
      S_sym = (S + S^T)/2
    返回 pos/neg top edges（去掉对角线，且必须出现过：syn_cnt>0）
    """
    M = syn_avg.shape[0]
    S = syn_avg.copy()
    C = syn_cnt.copy()
    # mask invalid
    valid = (C > 0)
    np.fill_diagonal(valid, False)

    Ssym = 0.5 * (S + S.T)
    # only keep upper triangle to avoid duplicates
    iu, ju = np.triu_indices(M, k=1)
    vals = []
    for i, j in zip(iu, ju):
        if valid[i, j] or valid[j, i]:
            vals.append((i, j, float(Ssym[i, j])))

    if len(vals) == 0:
        return Ssym, [], []

    vals_sorted = sorted(vals, key=lambda x: x[2], reverse=True)
    pos = [(i, j, v) for (i, j, v) in vals_sorted if v > 0][:topk_each]
    neg = [(i, j, v) for (i, j, v) in sorted(vals, key=lambda x: x[2]) if v < 0][:topk_each]
    return Ssym, pos, neg


def plot_interaction_global_combo(args):
    """
    依赖 interaction_matrix_global.csv 存在（由 exp_interaction_synergy_global 生成）
    输出：fig_m3_interaction_global_combo.png
    """
    csv_path = os.path.join(args.out_dir, "interaction_matrix_global.csv")
    if not os.path.exists(csv_path):
        return

    dfG = pd.read_csv(csv_path, index_col=0)
    syn_avg = dfG.values.astype(float)

    # 读取 cnt（若无则用 >0 mask 近似）
    cnt_path = os.path.join(args.out_dir, "interaction_meta_global.json")
    # 这里没有逐格 cnt 保存；我们用 syn_avg!=0 近似 valid（保守）
    syn_cnt = (np.abs(syn_avg) > 0).astype(float)

    # parse top_concepts from index names: "cpt_12"
    top_concepts = [int(s.replace("cpt_", "")) for s in dfG.index.tolist()]

    Ssym, pos, neg = _top_edges_from_global(syn_avg, syn_cnt, top_concepts, topk_each=args.global_top_edges_each)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    ax1, ax2 = axes

    # (a) sym heatmap
    im = ax1.imshow(Ssym, aspect="auto", cmap=cm.coolwarm)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("Synergy (sym avg)")
    step = max(1, len(top_concepts) // 12)
    ticks = list(range(0, len(top_concepts), step))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(top_concepts[t]) for t in ticks], rotation=45, fontsize=8)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([str(top_concepts[t]) for t in ticks], fontsize=8)
    ax1.set_title("(a) Global Interaction Heatmap (Sym)")

    # (b) top edges bar (pos+neg)
    bars = []
    labels = []

    # neg first (more negative at bottom), then pos
    for i, j, v in neg[::-1]:
        bars.append(v)
        labels.append(f"{top_concepts[i]}-{top_concepts[j]}")
    for i, j, v in pos:
        bars.append(v)
        labels.append(f"{top_concepts[i]}-{top_concepts[j]}")

    if len(bars) == 0:
        ax2.text(0.5, 0.5, "No valid edges for top-bar", ha="center", va="center")
        ax2.axis("off")
    else:
        y = np.arange(len(bars))
        ax2.barh(y, bars)
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.axvline(0.0, linewidth=1.0, color="#333333", alpha=0.8)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax2.set_title(f"(b) Top Synergy Edges (neg {len(neg)} + pos {len(pos)})")
        ax2.set_xlabel("Synergy value")

    plt.suptitle("Global Concept Interaction (Heatmap + Top Edges)", y=1.02, fontsize=12)
    savefig(os.path.join(args.out_dir, "fig_m3_interaction_global_combo.png"))


# -----------------------------
# Args / Main
# -----------------------------
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
    p.add_argument("--hard_pool", type=int, default=50)
    p.add_argument("--max_edges", type=int, default=24)

    # sanity / mask defect
    p.add_argument("--sanity_trials", type=int, default=200)
    p.add_argument("--sanity_keep", type=int, default=80)
    p.add_argument("--mask_effect_eps", type=float, default=1e-4)

    # interaction mode
    p.add_argument("--min_concepts_for_single", type=int, default=4,
                   help="单题交互图最低 concept 数；不足则转 global aggregate")

    # global aggregate params
    p.add_argument("--global_top_concepts", type=int, default=30)
    p.add_argument("--global_max_exercises", type=int, default=80)
    p.add_argument("--global_users_per_exer", type=int, default=64)

    # plotting: top edges each side for combo bar
    p.add_argument("--global_top_edges_each", type=int, default=10)

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
        args.out_dir = os.path.join(root, "exp_m3_out", dataset)
    if args.model_path is None:
        model_path_dataset = os.path.join(root, "saved_models", dataset, "best_model.pth")
        model_path_default = os.path.join(root, "saved_models", "best_model.pth")
        args.model_path = model_path_dataset if os.path.exists(model_path_dataset) else model_path_default

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"[ERR] model_path not found: {args.model_path}\n"
            f"请先运行: python main.py 训练生成模型，或手动指定 --model_path"
        )
    return args


def main():
    args = get_args()
    set_pub_style()
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

    # 预计算 embedding
    stu_final, exer_final, cpt_final = precompute_embeddings(model, graphs)

    # 先做 mask sanity：若模型缺陷则跳过 3B/3C
    sanity = quick_mask_sanity_check(args, model, test_df, device, stu_final, exer_final, cpt_final)
    mask_effective = bool(sanity.get("mask_effective", False))

    # Exp-3A：Q-noise curve（单图保留）
    df_curve = exp_qnoise_curve(args, model, graphs, test_df, device)

    # Exp-3A++：Hard false（单图保留）
    df_hard = exp_qnoise_hard_false_curve(args, model, graphs, test_df, device, cpt_final)

    # 新增：组合主图（不影响单图）
    try:
        plot_qnoise_combo(args, df_curve, df_hard)
    except Exception as e:
        print("[Warn] plot_qnoise_combo failed:", e)

    # 3B/3C：只有在 mask 生效时才生成（否则视为模型缺陷，自动跳过）
    df_attr = pd.DataFrame()
    focus_single = None
    global_ok = False
    if mask_effective:
        df_attr = exp_attribution_mask_in_place(args, model, test_df, device, stu_final, exer_final, cpt_final)

        focus_single = exp_interaction_synergy_single(args, model, test_df, device, stu_final, exer_final, cpt_final)
        if focus_single is None:
            global_ok = exp_interaction_synergy_global(args, model, test_df, device, stu_final, exer_final, cpt_final)
        else:
            # 即便 single 有，也仍可额外生成 global（如果你后期需要）
            # 这里保持默认不强制，避免额外算力；如需可手动把注释去掉
            # global_ok = exp_interaction_synergy_global(args, model, test_df, device, stu_final, exer_final, cpt_final)
            global_ok = os.path.exists(os.path.join(args.out_dir, "interaction_matrix_global.csv"))

        # 若 global csv 存在则生成 combo（heatmap+top edges bar）
        try:
            plot_interaction_global_combo(args)
        except Exception as e:
            print("[Warn] plot_interaction_global_combo failed:", e)

    # 汇总 + Excel
    summary = {
        "out_dir": args.out_dir,
        "num_concepts": int(args.num_concepts),
        "mask_sanity": sanity,
        "generated_attribution_and_interaction": bool(mask_effective),
        "focus_exercise_single": focus_single,
        "generated_global_interaction": bool(global_ok),
        "qnoise_rows": int(len(df_curve)),
        "hard_rows": int(len(df_hard)),
        "attr_rows": int(len(df_attr)) if mask_effective else 0,
        "note": "If mask_effective=False, Exp-3B/3C are skipped as model-head defect.",
        "figs": {
            "single_keep": [
                "qnoise_curve.png",
                "qnoise_hard_curve.png",
                "attribution_violin.png",
                "attribution_ridge.png",
                "interaction_heatmap.png",
                "interaction_network.png",
                "interaction_heatmap_global.png"
            ],
            "combo_added": [
                "fig_m3_qnoise_combo.png",
                "fig_m3_interaction_global_combo.png"
            ]
        }
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sheet_map = {
        "qnoise_curve": df_curve,
        "qnoise_hard_curve": df_hard,
    }
    if mask_effective and len(df_attr) > 0:
        sheet_map["attribution_table"] = df_attr

    # 如果 interaction csv 存在则也写入 excel（single/global）
    inter_single_csv = os.path.join(args.out_dir, "interaction_matrix.csv")
    inter_global_csv = os.path.join(args.out_dir, "interaction_matrix_global.csv")
    if os.path.exists(inter_single_csv):
        dfI = pd.read_csv(inter_single_csv, index_col=0).reset_index().rename(columns={"index": "row"})
        sheet_map["interaction_matrix_single"] = dfI
    if os.path.exists(inter_global_csv):
        dfG = pd.read_csv(inter_global_csv, index_col=0).reset_index().rename(columns={"index": "row"})
        sheet_map["interaction_matrix_global"] = dfG

    excel_path = os.path.join(args.out_dir, "module3_full.xlsx")
    write_full_excel(excel_path, sheet_map, summary)

    print("[OK] Module-3 experiments finished:", args.out_dir)
    print("[OK] Excel exported:", excel_path)
    if os.path.exists(os.path.join(args.out_dir, "fig_m3_qnoise_combo.png")):
        print("[OK] Combo fig (qnoise): fig_m3_qnoise_combo.png")
    if os.path.exists(os.path.join(args.out_dir, "fig_m3_interaction_global_combo.png")):
        print("[OK] Combo fig (interaction global): fig_m3_interaction_global_combo.png")


if __name__ == "__main__":
    main()
