#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-1 Experiments (No item name/text required):
  Exp-1A: Independence (MI/NMI over knowledge_state dims)
  Exp-1B: Semantic Anchor Alignment (corr(knowledge_state[:,i], empirical_acc[:,k]))
  Exp-1C: Counterfactual traversal via gradient direction on stu_emb to increase skill-k in knowledge_diagnosis

Outputs (out_dir):
  - cmig_pairs.csv, mi_hist.png
  - alignment_heatmap.png, alignment_metrics.json
  - traversal_heatmap.png, traversal_metrics.json
"""

import os
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

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


def binary_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


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
    hxy, _, _ = np.histogram2d(x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]], density=False)
    pxy = hxy.astype(np.float64)
    pxy = pxy / (pxy.sum() + 1e-12)

    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    nz = pxy > 0
    mi = (pxy[nz] * (np.log(pxy[nz] + 1e-12) - np.log(px[nz.any(axis=1), :] + 1e-12).repeat(py.shape[1], axis=1)[nz]
                     - np.log(py[:, nz.any(axis=0)] + 1e-12).repeat(px.shape[0], axis=0)[nz])).sum()
    # 上面为了避免复杂索引写法显得怪；更稳妥写法如下（更慢但清晰）
    # 这里用清晰写法重算一次：
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                mi += pxy[i, j] * (math.log(pxy[i, j] + 1e-12) - math.log(px[i, 0] + 1e-12) - math.log(py[0, j] + 1e-12))
    return float(mi)


def cosine_reliance(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> Tuple[float, float]:
    # reliance proxy: normalized cos similarity
    ca = F.cosine_similarity(out, a, dim=-1).mean().item()
    cb = F.cosine_similarity(out, b, dim=-1).mean().item()
    s = ca + cb + 1e-12
    return ca / s, cb / s


# -----------------------------
# Data + Graphs (mirror main.prepare_data but return dfs)
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


def load_and_map_data(train_file: str, valid_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict, Dict]:
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    student_ids = sorted(all_df["stu_id"].unique())
    exercise_ids = sorted(all_df["exer_id"].unique())
    all_cpts = set()
    for cpt_seq in all_df["cpt_seq"]:
        if isinstance(cpt_seq, str):
            all_cpts.update(int(c) for c in cpt_seq.split(","))
        else:
            all_cpts.add(int(cpt_seq))
    concept_ids = sorted(all_cpts)

    student_id_map = {old: new for new, old in enumerate(student_ids)}
    exercise_id_map = {old: new for new, old in enumerate(exercise_ids)}
    concept_id_map = {old: new for new, old in enumerate(concept_ids)}

    def map_concepts(cpt_seq):
        if isinstance(cpt_seq, str):
            cpts = [concept_id_map[int(c)] for c in cpt_seq.split(",")]
            return ",".join(str(c) for c in cpts)
        return str(concept_id_map[int(cpt_seq)])

    for df in [train_df, valid_df, test_df]:
        df["stu_id"] = df["stu_id"].map(student_id_map)
        df["exer_id"] = df["exer_id"].map(exercise_id_map)
        df["cpt_seq"] = df["cpt_seq"].apply(map_concepts)

    return train_df, valid_df, test_df, student_id_map, exercise_id_map, concept_id_map


def build_all_graphs(train_df: pd.DataFrame, num_students: int, num_exercises: int, num_concepts: int,
                     graph_dir: str, device: torch.device):
    adj_correct_se = build_graph(train_df, num_students, num_exercises, correct=True, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_wrong_se   = build_graph(train_df, num_students, num_exercises, correct=False, item_type="exercise", graph_dir=graph_dir).to(device)
    adj_correct_sc = build_graph(train_df, num_students, num_concepts, correct=True, item_type="concept", graph_dir=graph_dir).to(device)
    adj_wrong_sc   = build_graph(train_df, num_students, num_concepts, correct=False, item_type="concept", graph_dir=graph_dir).to(device)
    return adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc


def make_bundle(args) -> DataBundle:
    train_df, valid_df, test_df, *_ = load_and_map_data(args.train_file, args.valid_file, args.test_file)

    num_students = int(pd.concat([train_df, valid_df, test_df])["stu_id"].nunique())
    num_exercises = int(pd.concat([train_df, valid_df, test_df])["exer_id"].nunique())

    # concept ids are already mapped to contiguous 0..K-1 in cpt_seq
    all_cpts = set()
    for cpt_seq in pd.concat([train_df, valid_df, test_df])["cpt_seq"]:
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

    for stu_ids, exer_ids, cpts_list, labels in loader:
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        labels = labels.to(device)
        cpts_tensors = [torch.LongTensor(c).to(device) for c in cpts_list]
        _, _, ks = model(stu_ids, exer_ids, cpts_tensors, labels, adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc)
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
    # precompute entropies
    H = np.array([hist_entropy_1d(z[:, k], bins=bins) for k in range(K)], dtype=np.float64)

    # sample pairs
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
        mi = hist_mi_2d(z[:, i], z[:, j], bins=bins)
        denom = math.sqrt(max(H[i] * H[j], 1e-12))
        nmi = mi / denom
        rows.append({"i": i, "j": j, "H_i": float(H[i]), "H_j": float(H[j]), "MI": float(mi), "NMI": float(nmi)})
        mi_vals.append(mi)
        nmi_vals.append(nmi)

    df_pairs = pd.DataFrame(rows)
    df_pairs.to_csv(os.path.join(out_dir, "cmig_pairs.csv"), index=False)

    plt.figure()
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
# Exp-1B: Semantic Anchor Alignment
# -----------------------------
def exp_anchor_alignment(z: np.ndarray, acc: np.ndarray, out_dir: str):
    # correlation matrix R[i,k]
    K = z.shape[1]
    R = np.zeros((K, K), dtype=np.float64)

    for i in range(K):
        for k in range(K):
            R[i, k] = spearman_corr(z[:, i], acc[:, k])

    diag = np.diag(R)
    off = R[~np.eye(K, dtype=bool)]

    ddi = float(np.mean(diag) - np.mean(off))
    top1 = float(np.mean(np.argmax(np.abs(R), axis=1) == np.arange(K)))

    # plot heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(R, aspect="auto")
    plt.colorbar()
    plt.title("Semantic Anchor Alignment: Spearman corr(z_i, acc_k)")
    plt.xlabel("Concept k (empirical accuracy)")
    plt.ylabel("Factor i (knowledge_state dim)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alignment_heatmap.png"), dpi=220)
    plt.close()

    metrics = {"DDI": ddi, "Top1_match_rate": top1}
    with open(os.path.join(out_dir, "alignment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


# -----------------------------
# Exp-1C: Counterfactual traversal (no direct z editing; use gradient dir on stu_emb)
# -----------------------------
def pick_median_student(train_df: pd.DataFrame) -> int:
    # median overall accuracy student
    g = train_df.groupby("stu_id")["label"].mean()
    u = int(g.sort_values().index[len(g) // 2])
    return u


def build_exer_concepts_map(df: pd.DataFrame) -> Dict[int, List[int]]:
    m = {}
    for exer_id, sub in df.groupby("exer_id"):
        # mapped ids -> same across rows, take first
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

    return stu_final, exer_final, cpt_final, (stu_se, stu_sc)


def exp_counterfactual_traversal(model: CognitiveDiagnosisModel, bundle: DataBundle, out_dir: str, device: torch.device,
                                 concept_k: int = None, student_u: int = None,
                                 steps: int = 21, n_items_each: int = 12):
    # setup
    model.eval()
    if student_u is None:
        student_u = pick_median_student(bundle.train_df)
    if concept_k is None:
        concept_k = int(np.random.randint(0, bundle.num_concepts))

    exer_map = build_exer_concepts_map(pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True))

    # select items A (contain k), C (not contain k), and B (contain k + others)
    contain_k = [q for q, cpts in exer_map.items() if concept_k in cpts]
    not_k = [q for q, cpts in exer_map.items() if concept_k not in cpts]
    both_k_and_other = [q for q, cpts in exer_map.items() if (concept_k in cpts and len(cpts) >= 2)]

    rng = np.random.default_rng(0)
    rng.shuffle(contain_k)
    rng.shuffle(not_k)
    rng.shuffle(both_k_and_other)

    A = contain_k[:n_items_each]
    B = both_k_and_other[:n_items_each]
    C = not_k[:n_items_each]
    items = A + B + C
    item_groups = (["A"] * len(A)) + (["B"] * len(B)) + (["C"] * len(C))

    # precompute global embeddings once
    stu_final, exer_final, cpt_final, (stu_se, stu_sc) = precompute_global_embeddings(model, bundle.graphs)

    # baseline student embedding and grad direction to increase knowledge_state[k]
    stu_emb = stu_final[student_u].detach().clone().requires_grad_(True)

    ks = model.knowledge_diagnosis(stu_emb.unsqueeze(0)).squeeze(0)  # [K]
    target = ks[concept_k]
    target.backward()
    grad = stu_emb.grad.detach()
    grad_norm = torch.norm(grad) + 1e-12
    direction = grad / grad_norm

    # define traversal scale based on projection stats (approx)
    # sample a few students to get std of projection
    with torch.no_grad():
        sample_ids = torch.randperm(bundle.num_students, device=device)[: min(512, bundle.num_students)]
        proj = torch.matmul(stu_final[sample_ids], direction)
        sigma = torch.std(proj).item() + 1e-6

    ts = np.linspace(-3.0, 3.0, steps).astype(np.float32)
    # heatmap: rows=items, cols=ts
    heat = np.zeros((len(items), steps), dtype=np.float32)

    for ti, t in enumerate(ts):
        stu_emb_t = (stu_final[student_u] + (t * sigma) * direction).unsqueeze(0)  # [1,D]
        for ri, q in enumerate(items):
            exer_emb = exer_final[q].unsqueeze(0)  # [1,D]
            cpts = exer_map[q]
            cpt_embs = cpt_final[cpts].unsqueeze(0)  # [1,L,D]
            # pad not required; diagnosis_head supports mask but expects padded; here single sample, no pad.
            # emulate same mask logic:
            mask = torch.ones(1, len(cpts), dtype=torch.bool, device=device)
            pred = model.diagnosis_head(stu_emb_t, exer_emb, cpt_embs, mask).item()
            heat[ri, ti] = pred

    # plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="P(correct)")
    plt.title(f"Counterfactual Traversal Heatmap (student={student_u}, concept_k={concept_k})")
    plt.xlabel("Traversal step (low -> high on skill-k direction)")
    plt.ylabel("Items (A: only k, B: k+others, C: not k)")
    plt.yticks(range(len(items)), [f"{item_groups[i]}:{items[i]}" for i in range(len(items))], fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traversal_heatmap.png"), dpi=220)
    plt.close()

    # metrics: monotonicity on A vs stability on C
    def monotonic_ratio(vals: np.ndarray) -> float:
        # fraction of positive finite diffs
        d = np.diff(vals)
        return float(np.mean(d > 0))

    A_rows = [i for i, g in enumerate(item_groups) if g == "A"]
    C_rows = [i for i, g in enumerate(item_groups) if g == "C"]

    A_mono = float(np.mean([monotonic_ratio(heat[i]) for i in A_rows])) if A_rows else 0.0
    C_var = float(np.mean([np.var(heat[i]) for i in C_rows])) if C_rows else 0.0

    metrics = {
        "student_u": int(student_u),
        "concept_k": int(concept_k),
        "steps": int(steps),
        "sigma_proj": float(sigma),
        "A_monotonicity_ratio_mean": A_mono,
        "C_prediction_variance_mean": C_var,
        "note": "A should be monotonic increasing; C should be stable (low variance)."
    }
    with open(os.path.join(out_dir, "traversal_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


# -----------------------------
# Main
# -----------------------------
def get_args():
    import os
    import argparse

    root = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default=os.path.join(root, "assist_09", "train.csv"))
    p.add_argument("--valid_file", type=str, default=os.path.join(root, "assist_09", "valid.csv"))
    p.add_argument("--test_file",  type=str, default=os.path.join(root, "assist_09", "test.csv"))
    p.add_argument("--graph_dir",  type=str, default=os.path.join(root, "graphs"))
    p.add_argument("--out_dir",    type=str, default=os.path.join(root, "exp_m1_out"))

    # 训练得到的checkpoint（main.py 会保存到 saved_models/best_model.pth）
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
    p.add_argument("--mi_pairs", type=int, default=512)
    p.add_argument("--mi_bins", type=int, default=20)
    p.add_argument("--traversal_steps", type=int, default=21)
    p.add_argument("--traversal_items_each", type=int, default=12)
    p.add_argument("--traversal_student", type=int, default=-1)
    p.add_argument("--traversal_concept", type=int, default=-1)

    args = p.parse_args()

    # 强制检查模型是否存在
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

    # collect z (knowledge_state) and empirical accuracy anchors
    full_df = pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True)
    full_loader = DataLoader(CDDataset(full_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    z = collect_student_knowledge_state(model, full_loader, bundle.graphs, device, bundle.num_students, bundle.num_concepts)
    acc = empirical_concept_accuracy(full_df, bundle.num_students, bundle.num_concepts)

    # Exp-1A
    mi_metrics = exp_mi_independence(z, args.out_dir, seed=args.seed, pairs=args.mi_pairs, bins=args.mi_bins)
    # Exp-1B
    align_metrics = exp_anchor_alignment(z, acc, args.out_dir)
    # Exp-1C
    student_u = None if args.traversal_student < 0 else args.traversal_student
    concept_k = None if args.traversal_concept < 0 else args.traversal_concept
    trav_metrics = exp_counterfactual_traversal(
        model, bundle, args.out_dir, device,
        concept_k=concept_k,
        student_u=student_u,
        steps=args.traversal_steps,
        n_items_each=args.traversal_items_each
    )

    summary = {"Exp-1A_MI": mi_metrics, "Exp-1B_Align": align_metrics, "Exp-1C_Traversal": trav_metrics}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] Module-1 experiments finished:", args.out_dir)


if __name__ == "__main__":
    main()
