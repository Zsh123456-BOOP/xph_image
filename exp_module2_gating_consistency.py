#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-2 Experiments (No item name/text required):
  Exp-2A: Graph-edge dropout robustness + "reliance proxy" (cosine-based)
  Exp-2B: Consistency-Accuracy Pareto by scanning lambda_contrastive (train multiple runs)
  Exp-2C: "Information flow" by grouping items (concept-count or difficulty) -> stacked bar (proxy Sankey)

Outputs (out_dir):
  - robust_curve.csv, robust_curve.png
  - pareto.csv, pareto.png
  - flow_group.csv, flow_group.png
"""

import os
import json
import argparse
import random
from typing import Dict, Tuple, List

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
    p = np.clip(preds, eps, 1 - eps)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


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


# -----------------------------
# Exp-2A: Robustness to graph noise
# -----------------------------
def exp_graph_robustness(args, model_path: str, train_df, valid_df, test_df,
                         num_students, num_exercises, num_concepts, device):
    # build clean graphs
    graphs_clean = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    # load model
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    test_loader = DataLoader(CDDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    rows = []
    for lam in args.graph_drop_rates:
        # drop edges in all four graphs for evaluation
        graphs_noisy = tuple(
            sparse_edge_dropout(g, lam, seed=args.seed + int(lam * 1000)) for g in graphs_clean
        )
        _, metrics, _ = evaluate(model, test_loader, device, *graphs_noisy)

        # reliance proxy
        rel = estimate_reliance_proxy_student(model, graphs_noisy)

        rows.append({
            "drop_rate": lam,
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "rmse": metrics["rmse"],
            **rel
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "robust_curve.csv"), index=False)

    # plot
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
# Exp-2B: Pareto scan over lambda_contrastive (train multiple models)
# -----------------------------
def train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device) -> str:
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, verbose=False)

    best_path = os.path.join(args.out_dir, f"tmp_best_lambda_{args.lambda_contrastive:.4f}.pth")

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, device, *graphs, args, epoch, verbose=False)
        _, v_metrics, _ = evaluate(model, valid_loader, device, *graphs)
        stopper(v_metrics["auc"], model)
        if stopper.early_stop:
            break

    torch.save(stopper.best_model_state, best_path)
    return best_path


def exp_pareto(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device):
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)
    train_loader = DataLoader(CDDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(CDDataset(valid_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(CDDataset(test_df),  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    rows = []
    for lam in args.contrastive_lambdas:
        args.lambda_contrastive = float(lam)

        # make contrastive weight ~ constant (avoid decay effect) by setting huge decay epochs
        args.contrastive_decay_epochs = int(10**9)
        args.contrastive_min_weight = 0.0

        best_path = train_one(args, train_loader, valid_loader, graphs, num_students, num_exercises, num_concepts, device)

        model = init_model(args, num_students, num_exercises, num_concepts, device)
        sd = torch.load(best_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()

        _, t_metrics, _ = evaluate(model, test_loader, device, *graphs)
        d = compute_view_distances(model, graphs)

        # prediction entropy probe
        # reuse evaluate outputs by running once more to collect preds
        preds = []
        labs = []
        with torch.no_grad():
            for stu_ids, exer_ids, cpts_list, labels in test_loader:
                stu_ids = stu_ids.to(device)
                exer_ids = exer_ids.to(device)
                labels = labels.to(device)
                cpts_tensors = [torch.LongTensor(c).to(device) for c in cpts_list]
                p, _, _ = model(stu_ids, exer_ids, cpts_tensors, labels, *graphs)
                preds.append(p.detach().cpu().numpy())
                labs.append(labels.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)

        rows.append({
            "lambda_contrastive": float(lam),
            "test_auc": float(t_metrics["auc"]),
            "test_acc": float(t_metrics["accuracy"]),
            "D_view_mean": d["D_view_mean"],
            "D_exercise": d["D_exercise"],
            "D_concept": d["D_concept"],
            "pred_entropy": binary_entropy_from_preds(preds),
            "model_path": best_path
        })

    df = pd.DataFrame(rows).sort_values("lambda_contrastive")
    df.to_csv(os.path.join(args.out_dir, "pareto.csv"), index=False)

    # pareto plot: X=D_view_mean (lower is more consistent), Y=1-AUC (lower is better)
    plt.figure()
    plt.plot(df["D_view_mean"], 1.0 - df["test_auc"], marker="o")
    for _, r in df.iterrows():
        plt.text(r["D_view_mean"], 1.0 - r["test_auc"], f"{r['lambda_contrastive']:.2g}", fontsize=8)
    plt.xlabel("View distance D_view_mean (↓ better consistency)")
    plt.ylabel("1 - AUC (↓ better performance)")
    plt.title("Consistency–Accuracy Pareto (scan lambda_contrastive)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pareto.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Exp-2C: Information flow proxy (group items)
# -----------------------------
def exp_flow_group(args, model_path: str, train_df, valid_df, test_df,
                   num_students, num_exercises, num_concepts, device):
    graphs = build_all_graphs(train_df, num_students, num_exercises, num_concepts, args.graph_dir, device)

    # load model
    model = init_model(args, num_students, num_exercises, num_concepts, device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # build exercise meta: concept count + difficulty
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    exer_cpts = full_df.groupby("exer_id")["cpt_seq"].first().apply(lambda s: len(str(s).split(",")))
    exer_diff = full_df.groupby("exer_id")["label"].mean()  # higher -> easier

    # group by concept count bucket
    def bucket_c(x):
        if x <= 1:
            return "c=1"
        if x == 2:
            return "c=2"
        return "c>=3"

    meta = pd.DataFrame({
        "exer_id": exer_cpts.index.astype(int),
        "cpt_cnt": exer_cpts.values,
        "difficulty": exer_diff.reindex(exer_cpts.index).values
    })
    meta["bucket"] = meta["cpt_cnt"].apply(bucket_c)

    # compute student reliance proxy (global, not per-item); for item-level, we approximate by student embedding only
    # Instead, show "structure reliance" proxy via view distances: lower D_concept indicates stronger alignment under contrastive.
    d = compute_view_distances(model, graphs)

    rows = []
    for b, sub in meta.groupby("bucket"):
        rows.append({
            "bucket": b,
            "n_items": int(len(sub)),
            "avg_cpt_cnt": float(sub["cpt_cnt"].mean()),
            "avg_difficulty": float(sub["difficulty"].mean()),
            "D_exercise": d["D_exercise"],
            "D_concept": d["D_concept"],
            "D_view_mean": d["D_view_mean"],
        })

    df = pd.DataFrame(rows).sort_values("bucket")
    df.to_csv(os.path.join(args.out_dir, "flow_group.csv"), index=False)

    # stacked bar style proxy: show D_exercise vs D_concept per bucket
    plt.figure()
    x = np.arange(len(df))
    plt.bar(x, df["D_exercise"], label="Exercise view distance")
    plt.bar(x, df["D_concept"], bottom=df["D_exercise"], label="Concept view distance")
    plt.xticks(x, df["bucket"])
    plt.ylabel("Distance (stacked)")
    plt.title("Information Flow Proxy by Item Group (stacked view distances)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "flow_group.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Args
# -----------------------------
def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_args():
    import os
    import argparse

    root = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default=os.path.join(root, "assist_09", "train.csv"))
    p.add_argument("--valid_file", type=str, default=os.path.join(root, "assist_09", "valid.csv"))
    p.add_argument("--test_file",  type=str, default=os.path.join(root, "assist_09", "test.csv"))
    p.add_argument("--graph_dir",  type=str, default=os.path.join(root, "graphs"))
    p.add_argument("--out_dir",    type=str, default=os.path.join(root, "exp_m2_out"))

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
    df_robust = exp_graph_robustness(args, args.model_path, train_df, valid_df, test_df,
                                     num_students, num_exercises, num_concepts, device)

    # Exp-2B
    df_pareto = exp_pareto(args, train_df, valid_df, test_df, num_students, num_exercises, num_concepts, device)

    # Exp-2C
    df_flow = exp_flow_group(args, args.model_path, train_df, valid_df, test_df,
                             num_students, num_exercises, num_concepts, device)

    summary = {
        "robust_curve_rows": int(len(df_robust)),
        "pareto_rows": int(len(df_pareto)),
        "flow_rows": int(len(df_flow)),
        "out_dir": args.out_dir
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Module-2 experiments finished:", args.out_dir)


if __name__ == "__main__":
    main()
