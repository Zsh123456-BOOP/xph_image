#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module-3 Experiments (No item name/text required):
  Exp-3A: Q-noise robustness at inference time (corrupt cpt_seq list, keep labels unchanged)
          - Missing (drop true concepts)
          - False (add random concepts)
  Exp-3B: Sensitivity score per exercise (SDI-like): average |p(full)-p(mask-one)| across students/concepts
          -> correlate with concept-count
  Exp-3C: Co-attention heatmap for a selected multi-concept exercise
          (use diagnosis_head attention scores proxy: cpt_emb dot stu_emb)

Outputs (out_dir):
  - qnoise_curve.csv, qnoise_curve.png
  - sensitivity_scatter.png, sensitivity_table.csv
  - coattention_heatmap.png (for selected exercise)
"""

import os
import json
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def corrupt_cpt_list(cpts: List[int], mode: str, rho: float, num_concepts: int, rng: np.random.Generator) -> List[int]:
    cpts = list(dict.fromkeys(cpts))  # unique keep order
    if mode == "missing":
        if len(cpts) <= 1:
            return cpts
        keep = [c for c in cpts if rng.random() > rho]
        if len(keep) == 0:
            keep = [cpts[rng.integers(0, len(cpts))]]
        return keep
    elif mode == "false":
        # add random concepts not in list
        add_n = max(1, int(round(rho * max(1, len(cpts)))))
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
# Exp-3A: Q-noise robustness (inference-time)
# -----------------------------
def exp_qnoise_curve(args, model, graphs, test_df, device):
    rows = []
    for mode in ["missing", "false"]:
        for rho in args.qnoise_rates:
            df_noisy = apply_q_noise(test_df, mode=mode, rho=rho, num_concepts=args.num_concepts, seed=args.seed + int(rho * 1000) + (0 if mode == "missing" else 999))
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

    # plot
    plt.figure()
    for mode in ["missing", "false"]:
        sub = df[df["mode"] == mode].sort_values("rho")
        plt.plot(sub["rho"], sub["auc"], marker="o", label=mode)
    plt.xlabel("Q-noise rate rho")
    plt.ylabel("Test AUC")
    plt.title("Q-noise Robustness (inference-time cpt_seq corruption)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "qnoise_curve.png"), dpi=220)
    plt.close()

    return df


# -----------------------------
# Exp-3B: Sensitivity / SDI-like (mask-one concept)
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


def exp_sensitivity(args, model, graphs, test_df, device):
    rng = np.random.default_rng(args.seed)
    m = exercise_concept_map(test_df)

    # sample exercises with >=2 concepts
    multi = [q for q, cpts in m.items() if len(cpts) >= 2]
    rng.shuffle(multi)
    multi = multi[: min(args.max_exercises, len(multi))]

    stu_final, exer_final, cpt_final = precompute_embeddings(model, graphs)

    # sample interactions for each exercise
    sens_rows = []
    for q in multi:
        sub = test_df[test_df["exer_id"] == q]
        if len(sub) == 0:
            continue
        sub = sub.sample(n=min(args.max_users_per_exer, len(sub)), random_state=args.seed)

        cpts_full = m[q]
        cpts_full = cpts_full[: args.max_concepts_per_exer]

        deltas = []
        for _, r in sub.iterrows():
            u = int(r["stu_id"])
            # baseline
            stu_emb = stu_final[u].unsqueeze(0)
            exer_emb = exer_final[q].unsqueeze(0)
            cpt_embs = cpt_final[cpts_full].unsqueeze(0)
            mask = torch.ones(1, len(cpts_full), dtype=torch.bool, device=device)
            p_full = model.diagnosis_head(stu_emb, exer_emb, cpt_embs, mask).item()

            # mask-one
            for j in range(len(cpts_full)):
                cpts_mask = [c for idx, c in enumerate(cpts_full) if idx != j]
                if len(cpts_mask) == 0:
                    continue
                cpt_embs2 = cpt_final[cpts_mask].unsqueeze(0)
                mask2 = torch.ones(1, len(cpts_mask), dtype=torch.bool, device=device)
                p2 = model.diagnosis_head(stu_emb, exer_emb, cpt_embs2, mask2).item()
                deltas.append(abs(p_full - p2))

        if len(deltas) == 0:
            continue

        sens = float(np.mean(deltas))
        sens_rows.append({
            "exer_id": int(q),
            "concept_count": int(len(m[q])),
            "concept_count_used": int(len(cpts_full)),
            "sensitivity": sens,
            "difficulty": float(sub["label"].mean())  # empirical
        })

    df = pd.DataFrame(sens_rows)
    df.to_csv(os.path.join(args.out_dir, "sensitivity_table.csv"), index=False)

    # scatter
    plt.figure()
    plt.scatter(df["concept_count"], df["sensitivity"])
    plt.xlabel("Concept count |K_q|")
    plt.ylabel("Sensitivity (mean |p(full)-p(mask-one)| )")
    plt.title("Interaction Sensitivity vs Concept Count (SDI-like)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sensitivity_scatter.png"), dpi=220)
    plt.close()

    # spearman corr
    corr = float(pd.Series(df["concept_count"]).corr(pd.Series(df["sensitivity"]), method="spearman")) if len(df) > 2 else 0.0
    with open(os.path.join(args.out_dir, "sensitivity_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman(concept_count,sensitivity)": corr, "n_exercises": int(len(df))}, f, ensure_ascii=False, indent=2)

    return df


# -----------------------------
# Exp-3C: Co-attention heatmap for one exercise
# -----------------------------
@torch.no_grad()
def exp_coattention_heatmap(args, model, graphs, test_df, device):
    m = exercise_concept_map(test_df)
    # pick an exercise with >=3 concepts (more interpretable)
    cand = [q for q, cpts in m.items() if len(cpts) >= 3]
    if len(cand) == 0:
        return None
    q = args.focus_exercise if args.focus_exercise >= 0 else int(np.random.choice(cand))

    stu_final, exer_final, cpt_final = precompute_embeddings(model, graphs)

    sub = test_df[test_df["exer_id"] == q]
    sub = sub.sample(n=min(args.max_users_per_exer, len(sub)), random_state=args.seed)

    cpts = m[q][: args.max_concepts_per_exer]
    L = len(cpts)
    if L < 2:
        return None

    # compute average attention weights (proxy) from attention_scores = cpt_emb @ stu_emb
    # We reproduce diagnosis_head attention_scores logic:
    # attention_scores = cpt_embs @ stu_emb
    att_sum = np.zeros((L,), dtype=np.float64)

    for _, r in sub.iterrows():
        u = int(r["stu_id"])
        stu_emb = stu_final[u]  # [D]
        cpt_embs = cpt_final[cpts]  # [L,D]
        scores = torch.matmul(cpt_embs, stu_emb.unsqueeze(-1)).squeeze(-1)  # [L]
        w = torch.softmax(scores, dim=0).detach().cpu().numpy()
        att_sum += w

    att = att_sum / (len(sub) + 1e-12)

    # build a co-attention style matrix: outer(att, att)
    M = np.outer(att, att)

    plt.figure(figsize=(6, 5))
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.title(f"Co-attention Heatmap (exercise={q})")
    plt.xlabel("Concept index in cpt_seq")
    plt.ylabel("Concept index in cpt_seq")
    plt.xticks(range(L), [str(c) for c in cpts], rotation=45, fontsize=8)
    plt.yticks(range(L), [str(c) for c in cpts], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "coattention_heatmap.png"), dpi=220)
    plt.close()

    with open(os.path.join(args.out_dir, "coattention_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"focus_exercise": int(q), "concepts": [int(c) for c in cpts], "avg_attention": att.tolist()},
                  f, ensure_ascii=False, indent=2)
    return q


# -----------------------------
# Args / Main
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

    # Exp-3A
    df_curve = exp_qnoise_curve(args, model, graphs, test_df, device)

    # Exp-3B
    df_sens = exp_sensitivity(args, model, graphs, test_df, device)

    # Exp-3C
    focus = exp_coattention_heatmap(args, model, graphs, test_df, device)

    summary = {"out_dir": args.out_dir, "focus_exercise": focus, "qnoise_rows": int(len(df_curve)), "sens_rows": int(len(df_sens))}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Module-3 experiments finished:", args.out_dir)


if __name__ == "__main__":
    main()
