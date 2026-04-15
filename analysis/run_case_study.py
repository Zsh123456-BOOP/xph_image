import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.analysis_utils import (
    DEFAULT_MODEL_CONFIG,
    build_annotated_prediction_frame,
    set_seeds,
)
from analysis.case_study_utils import (
    align_cases_to_reference,
    select_conflict_cases,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run representative case-study extraction for xph_image Prism-CD.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_root", type=str, default="saved_models")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--graph_root", type=str, default="graphs")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/prism_xph_image_supp_20260415")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_MODEL_CONFIG["batch_size"])
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_MODEL_CONFIG["embedding_dim"])
    parser.add_argument("--num_layers", type=int, default=DEFAULT_MODEL_CONFIG["num_layers"])
    parser.add_argument("--fusion_type", type=str, default=DEFAULT_MODEL_CONFIG["fusion_type"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_MODEL_CONFIG["temperature"])
    parser.add_argument("--num_heads", type=int, default=DEFAULT_MODEL_CONFIG["num_heads"])
    parser.add_argument("--gated_num_gates", type=int, default=DEFAULT_MODEL_CONFIG["gated_num_gates"])
    parser.add_argument("--ortho_weight", type=float, default=DEFAULT_MODEL_CONFIG["ortho_weight"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_MODEL_CONFIG["dropout"])
    parser.add_argument("--hist_threshold", type=float, default=0.7)
    parser.add_argument("--min_concept_support", type=int, default=3)
    parser.add_argument("--max_concepts", type=int, default=2)
    parser.add_argument("--min_item_pred", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--case_file", type=str, default="")
    parser.add_argument("--item_drop_floor", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_model_config(args):
    return {
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "fusion_type": args.fusion_type,
        "temperature": args.temperature,
        "num_heads": args.num_heads,
        "gated_num_gates": args.gated_num_gates,
        "ortho_weight": args.ortho_weight,
        "dropout": args.dropout,
    }


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    analysis_state = build_annotated_prediction_frame(
        dataset=args.dataset,
        split=args.split,
        device=device,
        checkpoint=args.checkpoint,
        save_root=args.save_root,
        data_root=args.data_root,
        graph_root=args.graph_root,
        batch_size=args.batch_size,
        model_config=build_model_config(args),
        item_drop_floor=args.item_drop_floor,
    )
    annotated = analysis_state["annotated"].copy()
    annotated["dataset"] = args.dataset
    if args.case_file:
        reference_df = pd.read_csv(args.case_file)
        cases = align_cases_to_reference(annotated, reference_df)
    else:
        cases = select_conflict_cases(
            annotated,
            hist_threshold=args.hist_threshold,
            min_concept_support=args.min_concept_support,
            max_concepts=args.max_concepts,
            min_item_pred=args.min_item_pred,
        )
    if args.top_k > 0:
        cases = cases.head(args.top_k).reset_index(drop=True)
    if cases.empty:
        raise RuntimeError("No conflict cases found under current thresholds.")

    export_cols = [
        "dataset",
        "stu_id",
        "exer_id",
        "cpt_seq",
        "concept_count",
        "hist_avg_rate",
        "min_cpt_hist",
        "p_pred",
        "concept_proxy_pred",
        "item_drop",
        "concept_drop",
        "stable_concept_drop_ratio",
        "decoupling_gap",
    ]
    case_df = cases[export_cols].copy()
    case_df.insert(1, "case_rank", range(1, len(case_df) + 1))
    case_df = case_df.rename(columns={"p_pred": "item_p_pred"})
    case_df["concept_drop_ratio"] = case_df["concept_drop"] / case_df["item_drop"].clip(lower=1e-8)

    output_dir = Path(args.output_dir) / "case_study"
    ensure_dir(output_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    prefix = f"case_study_{args.dataset}_{args.split}_seed{args.seed}{suffix}"
    csv_path = output_dir / f"{prefix}.csv"
    case_df.to_csv(csv_path, index=False)

    print(f"Saved case-study table to {csv_path}")


if __name__ == "__main__":
    main()
