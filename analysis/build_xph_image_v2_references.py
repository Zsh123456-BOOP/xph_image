import argparse
import os
import sys
from pathlib import Path

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
from analysis.case_study_utils import select_conflict_cases
from analysis.slipping_utils import select_strong_positive_candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Build shared slipping/case-study references for xph_image v2.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--model_type", type=str, default="prism", choices=["prism", "neuralcd"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_root", type=str, default="saved_models")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--graph_root", type=str, default="graphs")
    parser.add_argument("--refs_dir", type=str, default="analysis_outputs/xph_image_refs_v2")
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
    parser.add_argument("--neuralcd_prednet_len1", type=int, default=512)
    parser.add_argument("--neuralcd_prednet_len2", type=int, default=256)
    parser.add_argument("--neuralcd_disc_scale", type=float, default=10.0)
    parser.add_argument("--hist_threshold", type=float, default=0.85)
    parser.add_argument("--min_concept_support", type=int, default=4)
    parser.add_argument("--pred_threshold", type=float, default=0.85)
    parser.add_argument("--max_item_pred", type=float, default=-1.0)
    parser.add_argument("--candidate_max_concepts", type=int, default=2)
    parser.add_argument("--require_all_mastery", action="store_true")
    parser.add_argument("--min_item_support", type=int, default=0)
    parser.add_argument("--min_item_acc", type=float, default=-1.0)
    parser.add_argument("--candidate_min_concept_proxy_pred", type=float, default=-1.0)
    parser.add_argument("--candidate_min_decoupling_gap", type=float, default=-1e9)
    parser.add_argument("--candidate_max_stable_ratio", type=float, default=-1.0)
    parser.add_argument("--case_max_concepts", type=int, default=2)
    parser.add_argument("--case_min_item_pred", type=float, default=0.75)
    parser.add_argument("--case_min_concept_proxy_pred", type=float, default=-1.0)
    parser.add_argument("--case_min_decoupling_gap", type=float, default=-1.0)
    parser.add_argument("--case_top_k", type=int, default=20)
    parser.add_argument("--item_drop_floor", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_model_config(args):
    if args.model_type == "neuralcd":
        return {
            "dropout": args.dropout,
            "prednet_len1": args.neuralcd_prednet_len1,
            "prednet_len2": args.neuralcd_prednet_len2,
            "discrimination_scale": args.neuralcd_disc_scale,
        }
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
        model_type=args.model_type,
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

    candidate_mask = select_strong_positive_candidates(
        annotated,
        hist_threshold=args.hist_threshold,
        min_concept_support=args.min_concept_support,
        pred_threshold=args.pred_threshold,
        max_item_pred=args.max_item_pred if args.max_item_pred >= 0 else None,
        max_concepts=args.candidate_max_concepts if args.candidate_max_concepts > 0 else None,
        require_all_mastery=args.require_all_mastery,
        min_item_support=args.min_item_support,
        min_item_acc=args.min_item_acc if args.min_item_acc >= 0 else None,
        min_concept_proxy_pred=(
            args.candidate_min_concept_proxy_pred
            if args.candidate_min_concept_proxy_pred >= 0
            else None
        ),
        min_decoupling_gap=(
            args.candidate_min_decoupling_gap
            if args.candidate_min_decoupling_gap > -1e8
            else None
        ),
        max_stable_concept_drop_ratio=(
            args.candidate_max_stable_ratio
            if args.candidate_max_stable_ratio >= 0
            else None
        ),
    )
    candidates = annotated.loc[candidate_mask].copy().reset_index(drop=True)

    cases = select_conflict_cases(
        annotated,
        hist_threshold=args.hist_threshold,
        min_concept_support=args.min_concept_support,
        max_concepts=args.case_max_concepts,
        min_item_pred=args.case_min_item_pred,
        min_concept_proxy_pred=args.case_min_concept_proxy_pred if args.case_min_concept_proxy_pred >= 0 else None,
        min_decoupling_gap=args.case_min_decoupling_gap if args.case_min_decoupling_gap >= 0 else None,
        min_item_support=args.min_item_support,
    )
    if args.case_top_k > 0:
        cases = cases.head(args.case_top_k).reset_index(drop=True)

    refs_dir = Path(args.refs_dir)
    ensure_dir(refs_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.dataset}_{args.split}_seed{args.seed}{suffix}"

    candidate_path = refs_dir / f"slipping_candidates_{stem}.csv"
    case_path = refs_dir / f"case_reference_{stem}.csv"
    summary_path = refs_dir / f"reference_summary_{stem}.csv"

    candidates.to_csv(candidate_path, index=False)
    cases.to_csv(case_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "split": args.split,
                "selector_model_type": args.model_type,
                "candidate_count": len(candidates),
                "case_count": len(cases),
                "hist_threshold": args.hist_threshold,
                "min_concept_support": args.min_concept_support,
                "pred_threshold": args.pred_threshold,
                "max_item_pred": None if args.max_item_pred < 0 else args.max_item_pred,
                "candidate_max_concepts": args.candidate_max_concepts,
                "require_all_mastery": bool(args.require_all_mastery),
                "min_item_support": args.min_item_support,
                "min_item_acc": None if args.min_item_acc < 0 else args.min_item_acc,
                "candidate_min_concept_proxy_pred": None
                if args.candidate_min_concept_proxy_pred < 0
                else args.candidate_min_concept_proxy_pred,
                "candidate_min_decoupling_gap": None
                if args.candidate_min_decoupling_gap <= -1e8
                else args.candidate_min_decoupling_gap,
                "candidate_max_stable_ratio": None
                if args.candidate_max_stable_ratio < 0
                else args.candidate_max_stable_ratio,
                "case_max_concepts": args.case_max_concepts,
                "case_min_item_pred": args.case_min_item_pred,
                "case_min_concept_proxy_pred": None
                if args.case_min_concept_proxy_pred < 0
                else args.case_min_concept_proxy_pred,
                "case_min_decoupling_gap": None
                if args.case_min_decoupling_gap < 0
                else args.case_min_decoupling_gap,
            }
        ]
    ).to_csv(summary_path, index=False)

    print(f"Saved slipping candidates to {candidate_path}")
    print(f"Saved case-study reference to {case_path}")
    print(f"Saved reference summary to {summary_path}")


if __name__ == "__main__":
    main()
