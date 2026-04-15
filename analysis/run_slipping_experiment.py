import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.analysis_utils import (
    DEFAULT_MODEL_CONFIG,
    build_annotated_prediction_frame,
    set_seeds,
)
from analysis.slipping_utils import (
    build_flipped_labels,
    build_stress_subset_indices,
    evaluate_binary_predictions,
    select_flip_indices,
    select_reference_candidates,
    select_strong_positive_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run controlled slip simulation for xph_image Prism-CD.")
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
    parser.add_argument("--pred_threshold", type=float, default=0.8)
    parser.add_argument("--flip_ratios", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--eval_seeds", type=str, default="888,889,890")
    parser.add_argument("--candidate_file", type=str, default="")
    parser.add_argument("--stress_negative_multiplier", type=float, default=1.0)
    parser.add_argument("--item_drop_floor", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def parse_list(raw, cast_fn):
    values = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            values.append(cast_fn(text))
    return values


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
    ratios = parse_list(args.flip_ratios, float)
    eval_seeds = parse_list(args.eval_seeds, int)
    if not ratios:
        raise ValueError("No flip ratios provided.")
    if not eval_seeds:
        raise ValueError("No eval seeds provided.")

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

    if args.candidate_file:
        reference_df = pd.read_csv(args.candidate_file)
        candidate_mask = select_reference_candidates(annotated, reference_df)
    else:
        candidate_mask = select_strong_positive_candidates(
            annotated,
            hist_threshold=args.hist_threshold,
            min_concept_support=args.min_concept_support,
            pred_threshold=args.pred_threshold,
        )
    annotated["is_candidate"] = candidate_mask.astype(bool)

    candidate_indices = annotated.index[annotated["is_candidate"]].tolist()
    if not candidate_indices:
        raise RuntimeError("No strong positive candidates matched the current thresholds.")

    original_labels = annotated["label"].astype(int).to_numpy()
    predictions = annotated["p_pred"].astype(float).to_numpy()

    summary_rows = []
    sample_rows = []
    for ratio in ratios:
        for eval_seed in eval_seeds:
            flip_indices = select_flip_indices(candidate_indices, ratio=ratio, seed=eval_seed)
            flipped_labels = build_flipped_labels(original_labels, flip_indices)
            stress_indices = build_stress_subset_indices(
                original_labels,
                candidate_mask.to_numpy(dtype=bool),
                seed=eval_seed,
                negative_multiplier=args.stress_negative_multiplier,
            )

            full_original = evaluate_binary_predictions(original_labels, predictions)
            full_flipped = evaluate_binary_predictions(flipped_labels, predictions)
            if stress_indices:
                stress_original = evaluate_binary_predictions(original_labels[stress_indices], predictions[stress_indices])
                stress_flipped = evaluate_binary_predictions(flipped_labels[stress_indices], predictions[stress_indices])
            else:
                stress_original = full_original
                stress_flipped = full_flipped

            flipped_frame = annotated.loc[flip_indices].copy()
            summary_rows.append(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": args.seed,
                    "eval_seed": eval_seed,
                    "ratio": ratio,
                    "candidate_count": len(candidate_indices),
                    "flipped_count": len(flip_indices),
                    "original_auc": full_original["auc"],
                    "pseudo_auc": full_flipped["auc"],
                    "pseudo_auc_delta": full_flipped["auc"] - full_original["auc"],
                    "original_acc": full_original["acc"],
                    "pseudo_acc": full_flipped["acc"],
                    "pseudo_acc_delta": full_flipped["acc"] - full_original["acc"],
                    "original_rmse": full_original["rmse"],
                    "pseudo_rmse": full_flipped["rmse"],
                    "pseudo_rmse_delta": full_flipped["rmse"] - full_original["rmse"],
                    "stress_original_auc": stress_original["auc"],
                    "stress_auc": stress_flipped["auc"],
                    "stress_auc_delta": stress_flipped["auc"] - stress_original["auc"],
                    "stress_original_acc": stress_original["acc"],
                    "stress_acc": stress_flipped["acc"],
                    "stress_acc_delta": stress_flipped["acc"] - stress_original["acc"],
                    "stress_original_rmse": stress_original["rmse"],
                    "stress_rmse": stress_flipped["rmse"],
                    "stress_rmse_delta": stress_flipped["rmse"] - stress_original["rmse"],
                    "flipped_mean_p_pred": float(flipped_frame["p_pred"].mean()) if not flipped_frame.empty else np.nan,
                    "flipped_mean_concept_proxy_pred": float(flipped_frame["concept_proxy_pred"].mean()) if not flipped_frame.empty else np.nan,
                    "flipped_mean_hist_avg_rate": float(flipped_frame["hist_avg_rate"].mean()) if not flipped_frame.empty else np.nan,
                    "flipped_mean_stable_concept_drop_ratio": float(flipped_frame["stable_concept_drop_ratio"].mean()) if not flipped_frame.empty else np.nan,
                    "flipped_p75_decoupling_gap": float(flipped_frame["decoupling_gap"].quantile(0.75)) if not flipped_frame.empty else np.nan,
                    "flipped_p90_decoupling_gap": float(flipped_frame["decoupling_gap"].quantile(0.90)) if not flipped_frame.empty else np.nan,
                }
            )
            if not flipped_frame.empty:
                export_cols = [
                    "dataset",
                    "stu_id",
                    "exer_id",
                    "cpt_seq",
                    "label",
                    "hist_avg_rate",
                    "min_cpt_hist",
                    "concept_count",
                    "p_pred",
                    "concept_proxy_pred",
                    "item_drop",
                    "concept_drop",
                    "stable_concept_drop_ratio",
                    "decoupling_gap",
                ]
                export_frame = flipped_frame[export_cols].copy()
                export_frame["ratio"] = ratio
                export_frame["eval_seed"] = eval_seed
                export_frame["flipped_label"] = 0
                sample_rows.extend(export_frame.to_dict(orient="records"))

    output_dir = Path(args.output_dir) / "slipping"
    ensure_dir(output_dir)
    suffix = f"_{args.tag}" if args.tag else ""
    prefix = f"slipping_{args.dataset}_{args.split}_seed{args.seed}{suffix}"

    summary_path = output_dir / f"{prefix}_summary.csv"
    flipped_path = output_dir / f"{prefix}_flipped_samples.csv"
    candidate_path = output_dir / f"{prefix}_candidate_pool.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(sample_rows).to_csv(flipped_path, index=False)
    annotated.loc[annotated["is_candidate"]].to_csv(candidate_path, index=False)

    print(f"Saved slipping summary to {summary_path}")
    print(f"Saved flipped samples to {flipped_path}")
    print(f"Saved candidate pool to {candidate_path}")


if __name__ == "__main__":
    main()
