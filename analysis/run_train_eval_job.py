import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from main import prepare_data, test, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate one xph_image job.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fusion_type", type=str, default="enhanced_gated")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--gated_num_gates", type=int, default=3)
    parser.add_argument("--ortho_weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_fusion", type=float, default=0.7)
    parser.add_argument("--lambda_contrastive", type=float, default=0.4)
    parser.add_argument("--fusion_warmup_epochs", type=int, default=1)
    parser.add_argument("--contrastive_decay_epochs", type=int, default=18)
    parser.add_argument("--contrastive_min_weight", type=float, default=0.12)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "step", "plateau"])
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--T_0", type=int, default=8)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--use_supervised_contrastive", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.graph_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    data_bundle = prepare_data(args, device)
    best_model_path = train(args, data_bundle, device)
    test_metrics = test(args, data_bundle, device, best_model_path)

    payload = {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Dataset": args.dataset,
        "Tag": args.tag,
        "Checkpoint": str(best_model_path),
        "Test_AUC": float(test_metrics["auc"]),
        "Test_ACC": float(test_metrics["accuracy"]),
        "Test_RMSE": float(test_metrics["rmse"]),
        "Config": {
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "fusion_type": args.fusion_type,
            "temperature": args.temperature,
            "num_heads": args.num_heads,
            "gated_num_gates": args.gated_num_gates,
            "ortho_weight": args.ortho_weight,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_fusion": args.lambda_fusion,
            "lambda_contrastive": args.lambda_contrastive,
            "fusion_warmup_epochs": args.fusion_warmup_epochs,
            "contrastive_decay_epochs": args.contrastive_decay_epochs,
            "contrastive_min_weight": args.contrastive_min_weight,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "scheduler_type": args.scheduler_type,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "T_0": args.T_0,
            "T_mult": args.T_mult,
            "seed": args.seed,
        },
    }
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved job result to {args.output_json}")


if __name__ == "__main__":
    main()
