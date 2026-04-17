import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.analysis_utils import build_prepare_args, set_seeds
from main import prepare_data
from neuralcd_baseline import NeuralCDModel
from utils import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal NeuralCD baseline on xph_image raw data.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--graph_root", type=str, default="graphs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--prednet_len1", type=int, default=512)
    parser.add_argument("--prednet_len2", type=int, default=256)
    parser.add_argument("--discrimination_scale", type=float, default=10.0)
    parser.add_argument("--save_root", type=str, default="saved_models/neuralcd")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def build_checkpoint_path(save_root, dataset, tag=""):
    dataset_dir = Path(save_root) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if tag:
        return dataset_dir / f"best_model_{tag}.pth"
    return dataset_dir / "best_model.pth"


def evaluate_loader(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            stu_ids, exer_ids, _cpt_ids_padded, _cpt_mask, batch_labels = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            batch_labels = batch_labels.to(device)
            batch_preds = model(stu_ids, exer_ids)
            preds.extend(batch_preds.detach().cpu().numpy().tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())

    if not labels:
        return {"auc": 0.5, "acc": 0.0, "rmse": 0.0}
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=int)
    try:
        auc = float(roc_auc_score(labels, preds))
    except ValueError:
        auc = 0.5
    acc = float(accuracy_score(labels, preds >= 0.5))
    rmse = float(math.sqrt(mean_squared_error(labels, preds)))
    return {"auc": auc, "acc": acc, "rmse": rmse}


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    prepare_args = build_prepare_args(
        dataset=args.dataset,
        data_root=args.data_root,
        graph_root=args.graph_root,
        batch_size=args.batch_size,
    )
    data_bundle = prepare_data(prepare_args, device)
    train_loader, valid_loader, test_loader = data_bundle["loaders"]
    q_matrix = data_bundle["q_matrix"].to(device)

    model = NeuralCDModel(
        num_students=data_bundle["num_students"],
        num_exercises=data_bundle["num_exercises"],
        num_concepts=data_bundle["num_concepts"],
        q_matrix=q_matrix,
        dropout=args.dropout,
        prednet_len1=args.prednet_len1,
        prednet_len2=args.prednet_len2,
        discrimination_scale=args.discrimination_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.BCELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    checkpoint_path = build_checkpoint_path(args.save_root, args.dataset, tag=args.tag)

    print(f"Using device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            stu_ids, exer_ids, _cpt_ids_padded, _cpt_mask, labels = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(stu_ids, exer_ids)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())

        valid_metrics = evaluate_loader(model, valid_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | valid_auc={valid_metrics['auc']:.4f} | "
            f"valid_acc={valid_metrics['acc']:.4f} | valid_rmse={valid_metrics['rmse']:.4f}"
        )

        early_stopping(valid_metrics["auc"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(early_stopping.best_model_state, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate_loader(model, test_loader, device)
    print(
        f"Final test | auc={test_metrics['auc']:.4f} | "
        f"acc={test_metrics['acc']:.4f} | rmse={test_metrics['rmse']:.4f}"
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
