from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import prepare_data
from model import CognitiveDiagnosisModel
from neuralcd_baseline import NeuralCDModel
from analysis.slipping_utils import (
    annotate_history_features,
    annotate_item_features,
    attach_concept_proxy_metrics,
    build_item_history,
    build_student_concept_history,
)
from analysis.slipping_utils import parse_cpt_seq


DEFAULT_MODEL_CONFIG = {
    "embedding_dim": 256,
    "num_layers": 3,
    "fusion_type": "enhanced_gated",
    "temperature": 0.1,
    "num_heads": 4,
    "use_supervised_contrastive": True,
    "gated_num_gates": 3,
    "ortho_weight": 0.5,
    "dropout": 0.3,
    "batch_size": 1024,
}

DEFAULT_NEURALCD_CONFIG = {
    "dropout": 0.5,
    "prednet_len1": 512,
    "prednet_len2": 256,
    "discrimination_scale": 10.0,
    "use_clipper": False,
}


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_type(model_type):
    alias_map = {
        "prism": "prism",
        "prism-cd": "prism",
        "cognitivediagnosismodel": "prism",
        "neuralcd": "neuralcd",
        "neural_cd": "neuralcd",
        "neural-cd": "neuralcd",
    }
    normalized = alias_map.get(str(model_type or "prism").strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return normalized


def resolve_checkpoint_path(dataset, checkpoint="", save_root="saved_models"):
    if checkpoint:
        return str(checkpoint)
    return str(Path(save_root) / dataset / "best_model.pth")


def build_prepare_args(dataset, data_root="data", graph_root="graphs", batch_size=1024):
    dataset_dir = Path(data_root) / dataset
    graph_dir = Path(graph_root) / dataset
    return Namespace(
        train_file=str(dataset_dir / "train.csv"),
        valid_file=str(dataset_dir / "valid.csv"),
        test_file=str(dataset_dir / "test.csv"),
        graph_dir=str(graph_dir),
        batch_size=int(batch_size),
    )


def prepare_analysis_bundle(dataset, device, data_root="data", graph_root="graphs", batch_size=1024):
    args = build_prepare_args(
        dataset=dataset,
        data_root=data_root,
        graph_root=graph_root,
        batch_size=batch_size,
    )
    return prepare_data(args, device)


def build_model(data_bundle, device, model_type="prism", model_config=None):
    model_type = normalize_model_type(model_type)
    if model_type == "neuralcd":
        cfg = dict(DEFAULT_NEURALCD_CONFIG)
        if model_config:
            cfg.update(model_config)
        q_matrix = data_bundle.get("q_matrix")
        if q_matrix is None:
            raise ValueError("NeuralCD analysis requires q_matrix in data_bundle.")
        if not torch.is_tensor(q_matrix):
            q_matrix = torch.as_tensor(q_matrix, dtype=torch.float32)
        model = NeuralCDModel(
            num_students=data_bundle["num_students"],
            num_exercises=data_bundle["num_exercises"],
            num_concepts=data_bundle["num_concepts"],
            q_matrix=q_matrix,
            dropout=float(cfg["dropout"]),
            prednet_len1=int(cfg["prednet_len1"]),
            prednet_len2=int(cfg["prednet_len2"]),
            discrimination_scale=float(cfg["discrimination_scale"]),
            use_clipper=bool(cfg["use_clipper"]),
        ).to(device)
        return model

    cfg = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        cfg.update(model_config)
    model = CognitiveDiagnosisModel(
        num_students=data_bundle["num_students"],
        num_exercises=data_bundle["num_exercises"],
        num_concepts=data_bundle["num_concepts"],
        embedding_dim=int(cfg["embedding_dim"]),
        num_layers=int(cfg["num_layers"]),
        concept_offset=int(data_bundle["concept_offset"]),
        fusion_type=str(cfg["fusion_type"]),
        temperature=float(cfg["temperature"]),
        num_heads=int(cfg["num_heads"]),
        use_supervised_contrastive=bool(cfg["use_supervised_contrastive"]),
        gated_num_gates=int(cfg["gated_num_gates"]),
        ortho_weight=float(cfg["ortho_weight"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    return model


def torch_load_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _is_state_dict(obj):
    if not isinstance(obj, dict):
        return False
    return all(torch.is_tensor(value) for value in obj.values())


def _extract_state_dict(checkpoint_obj):
    if _is_state_dict(checkpoint_obj):
        return checkpoint_obj
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint_obj and _is_state_dict(checkpoint_obj[key]):
                return checkpoint_obj[key]
    return None


def load_model_for_analysis(data_bundle, device, checkpoint_path, model_config=None, model_type="prism"):
    model_type = normalize_model_type(model_type)
    model = build_model(
        data_bundle,
        device,
        model_type=model_type,
        model_config=model_config,
    )
    checkpoint_obj = torch_load_compat(checkpoint_path, device)
    state_dict = _extract_state_dict(checkpoint_obj)
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    if len(missing) == len(model.state_dict()):
        raise RuntimeError(f"Checkpoint does not match model architecture: {checkpoint_path}")
    model.eval()
    return model


def _select_loader(data_bundle, split):
    split = str(split)
    if split == "train":
        return data_bundle["loaders"][0]
    if split == "valid":
        return data_bundle["loaders"][1]
    if split == "test":
        return data_bundle["loaders"][2]
    raise ValueError(f"Unsupported split: {split}")


def collect_split_predictions(model, data_bundle, split, device, model_type="prism"):
    model_type = normalize_model_type(model_type)
    loader = _select_loader(data_bundle, split)
    mapped_frame = data_bundle["mapped_frames"][split].reset_index(drop=True)
    mapped_concept_lists = [parse_cpt_seq(value) for value in mapped_frame["cpt_seq"]]

    all_predictions = []
    all_knowledge_states = []
    with torch.no_grad():
        for batch in loader:
            stu_ids, exer_ids, cpt_ids_padded, cpt_mask, labels = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            if model_type == "neuralcd":
                predictions = model(stu_ids, exer_ids)
                knowledge_states = model.get_knowledge_status(stu_ids)
            else:
                adj_correct_se, adj_wrong_se, adj_correct_sc, adj_wrong_sc = data_bundle["adj_graphs"]
                cpt_ids_padded = cpt_ids_padded.to(device)
                cpt_mask = cpt_mask.to(device)
                labels = labels.to(device)

                predictions, _aux_losses, knowledge_states = model(
                    stu_ids,
                    exer_ids,
                    cpt_ids_padded,
                    cpt_mask,
                    labels,
                    adj_correct_se,
                    adj_wrong_se,
                    adj_correct_sc,
                    adj_wrong_sc,
                )
            all_predictions.append(predictions.detach().cpu().numpy())
            all_knowledge_states.append(knowledge_states.detach().cpu().numpy())

    if all_predictions:
        predictions_np = np.concatenate(all_predictions, axis=0).astype(np.float32)
        knowledge_states_np = np.concatenate(all_knowledge_states, axis=0).astype(np.float32)
    else:
        predictions_np = np.array([], dtype=np.float32)
        knowledge_states_np = np.empty((0, data_bundle["num_concepts"]), dtype=np.float32)

    if len(mapped_frame) != len(predictions_np):
        raise RuntimeError(
            f"Split rows ({len(mapped_frame)}) do not match prediction rows ({len(predictions_np)})."
        )
    return predictions_np, knowledge_states_np, mapped_concept_lists


def compute_concept_proxies(knowledge_states, concept_id_lists):
    knowledge_states = np.asarray(knowledge_states, dtype=float)
    if knowledge_states.ndim != 2:
        raise ValueError("knowledge_states must be a 2D array.")
    if knowledge_states.shape[0] != len(concept_id_lists):
        raise ValueError("knowledge_states rows must match concept_id_lists length.")

    proxies = np.full(knowledge_states.shape[0], np.nan, dtype=float)
    for idx, concept_ids in enumerate(concept_id_lists):
        valid_ids = [int(c) for c in concept_ids if 0 <= int(c) < knowledge_states.shape[1]]
        if valid_ids:
            proxies[idx] = float(np.mean(knowledge_states[idx, valid_ids]))
    return proxies


def attach_predictions_to_raw_frame(raw_frame, predictions, knowledge_states, mapped_concept_lists):
    attached = raw_frame.reset_index(drop=True).copy()
    predictions = np.asarray(predictions, dtype=float)
    if len(attached) != len(predictions):
        raise ValueError("raw_frame row count does not match predictions length.")

    concept_proxies = compute_concept_proxies(knowledge_states, mapped_concept_lists)
    attached["concept_ids"] = list(mapped_concept_lists)
    attached["p_pred"] = predictions
    attached["concept_proxy_pred"] = concept_proxies
    return attached


def build_split_prediction_frame(model, data_bundle, split, device, model_type="prism"):
    predictions, knowledge_states, mapped_concept_lists = collect_split_predictions(
        model,
        data_bundle,
        split,
        device,
        model_type=model_type,
    )
    raw_frame = data_bundle["raw_frames"][split]
    attached = attach_predictions_to_raw_frame(
        raw_frame=raw_frame,
        predictions=predictions,
        knowledge_states=knowledge_states,
        mapped_concept_lists=mapped_concept_lists,
    )
    return attached, knowledge_states, mapped_concept_lists


def build_annotated_prediction_frame(
    dataset,
    split,
    device,
    model_type="prism",
    checkpoint="",
    save_root="saved_models",
    data_root="data",
    graph_root="graphs",
    batch_size=1024,
    model_config=None,
    item_drop_floor=0.05,
):
    model_type = normalize_model_type(model_type)
    data_bundle = prepare_analysis_bundle(
        dataset=dataset,
        device=device,
        data_root=data_root,
        graph_root=graph_root,
        batch_size=batch_size,
    )
    checkpoint_path = resolve_checkpoint_path(dataset, checkpoint=checkpoint, save_root=save_root)
    model = load_model_for_analysis(
        data_bundle=data_bundle,
        device=device,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        model_type=model_type,
    )
    attached, knowledge_states, mapped_concept_lists = build_split_prediction_frame(
        model=model,
        data_bundle=data_bundle,
        split=split,
        device=device,
        model_type=model_type,
    )
    history = build_student_concept_history(data_bundle["raw_frames"]["train"])
    annotated = annotate_history_features(attached, history)
    item_history = build_item_history(data_bundle["raw_frames"]["train"])
    annotated = annotate_item_features(annotated, item_history)
    annotated = attach_concept_proxy_metrics(
        annotated,
        concept_proxy_pred=annotated["concept_proxy_pred"].to_numpy(dtype=float),
        item_drop_floor=item_drop_floor,
        pred_col="p_pred",
    )
    return {
        "data_bundle": data_bundle,
        "model": model,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "knowledge_states": knowledge_states,
        "mapped_concept_lists": mapped_concept_lists,
        "annotated": annotated,
    }
