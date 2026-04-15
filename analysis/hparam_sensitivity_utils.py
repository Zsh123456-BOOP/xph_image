from __future__ import annotations

from copy import deepcopy

import pandas as pd


PRISM_BASE_CONFIGS = {
    "assist_09": {
        "batch_size": 1024,
        "epochs": 100,
        "patience": 7,
        "embedding_dim": 256,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_fusion": 0.7,
        "lambda_contrastive": 0.4,
        "fusion_warmup_epochs": 1,
        "contrastive_decay_epochs": 18,
        "contrastive_min_weight": 0.12,
        "grad_clip": 0.1,
        "scheduler_type": "cosine",
        "step_size": 10,
        "gamma": 0.5,
        "T_0": 8,
        "T_mult": 2,
        "fusion_type": "enhanced_gated",
        "temperature": 0.1,
        "num_heads": 4,
        "gated_num_gates": 3,
        "ortho_weight": 0.5,
    },
    "assist_17": {
        "batch_size": 1024,
        "epochs": 100,
        "patience": 7,
        "embedding_dim": 256,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_fusion": 0.7,
        "lambda_contrastive": 0.4,
        "fusion_warmup_epochs": 1,
        "contrastive_decay_epochs": 18,
        "contrastive_min_weight": 0.12,
        "grad_clip": 0.1,
        "scheduler_type": "cosine",
        "step_size": 10,
        "gamma": 0.5,
        "T_0": 8,
        "T_mult": 2,
        "fusion_type": "enhanced_gated",
        "temperature": 0.1,
        "num_heads": 4,
        "gated_num_gates": 3,
        "ortho_weight": 0.5,
    },
    "junyi": {
        "batch_size": 1024,
        "epochs": 100,
        "patience": 7,
        "embedding_dim": 256,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_fusion": 0.7,
        "lambda_contrastive": 0.4,
        "fusion_warmup_epochs": 1,
        "contrastive_decay_epochs": 18,
        "contrastive_min_weight": 0.12,
        "grad_clip": 0.1,
        "scheduler_type": "cosine",
        "step_size": 10,
        "gamma": 0.5,
        "T_0": 8,
        "T_mult": 2,
        "fusion_type": "enhanced_gated",
        "temperature": 0.1,
        "num_heads": 4,
        "gated_num_gates": 3,
        "ortho_weight": 0.5,
    },
}

PRISM_HPARAM_GRIDS = {
    "ortho_weight": [0.1, 0.3, 0.5, 0.7, 1.0],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "embedding_dim": [128, 192, 256, 320, 384],
}


def _format_tag_value(value):
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def make_hparam_tag(dataset, hparam, value):
    return f"hparam_{dataset}_{hparam}_{_format_tag_value(value)}"


def parse_hparam_tag(tag):
    prefix = "hparam_"
    if not str(tag).startswith(prefix):
        raise ValueError(f"Unsupported hparam tag: {tag}")

    payload = str(tag)[len(prefix):]
    last_sep = payload.rfind("_")
    if last_sep < 0:
        raise ValueError(f"Malformed hparam tag: {tag}")
    value_token = payload[last_sep + 1 :]
    dataset_hparam = payload[:last_sep]

    for hparam in ("ortho_weight", "embedding_dim", "dropout"):
        suffix = f"_{hparam}"
        if dataset_hparam.endswith(suffix):
            dataset = dataset_hparam[: -len(suffix)]
            value = float(value_token.replace("p", "."))
            return {"dataset": dataset, "hparam": hparam, "value": value}

    raise ValueError(f"Unable to parse hparam from tag: {tag}")


def build_prism_hparam_sweep_jobs(seed=888):
    jobs = []
    for dataset, base_config in PRISM_BASE_CONFIGS.items():
        for hparam, grid in PRISM_HPARAM_GRIDS.items():
            for value in grid:
                cfg = deepcopy(base_config)
                cfg[hparam] = value
                jobs.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "hparam": hparam,
                        "value": value,
                        "tag": make_hparam_tag(dataset, hparam, value),
                        "config": cfg,
                    }
                )
    return jobs


def build_hparam_result_summary(frame):
    hparam_rows = frame[frame["Tag"].astype(str).str.startswith("hparam_")].copy().reset_index(drop=True)
    if hparam_rows.empty:
        return pd.DataFrame(
            columns=["dataset", "hparam", "value", "tag", "test_auc", "test_acc", "test_rmse"]
        )
    if "Timestamp" in hparam_rows.columns:
        hparam_rows["Timestamp"] = pd.to_datetime(hparam_rows["Timestamp"], errors="coerce")
        hparam_rows = (
            hparam_rows.sort_values(["Tag", "Timestamp"])
            .drop_duplicates(subset=["Tag"], keep="last")
            .reset_index(drop=True)
        )

    parsed = hparam_rows["Tag"].apply(parse_hparam_tag).tolist()
    parsed_frame = pd.DataFrame(parsed)
    summary = pd.DataFrame(
        {
            "dataset": parsed_frame["dataset"],
            "hparam": parsed_frame["hparam"],
            "value": parsed_frame["value"].astype(float),
            "tag": hparam_rows["Tag"].astype(str).tolist(),
            "test_auc": pd.to_numeric(hparam_rows["Test_AUC"], errors="coerce"),
            "test_acc": pd.to_numeric(hparam_rows["Test_ACC"], errors="coerce"),
            "test_rmse": pd.to_numeric(hparam_rows["Test_RMSE"], errors="coerce"),
        }
    )
    return summary.sort_values(["hparam", "dataset", "value"]).reset_index(drop=True)
