import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, roc_auc_score


def parse_cpt_seq(value):
    if pd.isna(value):
        return []
    text = str(value).strip('"').strip("'").strip()
    if not text:
        return []
    try:
        return [int(part) for part in text.split(",") if str(part).strip() != ""]
    except ValueError:
        return []


def _pair_keys(df):
    frame = df.copy()
    frame["stu_id_key"] = frame["stu_id"].astype(str)
    frame["exer_id_key"] = frame["exer_id"].astype(str)
    return list(zip(frame["stu_id_key"], frame["exer_id_key"]))


def build_student_concept_history(train_df):
    history = {}
    if train_df is None or train_df.empty:
        return history

    for row in train_df.itertuples(index=False):
        stu_id = str(getattr(row, "stu_id"))
        label = int(getattr(row, "label"))
        for concept in parse_cpt_seq(getattr(row, "cpt_seq", "")):
            key = (stu_id, concept)
            correct, total = history.get(key, (0, 0))
            history[key] = (correct + label, total + 1)
    return history


def build_item_history(train_df):
    history = {}
    if train_df is None or train_df.empty or "exer_id" not in train_df.columns:
        return history

    for row in train_df.itertuples(index=False):
        exer_id = str(getattr(row, "exer_id"))
        label = int(getattr(row, "label"))
        correct, total = history.get(exer_id, (0, 0))
        history[exer_id] = (correct + label, total + 1)
    return history


def annotate_history_features(df, history):
    annotated = df.copy()
    concept_lists = annotated["cpt_seq"].map(parse_cpt_seq)

    hist_avg_rates = []
    min_hist_mastery_rates = []
    min_supports = []
    concept_counts = []

    for stu_id, concepts in zip(annotated["stu_id"].astype(str), concept_lists):
        rates = []
        supports = []
        for concept in concepts:
            correct, total = history.get((stu_id, concept), (0, 0))
            if total > 0:
                rates.append(correct / total)
                supports.append(total)
        concept_counts.append(len(concepts))
        if rates:
            hist_avg_rates.append(float(np.mean(rates)))
            min_hist_mastery_rates.append(float(np.min(rates)))
            min_supports.append(int(min(supports)))
        else:
            hist_avg_rates.append(np.nan)
            min_hist_mastery_rates.append(np.nan)
            min_supports.append(0)

    annotated["concept_ids"] = concept_lists
    annotated["concept_count"] = concept_counts
    annotated["hist_avg_rate"] = hist_avg_rates
    annotated["min_hist_mastery_rate"] = min_hist_mastery_rates
    annotated["min_cpt_hist"] = min_supports
    return annotated


def annotate_item_features(df, item_history):
    annotated = df.copy()
    item_acc = []
    item_support = []

    for exer_id in annotated["exer_id"].astype(str):
        correct, total = item_history.get(exer_id, (0, 0))
        if total > 0:
            item_acc.append(float(correct / total))
            item_support.append(int(total))
        else:
            item_acc.append(np.nan)
            item_support.append(0)

    annotated["item_train_acc"] = item_acc
    annotated["item_train_support"] = item_support
    return annotated


def attach_concept_proxy_metrics(df, concept_proxy_pred, item_drop_floor=0.05, pred_col="p_pred"):
    annotated = df.copy()
    concept_proxy = np.asarray(concept_proxy_pred, dtype=float)
    if len(annotated) != len(concept_proxy):
        raise ValueError(
            f"Concept proxy length ({len(concept_proxy)}) does not match rows ({len(annotated)})."
        )
    if "hist_avg_rate" not in annotated.columns or pred_col not in annotated.columns:
        raise ValueError("attach_concept_proxy_metrics requires 'hist_avg_rate' and prediction columns.")

    hist = pd.to_numeric(annotated["hist_avg_rate"], errors="coerce").to_numpy(dtype=float)
    item_pred = pd.to_numeric(annotated[pred_col], errors="coerce").to_numpy(dtype=float)
    item_drop = np.maximum(hist - item_pred, 0.0)
    concept_drop = np.maximum(hist - concept_proxy, 0.0)
    stable_ratio = np.full(len(annotated), np.nan, dtype=float)
    valid = np.isfinite(item_drop) & np.isfinite(concept_drop)
    stable_ratio[valid] = concept_drop[valid] / np.maximum(item_drop[valid], float(item_drop_floor))
    decoupling_gap = concept_proxy - item_pred

    annotated["concept_proxy_pred"] = concept_proxy
    annotated["item_drop"] = item_drop
    annotated["concept_drop"] = concept_drop
    annotated["stable_concept_drop_ratio"] = stable_ratio
    annotated["decoupling_gap"] = decoupling_gap
    return annotated


def select_strong_positive_candidates(
    df,
    hist_threshold,
    min_concept_support,
    pred_threshold=None,
    max_item_pred=None,
    max_concepts=None,
    require_all_mastery=False,
    min_item_support=0,
    min_item_acc=None,
    min_concept_proxy_pred=None,
    min_decoupling_gap=None,
    max_stable_concept_drop_ratio=None,
):
    mask = (
        (df["label"].astype(int) == 1)
        & (df["concept_count"].astype(int) > 0)
        & (df["hist_avg_rate"].fillna(-1.0) >= float(hist_threshold))
        & (df["min_cpt_hist"].astype(int) >= int(min_concept_support))
    )
    if pred_threshold is not None and "p_pred" in df.columns:
        mask = mask & (df["p_pred"].astype(float) >= float(pred_threshold))
    if max_item_pred is not None and "p_pred" in df.columns:
        mask = mask & (df["p_pred"].astype(float) <= float(max_item_pred))
    if max_concepts is not None:
        mask = mask & (df["concept_count"].astype(int) <= int(max_concepts))
    if require_all_mastery and "min_hist_mastery_rate" in df.columns:
        mask = mask & (df["min_hist_mastery_rate"].fillna(-1.0) >= float(hist_threshold))
    if int(min_item_support) > 0 and "item_train_support" in df.columns:
        mask = mask & (df["item_train_support"].astype(int) >= int(min_item_support))
    if min_item_acc is not None and "item_train_acc" in df.columns:
        mask = mask & (df["item_train_acc"].fillna(-1.0) >= float(min_item_acc))
    if min_concept_proxy_pred is not None and "concept_proxy_pred" in df.columns:
        mask = mask & (df["concept_proxy_pred"].fillna(-1.0) >= float(min_concept_proxy_pred))
    if min_decoupling_gap is not None and "decoupling_gap" in df.columns:
        mask = mask & (df["decoupling_gap"].fillna(-1e9) >= float(min_decoupling_gap))
    if max_stable_concept_drop_ratio is not None and "stable_concept_drop_ratio" in df.columns:
        mask = mask & (
            df["stable_concept_drop_ratio"].fillna(float("inf"))
            <= float(max_stable_concept_drop_ratio)
        )
    return mask


def select_reference_candidates(df, reference_df):
    if reference_df is None or reference_df.empty:
        return pd.Series(False, index=df.index)

    ref_pairs = set(_pair_keys(reference_df))
    mask = [pair in ref_pairs for pair in _pair_keys(df)]
    return pd.Series(mask, index=df.index)


def select_flip_indices(candidate_indices, ratio, seed):
    candidate_indices = list(candidate_indices)
    if not candidate_indices or ratio <= 0:
        return []

    sample_size = int(round(len(candidate_indices) * float(ratio)))
    if sample_size <= 0:
        sample_size = 1
    sample_size = min(sample_size, len(candidate_indices))

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.array(candidate_indices, dtype=int), size=sample_size, replace=False)
    return sorted(int(idx) for idx in chosen.tolist())


def _sample_negative_indices(pool_indices, sample_size, rng, negative_strategy="random", negative_scores=None):
    if sample_size <= 0:
        return np.array([], dtype=int)

    pool_indices = np.asarray(sorted({int(idx) for idx in pool_indices}), dtype=int)
    if pool_indices.size == 0:
        return np.array([], dtype=int)

    sample_size = min(int(sample_size), int(pool_indices.size))
    if sample_size <= 0:
        return np.array([], dtype=int)

    if negative_strategy == "random":
        chosen = rng.choice(pool_indices, size=sample_size, replace=False)
        return np.sort(chosen.astype(int))

    if negative_strategy == "hard":
        if negative_scores is None:
            raise ValueError("negative_scores is required when negative_strategy='hard'.")
        scores = np.asarray(negative_scores, dtype=float)
        if scores.shape[0] <= int(pool_indices.max()):
            raise ValueError("negative_scores length does not cover all pool indices.")
        tie_break = rng.random(pool_indices.size)
        order = np.lexsort((tie_break, -scores[pool_indices]))
        chosen = pool_indices[order[:sample_size]]
        return np.sort(chosen.astype(int))

    raise ValueError(f"Unsupported negative_strategy: {negative_strategy}")


def _allocate_match_targets(candidate_values, sample_size, rng):
    candidate_values = np.asarray(candidate_values)
    unique_values, counts = np.unique(candidate_values, return_counts=True)
    expected = counts.astype(float) * float(sample_size) / float(counts.sum())
    base = np.floor(expected).astype(int)
    remainder = int(sample_size - base.sum())
    if remainder > 0:
        fractional = expected - base
        tie_break = rng.random(len(unique_values))
        order = np.lexsort((tie_break, -fractional))
        for idx in order[:remainder]:
            base[idx] += 1
    return unique_values, base


def build_stress_subset_indices(
    labels,
    candidate_mask,
    seed,
    negative_multiplier=1.0,
    negative_scores=None,
    concept_counts=None,
    negative_strategy="random",
    match_concept_counts=False,
):
    labels = np.asarray(labels, dtype=int)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return []

    negative_indices = np.flatnonzero((~candidate_mask) & (labels == 0))
    sample_size = int(round(candidate_indices.size * float(negative_multiplier)))
    if negative_indices.size > 0 and float(negative_multiplier) > 0 and sample_size <= 0:
        sample_size = 1
    sample_size = min(sample_size, int(negative_indices.size))

    chosen_negatives = np.array([], dtype=int)
    rng = np.random.default_rng(int(seed))
    if sample_size > 0:
        if match_concept_counts:
            if concept_counts is None:
                raise ValueError("concept_counts is required when match_concept_counts=True.")
            concept_counts = np.asarray(concept_counts)
            if concept_counts.shape[0] != labels.shape[0]:
                raise ValueError("concept_counts must have the same length as labels.")
            candidate_concepts = concept_counts[candidate_indices]
            unique_values, concept_targets = _allocate_match_targets(candidate_concepts, sample_size, rng)
            chosen_parts = []
            remaining_negatives = set(int(idx) for idx in negative_indices.tolist())
            missing = 0

            for concept_value, target in zip(unique_values, concept_targets):
                if target <= 0:
                    continue
                bucket = [idx for idx in remaining_negatives if concept_counts[idx] == concept_value]
                chosen = _sample_negative_indices(
                    bucket,
                    target,
                    rng,
                    negative_strategy=negative_strategy,
                    negative_scores=negative_scores,
                )
                chosen_parts.extend(int(idx) for idx in chosen.tolist())
                remaining_negatives.difference_update(int(idx) for idx in chosen.tolist())
                missing += max(int(target) - int(len(chosen)), 0)

            if missing > 0:
                fallback = _sample_negative_indices(
                    sorted(remaining_negatives),
                    missing,
                    rng,
                    negative_strategy=negative_strategy,
                    negative_scores=negative_scores,
                )
                chosen_parts.extend(int(idx) for idx in fallback.tolist())

            chosen_negatives = np.asarray(sorted(chosen_parts), dtype=int)
        else:
            chosen_negatives = _sample_negative_indices(
                negative_indices,
                sample_size,
                rng,
                negative_strategy=negative_strategy,
                negative_scores=negative_scores,
            )

    combined = np.concatenate([candidate_indices.astype(int), np.sort(chosen_negatives.astype(int))])
    return sorted(int(idx) for idx in combined.tolist())


def build_flipped_labels(labels, flip_indices):
    flipped = np.asarray(labels, dtype=int).copy()
    if len(flip_indices) == 0:
        return flipped
    flipped[np.asarray(flip_indices, dtype=int)] = 0
    return flipped


def _candidate_thresholds(p_pred):
    scores = np.asarray(p_pred, dtype=float)
    finite_scores = np.unique(scores[np.isfinite(scores)])
    if finite_scores.size == 0:
        return np.array([0.5], dtype=float)
    if finite_scores.size == 1:
        single = float(finite_scores[0])
        return np.array(sorted({0.0, 0.5, 1.0, single}), dtype=float)
    midpoints = (finite_scores[:-1] + finite_scores[1:]) / 2.0
    thresholds = np.concatenate([[0.0], midpoints, [1.0]])
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return np.unique(thresholds.astype(float))


def find_optimal_threshold(labels, p_pred, metric="acc"):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(p_pred, dtype=float)
    if labels.size == 0:
        return 0.5

    metric = str(metric).strip().lower()
    if metric not in {"acc", "balanced_acc"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    best_threshold = 0.5
    best_score = -float("inf")
    best_distance = float("inf")
    for threshold in _candidate_thresholds(scores):
        preds = scores > float(threshold)
        if metric == "acc":
            score = float(accuracy_score(labels, preds))
        else:
            score = float(balanced_accuracy_score(labels, preds))
        distance = abs(float(threshold) - 0.5)
        if (
            score > best_score + 1e-12
            or (abs(score - best_score) <= 1e-12 and distance < best_distance - 1e-12)
            or (
                abs(score - best_score) <= 1e-12
                and abs(distance - best_distance) <= 1e-12
                and float(threshold) < float(best_threshold)
            )
        ):
            best_threshold = float(threshold)
            best_score = score
            best_distance = distance
    return float(best_threshold)


def evaluate_binary_predictions(labels, p_pred, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)
    if labels.size == 0:
        return {"auc": 0.0, "acc": 0.0, "balanced_acc": 0.0, "rmse": 0.0}

    try:
        auc = float(roc_auc_score(labels, p_pred))
    except ValueError:
        auc = 0.5

    binary_preds = p_pred > float(threshold)
    acc = float(accuracy_score(labels, binary_preds))
    balanced_acc = float(balanced_accuracy_score(labels, binary_preds))
    rmse = float(math.sqrt(mean_squared_error(labels, p_pred)))
    return {"auc": auc, "acc": acc, "balanced_acc": balanced_acc, "rmse": rmse}
