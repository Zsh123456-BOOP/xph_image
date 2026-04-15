import numpy as np
import pandas as pd


def _normalize_id(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")

    text = str(value).strip()
    if not text:
        return text

    try:
        number = float(text)
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _pair_key(row):
    return _normalize_id(row["stu_id"]), _normalize_id(row["exer_id"])


def select_conflict_cases(
    df,
    hist_threshold,
    min_concept_support,
    max_concepts=2,
    min_item_pred=None,
):
    cases = df.copy()
    cases = cases[
        (cases["label"].astype(int) == 0)
        & (cases["concept_count"].astype(int) >= 1)
        & (cases["concept_count"].astype(int) <= int(max_concepts))
        & (cases["hist_avg_rate"].fillna(-1.0) >= float(hist_threshold))
        & (cases["min_cpt_hist"].astype(int) >= int(min_concept_support))
    ]
    if min_item_pred is not None and "p_pred" in cases.columns:
        cases = cases[cases["p_pred"].astype(float) >= float(min_item_pred)]

    return cases.sort_values(
        by=["p_pred", "hist_avg_rate", "min_cpt_hist"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def align_cases_to_reference(cases, reference_df):
    if reference_df is None or reference_df.empty:
        return cases.iloc[0:0].copy()

    indexed = {}
    for _, row in cases.iterrows():
        indexed[_pair_key(row)] = row

    ordered_rows = []
    for _, row in reference_df.iterrows():
        key = _pair_key(row)
        if key in indexed:
            ordered_rows.append(indexed[key])

    if not ordered_rows:
        return cases.iloc[0:0].copy()
    return pd.DataFrame(ordered_rows).reset_index(drop=True)


def compute_concept_proxy_scores(item_scores, q_matrix, concept_indices):
    item_scores = np.asarray(item_scores, dtype=float)
    q_matrix = np.asarray(q_matrix, dtype=int)
    concept_indices = [int(idx) for idx in concept_indices]
    concept_scores = {}
    item_concept_counts = q_matrix.sum(axis=1)

    for concept_idx in concept_indices:
        exact_mask = (q_matrix[:, concept_idx] == 1) & (item_concept_counts == 1)
        if np.any(exact_mask):
            selected_scores = item_scores[exact_mask]
        else:
            contain_mask = q_matrix[:, concept_idx] == 1
            selected_scores = item_scores[contain_mask]

        if selected_scores.size == 0:
            concept_scores[concept_idx] = np.nan
        else:
            concept_scores[concept_idx] = float(np.mean(selected_scores))

    valid_scores = [score for score in concept_scores.values() if np.isfinite(score)]
    overall_proxy = float(np.mean(valid_scores)) if valid_scores else np.nan
    return {"concept_scores": concept_scores, "overall_proxy": overall_proxy}
