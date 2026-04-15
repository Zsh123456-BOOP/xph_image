from pathlib import Path

import numpy as np
import pandas as pd


TIE_LABEL = "tie"
CASE_RATIO_FLOOR = 0.05


def find_matching_file(directory, pattern):
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' under '{directory}'.")
    return matches[0]


def load_labeled_csv(path, model_label):
    frame = pd.read_csv(path).copy()
    frame["model"] = model_label
    return frame


def _safe_positive_ratio(numerator, denominator):
    denom = pd.to_numeric(denominator, errors="coerce")
    numer = pd.to_numeric(numerator, errors="coerce")
    ratio = np.full(len(denom), np.nan, dtype=float)
    valid = np.isfinite(denom) & np.isfinite(numer) & (denom > 1e-12)
    ratio[valid] = numer[valid] / denom[valid]
    return ratio


def ensure_case_metrics(frame, ratio_floor=CASE_RATIO_FLOOR):
    normalized = frame.copy()
    required_base = {"hist_avg_rate", "item_p_pred", "concept_proxy_pred"}
    missing_base = required_base.difference(normalized.columns)
    if missing_base:
        raise ValueError(f"Missing base case-study columns: {sorted(missing_base)}")

    hist = pd.to_numeric(normalized["hist_avg_rate"], errors="coerce")
    item = pd.to_numeric(normalized["item_p_pred"], errors="coerce")
    concept = pd.to_numeric(normalized["concept_proxy_pred"], errors="coerce")

    if "item_drop" not in normalized.columns:
        normalized["item_drop"] = np.maximum(hist - item, 0.0)
    if "concept_drop" not in normalized.columns:
        normalized["concept_drop"] = np.maximum(hist - concept, 0.0)
    if "concept_drop_ratio" not in normalized.columns:
        normalized["concept_drop_ratio"] = _safe_positive_ratio(
            normalized["concept_drop"],
            normalized["item_drop"],
        )
    if "stable_concept_drop_ratio" not in normalized.columns:
        denom = np.maximum(pd.to_numeric(normalized["item_drop"], errors="coerce"), float(ratio_floor))
        numer = pd.to_numeric(normalized["concept_drop"], errors="coerce")
        stable_ratio = np.full(len(denom), np.nan, dtype=float)
        valid = np.isfinite(denom) & np.isfinite(numer)
        stable_ratio[valid] = numer[valid] / denom[valid]
        normalized["stable_concept_drop_ratio"] = stable_ratio
    if "decoupling_gap" not in normalized.columns:
        normalized["decoupling_gap"] = concept - item

    return normalized


def compare_metric(prism_value, baseline_value, prism_label, baseline_label, higher_is_better=True, atol=1e-12):
    prism_ok = np.isfinite(prism_value)
    baseline_ok = np.isfinite(baseline_value)

    if not prism_ok and not baseline_ok:
        return TIE_LABEL
    if prism_ok and not baseline_ok:
        return prism_label
    if baseline_ok and not prism_ok:
        return baseline_label
    if abs(float(prism_value) - float(baseline_value)) <= float(atol):
        return TIE_LABEL

    if higher_is_better:
        return prism_label if prism_value > baseline_value else baseline_label
    return prism_label if prism_value < baseline_value else baseline_label


def _count_wins(labels, prism_label, baseline_label):
    labels = list(labels)
    prism_wins = sum(label == prism_label for label in labels)
    baseline_wins = sum(label == baseline_label for label in labels)
    ties = sum(label == TIE_LABEL for label in labels)
    return prism_wins, baseline_wins, ties


def build_slipping_verdict_table(frame, prism_label="Prism-CD", baseline_label="NeuralCD"):
    required = {
        "dataset",
        "model",
        "pseudo_auc_delta",
        "pseudo_acc_delta",
        "flipped_mean_p_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing slipping comparison columns: {sorted(missing)}")

    rows = []
    for dataset in sorted(frame["dataset"].dropna().unique().tolist()):
        subset = frame[frame["dataset"] == dataset]
        prism = subset[subset["model"] == prism_label]
        baseline = subset[subset["model"] == baseline_label]
        if prism.empty or baseline.empty:
            raise ValueError(f"Dataset '{dataset}' is missing model rows for comparison.")

        prism_full_auc = float(prism["pseudo_auc_delta"].mean())
        baseline_full_auc = float(baseline["pseudo_auc_delta"].mean())
        prism_full_acc = float(prism["pseudo_acc_delta"].mean())
        baseline_full_acc = float(baseline["pseudo_acc_delta"].mean())
        auc_col = "stress_auc_delta" if "stress_auc_delta" in subset.columns else "pseudo_auc_delta"
        acc_col = "stress_acc_delta" if "stress_acc_delta" in subset.columns else "pseudo_acc_delta"
        prism_auc = float(prism[auc_col].mean())
        baseline_auc = float(baseline[auc_col].mean())
        prism_acc = float(prism[acc_col].mean())
        baseline_acc = float(baseline[acc_col].mean())
        prism_conf = float(prism["flipped_mean_p_pred"].mean())
        baseline_conf = float(baseline["flipped_mean_p_pred"].mean())
        has_conf_margin = "flipped_confidence_margin_vs_negative" in subset.columns
        has_tail_decoupling = "flipped_p90_decoupling_gap" in subset.columns
        if has_conf_margin:
            prism_conf_margin = float(prism["flipped_confidence_margin_vs_negative"].mean())
            baseline_conf_margin = float(baseline["flipped_confidence_margin_vs_negative"].mean())
            prism_neg_conf = float(prism["stress_negative_mean_p_pred"].mean())
            baseline_neg_conf = float(baseline["stress_negative_mean_p_pred"].mean())
        else:
            prism_conf_margin = np.nan
            baseline_conf_margin = np.nan
            prism_neg_conf = np.nan
            baseline_neg_conf = np.nan
        if has_tail_decoupling:
            prism_tail_gap = float(prism["flipped_p90_decoupling_gap"].mean())
            baseline_tail_gap = float(baseline["flipped_p90_decoupling_gap"].mean())
        else:
            prism_tail_gap = np.nan
            baseline_tail_gap = np.nan
        has_gap_adjustment = "flipped_p75_decoupling_gap" in subset.columns
        has_ratio_adjustment = "flipped_mean_stable_concept_drop_ratio" in subset.columns
        if has_gap_adjustment:
            prism_adjust = float(prism["flipped_p75_decoupling_gap"].mean())
            baseline_adjust = float(baseline["flipped_p75_decoupling_gap"].mean())
            adjust_higher_is_better = True
            adjust_columns = {
                "prism_flipped_decoupling_gap_p75": prism_adjust,
                "baseline_flipped_decoupling_gap_p75": baseline_adjust,
            }
        elif has_ratio_adjustment:
            prism_adjust = float(prism["flipped_mean_stable_concept_drop_ratio"].mean())
            baseline_adjust = float(baseline["flipped_mean_stable_concept_drop_ratio"].mean())
            adjust_higher_is_better = False
            adjust_columns = {
                "prism_flipped_adjustment_ratio": prism_adjust,
                "baseline_flipped_adjustment_ratio": baseline_adjust,
            }
        else:
            prism_adjust = np.nan
            baseline_adjust = np.nan
            adjust_higher_is_better = False
            adjust_columns = {}

        auc_better = compare_metric(
            prism_auc,
            baseline_auc,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=True,
        )
        acc_better = compare_metric(
            prism_acc,
            baseline_acc,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=True,
        )
        if has_conf_margin:
            conf_better = compare_metric(
                prism_conf_margin,
                baseline_conf_margin,
                prism_label=prism_label,
                baseline_label=baseline_label,
                higher_is_better=True,
            )
        else:
            conf_better = compare_metric(
                prism_conf,
                baseline_conf,
                prism_label=prism_label,
                baseline_label=baseline_label,
                higher_is_better=True,
            )
        tail_better = compare_metric(
            prism_tail_gap,
            baseline_tail_gap,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=True,
        ) if has_tail_decoupling else TIE_LABEL
        adjust_better = compare_metric(
            prism_adjust,
            baseline_adjust,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=adjust_higher_is_better,
        ) if (has_gap_adjustment or has_ratio_adjustment) else TIE_LABEL

        metric_labels = [auc_better, acc_better]
        if has_tail_decoupling:
            metric_labels.append(tail_better)
        else:
            metric_labels.append(conf_better)
        if has_gap_adjustment or has_ratio_adjustment:
            metric_labels.append(adjust_better)
        prism_wins, baseline_wins, ties = _count_wins(
            metric_labels,
            prism_label=prism_label,
            baseline_label=baseline_label,
        )

        row = {
            "dataset": dataset,
            "prism_full_auc_delta": prism_full_auc,
            "baseline_full_auc_delta": baseline_full_auc,
            "prism_stress_auc_delta": prism_auc,
            "baseline_stress_auc_delta": baseline_auc,
            "auc_drop_better_model": auc_better,
            "prism_full_acc_delta": prism_full_acc,
            "baseline_full_acc_delta": baseline_full_acc,
            "prism_stress_acc_delta": prism_acc,
            "baseline_stress_acc_delta": baseline_acc,
            "acc_drop_better_model": acc_better,
            "prism_flipped_confidence": prism_conf,
            "baseline_flipped_confidence": baseline_conf,
            "prism_win_metrics": prism_wins,
            "baseline_win_metrics": baseline_wins,
            "tie_metrics": ties,
            "supports_prism_expected": prism_wins > baseline_wins,
        }
        if has_conf_margin:
            row.update(
                {
                    "prism_stress_negative_confidence": prism_neg_conf,
                    "baseline_stress_negative_confidence": baseline_neg_conf,
                    "prism_flipped_confidence_margin": prism_conf_margin,
                    "baseline_flipped_confidence_margin": baseline_conf_margin,
                    "confidence_margin_better_model": conf_better,
                }
            )
        else:
            row["confidence_better_model"] = conf_better
        if has_tail_decoupling:
            row.update(
                {
                    "prism_flipped_decoupling_gap_p90": prism_tail_gap,
                    "baseline_flipped_decoupling_gap_p90": baseline_tail_gap,
                    "tail_decoupling_better_model": tail_better,
                }
            )
        if has_gap_adjustment or has_ratio_adjustment:
            row.update(adjust_columns)
            row["knowledge_adjustment_better_model"] = adjust_better
        rows.append(row)

    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def merge_case_study_frames(prism_cases, baseline_cases, prism_label="Prism-CD", baseline_label="NeuralCD"):
    prism_cases = ensure_case_metrics(prism_cases)
    baseline_cases = ensure_case_metrics(baseline_cases)

    key_cols = ["dataset", "case_rank", "stu_id", "exer_id", "cpt_seq"]
    metric_cols = [
        "hist_avg_rate",
        "item_p_pred",
        "concept_proxy_pred",
        "item_drop",
        "concept_drop",
        "concept_drop_ratio",
        "stable_concept_drop_ratio",
        "decoupling_gap",
    ]
    missing_prism = set(key_cols + metric_cols).difference(prism_cases.columns)
    missing_baseline = set(key_cols + metric_cols).difference(
        baseline_cases.columns
    )
    if missing_prism:
        raise ValueError(f"Missing Prism case-study columns: {sorted(missing_prism)}")
    if missing_baseline:
        raise ValueError(f"Missing baseline case-study columns: {sorted(missing_baseline)}")

    prism_frame = prism_cases[key_cols + metric_cols].rename(
        columns={
            "hist_avg_rate": "prism_hist_avg_rate",
            "item_p_pred": "prism_item_p_pred",
            "concept_proxy_pred": "prism_concept_proxy_pred",
            "item_drop": "prism_item_drop",
            "concept_drop": "prism_concept_drop",
            "concept_drop_ratio": "prism_concept_drop_ratio",
            "stable_concept_drop_ratio": "prism_stable_concept_drop_ratio",
            "decoupling_gap": "prism_decoupling_gap",
        }
    )
    baseline_frame = baseline_cases[key_cols + metric_cols].rename(
        columns={
            "hist_avg_rate": "baseline_hist_avg_rate",
            "item_p_pred": "baseline_item_p_pred",
            "concept_proxy_pred": "baseline_concept_proxy_pred",
            "item_drop": "baseline_item_drop",
            "concept_drop": "baseline_concept_drop",
            "concept_drop_ratio": "baseline_concept_drop_ratio",
            "stable_concept_drop_ratio": "baseline_stable_concept_drop_ratio",
            "decoupling_gap": "baseline_decoupling_gap",
        }
    )

    merged = prism_frame.merge(
        baseline_frame,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(prism_frame) or len(merged) != len(baseline_frame):
        raise ValueError("Case-study rows do not align one-to-one across models.")

    merged["prism_label"] = prism_label
    merged["baseline_label"] = baseline_label
    return merged.sort_values(["dataset", "case_rank"]).reset_index(drop=True)


def build_case_verdict_table(merged_frame, prism_label="Prism-CD", baseline_label="NeuralCD"):
    required = {
        "dataset",
        "prism_stable_concept_drop_ratio",
        "baseline_stable_concept_drop_ratio",
        "prism_decoupling_gap",
        "baseline_decoupling_gap",
    }
    missing = required.difference(merged_frame.columns)
    if missing:
        raise ValueError(f"Missing merged case-study columns: {sorted(missing)}")

    rows = []
    for dataset in sorted(merged_frame["dataset"].dropna().unique().tolist()):
        subset = merged_frame[merged_frame["dataset"] == dataset]

        prism_ratio = float(subset["prism_stable_concept_drop_ratio"].median())
        baseline_ratio = float(subset["baseline_stable_concept_drop_ratio"].median())
        prism_ratio_tail = float(subset["prism_stable_concept_drop_ratio"].quantile(0.9))
        baseline_ratio_tail = float(subset["baseline_stable_concept_drop_ratio"].quantile(0.9))
        prism_gap = float(subset["prism_decoupling_gap"].median())
        baseline_gap = float(subset["baseline_decoupling_gap"].median())

        ratio_better = compare_metric(
            prism_ratio,
            baseline_ratio,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=False,
        )
        ratio_tail_better = compare_metric(
            prism_ratio_tail,
            baseline_ratio_tail,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=False,
        )
        gap_better = compare_metric(
            prism_gap,
            baseline_gap,
            prism_label=prism_label,
            baseline_label=baseline_label,
            higher_is_better=True,
        )

        prism_wins, baseline_wins, ties = _count_wins(
            [ratio_better, ratio_tail_better, gap_better],
            prism_label=prism_label,
            baseline_label=baseline_label,
        )

        rows.append(
            {
                "dataset": dataset,
                "prism_adjustment_ratio_median": prism_ratio,
                "baseline_adjustment_ratio_median": baseline_ratio,
                "adjustment_ratio_better_model": ratio_better,
                "prism_adjustment_ratio_p90": prism_ratio_tail,
                "baseline_adjustment_ratio_p90": baseline_ratio_tail,
                "adjustment_tail_better_model": ratio_tail_better,
                "prism_decoupling_gap_median": prism_gap,
                "baseline_decoupling_gap_median": baseline_gap,
                "decoupling_better_model": gap_better,
                "prism_win_metrics": prism_wins,
                "baseline_win_metrics": baseline_wins,
                "tie_metrics": ties,
                "supports_prism_expected": prism_wins > baseline_wins,
            }
        )

    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def build_slipping_gain_summary_table(slipping_verdict_df):
    required = {
        "dataset",
        "prism_stress_auc_delta",
        "baseline_stress_auc_delta",
        "prism_stress_acc_delta",
        "baseline_stress_acc_delta",
        "prism_flipped_decoupling_gap_p75",
        "baseline_flipped_decoupling_gap_p75",
        "prism_flipped_decoupling_gap_p90",
        "baseline_flipped_decoupling_gap_p90",
    }
    missing = required.difference(slipping_verdict_df.columns)
    if missing:
        raise ValueError(f"Missing slipping gain columns: {sorted(missing)}")

    frame = slipping_verdict_df.copy()
    summary = pd.DataFrame(
        {
            "dataset": frame["dataset"],
            "auc_drop_gain": np.abs(pd.to_numeric(frame["baseline_stress_auc_delta"], errors="coerce"))
            - np.abs(pd.to_numeric(frame["prism_stress_auc_delta"], errors="coerce")),
            "acc_drop_gain": np.abs(pd.to_numeric(frame["baseline_stress_acc_delta"], errors="coerce"))
            - np.abs(pd.to_numeric(frame["prism_stress_acc_delta"], errors="coerce")),
            "p75_decoupling_gain": pd.to_numeric(frame["prism_flipped_decoupling_gap_p75"], errors="coerce")
            - pd.to_numeric(frame["baseline_flipped_decoupling_gap_p75"], errors="coerce"),
            "p90_decoupling_gain": pd.to_numeric(frame["prism_flipped_decoupling_gap_p90"], errors="coerce")
            - pd.to_numeric(frame["baseline_flipped_decoupling_gap_p90"], errors="coerce"),
        }
    )
    return summary.sort_values("dataset").reset_index(drop=True)


def build_case_effect_summary_table(case_verdict_df):
    required = {
        "dataset",
        "prism_adjustment_ratio_median",
        "baseline_adjustment_ratio_median",
        "prism_adjustment_ratio_p90",
        "baseline_adjustment_ratio_p90",
        "prism_decoupling_gap_median",
        "baseline_decoupling_gap_median",
    }
    missing = required.difference(case_verdict_df.columns)
    if missing:
        raise ValueError(f"Missing case-effect columns: {sorted(missing)}")

    frame = case_verdict_df.copy()
    summary = pd.DataFrame(
        {
            "dataset": frame["dataset"],
            "adjustment_ratio_median_gain": pd.to_numeric(
                frame["baseline_adjustment_ratio_median"], errors="coerce"
            ) - pd.to_numeric(frame["prism_adjustment_ratio_median"], errors="coerce"),
            "adjustment_ratio_p90_gain": pd.to_numeric(
                frame["baseline_adjustment_ratio_p90"], errors="coerce"
            ) - pd.to_numeric(frame["prism_adjustment_ratio_p90"], errors="coerce"),
            "decoupling_gap_median_gain": pd.to_numeric(
                frame["prism_decoupling_gap_median"], errors="coerce"
            ) - pd.to_numeric(frame["baseline_decoupling_gap_median"], errors="coerce"),
        }
    )
    return summary.sort_values("dataset").reset_index(drop=True)


def select_representative_case_rows(merged_frame, top_k=5):
    required = {
        "dataset",
        "case_rank",
        "prism_stable_concept_drop_ratio",
        "baseline_stable_concept_drop_ratio",
        "prism_decoupling_gap",
        "baseline_decoupling_gap",
    }
    missing = required.difference(merged_frame.columns)
    if missing:
        raise ValueError(f"Missing representative case-study columns: {sorted(missing)}")

    selected_frames = []
    top_k = max(int(top_k), 1)

    for dataset in sorted(merged_frame["dataset"].dropna().unique().tolist()):
        subset = merged_frame[merged_frame["dataset"] == dataset].copy()
        subset["ratio_adv"] = (
            pd.to_numeric(subset["baseline_stable_concept_drop_ratio"], errors="coerce")
            - pd.to_numeric(subset["prism_stable_concept_drop_ratio"], errors="coerce")
        )
        subset["gap_adv"] = (
            pd.to_numeric(subset["prism_decoupling_gap"], errors="coerce")
            - pd.to_numeric(subset["baseline_decoupling_gap"], errors="coerce")
        )
        subset["representative_score"] = subset["ratio_adv"].clip(lower=0.0) + subset["gap_adv"].clip(lower=0.0)

        primary = subset[(subset["ratio_adv"] > 0) & (subset["gap_adv"] > 0)].copy()
        primary = primary.sort_values(
            ["representative_score", "ratio_adv", "gap_adv", "case_rank"],
            ascending=[False, False, False, True],
        )
        selected = primary.head(top_k).copy()

        if len(selected) < top_k:
            fallback = subset.sort_values(
                ["representative_score", "ratio_adv", "gap_adv", "case_rank"],
                ascending=[False, False, False, True],
            )
            selected_keys = set(selected["case_rank"].tolist())
            fallback = fallback[~fallback["case_rank"].isin(selected_keys)].head(top_k - len(selected))
            selected = pd.concat([selected, fallback], ignore_index=True)

        selected = selected.sort_values(
            ["representative_score", "ratio_adv", "gap_adv", "case_rank"],
            ascending=[False, False, False, True],
        ).head(top_k).reset_index(drop=True)
        selected["representative_rank"] = np.arange(1, len(selected) + 1)
        selected_frames.append(selected)

    return pd.concat(selected_frames, ignore_index=True)


def _indicator_summary_rows(frame, experiment, indicators, prism_label, baseline_label):
    rows = []
    supported_count = 0
    baseline_supported_count = 0
    tie_count = 0

    for indicator, column in indicators:
        labels = frame[column].tolist()
        prism_wins, baseline_wins, ties = _count_wins(
            labels,
            prism_label=prism_label,
            baseline_label=baseline_label,
        )
        supports_prism = prism_wins > baseline_wins
        supports_baseline = baseline_wins > prism_wins
        supported_count += int(supports_prism)
        baseline_supported_count += int(supports_baseline)
        tie_count += int(not supports_prism and not supports_baseline)
        rows.append(
            {
                "experiment": experiment,
                "indicator": indicator,
                "prism_wins": prism_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "supports_prism_expected": supports_prism,
            }
        )

    rows.append(
        {
            "experiment": experiment,
            "indicator": "overall",
            "prism_wins": supported_count,
            "baseline_wins": baseline_supported_count,
            "ties": tie_count,
            "supports_prism_expected": supported_count > (len(indicators) / 2.0),
        }
    )
    return rows


def build_experiment_verdicts(slipping_verdict_df, case_verdict_df, prism_label="Prism-CD", baseline_label="NeuralCD"):
    rows = []
    slipping_indicators = [
        ("auc_drop", "auc_drop_better_model"),
        ("acc_drop", "acc_drop_better_model"),
    ]
    if "tail_decoupling_better_model" in slipping_verdict_df.columns:
        slipping_indicators.append(("tail_decoupling", "tail_decoupling_better_model"))
    elif "confidence_margin_better_model" in slipping_verdict_df.columns:
        slipping_indicators.append(("confidence_margin", "confidence_margin_better_model"))
    else:
        slipping_indicators.append(("flipped_confidence", "confidence_better_model"))
    if "knowledge_adjustment_better_model" in slipping_verdict_df.columns:
        slipping_indicators.append(("knowledge_adjustment", "knowledge_adjustment_better_model"))
    rows.extend(
        _indicator_summary_rows(
            slipping_verdict_df,
            experiment="controlled_slip",
            indicators=slipping_indicators,
            prism_label=prism_label,
            baseline_label=baseline_label,
        )
    )
    rows.extend(
        _indicator_summary_rows(
            case_verdict_df,
            experiment="case_study",
            indicators=[
                ("adjustment_ratio", "adjustment_ratio_better_model"),
                ("adjustment_tail", "adjustment_tail_better_model"),
                ("decoupling_gap", "decoupling_better_model"),
            ],
            prism_label=prism_label,
            baseline_label=baseline_label,
        )
    )
    return pd.DataFrame(rows)
