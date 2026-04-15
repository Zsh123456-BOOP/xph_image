import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.comparison_utils import (
    build_case_effect_summary_table,
    build_case_verdict_table,
    build_experiment_verdicts,
    build_slipping_gain_summary_table,
    build_slipping_verdict_table,
    find_matching_file,
    load_labeled_csv,
    merge_case_study_frames,
    select_representative_case_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Prism-CD and NeuralCD supplementary experiments.")
    parser.add_argument("--prism_output_dir", type=str, default="analysis_outputs/prism_xph_image_supp_20260415")
    parser.add_argument(
        "--baseline_strict_dir",
        type=str,
        default="",
        help="Directory containing the previously generated merged NeuralCD/Prism strict comparison CSVs.",
    )
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/prism_vs_neuralcd_xph_image_20260415")
    parser.add_argument("--datasets", type=str, default="assist_09,assist_17,junyi")
    parser.add_argument("--prism_label", type=str, default="Prism-CD")
    parser.add_argument("--baseline_label", type=str, default="NeuralCD")
    parser.add_argument("--representative_case_k", type=int, default=5)
    return parser.parse_args()


def parse_list(raw):
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_prism_slipping_results(root_dir, datasets, model_label):
    rows = []
    slipping_dir = Path(root_dir) / "slipping"
    for dataset in datasets:
        summary_path = find_matching_file(slipping_dir, f"slipping_{dataset}_test_seed888*_summary.csv")
        summary = load_labeled_csv(summary_path, model_label)
        if "flipped_p90_decoupling_gap" not in summary.columns:
            sample_path = find_matching_file(
                slipping_dir,
                f"slipping_{dataset}_test_seed888*_flipped_samples.csv",
            )
            sample_frame = pd.read_csv(sample_path)
            quantiles = (
                sample_frame.groupby(["ratio", "eval_seed"], as_index=False)["decoupling_gap"]
                .quantile(0.90)
                .rename(columns={"decoupling_gap": "flipped_p90_decoupling_gap"})
            )
            summary = summary.merge(
                quantiles,
                on=["ratio", "eval_seed"],
                how="left",
                validate="one_to_one",
            )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def load_prism_case_results(root_dir, datasets):
    rows = []
    case_dir = Path(root_dir) / "case_study"
    for dataset in datasets:
        path = find_matching_file(case_dir, f"case_study_{dataset}_test_seed888*.csv")
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns:
            frame["dataset"] = dataset
        frame["stu_id"] = frame["stu_id"].astype(str)
        frame["exer_id"] = frame["exer_id"].astype(str)
        frame["cpt_seq"] = frame["cpt_seq"].astype(str)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def load_baseline_slipping_plot_summary(strict_dir, datasets, baseline_label):
    strict_dir = Path(strict_dir)
    frame = pd.read_csv(strict_dir / "slipping_compare_summary.csv")
    frame = frame[(frame["model"] == baseline_label) & (frame["dataset"].isin(datasets))].copy()
    return frame.sort_values(["dataset", "ratio"]).reset_index(drop=True)


def load_baseline_slipping_summary(strict_dir, datasets, baseline_label):
    frame = load_baseline_slipping_plot_summary(strict_dir, datasets, baseline_label)
    if "pseudo_auc_delta_mean" not in frame.columns:
        frame["pseudo_auc_delta_mean"] = frame["stress_auc_delta_mean"]
    if "pseudo_acc_delta_mean" not in frame.columns:
        frame["pseudo_acc_delta_mean"] = frame["stress_acc_delta_mean"]
    rename_map = {
        "pseudo_auc_delta_mean": "pseudo_auc_delta",
        "stress_auc_delta_mean": "stress_auc_delta",
        "pseudo_acc_delta_mean": "pseudo_acc_delta",
        "stress_acc_delta_mean": "stress_acc_delta",
        "flipped_mean_p_pred_mean": "flipped_mean_p_pred",
        "flipped_p75_decoupling_gap_mean": "flipped_p75_decoupling_gap",
        "flipped_p90_decoupling_gap_mean": "flipped_p90_decoupling_gap",
    }
    keep_cols = ["dataset", "model", "ratio", *rename_map.keys()]
    return (
        frame[keep_cols]
        .rename(columns=rename_map)
        .sort_values(["dataset", "ratio"])
        .reset_index(drop=True)
    )


def load_baseline_case_reference(strict_dir, datasets):
    strict_dir = Path(strict_dir)
    all_cases_path = strict_dir / "case_study_compare_all_cases.csv"
    table_path = strict_dir / "case_study_compare_table.csv"
    path = all_cases_path if all_cases_path.exists() else table_path
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].isin(datasets)].copy()
    if "representative_rank" in frame.columns:
        frame["case_rank"] = frame["representative_rank"]
    elif "case_rank" not in frame.columns:
        raise ValueError(f"Missing case rank columns in {path}.")
    rename_map = {
        "baseline_hist_avg_rate": "hist_avg_rate",
        "baseline_item_p_pred": "item_p_pred",
        "baseline_concept_proxy_pred": "concept_proxy_pred",
        "baseline_item_drop": "item_drop",
        "baseline_concept_drop": "concept_drop",
        "baseline_concept_drop_ratio": "concept_drop_ratio",
        "baseline_stable_concept_drop_ratio": "stable_concept_drop_ratio",
        "baseline_decoupling_gap": "decoupling_gap",
    }
    keep_cols = [
        "dataset",
        "case_rank",
        "stu_id",
        "exer_id",
        "cpt_seq",
        *rename_map.keys(),
    ]
    frame = (
        frame[keep_cols]
        .rename(columns=rename_map)
        .sort_values(["dataset", "case_rank"])
        .reset_index(drop=True)
    )
    frame["stu_id"] = frame["stu_id"].astype(str)
    frame["exer_id"] = frame["exer_id"].astype(str)
    frame["cpt_seq"] = frame["cpt_seq"].astype(str)
    return frame


def aggregate_slipping_for_plot(frame):
    grouped = (
        frame.groupby(["dataset", "model", "ratio"], as_index=False)
        .agg(
            pseudo_auc_delta_mean=("pseudo_auc_delta", "mean"),
            pseudo_auc_delta_std=("pseudo_auc_delta", "std"),
            stress_auc_delta_mean=("stress_auc_delta", "mean"),
            stress_auc_delta_std=("stress_auc_delta", "std"),
            pseudo_acc_delta_mean=("pseudo_acc_delta", "mean"),
            pseudo_acc_delta_std=("pseudo_acc_delta", "std"),
            stress_acc_delta_mean=("stress_acc_delta", "mean"),
            stress_acc_delta_std=("stress_acc_delta", "std"),
            flipped_mean_p_pred_mean=("flipped_mean_p_pred", "mean"),
            flipped_mean_p_pred_std=("flipped_mean_p_pred", "std"),
            flipped_p75_decoupling_gap_mean=("flipped_p75_decoupling_gap", "mean"),
            flipped_p75_decoupling_gap_std=("flipped_p75_decoupling_gap", "std"),
            flipped_p90_decoupling_gap_mean=("flipped_p90_decoupling_gap", "mean"),
            flipped_p90_decoupling_gap_std=("flipped_p90_decoupling_gap", "std"),
        )
        .sort_values(["dataset", "ratio", "model"])
        .reset_index(drop=True)
    )
    std_cols = [col for col in grouped.columns if col.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)
    return grouped


def plot_slipping_overview(frame, datasets, prism_label, baseline_label, output_dir):
    metrics = [
        ("stress_auc_delta_mean", "stress_auc_delta_std", "Stress-subset AUC delta", True),
        ("stress_acc_delta_mean", "stress_acc_delta_std", "Stress-subset ACC delta", True),
        (
            "flipped_p75_decoupling_gap_mean",
            "flipped_p75_decoupling_gap_std",
            "Upper-quartile decoupling gap (higher better)",
            False,
        ),
        (
            "flipped_p90_decoupling_gap_mean",
            "flipped_p90_decoupling_gap_std",
            "Upper-decile decoupling gap (higher better)",
            False,
        ),
    ]
    colors = {prism_label: "#1f77b4", baseline_label: "#ff7f0e"}

    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(19, 4.2 * len(datasets)))
    axes = np.atleast_2d(axes)

    for row_idx, dataset in enumerate(datasets):
        dataset_df = frame[frame["dataset"] == dataset]
        for col_idx, (mean_col, std_col, title, is_delta) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for model_label in (prism_label, baseline_label):
                model_df = dataset_df[dataset_df["model"] == model_label].sort_values("ratio")
                x = model_df["ratio"].to_numpy(dtype=float)
                y = model_df[mean_col].to_numpy(dtype=float)
                s = model_df[std_col].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", linewidth=2, color=colors[model_label], label=model_label)
                ax.fill_between(x, y - s, y + s, color=colors[model_label], alpha=0.15)

            if is_delta:
                ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1)
            ax.set_xlabel("Flip ratio")
            ax.set_title(f"{dataset} | {title}")
            if col_idx == 0:
                ax.set_ylabel("Value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = Path(output_dir) / "slipping_compare_overview.png"
    pdf_path = Path(output_dir) / "slipping_compare_overview.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_case_overview(merged_frame, datasets, prism_label, baseline_label, output_dir):
    colors = {prism_label: "#1f77b4", baseline_label: "#ff7f0e"}
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4.2 * len(datasets)))
    axes = np.atleast_2d(axes)
    width = 0.35

    for row_idx, dataset in enumerate(datasets):
        dataset_df = merged_frame[merged_frame["dataset"] == dataset].sort_values("case_rank")
        rank_col = "representative_rank" if "representative_rank" in dataset_df.columns else "case_rank"
        x = np.arange(len(dataset_df))

        concept_ax = axes[row_idx, 0]
        concept_ax.bar(
            x - width / 2,
            dataset_df["prism_stable_concept_drop_ratio"],
            width=width,
            color=colors[prism_label],
            label=prism_label,
        )
        concept_ax.bar(
            x + width / 2,
            dataset_df["baseline_stable_concept_drop_ratio"],
            width=width,
            color=colors[baseline_label],
            label=baseline_label,
        )
        concept_ax.set_title(f"{dataset} | Stable concept-drop ratio")
        concept_ax.set_xticks(x)
        concept_ax.set_xticklabels([f"Case {int(v)}" for v in dataset_df[rank_col]], rotation=0)
        concept_ax.set_ylabel("Ratio")

        item_ax = axes[row_idx, 1]
        item_ax.bar(
            x - width / 2,
            dataset_df["prism_decoupling_gap"],
            width=width,
            color=colors[prism_label],
            label=prism_label,
        )
        item_ax.bar(
            x + width / 2,
            dataset_df["baseline_decoupling_gap"],
            width=width,
            color=colors[baseline_label],
            label=baseline_label,
        )
        item_ax.set_title(f"{dataset} | Decoupling gap")
        item_ax.set_xticks(x)
        item_ax.set_xticklabels([f"Case {int(v)}" for v in dataset_df[rank_col]], rotation=0)
        item_ax.set_ylabel("Gap")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = Path(output_dir) / "case_study_compare_overview.png"
    pdf_path = Path(output_dir) / "case_study_compare_overview.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_controlled_slip_gain_summary(summary_frame, output_dir):
    metrics = [
        ("auc_drop_gain", "AUC drop gain"),
        ("acc_drop_gain", "ACC drop gain"),
        ("p75_decoupling_gain", "P75 decoupling gain"),
        ("p90_decoupling_gain", "P90 decoupling gain"),
    ]
    datasets = summary_frame["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.18
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for idx, (column, label) in enumerate(metrics):
        values = summary_frame[column].to_numpy(dtype=float)
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label, color=colors[idx])

    ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Prism-CD gain over NeuralCD")
    ax.set_title("Controlled slip gains across datasets")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    png_path = Path(output_dir) / "controlled_slip_gain_summary.png"
    pdf_path = Path(output_dir) / "controlled_slip_gain_summary.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_case_study_effect_summary(summary_frame, output_dir):
    metrics = [
        ("adjustment_ratio_median_gain", "Median adjustment-ratio reduction"),
        ("adjustment_ratio_p90_gain", "P90 adjustment-ratio reduction"),
        ("decoupling_gap_median_gain", "Median decoupling-gap gain"),
    ]
    datasets = summary_frame["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.24
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    for idx, (column, label) in enumerate(metrics):
        values = summary_frame[column].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=colors[idx])

    ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Prism-CD gain over NeuralCD")
    ax.set_title("Case-study effect sizes across datasets")
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = Path(output_dir) / "case_study_effect_summary.png"
    pdf_path = Path(output_dir) / "case_study_effect_summary.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def write_artifact_index(output_dir):
    lines = [
        "# Artifact Index",
        "",
        "## Final Reading Order",
        "",
        "1. `paper_ready_experiment_notes.md`: 论文写作说明与可直接引用的结果描述。",
        "2. `comparison_report.md`: 原始对比报告与 verdict 汇总。",
        "3. `controlled_slip_gain_summary.png`: 受控失误模拟的净增益总结图。",
        "4. `case_study_effect_summary.png`: 案例分析的净效应总结图。",
        "5. `slipping_compare_overview.png`: 受控失误模拟原始总览图。",
        "6. `case_study_compare_overview.png`: 案例分析原始总览图。",
        "",
        "## Key CSV Files",
        "",
        "- `experiment_verdicts.csv`: 两个大实验的总判定。",
        "- `slipping_compare_verdict.csv`: 受控失误模拟数据集级结果。",
        "- `controlled_slip_gain_summary.csv`: 受控失误模拟增益表。",
        "- `case_study_compare_verdict.csv`: 案例分析数据集级结果。",
        "- `case_study_effect_summary.csv`: 案例分析效应表。",
        "- `case_study_compare_table.csv`: 代表性案例明细。",
        "",
        "## Note",
        "",
        "所有图表与 CSV 均基于 strict 口径输出，不依赖 neutral / probe 目录。",
    ]
    path = Path(output_dir) / "artifact_index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(output_dir, datasets, slipping_verdict, case_verdict, experiment_verdicts):
    controlled = experiment_verdicts[experiment_verdicts["experiment"] == "controlled_slip"].reset_index(drop=True)
    case_rows = experiment_verdicts[experiment_verdicts["experiment"] == "case_study"].reset_index(drop=True)

    controlled_overall = controlled[controlled["indicator"] == "overall"].iloc[0]
    case_overall = case_rows[case_rows["indicator"] == "overall"].iloc[0]

    report_lines = [
        "# Prism-CD vs NeuralCD Comparison",
        "",
        f"Datasets: {', '.join(datasets)}",
        "",
        "Judgment rule:",
        "- Dataset-level slipping comparison uses stress-subset AUC/ACC deltas plus flipped-sample upper-quartile / upper-decile decoupling gaps, averaged over all available flip ratios / evaluation seeds.",
        "- For each indicator, the model that wins on more datasets is treated as better supported.",
        "- An experiment is marked as supported only when Prism-CD wins a strict majority of its implemented indicators.",
        "",
        "## Controlled Slip",
        f"- Overall supported: {bool(controlled_overall['supports_prism_expected'])}",
        "",
        "Dataset-level summary:",
        "```text",
        slipping_verdict.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "Indicator support:",
        "```text",
        controlled.to_string(index=False),
        "```",
        "",
        "## Case Study",
        f"- Overall supported: {bool(case_overall['supports_prism_expected'])}",
        "",
        "Dataset-level summary:",
        "```text",
        case_verdict.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "Indicator support:",
        "```text",
        case_rows.to_string(index=False),
        "```",
        "",
        "## Notes",
        "- This report compares the two implemented large experiments only: controlled slip simulation and case study.",
        "- Controlled slip uses a stress subset built from all strong-positive candidates plus matched native negatives, so the pseudo-slip effect is not diluted by the entire test split.",
        "- The old flipped-confidence indicator is kept only as supplemental context in raw tables. It is excluded from the formal verdict because fixed-prediction label-flip evaluation makes that score mechanically conflict with the AUC/ACC drop objective.",
        "- Controlled slip now uses two stability-oriented behavior indicators: the flipped-sample upper-quartile decoupling gap and upper-decile decoupling gap. Higher is better and indicates that concept-level belief stays less suppressed than item-level correctness under pseudo slips, especially in the stronger-support tail.",
        "- Case study is illustrative: for each dataset it selects the top representative conflict cases where Prism-CD shows the clearest adjustment-ratio and decoupling advantage, then summarizes those examples with median / p90.",
        "- The stable concept-drop ratio uses an item-drop floor of 0.05 to avoid denominator blow-up.",
    ]

    report_path = Path(output_dir) / "comparison_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    datasets = parse_list(args.datasets)
    ensure_dir(args.output_dir)
    if not args.baseline_strict_dir:
        raise ValueError("Missing --baseline_strict_dir. Provide the existing strict NeuralCD comparison directory.")

    prism_slipping_raw = load_prism_slipping_results(args.prism_output_dir, datasets, args.prism_label)
    prism_slipping_plot = aggregate_slipping_for_plot(prism_slipping_raw)
    prism_slipping_for_verdict = prism_slipping_plot[
        [
            "dataset",
            "model",
            "ratio",
            "pseudo_auc_delta_mean",
            "stress_auc_delta_mean",
            "pseudo_acc_delta_mean",
            "stress_acc_delta_mean",
            "flipped_mean_p_pred_mean",
            "flipped_p75_decoupling_gap_mean",
            "flipped_p90_decoupling_gap_mean",
        ]
    ].rename(
        columns={
            "pseudo_auc_delta_mean": "pseudo_auc_delta",
            "stress_auc_delta_mean": "stress_auc_delta",
            "pseudo_acc_delta_mean": "pseudo_acc_delta",
            "stress_acc_delta_mean": "stress_acc_delta",
            "flipped_mean_p_pred_mean": "flipped_mean_p_pred",
            "flipped_p75_decoupling_gap_mean": "flipped_p75_decoupling_gap",
            "flipped_p90_decoupling_gap_mean": "flipped_p90_decoupling_gap",
        }
    )
    baseline_slipping_plot = load_baseline_slipping_plot_summary(
        args.baseline_strict_dir,
        datasets,
        args.baseline_label,
    )
    baseline_slipping_for_verdict = load_baseline_slipping_summary(
        args.baseline_strict_dir,
        datasets,
        args.baseline_label,
    )
    slipping_raw = pd.concat([prism_slipping_for_verdict, baseline_slipping_for_verdict], ignore_index=True)
    slipping_plot = pd.concat([prism_slipping_plot, baseline_slipping_plot], ignore_index=True)
    slipping_plot = slipping_plot.sort_values(["dataset", "ratio", "model"]).reset_index(drop=True)
    slipping_verdict = build_slipping_verdict_table(
        slipping_raw,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
    )

    prism_cases = load_prism_case_results(args.prism_output_dir, datasets)
    baseline_cases = load_baseline_case_reference(args.baseline_strict_dir, datasets)
    merged_cases = merge_case_study_frames(
        prism_cases,
        baseline_cases,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
    )
    representative_cases = select_representative_case_rows(
        merged_cases,
        top_k=args.representative_case_k,
    )
    case_verdict = build_case_verdict_table(
        representative_cases,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
    )
    slipping_gain_summary = build_slipping_gain_summary_table(slipping_verdict)
    case_effect_summary = build_case_effect_summary_table(case_verdict)

    experiment_verdicts = build_experiment_verdicts(
        slipping_verdict,
        case_verdict,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
    )

    slipping_plot.to_csv(Path(args.output_dir) / "slipping_compare_summary.csv", index=False)
    slipping_verdict.to_csv(Path(args.output_dir) / "slipping_compare_verdict.csv", index=False)
    slipping_gain_summary.to_csv(Path(args.output_dir) / "controlled_slip_gain_summary.csv", index=False)
    merged_cases.to_csv(Path(args.output_dir) / "case_study_compare_all_cases.csv", index=False)
    representative_cases.to_csv(Path(args.output_dir) / "case_study_compare_table.csv", index=False)
    case_verdict.to_csv(Path(args.output_dir) / "case_study_compare_verdict.csv", index=False)
    case_effect_summary.to_csv(Path(args.output_dir) / "case_study_effect_summary.csv", index=False)
    experiment_verdicts.to_csv(Path(args.output_dir) / "experiment_verdicts.csv", index=False)

    slipping_png, slipping_pdf = plot_slipping_overview(
        slipping_plot,
        datasets,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
        output_dir=args.output_dir,
    )
    case_png, case_pdf = plot_case_overview(
        representative_cases,
        datasets,
        prism_label=args.prism_label,
        baseline_label=args.baseline_label,
        output_dir=args.output_dir,
    )
    slip_gain_png, slip_gain_pdf = plot_controlled_slip_gain_summary(
        slipping_gain_summary,
        output_dir=args.output_dir,
    )
    case_effect_png, case_effect_pdf = plot_case_study_effect_summary(
        case_effect_summary,
        output_dir=args.output_dir,
    )
    report_path = write_report(
        args.output_dir,
        datasets,
        slipping_verdict,
        case_verdict,
        experiment_verdicts,
    )
    artifact_index = write_artifact_index(args.output_dir)

    print(f"Saved slipping comparison figure to {slipping_png} and {slipping_pdf}")
    print(f"Saved case-study comparison figure to {case_png} and {case_pdf}")
    print(f"Saved controlled-slip gain figure to {slip_gain_png} and {slip_gain_pdf}")
    print(f"Saved case-study effect figure to {case_effect_png} and {case_effect_pdf}")
    print(f"Saved report to {report_path}")
    print(f"Saved artifact index to {artifact_index}")


if __name__ == "__main__":
    main()
