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

from analysis.hparam_sensitivity_utils import (
    PRISM_BASE_CONFIGS,
    build_hparam_result_summary,
)

HPARAM_DISPLAY_NAMES = {
    "ortho_weight": "Orthogonal Weight",
    "dropout": "Dropout",
    "embedding_dim": "Embedding Dimension",
}

HPARAM_ORDER = ["ortho_weight", "dropout", "embedding_dim"]


def parse_list(raw):
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Prism-CD hyperparameter sensitivity results.")
    parser.add_argument("--results_csv", type=str, default="results/all_datasets_results.csv")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs/prism_hparam_sensitivity_20260413")
    parser.add_argument("--datasets", type=str, default="assist_09,assist_17,junyi")
    parser.add_argument("--hparams", type=str, default="ortho_weight,dropout,embedding_dim")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_value_label(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def annotate_best(ax, xs, ys, color, label):
    if len(xs) == 0:
        return
    best_idx = int(np.nanargmax(ys))
    ax.scatter([xs[best_idx]], [ys[best_idx]], color=color, marker="*", s=160, zorder=5)
    ax.annotate(
        f"{label} best={ys[best_idx]:.4f}",
        xy=(xs[best_idx], ys[best_idx]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=color,
    )


def build_best_summary(summary_df):
    rows = []
    for (hparam, dataset), subset in summary_df.groupby(["hparam", "dataset"], sort=True):
        subset = subset.sort_values("value").reset_index(drop=True)
        best_idx = int(subset["test_auc"].astype(float).idxmax())
        best_row = subset.loc[best_idx]
        default_value = PRISM_BASE_CONFIGS[dataset][hparam]
        default_subset = subset[np.isclose(subset["value"].astype(float), float(default_value))]
        if default_subset.empty:
            default_row = subset.iloc[0]
        else:
            default_row = default_subset.iloc[0]
        rows.append(
            {
                "hparam": hparam,
                "dataset": dataset,
                "default_value": default_value,
                "best_value_by_auc": best_row["value"],
                "default_auc": default_row["test_auc"],
                "best_auc": best_row["test_auc"],
                "auc_gain_vs_default": float(best_row["test_auc"]) - float(default_row["test_auc"]),
                "default_acc": default_row["test_acc"],
                "best_acc": best_row["test_acc"],
                "acc_gain_vs_default": float(best_row["test_acc"]) - float(default_row["test_acc"]),
                "default_rmse": default_row["test_rmse"],
                "best_rmse": best_row["test_rmse"],
                "rmse_delta_vs_default": float(best_row["test_rmse"]) - float(default_row["test_rmse"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["hparam", "dataset"]).reset_index(drop=True)


def build_gain_summary_frame(best_summary):
    gain_summary = best_summary.copy()
    gain_summary["best_value_label"] = gain_summary["best_value_by_auc"].apply(format_value_label)
    return (
        gain_summary[
            [
                "dataset",
                "hparam",
                "best_value_label",
                "default_value",
                "best_value_by_auc",
                "default_auc",
                "best_auc",
                "auc_gain_vs_default",
                "default_acc",
                "best_acc",
                "acc_gain_vs_default",
                "default_rmse",
                "best_rmse",
                "rmse_delta_vs_default",
            ]
        ]
        .sort_values(["dataset", "hparam"])
        .reset_index(drop=True)
    )


def plot_hparam_figure(summary_df, hparam, datasets, output_dir):
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"AUC": "#1f77b4", "ACC": "#ff7f0e"}
    display_name = HPARAM_DISPLAY_NAMES.get(hparam, hparam)

    for ax, dataset in zip(axes, datasets):
        subset = summary_df[(summary_df["hparam"] == hparam) & (summary_df["dataset"] == dataset)].sort_values("value")
        if subset.empty:
            ax.set_visible(False)
            continue
        xs = subset["value"].to_numpy(dtype=float)
        aucs = subset["test_auc"].to_numpy(dtype=float)
        accs = subset["test_acc"].to_numpy(dtype=float)
        default_value = float(PRISM_BASE_CONFIGS[dataset][hparam])

        ax.plot(xs, aucs, marker="o", linewidth=2.2, color=colors["AUC"], label="AUC")
        ax.plot(xs, accs, marker="s", linewidth=2.0, linestyle="--", color=colors["ACC"], label="ACC")
        ax.axvline(default_value, color="#666666", linestyle=":", linewidth=1.2, label="Default")
        annotate_best(ax, xs, aucs, colors["AUC"], "AUC")
        ax.set_title(dataset)
        ax.set_xlabel(display_name)
        ax.set_xticks(xs)
        ax.set_ylim(
            max(0.0, min(np.nanmin(aucs), np.nanmin(accs)) - 0.02),
            min(1.0, max(np.nanmax(aucs), np.nanmax(accs)) + 0.02),
        )

    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Prism-CD sensitivity to {display_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    png_path = Path(output_dir) / f"{hparam}_sensitivity.png"
    pdf_path = Path(output_dir) / f"{hparam}_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_gain_summary(gain_summary, datasets, output_dir):
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.34
    colors = {"auc": "#1f77b4", "acc": "#ff7f0e"}

    for ax, dataset in zip(axes, datasets):
        subset = gain_summary[gain_summary["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        subset["hparam"] = pd.Categorical(subset["hparam"], categories=HPARAM_ORDER, ordered=True)
        subset = subset.sort_values("hparam").reset_index(drop=True)
        x = np.arange(len(subset))

        ax.bar(
            x - width / 2,
            subset["auc_gain_vs_default"].to_numpy(dtype=float),
            width=width,
            color=colors["auc"],
            label="AUC gain",
        )
        ax.bar(
            x + width / 2,
            subset["acc_gain_vs_default"].to_numpy(dtype=float),
            width=width,
            color=colors["acc"],
            label="ACC gain",
        )
        ax.axhline(0.0, linestyle="--", color="#666666", linewidth=1.0)
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{HPARAM_DISPLAY_NAMES.get(row.hparam, row.hparam)}\n(best={row.best_value_label})"
                for row in subset.itertuples(index=False)
            ]
        )

    axes[0].set_ylabel("Gain vs default")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Prism-CD best-point gain over default", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    png_path = Path(output_dir) / "best_vs_default_gain_summary.png"
    pdf_path = Path(output_dir) / "best_vs_default_gain_summary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_notes(best_summary, gain_summary, output_dir):
    top_auc_row = gain_summary.sort_values("auc_gain_vs_default", ascending=False).iloc[0]
    top_acc_row = gain_summary.sort_values("acc_gain_vs_default", ascending=False).iloc[0]
    mean_gain = (
        gain_summary.groupby("hparam", as_index=False)[["auc_gain_vs_default", "acc_gain_vs_default"]]
        .mean()
        .sort_values("auc_gain_vs_default", ascending=False)
        .reset_index(drop=True)
    )
    most_auc_sensitive = mean_gain.iloc[0]

    lines = [
        "# Prism-CD 参数敏感性实验说明",
        "",
        "本文档汇总 Prism-CD 在三个数据集上的单因子参数敏感性实验结果。",
        "",
        "## Hyperparameters",
        "",
        "- `ortho_weight`",
        "- `dropout`",
        "- `embedding_dim`",
        "",
        "## Main Findings",
        "",
        f"- 按平均 AUC 增益看，`{most_auc_sensitive['hparam']}` 的波动最明显，其平均 AUC 增益为 {most_auc_sensitive['auc_gain_vs_default']:.4f}。",
        f"- 单项最大 AUC 增益出现在 `{top_auc_row['dataset']}` 的 `{top_auc_row['hparam']}`，默认值到最优值带来 {top_auc_row['auc_gain_vs_default']:.4f} 的 AUC 提升。",
        f"- 单项最大 ACC 增益出现在 `{top_acc_row['dataset']}` 的 `{top_acc_row['hparam']}`，默认值到最优值带来 {top_acc_row['acc_gain_vs_default']:.4f} 的 ACC 提升。",
        "- 从跨数据集一致性与 ACC 受益看，`ortho_weight` 更值得优先调节，其中 `assist_09` 的提升最明显。",
        "- `dropout` 只在 `assist_17` 上出现轻微改善，`assist_09` 和 `junyi` 的默认值已经处于最优点。",
        "- `embedding_dim` 的最优 AUC 在不同数据集上略有变化，但收益整体小于 `ortho_weight`，说明它更适合作为次级调节参数。",
        "",
        "## Best-by-AUC Summary",
        "",
        "```text",
        best_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "## Recommendation",
        "",
        "- 如果只允许优先调一个参数，建议先调 `ortho_weight`。",
        "- 如果目标是保持当前主实验口径稳定，默认配置整体已经足够稳，不建议仅凭单因子 sweep 直接改动全部正式实验配置。",
        "- `assist_17` 可以把 `dropout` 从默认值继续下探，但收益通常较小，应与正式对比实验重新联动验证。",
        "- `embedding_dim` 的变化对不同指标并不总是同向，除非后续进行联合调参，否则建议保持现有默认值。",
        "",
        "## Reading Guide",
        "",
        "- 主图同时展示 `AUC` 与 `ACC`。",
        "- 虚线表示当前默认配置位置。",
        "- 星号标出该数据集在当前超参上的最佳 AUC 点。",
        "- `best_vs_default_gain_summary.png` 用于展示默认配置到最佳点的净增益，更适合论文正文快速引用。",
        "- `RMSE` 收录在 summary CSV 中，供正文或附录写作使用。",
    ]
    path = Path(output_dir) / "prism_hparam_sensitivity_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_artifact_index(output_dir):
    lines = [
        "# Artifact Index",
        "",
        "- `prism_hparam_sensitivity_summary.csv`: sweep 明细结果。",
        "- `prism_hparam_sensitivity_best.csv`: 每个数据集/超参按 AUC 选出的最佳点。",
        "- `best_vs_default_gain_summary.png`: 默认值到最佳点的增益总览图。",
        "- `hyper_weight_sensitivity.png`: `hyper_weight` 敏感性主图。",
        "- `dropout_sensitivity.png`: `dropout` 敏感性主图。",
        "- `hidden_dim_sensitivity.png`: `hidden_dim` 敏感性主图。",
        "- `prism_hparam_sensitivity_notes.md`: 论文写作说明。",
    ]
    path = Path(output_dir) / "artifact_index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    datasets = parse_list(args.datasets)
    hparams = parse_list(args.hparams)

    result_frame = pd.read_csv(args.results_csv)
    summary = build_hparam_result_summary(result_frame)
    summary = summary[summary["dataset"].isin(datasets) & summary["hparam"].isin(hparams)].copy()
    if summary.empty:
        raise RuntimeError("No Prism hyperparameter sweep rows found in results CSV.")

    best_summary = build_best_summary(summary)
    gain_summary = build_gain_summary_frame(best_summary)
    summary_path = Path(args.output_dir) / "prism_hparam_sensitivity_summary.csv"
    best_path = Path(args.output_dir) / "prism_hparam_sensitivity_best.csv"
    gain_path = Path(args.output_dir) / "prism_hparam_gain_summary.csv"
    summary.to_csv(summary_path, index=False)
    best_summary.to_csv(best_path, index=False)
    gain_summary.to_csv(gain_path, index=False)

    generated = []
    for hparam in hparams:
        generated.extend(plot_hparam_figure(summary, hparam, datasets, args.output_dir))
    generated.extend(plot_gain_summary(gain_summary, datasets, args.output_dir))

    notes_path = write_notes(best_summary, gain_summary, args.output_dir)
    index_path = write_artifact_index(args.output_dir)

    print(f"Saved summary to {summary_path}")
    print(f"Saved best summary to {best_path}")
    print(f"Saved gain summary to {gain_path}")
    for path in generated:
        print(f"Saved figure to {path}")
    print(f"Saved notes to {notes_path}")
    print(f"Saved artifact index to {index_path}")


if __name__ == "__main__":
    main()
