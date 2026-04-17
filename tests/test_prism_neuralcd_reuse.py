import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.run_prism_neuralcd_comparison import (
    merge_case_results_with_fallback,
    load_baseline_case_reference,
    load_baseline_slipping_summary,
    load_output_dir_case_results,
    load_output_dir_slipping_results,
)


class PrismNeuralCDReuseTests(unittest.TestCase):
    def test_load_output_dir_slipping_results_derives_tail_gap_from_flipped_samples(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            slipping_dir = output_dir / "slipping"
            slipping_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "dataset": "assist_09",
                        "ratio": 0.2,
                        "eval_seed": 888,
                        "pseudo_auc_delta": -0.03,
                        "stress_auc_delta": -0.02,
                        "pseudo_acc_delta": -0.04,
                        "stress_acc_delta": -0.03,
                        "flipped_mean_p_pred": 0.71,
                        "flipped_p75_decoupling_gap": 0.10,
                    }
                ]
            ).to_csv(slipping_dir / "slipping_assist_09_test_seed888_summary.csv", index=False)
            pd.DataFrame(
                [
                    {"ratio": 0.2, "eval_seed": 888, "decoupling_gap": 0.10},
                    {"ratio": 0.2, "eval_seed": 888, "decoupling_gap": 0.35},
                    {"ratio": 0.2, "eval_seed": 888, "decoupling_gap": 0.50},
                ]
            ).to_csv(slipping_dir / "slipping_assist_09_test_seed888_flipped_samples.csv", index=False)

            frame = load_output_dir_slipping_results(output_dir, ["assist_09"], "NeuralCD")

        self.assertEqual(frame["model"].tolist(), ["NeuralCD"])
        self.assertEqual(frame["dataset"].tolist(), ["assist_09"])
        self.assertAlmostEqual(frame.loc[0, "flipped_p90_decoupling_gap"], 0.47, places=6)

    def test_load_output_dir_case_results_reads_local_case_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            case_dir = output_dir / "case_study"
            case_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "stu_id": 7,
                        "exer_id": 11,
                        "cpt_seq": "2",
                        "hist_avg_rate": 0.92,
                        "item_p_pred": 0.70,
                        "concept_proxy_pred": 0.81,
                    }
                ]
            ).to_csv(case_dir / "case_study_assist_17_test_seed888.csv", index=False)

            frame = load_output_dir_case_results(output_dir, ["assist_17"])

        self.assertEqual(frame["dataset"].tolist(), ["assist_17"])
        self.assertEqual(frame["stu_id"].tolist(), ["7"])
        self.assertEqual(frame["exer_id"].tolist(), ["11"])

    def test_load_baseline_slipping_summary_extracts_only_neuralcd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            strict_dir = Path(tmp_dir)
            pd.DataFrame(
                [
                    {
                        "dataset": "assist_09",
                        "model": "NeuralCD",
                        "ratio": 0.1,
                        "stress_auc_delta_mean": -0.04,
                        "stress_auc_delta_std": 0.001,
                        "stress_acc_delta_mean": -0.05,
                        "stress_acc_delta_std": 0.002,
                        "flipped_mean_p_pred_mean": 0.94,
                        "flipped_mean_p_pred_std": 0.01,
                        "flipped_p75_decoupling_gap_mean": -0.02,
                        "flipped_p75_decoupling_gap_std": 0.003,
                        "flipped_p90_decoupling_gap_mean": 0.01,
                        "flipped_p90_decoupling_gap_std": 0.004,
                    },
                    {
                        "dataset": "assist_09",
                        "model": "Prism-CD",
                        "ratio": 0.1,
                        "stress_auc_delta_mean": -0.03,
                        "stress_auc_delta_std": 0.001,
                        "stress_acc_delta_mean": -0.04,
                        "stress_acc_delta_std": 0.002,
                        "flipped_mean_p_pred_mean": 0.82,
                        "flipped_mean_p_pred_std": 0.02,
                        "flipped_p75_decoupling_gap_mean": 0.00,
                        "flipped_p75_decoupling_gap_std": 0.004,
                        "flipped_p90_decoupling_gap_mean": 0.09,
                        "flipped_p90_decoupling_gap_std": 0.010,
                    },
                ]
            ).to_csv(strict_dir / "slipping_compare_summary.csv", index=False)

            baseline = load_baseline_slipping_summary(strict_dir, ["assist_09"], "NeuralCD")

        self.assertEqual(baseline["model"].tolist(), ["NeuralCD"])
        self.assertEqual(baseline["dataset"].tolist(), ["assist_09"])
        self.assertAlmostEqual(baseline.loc[0, "stress_auc_delta"], -0.04, places=6)
        self.assertAlmostEqual(baseline.loc[0, "flipped_p90_decoupling_gap"], 0.01, places=6)

    def test_load_baseline_case_reference_renames_baseline_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            strict_dir = Path(tmp_dir)
            pd.DataFrame(
                [
                    {
                        "dataset": "assist_17",
                        "case_rank": 1,
                        "stu_id": 7,
                        "exer_id": 11,
                        "cpt_seq": "2",
                        "baseline_hist_avg_rate": 0.9,
                        "baseline_item_p_pred": 0.8,
                        "baseline_concept_proxy_pred": 0.7,
                        "baseline_item_drop": 0.1,
                        "baseline_concept_drop": 0.2,
                        "baseline_concept_drop_ratio": 2.0,
                        "baseline_stable_concept_drop_ratio": 2.0,
                        "baseline_decoupling_gap": -0.1,
                    }
                ]
            ).to_csv(strict_dir / "case_study_compare_table.csv", index=False)

            baseline = load_baseline_case_reference(strict_dir, ["assist_17"])

        self.assertEqual(
            baseline.columns.tolist(),
            [
                "dataset",
                "case_rank",
                "stu_id",
                "exer_id",
                "cpt_seq",
                "hist_avg_rate",
                "item_p_pred",
                "concept_proxy_pred",
                "item_drop",
                "concept_drop",
                "concept_drop_ratio",
                "stable_concept_drop_ratio",
                "decoupling_gap",
            ],
        )
        self.assertEqual(baseline.loc[0, "stu_id"], "7")
        self.assertAlmostEqual(baseline.loc[0, "stable_concept_drop_ratio"], 2.0, places=6)

    def test_merge_case_results_with_fallback_uses_rank_when_no_exact_overlap(self):
        prism_cases = pd.DataFrame(
            [
                {
                    "dataset": "junyi",
                    "case_rank": 1,
                    "stu_id": "1",
                    "exer_id": "101",
                    "cpt_seq": "2",
                    "hist_avg_rate": 0.9,
                    "item_p_pred": 0.6,
                    "concept_proxy_pred": 0.8,
                    "item_drop": 0.3,
                    "concept_drop": 0.1,
                    "concept_drop_ratio": 0.3333,
                    "stable_concept_drop_ratio": 0.3333,
                    "decoupling_gap": 0.2,
                }
            ]
        )
        baseline_cases = pd.DataFrame(
            [
                {
                    "dataset": "junyi",
                    "case_rank": 1,
                    "stu_id": "999",
                    "exer_id": "888",
                    "cpt_seq": "7",
                    "hist_avg_rate": 0.9,
                    "item_p_pred": 0.8,
                    "concept_proxy_pred": 0.7,
                    "item_drop": 0.1,
                    "concept_drop": 0.2,
                    "concept_drop_ratio": 2.0,
                    "stable_concept_drop_ratio": 2.0,
                    "decoupling_gap": -0.1,
                }
            ]
        )

        merged = merge_case_results_with_fallback(
            prism_cases,
            baseline_cases,
            prism_label="Prism-CD",
            baseline_label="NeuralCD",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "dataset"], "junyi")
        self.assertEqual(merged.loc[0, "reference_mode"], "rank_fallback")
        self.assertAlmostEqual(merged.loc[0, "prism_decoupling_gap"], 0.2, places=6)
        self.assertAlmostEqual(merged.loc[0, "baseline_decoupling_gap"], -0.1, places=6)


if __name__ == "__main__":
    unittest.main()
