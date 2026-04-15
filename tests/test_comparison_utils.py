import unittest

import pandas as pd

from analysis.comparison_utils import (
    build_case_effect_summary_table,
    build_case_verdict_table,
    build_experiment_verdicts,
    build_slipping_gain_summary_table,
    build_slipping_verdict_table,
    ensure_case_metrics,
    merge_case_study_frames,
    select_representative_case_rows,
)


class ComparisonUtilsTests(unittest.TestCase):
    def test_build_slipping_verdict_prefers_smaller_drops_and_stronger_tail_decoupling(self):
        frame = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "model": "Prism-CD",
                    "pseudo_auc_delta": -0.08,
                    "pseudo_acc_delta": -0.06,
                    "stress_auc_delta": -0.12,
                    "stress_acc_delta": -0.10,
                    "flipped_mean_p_pred": 0.90,
                    "flipped_p90_decoupling_gap": 0.08,
                    "flipped_p75_decoupling_gap": -0.03,
                },
                {
                    "dataset": "assist_09",
                    "model": "Prism-CD",
                    "pseudo_auc_delta": -0.09,
                    "pseudo_acc_delta": -0.07,
                    "stress_auc_delta": -0.14,
                    "stress_acc_delta": -0.11,
                    "flipped_mean_p_pred": 0.91,
                    "flipped_p90_decoupling_gap": 0.10,
                    "flipped_p75_decoupling_gap": -0.01,
                },
                {
                    "dataset": "assist_09",
                    "model": "NeuralCD",
                    "pseudo_auc_delta": -0.07,
                    "pseudo_acc_delta": -0.05,
                    "stress_auc_delta": -0.20,
                    "stress_acc_delta": -0.16,
                    "flipped_mean_p_pred": 0.84,
                    "flipped_p90_decoupling_gap": 0.02,
                    "flipped_p75_decoupling_gap": -0.08,
                },
                {
                    "dataset": "assist_09",
                    "model": "NeuralCD",
                    "pseudo_auc_delta": -0.08,
                    "pseudo_acc_delta": -0.06,
                    "stress_auc_delta": -0.22,
                    "stress_acc_delta": -0.18,
                    "flipped_mean_p_pred": 0.85,
                    "flipped_p90_decoupling_gap": 0.03,
                    "flipped_p75_decoupling_gap": -0.07,
                },
            ]
        )

        verdict = build_slipping_verdict_table(frame, prism_label="Prism-CD", baseline_label="NeuralCD")

        row = verdict.iloc[0]
        self.assertAlmostEqual(row["prism_full_auc_delta"], -0.085, places=6)
        self.assertAlmostEqual(row["baseline_full_auc_delta"], -0.075, places=6)
        self.assertAlmostEqual(row["prism_stress_auc_delta"], -0.13, places=6)
        self.assertAlmostEqual(row["baseline_stress_auc_delta"], -0.21, places=6)
        self.assertEqual(row["auc_drop_better_model"], "Prism-CD")
        self.assertEqual(row["acc_drop_better_model"], "Prism-CD")
        self.assertAlmostEqual(row["prism_flipped_decoupling_gap_p90"], 0.09, places=6)
        self.assertAlmostEqual(row["baseline_flipped_decoupling_gap_p90"], 0.025, places=6)
        self.assertEqual(row["tail_decoupling_better_model"], "Prism-CD")
        self.assertAlmostEqual(row["prism_flipped_decoupling_gap_p75"], -0.02, places=6)
        self.assertAlmostEqual(row["baseline_flipped_decoupling_gap_p75"], -0.075, places=6)
        self.assertEqual(row["knowledge_adjustment_better_model"], "Prism-CD")
        self.assertTrue(bool(row["supports_prism_expected"]))

    def test_merge_case_study_frames_and_build_case_verdict(self):
        prism_cases = pd.DataFrame(
            [
                {
                    "dataset": "junyi",
                    "case_rank": 1,
                    "stu_id": "s1",
                    "exer_id": "e1",
                    "cpt_seq": "1",
                    "hist_avg_rate": 0.90,
                    "item_p_pred": 0.62,
                    "concept_proxy_pred": 0.82,
                    "item_drop": 0.28,
                    "concept_drop": 0.08,
                    "concept_drop_ratio": 0.2857,
                    "stable_concept_drop_ratio": 0.2857,
                    "decoupling_gap": 0.20,
                },
                {
                    "dataset": "junyi",
                    "case_rank": 2,
                    "stu_id": "s2",
                    "exer_id": "e2",
                    "cpt_seq": "2",
                    "hist_avg_rate": 0.88,
                    "item_p_pred": 0.58,
                    "concept_proxy_pred": 0.78,
                    "item_drop": 0.30,
                    "concept_drop": 0.10,
                    "concept_drop_ratio": 0.3333,
                    "stable_concept_drop_ratio": 0.3333,
                    "decoupling_gap": 0.20,
                },
            ]
        )
        neural_cases = pd.DataFrame(
            [
                {
                    "dataset": "junyi",
                    "case_rank": 1,
                    "stu_id": "s1",
                    "exer_id": "e1",
                    "cpt_seq": "1",
                    "hist_avg_rate": 0.90,
                    "item_p_pred": 0.74,
                    "concept_proxy_pred": 0.79,
                    "item_drop": 0.16,
                    "concept_drop": 0.11,
                    "concept_drop_ratio": 0.6875,
                    "stable_concept_drop_ratio": 0.6875,
                    "decoupling_gap": 0.05,
                },
                {
                    "dataset": "junyi",
                    "case_rank": 2,
                    "stu_id": "s2",
                    "exer_id": "e2",
                    "cpt_seq": "2",
                    "hist_avg_rate": 0.88,
                    "item_p_pred": 0.73,
                    "concept_proxy_pred": 0.76,
                    "item_drop": 0.15,
                    "concept_drop": 0.12,
                    "concept_drop_ratio": 0.8,
                    "stable_concept_drop_ratio": 0.8,
                    "decoupling_gap": 0.03,
                },
            ]
        )

        merged = merge_case_study_frames(
            prism_cases,
            neural_cases,
            prism_label="Prism-CD",
            baseline_label="NeuralCD",
        )
        verdict = build_case_verdict_table(
            merged,
            prism_label="Prism-CD",
            baseline_label="NeuralCD",
        )

        self.assertEqual(merged["case_rank"].tolist(), [1, 2])
        row = verdict.iloc[0]
        self.assertEqual(row["adjustment_ratio_better_model"], "Prism-CD")
        self.assertEqual(row["adjustment_tail_better_model"], "Prism-CD")
        self.assertEqual(row["decoupling_better_model"], "Prism-CD")
        self.assertTrue(bool(row["supports_prism_expected"]))

    def test_select_representative_case_rows_prefers_prism_advantage_cases(self):
        merged = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "case_rank": 1,
                    "prism_stable_concept_drop_ratio": 0.3,
                    "baseline_stable_concept_drop_ratio": 1.2,
                    "prism_decoupling_gap": 0.5,
                    "baseline_decoupling_gap": -0.1,
                },
                {
                    "dataset": "assist_09",
                    "case_rank": 2,
                    "prism_stable_concept_drop_ratio": 0.4,
                    "baseline_stable_concept_drop_ratio": 1.0,
                    "prism_decoupling_gap": 0.4,
                    "baseline_decoupling_gap": 0.0,
                },
                {
                    "dataset": "assist_09",
                    "case_rank": 3,
                    "prism_stable_concept_drop_ratio": 1.5,
                    "baseline_stable_concept_drop_ratio": 0.8,
                    "prism_decoupling_gap": -0.2,
                    "baseline_decoupling_gap": -0.1,
                },
            ]
        )

        selected = select_representative_case_rows(merged, top_k=2)

        self.assertEqual(selected["case_rank"].tolist(), [1, 2])
        self.assertEqual(selected["representative_rank"].tolist(), [1, 2])
        self.assertTrue((selected["representative_score"] > 0).all())

    def test_ensure_case_metrics_backfills_legacy_columns_with_ratio_floor(self):
        legacy = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "case_rank": 1,
                    "stu_id": "s1",
                    "exer_id": "e1",
                    "cpt_seq": "1_2",
                    "hist_avg_rate": 0.90,
                    "item_p_pred": 0.60,
                    "concept_proxy_pred": 0.78,
                }
            ]
        )

        normalized = ensure_case_metrics(legacy)

        self.assertAlmostEqual(normalized.loc[0, "item_drop"], 0.30, places=6)
        self.assertAlmostEqual(normalized.loc[0, "concept_drop"], 0.12, places=6)
        self.assertAlmostEqual(normalized.loc[0, "concept_drop_ratio"], 0.4, places=6)
        self.assertAlmostEqual(normalized.loc[0, "stable_concept_drop_ratio"], 0.4, places=6)
        self.assertAlmostEqual(normalized.loc[0, "decoupling_gap"], 0.18, places=6)

        floored = ensure_case_metrics(
            pd.DataFrame(
                [
                    {
                        "dataset": "assist_17",
                        "case_rank": 1,
                        "stu_id": "s2",
                        "exer_id": "e9",
                        "cpt_seq": "9",
                        "hist_avg_rate": 0.90,
                        "item_p_pred": 0.88,
                        "concept_proxy_pred": 0.70,
                    }
                ]
            )
        )
        self.assertAlmostEqual(floored.loc[0, "concept_drop_ratio"], 10.0, places=6)
        self.assertAlmostEqual(floored.loc[0, "stable_concept_drop_ratio"], 4.0, places=6)

    def test_build_experiment_verdicts_requires_majority_support(self):
        slipping_verdict = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "auc_drop_better_model": "Prism-CD",
                    "acc_drop_better_model": "Prism-CD",
                    "tail_decoupling_better_model": "Prism-CD",
                    "knowledge_adjustment_better_model": "Prism-CD",
                },
                {
                    "dataset": "assist_17",
                    "auc_drop_better_model": "NeuralCD",
                    "acc_drop_better_model": "NeuralCD",
                    "tail_decoupling_better_model": "Prism-CD",
                    "knowledge_adjustment_better_model": "Prism-CD",
                },
                {
                    "dataset": "junyi",
                    "auc_drop_better_model": "Prism-CD",
                    "acc_drop_better_model": "Prism-CD",
                    "tail_decoupling_better_model": "Prism-CD",
                    "knowledge_adjustment_better_model": "Prism-CD",
                },
            ]
        )
        case_verdict = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "adjustment_ratio_better_model": "Prism-CD",
                    "adjustment_tail_better_model": "Prism-CD",
                    "decoupling_better_model": "Prism-CD",
                },
                {
                    "dataset": "assist_17",
                    "adjustment_ratio_better_model": "Prism-CD",
                    "adjustment_tail_better_model": "NeuralCD",
                    "decoupling_better_model": "Prism-CD",
                },
                {
                    "dataset": "junyi",
                    "adjustment_ratio_better_model": "Prism-CD",
                    "adjustment_tail_better_model": "Prism-CD",
                    "decoupling_better_model": "NeuralCD",
                },
            ]
        )

        verdicts = build_experiment_verdicts(
            slipping_verdict,
            case_verdict,
            prism_label="Prism-CD",
            baseline_label="NeuralCD",
        )

        controlled_overall = verdicts[
            (verdicts["experiment"] == "controlled_slip")
            & (verdicts["indicator"] == "overall")
        ].iloc[0]
        case_overall = verdicts[
            (verdicts["experiment"] == "case_study")
            & (verdicts["indicator"] == "overall")
        ].iloc[0]
        adjustment_row = verdicts[
            (verdicts["experiment"] == "controlled_slip")
            & (verdicts["indicator"] == "knowledge_adjustment")
        ].iloc[0]

        self.assertTrue(bool(controlled_overall["supports_prism_expected"]))
        self.assertTrue(bool(case_overall["supports_prism_expected"]))
        self.assertTrue(bool(adjustment_row["supports_prism_expected"]))

    def test_build_slipping_gain_summary_table_uses_positive_gain_direction(self):
        verdict = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "prism_stress_auc_delta": -0.06,
                    "baseline_stress_auc_delta": -0.08,
                    "prism_stress_acc_delta": -0.09,
                    "baseline_stress_acc_delta": -0.10,
                    "prism_flipped_decoupling_gap_p75": -0.01,
                    "baseline_flipped_decoupling_gap_p75": -0.03,
                    "prism_flipped_decoupling_gap_p90": 0.12,
                    "baseline_flipped_decoupling_gap_p90": 0.02,
                }
            ]
        )

        summary = build_slipping_gain_summary_table(verdict)
        row = summary.iloc[0]

        self.assertAlmostEqual(row["auc_drop_gain"], 0.02, places=6)
        self.assertAlmostEqual(row["acc_drop_gain"], 0.01, places=6)
        self.assertAlmostEqual(row["p75_decoupling_gain"], 0.02, places=6)
        self.assertAlmostEqual(row["p90_decoupling_gain"], 0.10, places=6)

    def test_build_case_effect_summary_table_uses_positive_effect_direction(self):
        verdict = pd.DataFrame(
            [
                {
                    "dataset": "junyi",
                    "prism_adjustment_ratio_median": 0.8,
                    "baseline_adjustment_ratio_median": 1.1,
                    "prism_adjustment_ratio_p90": 1.2,
                    "baseline_adjustment_ratio_p90": 2.0,
                    "prism_decoupling_gap_median": 0.06,
                    "baseline_decoupling_gap_median": -0.02,
                }
            ]
        )

        summary = build_case_effect_summary_table(verdict)
        row = summary.iloc[0]

        self.assertAlmostEqual(row["adjustment_ratio_median_gain"], 0.3, places=6)
        self.assertAlmostEqual(row["adjustment_ratio_p90_gain"], 0.8, places=6)
        self.assertAlmostEqual(row["decoupling_gap_median_gain"], 0.08, places=6)


if __name__ == "__main__":
    unittest.main()
