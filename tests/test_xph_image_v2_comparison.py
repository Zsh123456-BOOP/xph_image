import unittest

import pandas as pd

from analysis.run_prism_neuralcd_comparison import (
    build_verdict_ready_slipping_frame,
    resolve_baseline_source_mode,
    resolve_slipping_acc_columns,
)


class XphImageV2ComparisonTests(unittest.TestCase):
    def test_resolve_baseline_source_mode_prefers_local_output_dir(self):
        mode = resolve_baseline_source_mode(
            baseline_output_dir="analysis_outputs/neuralcd_xph_image_supp_v2",
            baseline_strict_dir="analysis_outputs/old_strict_dir",
        )

        self.assertEqual(mode, "output_dir")

    def test_resolve_baseline_source_mode_falls_back_to_strict_dir(self):
        mode = resolve_baseline_source_mode(
            baseline_output_dir="",
            baseline_strict_dir="analysis_outputs/old_strict_dir",
        )

        self.assertEqual(mode, "strict_dir")

    def test_resolve_slipping_acc_columns_supports_new_metrics(self):
        self.assertEqual(resolve_slipping_acc_columns("raw")[-1], "ACC")
        self.assertEqual(resolve_slipping_acc_columns("calibrated")[-1], "Calibrated ACC")
        self.assertEqual(resolve_slipping_acc_columns("balanced")[-1], "Balanced ACC")

    def test_build_verdict_ready_slipping_frame_can_use_calibrated_acc(self):
        plot_frame = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "model": "Prism-CD",
                    "ratio": 0.2,
                    "pseudo_auc_delta_mean": -0.03,
                    "stress_auc_delta_mean": -0.05,
                    "pseudo_acc_delta_mean": -0.10,
                    "stress_acc_delta_mean": -0.10,
                    "pseudo_calibrated_acc_delta_mean": -0.04,
                    "stress_calibrated_acc_delta_mean": -0.06,
                    "pseudo_balanced_acc_delta_mean": -0.02,
                    "stress_balanced_acc_delta_mean": -0.03,
                    "flipped_mean_p_pred_mean": 0.8,
                    "flipped_p75_decoupling_gap_mean": -0.1,
                    "flipped_p90_decoupling_gap_mean": -0.05,
                }
            ]
        )

        calibrated = build_verdict_ready_slipping_frame(plot_frame, acc_metric="calibrated")
        balanced = build_verdict_ready_slipping_frame(plot_frame, acc_metric="balanced")

        self.assertAlmostEqual(calibrated.loc[0, "pseudo_acc_delta"], -0.04, places=6)
        self.assertAlmostEqual(calibrated.loc[0, "stress_acc_delta"], -0.06, places=6)
        self.assertAlmostEqual(balanced.loc[0, "pseudo_acc_delta"], -0.02, places=6)
        self.assertAlmostEqual(balanced.loc[0, "stress_acc_delta"], -0.03, places=6)


if __name__ == "__main__":
    unittest.main()
