import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.run_prism_hparam_sensitivity_analysis import (
    build_best_summary,
    build_gain_summary_frame,
    plot_gain_summary,
    write_notes,
)


class PrismHparamAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.summary = pd.DataFrame(
            [
                {
                    "dataset": "assist_09",
                    "hparam": "ortho_weight",
                    "value": 0.5,
                    "tag": "hparam_assist_09_ortho_weight_0p5",
                    "test_auc": 0.7903,
                    "test_acc": 0.7430,
                    "test_rmse": 0.4249,
                },
                {
                    "dataset": "assist_09",
                    "hparam": "ortho_weight",
                    "value": 0.7,
                    "tag": "hparam_assist_09_ortho_weight_0p7",
                    "test_auc": 0.7917,
                    "test_acc": 0.7499,
                    "test_rmse": 0.4224,
                },
                {
                    "dataset": "assist_17",
                    "hparam": "dropout",
                    "value": 0.3,
                    "tag": "hparam_assist_17_dropout_0p3",
                    "test_auc": 0.7950,
                    "test_acc": 0.7243,
                    "test_rmse": 0.4313,
                },
                {
                    "dataset": "assist_17",
                    "hparam": "dropout",
                    "value": 0.1,
                    "tag": "hparam_assist_17_dropout_0p1",
                    "test_auc": 0.7956,
                    "test_acc": 0.7258,
                    "test_rmse": 0.4323,
                },
            ]
        )

    def test_build_gain_summary_frame_outputs_readable_labels(self):
        best = build_best_summary(self.summary)

        gain = build_gain_summary_frame(best)

        self.assertEqual(
            gain[["dataset", "hparam", "best_value_label"]].to_dict(orient="records"),
            [
                {"dataset": "assist_09", "hparam": "ortho_weight", "best_value_label": "0.7"},
                {"dataset": "assist_17", "hparam": "dropout", "best_value_label": "0.1"},
            ],
        )
        self.assertAlmostEqual(gain.loc[0, "auc_gain_vs_default"], 0.0014, places=6)
        self.assertAlmostEqual(gain.loc[1, "acc_gain_vs_default"], 0.0015, places=6)

    def test_plot_gain_summary_and_notes_are_generated(self):
        best = build_best_summary(self.summary)
        gain = build_gain_summary_frame(best)

        with tempfile.TemporaryDirectory() as tmp_dir:
            png_path, pdf_path = plot_gain_summary(
                gain_summary=gain,
                datasets=["assist_09", "assist_17"],
                output_dir=tmp_dir,
            )
            notes_path = write_notes(best, gain, tmp_dir)

            self.assertTrue(Path(png_path).exists())
            self.assertTrue(Path(pdf_path).exists())
            self.assertTrue(Path(notes_path).exists())

            notes_text = Path(notes_path).read_text(encoding="utf-8")
            self.assertIn("## Main Findings", notes_text)
            self.assertIn("## Recommendation", notes_text)
            self.assertIn("assist_09", notes_text)
            self.assertIn("ortho_weight", notes_text)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
