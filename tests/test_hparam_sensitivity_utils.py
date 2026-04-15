import unittest

import pandas as pd

from analysis.hparam_sensitivity_utils import (
    build_hparam_result_summary,
    build_prism_hparam_sweep_jobs,
    make_hparam_tag,
    parse_hparam_tag,
)


class HparamSensitivityUtilsTests(unittest.TestCase):
    def test_make_and_parse_hparam_tag_round_trip(self):
        tag = make_hparam_tag("assist_09", "ortho_weight", 0.5)
        parsed = parse_hparam_tag(tag)

        self.assertEqual(tag, "hparam_assist_09_ortho_weight_0p5")
        self.assertEqual(parsed["dataset"], "assist_09")
        self.assertEqual(parsed["hparam"], "ortho_weight")
        self.assertAlmostEqual(parsed["value"], 0.5, places=6)

    def test_build_prism_hparam_sweep_jobs_creates_45_jobs(self):
        jobs = build_prism_hparam_sweep_jobs(seed=888)

        self.assertEqual(len(jobs), 45)
        tags = [job["tag"] for job in jobs]
        self.assertEqual(len(tags), len(set(tags)))
        self.assertTrue(all(job["seed"] == 888 for job in jobs))
        self.assertEqual(
            sorted({job["hparam"] for job in jobs}),
            ["dropout", "embedding_dim", "ortho_weight"],
        )

    def test_build_hparam_result_summary_filters_and_sorts_hparam_runs(self):
        frame = pd.DataFrame(
            [
                {
                    "Dataset": "assist_09",
                    "Tag": "assist_09_full",
                    "Test_AUC": 0.79,
                    "Test_ACC": 0.74,
                    "Test_RMSE": 0.42,
                },
                {
                    "Dataset": "assist_09",
                    "Tag": "hparam_assist_09_dropout_0p3",
                    "Test_AUC": 0.80,
                    "Test_ACC": 0.75,
                    "Test_RMSE": 0.41,
                },
                {
                    "Dataset": "assist_09",
                    "Tag": "hparam_assist_09_dropout_0p1",
                    "Test_AUC": 0.78,
                    "Test_ACC": 0.73,
                    "Test_RMSE": 0.43,
                },
                {
                    "Dataset": "junyi",
                    "Tag": "hparam_junyi_embedding_dim_256",
                    "Test_AUC": 0.81,
                    "Test_ACC": 0.77,
                    "Test_RMSE": 0.40,
                },
            ]
        )

        summary = build_hparam_result_summary(frame)

        self.assertEqual(summary["tag"].tolist(), [
            "hparam_assist_09_dropout_0p1",
            "hparam_assist_09_dropout_0p3",
            "hparam_junyi_embedding_dim_256",
        ])
        self.assertEqual(summary["dataset"].tolist(), ["assist_09", "assist_09", "junyi"])
        self.assertEqual(summary["hparam"].tolist(), ["dropout", "dropout", "embedding_dim"])
        self.assertEqual(summary["value"].tolist(), [0.1, 0.3, 256.0])

    def test_build_hparam_result_summary_keeps_latest_duplicate_tag(self):
        frame = pd.DataFrame(
            [
                {
                    "Timestamp": "2026-04-13 10:00:00",
                    "Dataset": "assist_09",
                    "Tag": "hparam_assist_09_ortho_weight_0p3",
                    "Test_AUC": 0.78,
                    "Test_ACC": 0.73,
                    "Test_RMSE": 0.43,
                },
                {
                    "Timestamp": "2026-04-13 12:00:00",
                    "Dataset": "assist_09",
                    "Tag": "hparam_assist_09_ortho_weight_0p3",
                    "Test_AUC": 0.80,
                    "Test_ACC": 0.75,
                    "Test_RMSE": 0.41,
                },
            ]
        )

        summary = build_hparam_result_summary(frame)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.loc[0, "test_auc"], 0.80, places=6)


if __name__ == "__main__":
    unittest.main()
