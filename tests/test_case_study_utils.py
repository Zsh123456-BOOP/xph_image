import unittest

import numpy as np
import pandas as pd

from analysis.case_study_utils import (
    align_cases_to_reference,
    compute_concept_proxy_scores,
    select_conflict_cases,
)


class CaseStudyUtilsTests(unittest.TestCase):
    def test_select_conflict_cases_filters_and_sorts(self):
        df = pd.DataFrame(
            [
                {
                    "stu_id": "s1",
                    "label": 0,
                    "concept_count": 1,
                    "hist_avg_rate": 0.9,
                    "min_cpt_hist": 4,
                    "p_pred": 0.82,
                },
                {
                    "stu_id": "s2",
                    "label": 0,
                    "concept_count": 3,
                    "hist_avg_rate": 0.95,
                    "min_cpt_hist": 8,
                    "p_pred": 0.91,
                },
                {
                    "stu_id": "s3",
                    "label": 0,
                    "concept_count": 2,
                    "hist_avg_rate": 0.8,
                    "min_cpt_hist": 3,
                    "p_pred": 0.73,
                },
            ]
        )

        cases = select_conflict_cases(
            df,
            hist_threshold=0.85,
            min_concept_support=4,
            max_concepts=2,
            min_item_pred=0.75,
        )

        self.assertEqual(cases["stu_id"].tolist(), ["s1"])

    def test_compute_concept_proxy_scores_prefers_exact_items(self):
        item_scores = np.array([0.9, 0.7, 0.4, 0.2], dtype=float)
        q_matrix = np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=int,
        )

        summary = compute_concept_proxy_scores(item_scores, q_matrix, [0, 1])

        self.assertAlmostEqual(summary["concept_scores"][0], 0.9, places=6)
        self.assertAlmostEqual(summary["concept_scores"][1], 0.3, places=6)
        self.assertAlmostEqual(summary["overall_proxy"], 0.6, places=6)

    def test_align_cases_to_reference_preserves_reference_order(self):
        cases = pd.DataFrame(
            [
                {"stu_id": "s1", "exer_id": 10, "p_pred": 0.7},
                {"stu_id": "s2", "exer_id": 20, "p_pred": 0.8},
                {"stu_id": "s3", "exer_id": 30, "p_pred": 0.9},
            ]
        )
        reference = pd.DataFrame(
            [
                {"stu_id": "s3", "exer_id": 30},
                {"stu_id": "s1", "exer_id": 10},
            ]
        )

        aligned = align_cases_to_reference(cases, reference)

        self.assertEqual(aligned["stu_id"].tolist(), ["s3", "s1"])
        self.assertEqual(aligned["exer_id"].tolist(), [30, 10])

    def test_align_cases_to_reference_normalizes_numeric_ids(self):
        cases = pd.DataFrame(
            [
                {"stu_id": "747", "exer_id": 158, "p_pred": 0.71},
                {"stu_id": "1284", "exer_id": 86, "p_pred": 0.69},
            ]
        )
        reference = pd.DataFrame(
            [
                {"stu_id": 1284, "exer_id": 86, "hist_avg_rate": 0.82},
                {"stu_id": 747, "exer_id": 158, "hist_avg_rate": 0.81},
            ]
        )

        aligned = align_cases_to_reference(cases, reference)

        self.assertEqual(aligned["stu_id"].tolist(), ["1284", "747"])
        self.assertEqual(aligned["exer_id"].tolist(), [86, 158])

    def test_select_conflict_cases_supports_stricter_gap_filters(self):
        df = pd.DataFrame(
            [
                {
                    "stu_id": "s1",
                    "label": 0,
                    "concept_count": 1,
                    "hist_avg_rate": 0.95,
                    "min_cpt_hist": 6,
                    "p_pred": 0.78,
                    "concept_proxy_pred": 0.88,
                    "decoupling_gap": 0.10,
                    "item_train_support": 5,
                },
                {
                    "stu_id": "s2",
                    "label": 0,
                    "concept_count": 1,
                    "hist_avg_rate": 0.94,
                    "min_cpt_hist": 6,
                    "p_pred": 0.80,
                    "concept_proxy_pred": 0.82,
                    "decoupling_gap": 0.02,
                    "item_train_support": 5,
                },
                {
                    "stu_id": "s3",
                    "label": 0,
                    "concept_count": 1,
                    "hist_avg_rate": 0.93,
                    "min_cpt_hist": 6,
                    "p_pred": 0.77,
                    "concept_proxy_pred": 0.86,
                    "decoupling_gap": 0.09,
                    "item_train_support": 1,
                },
            ]
        )

        cases = select_conflict_cases(
            df,
            hist_threshold=0.9,
            min_concept_support=5,
            max_concepts=2,
            min_item_pred=0.75,
            min_concept_proxy_pred=0.85,
            min_decoupling_gap=0.08,
            min_item_support=3,
        )

        self.assertEqual(cases["stu_id"].tolist(), ["s1"])


if __name__ == "__main__":
    unittest.main()
