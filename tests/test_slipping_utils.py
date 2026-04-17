import unittest

import pandas as pd

from analysis.slipping_utils import (
    annotate_history_features,
    annotate_item_features,
    attach_concept_proxy_metrics,
    build_stress_subset_indices,
    build_item_history,
    build_student_concept_history,
    evaluate_binary_predictions,
    find_optimal_threshold,
    select_reference_candidates,
    select_strong_positive_candidates,
    select_flip_indices,
)


class SlippingUtilsTests(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame(
            [
                {"stu_id": "s1", "exer_id": 11, "label": 1, "cpt_seq": "1"},
                {"stu_id": "s1", "exer_id": 12, "label": 1, "cpt_seq": "1,2"},
                {"stu_id": "s1", "exer_id": 13, "label": 0, "cpt_seq": "2"},
                {"stu_id": "s2", "exer_id": 21, "label": 0, "cpt_seq": "1"},
                {"stu_id": "s2", "exer_id": 22, "label": 1, "cpt_seq": "2"},
            ]
        )
        self.test_df = pd.DataFrame(
            [
                {"stu_id": "s1", "exer_id": 11, "label": 1, "cpt_seq": "1", "p_pred": 0.92},
                {"stu_id": "s1", "exer_id": 22, "label": 1, "cpt_seq": "2", "p_pred": 0.88},
                {"stu_id": "s1", "exer_id": 12, "label": 0, "cpt_seq": "1,2", "p_pred": 0.74},
                {"stu_id": "s2", "exer_id": 21, "label": 1, "cpt_seq": "1", "p_pred": 0.81},
            ]
        )

    def test_annotate_history_features(self):
        history = build_student_concept_history(self.train_df)
        annotated = annotate_history_features(self.test_df, history)

        first = annotated.iloc[0]
        self.assertAlmostEqual(first["hist_avg_rate"], 1.0, places=6)
        self.assertEqual(first["min_cpt_hist"], 2)
        self.assertEqual(first["concept_count"], 1)

        second = annotated.iloc[1]
        self.assertAlmostEqual(second["hist_avg_rate"], 0.5, places=6)
        self.assertEqual(second["min_cpt_hist"], 2)
        self.assertAlmostEqual(second["min_hist_mastery_rate"], 0.5, places=6)

    def test_annotate_item_features(self):
        item_history = build_item_history(self.train_df)
        annotated = annotate_item_features(self.test_df, item_history)

        first = annotated.iloc[0]
        self.assertAlmostEqual(first["item_train_acc"], 1.0, places=6)
        self.assertEqual(first["item_train_support"], 1)

        second = annotated.iloc[1]
        self.assertAlmostEqual(second["item_train_acc"], 1.0, places=6)
        self.assertEqual(second["item_train_support"], 1)

    def test_select_strong_positive_candidates(self):
        history = build_student_concept_history(self.train_df)
        annotated = annotate_history_features(self.test_df, history)

        candidate_mask = select_strong_positive_candidates(
            annotated,
            hist_threshold=0.7,
            min_concept_support=2,
            pred_threshold=0.9,
        )

        self.assertEqual(candidate_mask.tolist(), [True, False, False, False])

    def test_select_flip_indices_is_deterministic(self):
        chosen = select_flip_indices([0, 1, 2, 3], ratio=0.5, seed=7)

        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen, select_flip_indices([0, 1, 2, 3], ratio=0.5, seed=7))
        self.assertTrue(set(chosen).issubset({0, 1, 2, 3}))

    def test_build_stress_subset_indices_keeps_candidates_and_balances_negatives(self):
        labels = [1, 1, 1, 0, 0, 0, 0]
        candidate_mask = [True, True, True, False, False, False, False]

        chosen = build_stress_subset_indices(
            labels,
            candidate_mask,
            seed=11,
            negative_multiplier=1.0,
        )

        self.assertEqual(chosen[:3], [0, 1, 2])
        self.assertEqual(len(chosen), 6)
        self.assertEqual(chosen, build_stress_subset_indices(labels, candidate_mask, seed=11, negative_multiplier=1.0))
        self.assertEqual(sum(idx >= 3 for idx in chosen), 3)

    def test_attach_concept_proxy_metrics_uses_item_drop_floor(self):
        flipped = pd.DataFrame(
            [
                {"hist_avg_rate": 0.90, "p_pred": 0.88},
                {"hist_avg_rate": 0.95, "p_pred": 0.60},
            ]
        )

        enriched = attach_concept_proxy_metrics(
            flipped,
            concept_proxy_pred=[0.70, 0.80],
            item_drop_floor=0.05,
        )

        self.assertAlmostEqual(enriched.loc[0, "item_drop"], 0.02, places=6)
        self.assertAlmostEqual(enriched.loc[0, "concept_drop"], 0.20, places=6)
        self.assertAlmostEqual(enriched.loc[0, "stable_concept_drop_ratio"], 4.0, places=6)
        self.assertAlmostEqual(enriched.loc[0, "decoupling_gap"], -0.18, places=6)
        self.assertAlmostEqual(enriched.loc[1, "stable_concept_drop_ratio"], 0.4285714286, places=6)

    def test_select_reference_candidates_matches_stu_and_exer(self):
        annotated = pd.DataFrame(
            [
                {"stu_id": "s1", "exer_id": 11, "label": 1},
                {"stu_id": "s2", "exer_id": 22, "label": 1},
                {"stu_id": "s3", "exer_id": 33, "label": 1},
            ]
        )
        reference = pd.DataFrame(
            [
                {"stu_id": "s2", "exer_id": 22},
                {"stu_id": "s3", "exer_id": 33},
            ]
        )

        mask = select_reference_candidates(annotated, reference)

        self.assertEqual(mask.tolist(), [False, True, True])

    def test_find_optimal_threshold_can_improve_accuracy_beyond_fixed_half(self):
        labels = [0, 0, 1, 1]
        predictions = [0.10, 0.20, 0.30, 0.40]

        threshold = find_optimal_threshold(labels, predictions, metric="acc")
        fixed = evaluate_binary_predictions(labels, predictions, threshold=0.5)
        calibrated = evaluate_binary_predictions(labels, predictions, threshold=threshold)

        self.assertLess(threshold, 0.5)
        self.assertAlmostEqual(fixed["acc"], 0.5, places=6)
        self.assertAlmostEqual(calibrated["acc"], 1.0, places=6)

    def test_evaluate_binary_predictions_reports_balanced_accuracy(self):
        labels = [0, 0, 1, 1]
        predictions = [0.60, 0.40, 0.70, 0.30]

        metrics = evaluate_binary_predictions(labels, predictions, threshold=0.5)

        self.assertAlmostEqual(metrics["acc"], 0.5, places=6)
        self.assertAlmostEqual(metrics["balanced_acc"], 0.5, places=6)

    def test_select_strong_positive_candidates_supports_stricter_filters(self):
        annotated = pd.DataFrame(
            [
                {
                    "label": 1,
                    "concept_count": 1,
                    "hist_avg_rate": 0.95,
                    "min_hist_mastery_rate": 0.95,
                    "min_cpt_hist": 5,
                    "p_pred": 0.93,
                    "concept_proxy_pred": 0.71,
                    "decoupling_gap": -0.22,
                    "item_train_support": 4,
                    "item_train_acc": 0.80,
                },
                {
                    "label": 1,
                    "concept_count": 2,
                    "hist_avg_rate": 0.92,
                    "min_hist_mastery_rate": 0.91,
                    "min_cpt_hist": 5,
                    "p_pred": 0.95,
                    "concept_proxy_pred": 0.84,
                    "decoupling_gap": -0.11,
                    "item_train_support": 6,
                    "item_train_acc": 0.90,
                },
                {
                    "label": 1,
                    "concept_count": 1,
                    "hist_avg_rate": 0.90,
                    "min_hist_mastery_rate": 0.60,
                    "min_cpt_hist": 5,
                    "p_pred": 0.94,
                    "concept_proxy_pred": 0.62,
                    "decoupling_gap": -0.32,
                    "item_train_support": 5,
                    "item_train_acc": 0.85,
                },
            ]
        )

        candidate_mask = select_strong_positive_candidates(
            annotated,
            hist_threshold=0.9,
            min_concept_support=4,
            pred_threshold=0.9,
            max_concepts=1,
            require_all_mastery=True,
            min_item_support=3,
            min_item_acc=0.75,
            min_concept_proxy_pred=0.7,
            min_decoupling_gap=-0.25,
        )

        self.assertEqual(candidate_mask.tolist(), [True, False, False])

    def test_select_strong_positive_candidates_respects_max_item_pred(self):
        annotated = pd.DataFrame(
            [
                {
                    "label": 1,
                    "concept_count": 1,
                    "hist_avg_rate": 0.93,
                    "min_hist_mastery_rate": 0.93,
                    "min_cpt_hist": 5,
                    "p_pred": 0.89,
                },
                {
                    "label": 1,
                    "concept_count": 1,
                    "hist_avg_rate": 0.94,
                    "min_hist_mastery_rate": 0.94,
                    "min_cpt_hist": 5,
                    "p_pred": 0.97,
                },
            ]
        )

        candidate_mask = select_strong_positive_candidates(
            annotated,
            hist_threshold=0.9,
            min_concept_support=4,
            pred_threshold=0.85,
            max_item_pred=0.95,
        )

        self.assertEqual(candidate_mask.tolist(), [True, False])

    def test_build_stress_subset_indices_hard_strategy_prefers_high_score_negatives(self):
        labels = [1, 1, 0, 0, 0]
        candidate_mask = [True, True, False, False, False]
        negative_scores = [0.91, 0.88, 0.41, 0.72, 0.65]

        chosen = build_stress_subset_indices(
            labels,
            candidate_mask,
            seed=5,
            negative_multiplier=1.0,
            negative_scores=negative_scores,
            negative_strategy="hard",
        )

        self.assertEqual(chosen[:2], [0, 1])
        self.assertEqual(chosen[2:], [3, 4])

    def test_build_stress_subset_indices_can_match_concept_counts(self):
        labels = [1, 1, 1, 0, 0, 0, 0]
        candidate_mask = [True, True, True, False, False, False, False]
        concept_counts = [1, 2, 2, 1, 1, 2, 3]
        negative_scores = [0.95, 0.94, 0.93, 0.40, 0.30, 0.80, 0.99]

        chosen = build_stress_subset_indices(
            labels,
            candidate_mask,
            seed=11,
            negative_multiplier=1.0,
            negative_scores=negative_scores,
            concept_counts=concept_counts,
            negative_strategy="hard",
            match_concept_counts=True,
        )

        negative_concepts = [concept_counts[idx] for idx in chosen if idx >= 3]
        self.assertEqual(sorted(negative_concepts), [1, 2, 3])
        self.assertIn(5, chosen)
        self.assertIn(6, chosen)


if __name__ == "__main__":
    unittest.main()
