import unittest

import numpy as np
import pandas as pd

from analysis.analysis_utils import (
    attach_predictions_to_raw_frame,
    compute_concept_proxies,
)


class XphImageAnalysisUtilsTests(unittest.TestCase):
    def test_compute_concept_proxies_averages_target_concepts(self):
        knowledge_states = np.array(
            [
                [0.8, 0.6, 0.2],
                [0.3, 0.5, 0.9],
            ],
            dtype=float,
        )
        concept_id_lists = [[0, 1], [2]]

        proxies = compute_concept_proxies(knowledge_states, concept_id_lists)

        np.testing.assert_allclose(proxies, np.array([0.7, 0.9], dtype=float))

    def test_attach_predictions_to_raw_frame_adds_prediction_columns(self):
        raw_frame = pd.DataFrame(
            [
                {"stu_id": 10, "exer_id": 101, "cpt_seq": "1,2", "label": 1},
                {"stu_id": 11, "exer_id": 102, "cpt_seq": "2", "label": 0},
            ]
        )
        predictions = np.array([0.91, 0.34], dtype=float)
        knowledge_states = np.array(
            [
                [0.8, 0.7],
                [0.2, 0.6],
            ],
            dtype=float,
        )
        mapped_concept_lists = [[0, 1], [1]]

        attached = attach_predictions_to_raw_frame(
            raw_frame=raw_frame,
            predictions=predictions,
            knowledge_states=knowledge_states,
            mapped_concept_lists=mapped_concept_lists,
        )

        self.assertEqual(attached["concept_ids"].tolist(), [[0, 1], [1]])
        self.assertAlmostEqual(attached.loc[0, "p_pred"], 0.91, places=6)
        self.assertAlmostEqual(attached.loc[0, "concept_proxy_pred"], 0.75, places=6)
        self.assertAlmostEqual(attached.loc[1, "concept_proxy_pred"], 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
