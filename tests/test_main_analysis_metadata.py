import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from main import prepare_data


class MainAnalysisMetadataTests(unittest.TestCase):
    def test_prepare_data_exposes_frames_and_id_maps_for_analysis(self):
        train_df = pd.DataFrame(
            [
                {"stu_id": 10, "exer_id": 101, "cpt_seq": "1,2", "label": 1},
                {"stu_id": 11, "exer_id": 102, "cpt_seq": "2", "label": 0},
            ]
        )
        valid_df = pd.DataFrame(
            [
                {"stu_id": 10, "exer_id": 103, "cpt_seq": "1", "label": 1},
            ]
        )
        test_df = pd.DataFrame(
            [
                {"stu_id": 12, "exer_id": 104, "cpt_seq": "2", "label": 1},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_path = tmp_path / "train.csv"
            valid_path = tmp_path / "valid.csv"
            test_path = tmp_path / "test.csv"
            graph_dir = tmp_path / "graphs"

            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
            test_df.to_csv(test_path, index=False)

            args = Namespace(
                train_file=str(train_path),
                valid_file=str(valid_path),
                test_file=str(test_path),
                graph_dir=str(graph_dir),
                batch_size=2,
            )

            bundle = prepare_data(args, torch.device("cpu"))

        self.assertIn("raw_frames", bundle)
        self.assertIn("mapped_frames", bundle)
        self.assertIn("id_maps", bundle)
        self.assertIn("reverse_id_maps", bundle)
        self.assertIn("q_matrix", bundle)
        self.assertEqual(bundle["raw_frames"]["train"]["stu_id"].tolist(), [10, 11])
        self.assertEqual(bundle["mapped_frames"]["train"]["stu_id"].tolist(), [0, 1])
        self.assertEqual(bundle["id_maps"]["student"][10], 0)
        self.assertEqual(bundle["reverse_id_maps"]["student"][0], 10)
        self.assertEqual(tuple(bundle["q_matrix"].shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
