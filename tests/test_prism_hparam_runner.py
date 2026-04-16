import unittest
from pathlib import Path
from types import SimpleNamespace

from run_prism_hparam_sensitivity import build_command, build_log_path, stagger_jobs_by_dataset


class PrismHparamRunnerTests(unittest.TestCase):
    def test_stagger_jobs_by_dataset_interleaves_first_wave(self):
        jobs = [
            {"dataset": "assist_09", "tag": "a1", "seed": 888, "config": {"epochs": 100, "patience": 7}},
            {"dataset": "assist_09", "tag": "a2", "seed": 888, "config": {"epochs": 100, "patience": 7}},
            {"dataset": "assist_17", "tag": "b1", "seed": 888, "config": {"epochs": 100, "patience": 7}},
            {"dataset": "assist_17", "tag": "b2", "seed": 888, "config": {"epochs": 100, "patience": 7}},
            {"dataset": "junyi", "tag": "c1", "seed": 888, "config": {"epochs": 100, "patience": 7}},
        ]

        staggered = stagger_jobs_by_dataset(jobs)

        self.assertEqual([job["tag"] for job in staggered[:3]], ["a1", "b1", "c1"])
        self.assertEqual([job["tag"] for job in staggered[3:]], ["a2", "b2"])

    def test_build_command_uses_unbuffered_python_and_applies_overrides(self):
        args = SimpleNamespace(
            save_root="saved_models/hparam_runs",
            graph_root="graphs_hparam",
            job_result_dir="results/hparam_job_results",
            data_root="data",
            override_epochs=3,
            override_patience=2,
        )
        job = {
            "dataset": "assist_09",
            "tag": "hparam_assist_09_ortho_weight_0p3",
            "seed": 888,
            "config": {"epochs": 100, "patience": 7, "dropout": 0.3},
        }

        cmd, output_json = build_command(job, args)

        self.assertEqual(cmd[1], "-u")
        self.assertIn("--epochs", cmd)
        self.assertIn("--patience", cmd)
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "3")
        self.assertEqual(cmd[cmd.index("--patience") + 1], "2")
        self.assertEqual(
            output_json,
            Path("results/hparam_job_results/hparam_assist_09_ortho_weight_0p3.json"),
        )

    def test_build_log_path_groups_logs_by_dataset(self):
        log_path = build_log_path("logs/hparam_runs", "assist_09", "job_1")

        self.assertEqual(log_path, Path("logs/hparam_runs/assist_09/job_1.log"))


if __name__ == "__main__":
    unittest.main()
