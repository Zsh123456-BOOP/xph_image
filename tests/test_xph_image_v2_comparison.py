import unittest

from analysis.run_prism_neuralcd_comparison import resolve_baseline_source_mode


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


if __name__ == "__main__":
    unittest.main()
