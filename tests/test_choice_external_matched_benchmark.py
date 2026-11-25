"""Release regressions for the published-checkpoint benchmark and exact training match."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from choice_external_benchmark import grade


class MatchedExternalBenchmark(unittest.TestCase):
    def test_multi_click_mrr_and_first_hit(self):
        # Two clicks at ranks 1 and 3. MIND averages their reciprocal ranks.
        row = grade(np.array([0, 1, 1], np.uint8), np.array([0, 3], np.int32),
                    np.array([2, 3, 1], np.int32))[0]
        self.assertAlmostEqual(row[0], .5)
        self.assertAlmostEqual(row[1], 2 / 3)
        self.assertEqual(row[4], 1)
        with self.assertRaises(ValueError):
            grade(np.array([0, 1, 1], np.uint8), np.array([0, 3], np.int32),
                  np.array([2, 1, 1], np.int32))

    def test_all_release_models_and_training_split(self):
        result = json.loads((ROOT / "results/choice_gains/mind_external_baselines.json").read_text())
        self.assertEqual(result["history50_fit"]["train_impressions"], 156965)
        self.assertEqual(result["matched50_fit"]["train_impressions"], 149117)
        self.assertEqual(result["matched50_fit"]["history_limit_train_and_score"], 50)
        self.assertEqual(result["matched50_fit"]["held_out_ordered_ids_sha256"],
                         "579fb90775d473b2bd184e1502af5c321267baa1bf59b8a06193691375616d1e")
        self.assertEqual(len(result["publisher_small_validation"]), 9)
        for row in result["publisher_small_validation"].values():
            self.assertLess(row["max_abs_local_difference"], .0002)
        for cohort in result["cohorts"].values():
            for model, row in cohort["models"].items():
                self.assertAlmostEqual(row["Hit@1"], row["first_hits"] / cohort["impressions"])
                check = row["original_metric_functions_all_lists"]
                self.assertEqual(check["impressions"], cohort["impressions"])
                self.assertLess(check["max_abs_error"], 1e-10)
            for name in ("Matched50", "Switch"):
                paired = cohort["paired"][name]["Neural-9"]
                count = cohort["models"][name]["first_hits"] - cohort["models"]["Neural-9"]["first_hits"]
                self.assertEqual(paired["extra_first_hits"], count)


if __name__ == "__main__":
    unittest.main()
