import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.choice_baselines import cases


class ChoiceBaselineTest(unittest.TestCase):
    def test_case_history_excludes_future_actions_and_concurrent_rows(self):
        start = pd.Timestamp("2023-09-04")
        def slate(first, picked=0, at=1):
            return [{"itemId": first+i, "selected": int(i==picked),
                     "timestamp": float(start.timestamp()+at) if i==picked else None} for i in range(10)]
        logs = pd.DataFrame(dict(userId=[1]*4, impressionId=[91,20,21,11],
            timestamp=[start + pd.Timedelta(seconds=i) for i in [6,2,2,1]],
            iteration=[2,1,1,np.nan],impressions=[slate(30,1,9),slate(20,2,5),slate(50,3,2),slate(10,4,3)]))
        users = pd.DataFrame(dict(userId=[1],domain=["movies"]))
        with patch("scripts.choice_baselines.pd.read_parquet", side_effect=[logs,users]):
            clean = cases(Path("."),"recommendation")
        by_id = {r.slate:r for r in clean}
        np.testing.assert_array_equal(by_id[91].past,[14,22,53])
        self.assertEqual(len(by_id[20].past),0)
        self.assertEqual(len(by_id[21].past),0)
        self.assertEqual(by_id[91].true, [31])
        with patch("scripts.choice_baselines.pd.read_parquet", side_effect=[logs,users]):
            original = cases(Path("."),"legacy")
        self.assertEqual(len(original),3)
        self.assertTrue(all(r.true == [31] for r in original))
        self.assertTrue(all(31 in r.past for r in original))


if __name__ == "__main__":
    unittest.main()
