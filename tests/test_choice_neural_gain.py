import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from choice_neural_gain_stream import posteriors, hour_key
from choice_neural_gain import rank_scores
from choice_neural_gain_first import differences


class NeuralGainTests(unittest.TestCase):
    def test_no_current_or_recent_hour_outcomes(self):
        n=np.zeros((12,3)); k=n.copy();cat=np.array([0,0,1])
        n[2,1]=3;k[2,1]=2; baseline=posteriors(n,k,cat,0,[5],1)[5]
        n[4:,0]=9999;k[4:,0]=8800
        np.testing.assert_allclose(baseline,posteriors(n,k,cat,0,[5],1)[5])
        n[3,0]=800;k[3,0]=799
        self.assertGreater(posteriors(n,k,cat,0,[5],1)[5][0],baseline[0])

    def test_hour_rollover(self):
        self.assertEqual(hour_key(b"11/15/2019 12:00:01 AM"),360)
        self.assertEqual(hour_key(b"11/15/2019 12:10:00 PM"),372)

    def test_rank_ties_get_average(self):
        np.testing.assert_allclose(rank_scores(np.array([1.,1.,5.]),np.array([0,3])),[.25,.25,1.])

    def test_first_slot_features_are_independent_of_current_clicks(self):
        components=np.arange(15,dtype=np.float32).reshape(3,5)
        static=np.array([.8,.4,.2]);live=np.array([.7,.9,.1])
        args=(components,static,live,np.array([1,2,3]),np.array([2,1,3]))
        left=differences(*args,np.array([1,0,0]),np.array([0,3]))
        right=differences(*args,np.array([0,1,0]),np.array([0,3]))
        np.testing.assert_array_equal(left[0],right[0])
        np.testing.assert_array_equal(left[2],right[2])
        self.assertEqual((left[1][0],right[1][0]),(-1,1))


if __name__=="__main__":unittest.main()
