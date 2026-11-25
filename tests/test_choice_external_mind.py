import unittest

import numpy as np

try:
    from scipy.optimize import approx_fprime
    from scripts.choice_external_mind import (
        build_ranks,
        evaluate,
        objective,
        to_percentile,
    )
except ModuleNotFoundError as missing:
    if missing.name not in {"numba", "scipy", "sklearn", "llvmlite"}:
        raise
    build_ranks = None


@unittest.skipIf(
    build_ranks is None, "Requires the optional MIND numerical environment"
)
class MindRankingTests(unittest.TestCase):
    def test_repeated_signals_get_average_percentile(self):
        np.testing.assert_allclose(
            to_percentile(np.array([3.0, 1.0, 3.0])), [0.75, 0.0, 0.75]
        )

    def test_one_days_labels_cannot_change_its_scores(self):
        topics = np.zeros((3, 2), dtype=np.int32)
        dense = np.zeros((3, 2), dtype=np.float32)
        hist = np.zeros((1, 1), dtype=np.int32)
        enc = np.zeros((1, 2), dtype=np.float32)
        days = np.array([9, 9, 10])
        offsets = np.array([0, 2, 4, 6])
        items = np.array([1, 2, 2, 1, 1, 2])

        def get_features(labels):
            state = tuple(np.zeros(3, np.int64) for _ in range(3))
            return build_ranks(
                days,
                np.zeros(3, np.int32),
                offsets,
                items,
                np.array(labels, np.uint8),
                topics,
                dense,
                hist,
                hist,
                enc,
                *state,
                True,
            )

        left = get_features([1, 0, 0, 1, 0, 1])
        right = get_features([0, 1, 1, 0, 0, 1])
        np.testing.assert_allclose(left[:4], right[:4])
        self.assertGreater(left[4, 0], left[5, 0])
        self.assertLess(right[4, 0], right[5, 0])

    def test_multi_click_auc_and_mean_reciprocal_rank(self):
        values, ranks = evaluate(
            np.array([3.0, 2.0, 1.0]),
            np.array([42]),
            np.array([0, 3]),
            np.array([10, 20, 30]),
            np.array([1, 0, 1], dtype=np.uint8),
        )
        np.testing.assert_array_equal(ranks, [1, 2, 3])
        self.assertAlmostEqual(values[0, 0], 0.5)
        self.assertAlmostEqual(values[0, 1], (1.0 + 1.0 / 3) / 2.0)

    def test_tied_choice_is_independent_of_source_order(self):
        offsets, slates = np.array([0, 3]), np.array([4])
        items, labels = np.array([4, 8, 22]), np.array([1, 1, 0], np.uint8)
        original, ranks = evaluate(np.zeros(3), slates, offsets, items, labels)
        order = [2, 0, 1]
        permuted, permuted_ranks = evaluate(
            np.zeros(3), slates, offsets, items[order], labels[order]
        )
        np.testing.assert_array_equal(ranks[order], permuted_ranks)
        np.testing.assert_allclose(original, permuted)

    def test_listwise_objective_gradient(self):
        x = np.array(
            [[0, 0.5], [1, 0.5], [0.2, 1], [0.8, 0], [0.9, 0.5]], dtype=np.float32
        )
        labels = np.array([1, 0, 0, 1, 1], dtype=np.uint8)
        offsets = np.array([0, 2, 5])
        cols = np.array([0, 1])
        beta = np.array([0.1, 0.2, -0.4, 0.3, 0.1, 0.5])
        _, analytic = objective(beta, x, labels, offsets, cols)
        finite_difference = approx_fprime(
            beta, lambda b: objective(b, x, labels, offsets, cols)[0], 1e-7
        )
        np.testing.assert_allclose(analytic, finite_difference, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
