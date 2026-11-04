"""Deterministic tests for Christoffersen transition and LR statistics."""
import unittest
import numpy as np
from christoffersen_var import christoffersen_tests


class ChristoffersenTests(unittest.TestCase):
    def test_transition_counts(self):
        r = christoffersen_tests([0, 0, 1, 1, 0], 0.2)
        self.assertEqual((r["n00"], r["n01"], r["n10"], r["n11"]), (1, 1, 1, 1))
        self.assertAlmostEqual(r["lr_ind"], 0.0)
        self.assertAlmostEqual(r["p_ind"], 1.0)

    def test_clustered_breaches(self):
        r = christoffersen_tests([0] * 50 + [1] * 10 + [0] * 50, 0.1)
        self.assertLess(r["p_ind"], 0.05)
        self.assertGreater(r["lr_cc"], r["lr_uc"])

    def test_no_breaches_finite(self):
        r = christoffersen_tests([0] * 100, 0.01)
        self.assertTrue(np.isfinite(r["lr_ind"]))
        self.assertTrue(np.isfinite(r["lr_cc"]))

    def test_all_breaches_finite(self):
        r = christoffersen_tests([1] * 20, 0.01)
        self.assertTrue(np.isfinite(r["lr_ind"]))

    def test_invalid_input(self):
        for x in ([0], [0, 2, 1], []):
            with self.assertRaises(ValueError):
                christoffersen_tests(x)


if __name__ == "__main__":
    unittest.main()
