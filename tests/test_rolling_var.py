"""Synthetic, deterministic rolling VaR and Kupiec test checks."""
import unittest
import numpy as np
from rolling_var_backtest import rolling_historical_var, kupiec_pof


class RollingVaRTests(unittest.TestCase):
    def test_forecast_excludes_current_return(self):
        data = [0.01, 0.02, 0.03, 0.04, -0.99]
        out = rolling_historical_var(data, window=4, alpha=0.25)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out["var"].iloc[0], 0.0175)
        self.assertTrue(bool(out["breach"].iloc[0]))

    def test_number_of_forecasts(self):
        out = rolling_historical_var(np.arange(20) / 100, window=5, alpha=0.1)
        self.assertEqual(len(out), 15)

    def test_kupiec_no_breaches_is_finite(self):
        result = kupiec_pof([False] * 100, 0.01)
        self.assertEqual(result["breaches"], 0)
        self.assertTrue(np.isfinite(result["lr"]))
        self.assertGreaterEqual(result["p_value"], 0)

    def test_kupiec_all_breaches_is_finite(self):
        result = kupiec_pof([True] * 10, 0.01)
        self.assertEqual(result["breaches"], 10)
        self.assertTrue(np.isfinite(result["lr"]))

    def test_kupiec_near_expected_rate(self):
        result = kupiec_pof([True] + [False] * 99, 0.01)
        self.assertAlmostEqual(result["lr"], 0.0, places=8)
        self.assertAlmostEqual(result["p_value"], 1.0, places=8)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            rolling_historical_var([0.01], window=5)
        with self.assertRaises(ValueError):
            kupiec_pof([], 0.01)


if __name__ == "__main__":
    unittest.main()
