"""Deterministic, offline model-comparison tests."""
import unittest
import numpy as np
from compare_var_models import (historical_forecasts, ewma_forecasts,
                                garch_forecasts, compare_models)


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        cls.returns = rng.normal(0.0003, 0.02, 160)

    def test_forecast_count_and_common_dates(self):
        hist = historical_forecasts(self.returns, 60)
        ewma = ewma_forecasts(self.returns, 60)
        garch, params = garch_forecasts(self.returns, 60)
        self.assertEqual(len(hist), 100)
        self.assertTrue(hist.index.equals(ewma.index))
        self.assertTrue(hist.index.equals(garch.index))
        self.assertLess(params["alpha"] + params["beta"], 1)

    def test_historical_no_lookahead(self):
        original = historical_forecasts(self.returns, 60)
        modified = self.returns.copy()
        modified[-1] = -0.8
        changed = historical_forecasts(modified, 60)
        self.assertAlmostEqual(original.iloc[-1], changed.iloc[-1])

    def test_ewma_no_lookahead(self):
        original = ewma_forecasts(self.returns, 60)
        modified = self.returns.copy()
        modified[-1] = -0.8
        changed = ewma_forecasts(modified, 60)
        self.assertAlmostEqual(original.iloc[-1], changed.iloc[-1])

    def test_garch_no_lookahead(self):
        original, _ = garch_forecasts(self.returns, 60)
        modified = self.returns.copy()
        modified[-1] = -0.8
        changed, _ = garch_forecasts(modified, 60)
        self.assertAlmostEqual(original.iloc[-1], changed.iloc[-1])

    def test_comparison_outputs(self):
        frame, summary, _ = compare_models(self.returns, 60)
        self.assertEqual(len(frame), 100)
        self.assertEqual(set(summary["model"]), {"historical", "ewma", "garch"})
        self.assertTrue((summary["forecasts"] == 100).all())

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            historical_forecasts(self.returns, 10)


if __name__ == "__main__":
    unittest.main()
