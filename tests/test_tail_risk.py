"""Unit tests of the tail-risk calculation conventions used in Exc1y2.py.

Synthetic data only; these tests do not validate Yahoo Finance prices,
fitted MELI parameters, or out-of-sample forecast performance.
Run: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
from scipy.stats import t


def historical_tail(returns, alpha):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite 1D return series")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    var = np.quantile(values, alpha)
    es = values[values <= var].mean()
    return float(var), float(es)


def student_tail(df, loc, scale, alpha):
    if df <= 1 or scale <= 0 or not 0 < alpha < 1:
        raise ValueError("ES requires df > 1, scale > 0 and 0 < alpha < 1")
    var = t.ppf(alpha, df, loc=loc, scale=scale)
    es = t.expect(lambda x: x, args=(df,), loc=loc, scale=scale,
                  lb=-np.inf, ub=var) / alpha
    return float(var), float(es)


class TailRiskTests(unittest.TestCase):
    def test_historical_known_sample(self):
        var, es = historical_tail([-0.20, -0.10, 0.0, 0.10, 0.20], 0.20)
        self.assertAlmostEqual(var, -0.12)
        self.assertAlmostEqual(es, -0.20)

    def test_historical_es_not_above_var(self):
        var, es = historical_tail([-0.3, -0.2, -0.1, 0, 0.1], 0.4)
        self.assertLessEqual(es, var)

    def test_student_tail_order(self):
        var, es = student_tail(5, 0.0, 0.03, 0.01)
        self.assertLess(es, var)
        self.assertLess(var, 0)

    def test_student_location_shift(self):
        var0, es0 = student_tail(5, 0, 0.03, 0.01)
        var1, es1 = student_tail(5, 0.02, 0.03, 0.01)
        self.assertAlmostEqual(var1 - var0, 0.02, places=8)
        self.assertAlmostEqual(es1 - es0, 0.02, places=8)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            historical_tail([], 0.01)
        with self.assertRaises(ValueError):
            student_tail(0.8, 0, 1, 0.01)


if __name__ == "__main__":
    unittest.main()
