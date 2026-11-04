"""Christoffersen (1998) first-order VaR exception independence and coverage tests.

LR independence tests whether P(breach_t | breach_(t-1)) is constant.
LR conditional coverage combines Kupiec unconditional coverage and independence.
Asymptotic chi-square approximations can be unreliable with sparse exceptions.
"""
import numpy as np
from scipy.stats import chi2
from rolling_var_backtest import kupiec_pof


def _bernoulli_loglik(successes, trials):
    if trials == 0:
        return 0.0
    p = successes / trials
    ll = 0.0
    if successes:
        ll += successes * np.log(p)
    if successes < trials:
        ll += (trials - successes) * np.log1p(-p)
    return float(ll)


def christoffersen_tests(breaches, alpha=0.01):
    """Return transition counts, independence LR(1), and coverage LR(2)."""
    x = np.asarray(breaches)
    if x.ndim != 1 or x.size < 2 or not np.isin(x, [0, 1, False, True]).all():
        raise ValueError("At least two binary breach indicators required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    x = x.astype(int)
    previous, current = x[:-1], x[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    n0, n1 = n00 + n01, n10 + n11
    pooled = (n01 + n11) / (n0 + n1)
    ll_null = _bernoulli_loglik(n01 + n11, n0 + n1)
    ll_alt = _bernoulli_loglik(n01, n0) + _bernoulli_loglik(n11, n1)
    lr_ind = max(0.0, 2 * (ll_alt - ll_null))
    uc = kupiec_pof(x, alpha)
    lr_cc = uc["lr"] + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "p01": n01 / n0 if n0 else None,
            "p11": n11 / n1 if n1 else None,
            "lr_ind": lr_ind, "p_ind": float(chi2.sf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(chi2.sf(lr_cc, 2)),
            "lr_uc": uc["lr"], "p_uc": uc["p_value"]}
