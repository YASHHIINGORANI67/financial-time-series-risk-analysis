"""Compare historical, EWMA-normal and GARCH(1,1)-normal one-day VaR.

All models forecast strictly before the realized return. GARCH parameters are
estimated once on the initial training window and filtered forward without
refitting; this is an explicitly fixed-parameter out-of-sample comparison.
Returns and VaR are in decimal daily log-return units.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from rolling_var_backtest import load_returns, kupiec_pof
from christoffersen_var import christoffersen_tests


def _series(returns, window, alpha):
    s = pd.Series(returns, dtype=float).dropna()
    if not np.isfinite(s.to_numpy()).all() or len(s) <= window:
        raise ValueError("Need finite returns with more observations than window")
    if window < 30 or not 0 < alpha < 0.5:
        raise ValueError("window >= 30 and 0 < alpha < 0.5 required")
    return s


def historical_forecasts(returns, window=250, alpha=0.01):
    s = _series(returns, window, alpha)
    return s.shift(1).rolling(window).quantile(alpha).iloc[window:]


def ewma_forecasts(returns, window=250, alpha=0.01, decay=0.94):
    s = _series(returns, window, alpha)
    if not 0 < decay < 1:
        raise ValueError("decay must lie between 0 and 1")
    r = s.to_numpy()
    # Initialize variance using only training returns, then update on each
    # realized observation before forecasting the *following* observation.
    mu = float(np.mean(r[:window]))
    variance = float(np.var(r[:window], ddof=1))
    if variance <= 0:
        raise ValueError("Training returns must have positive variance")
    z = norm.ppf(alpha)
    forecast = []
    for t in range(window, len(r)):
        forecast.append(mu + z * np.sqrt(variance))
        variance = decay * variance + (1 - decay) * (r[t] - mu) ** 2
    return pd.Series(forecast, index=s.index[window:])


def _garch_filter(params, residuals, initial_variance):
    omega, a, b = params
    h = np.empty(len(residuals))
    h[0] = initial_variance
    for t in range(1, len(residuals)):
        h[t] = omega + a * residuals[t - 1] ** 2 + b * h[t - 1]
    return h


def garch_forecasts(returns, window=250, alpha=0.01):
    s = _series(returns, window, alpha)
    r = s.to_numpy()
    # Scale returns to percent to improve optimizer conditioning.
    scaled = r * 100
    mu = float(np.mean(scaled[:window]))
    e = scaled - mu
    training = e[:window]
    v = float(np.var(training, ddof=1))
    if v <= 0:
        raise ValueError("Training returns must have positive variance")

    def objective(params):
        h = _garch_filter(params, training, v)
        if not np.isfinite(h).all() or np.any(h <= 0):
            return 1e12
        return float(0.5 * np.sum(np.log(2 * np.pi * h) + training ** 2 / h))

    fit = minimize(objective, [v * 0.05, 0.07, 0.88],
                   method="SLSQP",
                   bounds=[(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)],
                   constraints=[{"type": "ineq",
                                 "fun": lambda p: 0.999 - p[1] - p[2]}],
                   options={"maxiter": 2000, "ftol": 1e-9})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"GARCH training optimization failed: {fit.message}")
    omega, a, b = fit.x
    if a + b >= 1:
        raise RuntimeError("Invalid GARCH persistence")
    h_train = _garch_filter(fit.x, training, v)
    # Last training observation updates the variance for first test date.
    next_h = omega + a * training[-1] ** 2 + b * h_train[-1]
    forecasts = []
    for t in range(window, len(r)):
        forecasts.append((mu + norm.ppf(alpha) * np.sqrt(next_h)) / 100)
        next_h = omega + a * e[t] ** 2 + b * next_h
    return pd.Series(forecasts, index=s.index[window:]), {
        "omega": float(omega), "alpha": float(a), "beta": float(b),
        "training_n": window, "fixed_parameters": True
    }


def compare_models(returns, window=250, alpha=0.01, decay=0.94):
    s = _series(returns, window, alpha)
    observed = s.iloc[window:]
    hist = historical_forecasts(s, window, alpha)
    ewma = ewma_forecasts(s, window, alpha, decay)
    garch, parameters = garch_forecasts(s, window, alpha)
    frame = pd.DataFrame({"return": observed, "historical_var": hist,
                          "ewma_var": ewma, "garch_var": garch})
    metrics = []
    for name in ("historical", "ewma", "garch"):
        breach = frame["return"] < frame[f"{name}_var"]
        frame[f"{name}_breach"] = breach
        uc = kupiec_pof(breach.to_numpy(), alpha)
        cc = christoffersen_tests(breach.to_numpy(), alpha)
        metrics.append({
            "model": name, "forecasts": uc["n"], "breaches": uc["breaches"],
            "expected": uc["expected"], "breach_rate": uc["rate"],
            "kupiec_lr": uc["lr"], "kupiec_p": uc["p_value"],
            "independence_lr": cc["lr_ind"], "independence_p": cc["p_ind"],
            "conditional_lr": cc["lr_cc"], "conditional_p": cc["p_cc"],
            "n00": cc["n00"], "n01": cc["n01"], "n10": cc["n10"], "n11": cc["n11"]
        })
    return frame, pd.DataFrame(metrics), parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--ticker", default="MELI")
    parser.add_argument("--start", default="2021-12-31")
    parser.add_argument("--end", default="2026-01-16")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.94)
    parser.add_argument("--outdir", type=Path, default=Path("model_comparison_results"))
    args = parser.parse_args()
    returns = load_returns(args.csv, args.ticker, args.start, args.end)
    frame, summary, garch_params = compare_models(returns, args.window,
                                                   args.alpha, args.decay)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "forecasts.csv", index_label="observation")
    summary.to_csv(args.outdir / "summary.csv", index=False)
    config = {"ticker": args.ticker, "start": args.start, "end_exclusive": args.end,
              "window": args.window, "alpha": args.alpha, "decay": args.decay,
              "garch": garch_params, "data_source": str(args.csv) if args.csv else "Yahoo Finance"}
    (args.outdir / "configuration.json").write_text(json.dumps(config, indent=2))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("Interpret p-values cautiously; tests are asymptotic and breaches may be sparse.")
    print("Models use the same dates. GARCH parameters fitted on training window only.")
    print(f"Saved results to {args.outdir}")


if __name__ == "__main__":
    main()
