"""Out-of-sample rolling historical VaR backtest for MELI daily log returns.

Uses only observations strictly before each forecast date. No future leakage.
The Kupiec POF test assesses unconditional exception frequency, not independence.
This is a research diagnostic, not an investment recommendation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


def rolling_historical_var(returns, window=250, alpha=0.01):
    """Return a frame with actual returns, one-step VaR and breach indicators."""
    if window < 2 or not 0 < alpha < 1:
        raise ValueError("window >= 2 and 0 < alpha < 1 required")
    series = pd.Series(returns, dtype=float).dropna()
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Returns must be finite")
    if len(series) <= window:
        raise ValueError("Need more observations than the estimation window")
    forecasts = series.shift(1).rolling(window=window, min_periods=window).quantile(alpha)
    result = pd.DataFrame({"return": series, "var": forecasts}).dropna()
    result["breach"] = result["return"] < result["var"]
    return result


def kupiec_pof(breaches, alpha):
    """Likelihood-ratio unconditional coverage test, chi-square(1) approximation.

    Exact binomial testing may be preferable for very small expected breach counts.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    x = np.asarray(breaches)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Nonempty one-dimensional breach array required")
    if not np.isin(x, [False, True, 0, 1]).all():
        raise ValueError("Breach indicators must be binary")
    n = int(x.size)
    k = int(x.sum())
    p_hat = k / n
    # Stable log likelihood at the boundary k=0 or k=n.
    ll_null = k * np.log(alpha) + (n - k) * np.log1p(-alpha)
    ll_alt = 0.0
    if k:
        ll_alt += k * np.log(p_hat)
    if k < n:
        ll_alt += (n - k) * np.log1p(-p_hat)
    lr = max(0.0, 2 * (ll_alt - ll_null))
    return {"n": n, "breaches": k, "expected": n * alpha,
            "rate": p_hat, "lr": lr, "p_value": float(chi2.sf(lr, 1))}


def load_returns(csv_path=None, ticker="MELI", start="2021-12-31", end="2026-01-16"):
    if csv_path:
        df = pd.read_csv(csv_path)
        if "log_return" in df.columns:
            return pd.Series(df["log_return"].to_numpy(), dtype=float)
        if "Close" not in df.columns:
            raise ValueError("CSV needs a 'log_return' or 'Close' column")
        prices = pd.to_numeric(df["Close"], errors="raise")
    else:
        import yfinance as yf
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if history.empty or "Close" not in history:
            raise ValueError("No adjusted close data returned; check connection/ticker")
        prices = history["Close"]
    if (prices <= 0).any():
        raise ValueError("Close prices must be positive")
    return np.log(prices / prices.shift(1)).dropna()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Local CSV with log_return or Close column")
    parser.add_argument("--ticker", default="MELI")
    parser.add_argument("--start", default="2021-12-31")
    parser.add_argument("--end", default="2026-01-16", help="Exclusive end date")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output", type=Path, help="Optional forecast CSV output")
    args = parser.parse_args()
    returns = load_returns(args.csv, args.ticker, args.start, args.end)
    result = rolling_historical_var(returns, args.window, args.alpha)
    metrics = kupiec_pof(result["breach"].to_numpy(), args.alpha)
    from christoffersen_var import christoffersen_tests
    coverage = christoffersen_tests(result["breach"].to_numpy(), args.alpha)
    print(f"One-day rolling historical VaR: window={args.window}, alpha={args.alpha:.3%}")
    print(f"Forecasts={metrics['n']} | breaches={metrics['breaches']} "
          f"| expected={metrics['expected']:.2f} | observed rate={metrics['rate']:.3%}")
    print(f"Kupiec LR={metrics['lr']:.4f} | asymptotic p-value={metrics['p_value']:.4g}")
    print("At 5%: " + ("reject unconditional coverage" if metrics["p_value"] < 0.05
                         else "do not reject unconditional coverage"))
    if metrics["expected"] < 5:
        print("WARNING: fewer than five expected breaches; chi-square approximation may be unreliable.")
    print(f"Christoffersen independence LR={coverage['lr_ind']:.4f} | asymptotic p-value={coverage['p_ind']:.4g}")
    print(f"Christoffersen conditional coverage LR={coverage['lr_cc']:.4f} | asymptotic p-value={coverage['p_cc']:.4g}")
    print(f"Transitions n00={coverage['n00']}, n01={coverage['n01']}, n10={coverage['n10']}, n11={coverage['n11']}")
    print("NOTE: independence is a first-order test; sparse breaches may weaken chi-square approximations.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index_label="observation")
        print(f"Saved forecasts to {args.output}")


if __name__ == "__main__":
    main()
