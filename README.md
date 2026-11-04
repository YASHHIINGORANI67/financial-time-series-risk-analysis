# Financial Time Series Risk & Trading Strategy Analysis

## Project Overview

An empirical financial risk analysis of MercadoLibre (NASDAQ: MELI), examining historical stock returns, extreme downside risk, return predictability, volatility dynamics, and technical trading strategies.

The project combines Python, Gretl, and Excel to apply statistical and econometric techniques to financial market data.

**Analysis period:** January 2022 – January 2026  
**Asset:** MercadoLibre (MELI)  
**Methods:** Historical VaR, Expected Shortfall, Student-t distribution, ARMA, GARCH, and MACD

## Objectives

- Analyze the distribution of daily stock returns.
- Estimate downside risk using Value at Risk (VaR) and Expected Shortfall (ES).
- Compare historical and parametric tail-risk estimates.
- Examine return predictability using ARMA models.
- Study conditional volatility using GARCH models.
- Evaluate a MACD trading strategy against buy-and-hold.

## Tools and Technologies

- **Python:** pandas, NumPy, SciPy, Matplotlib, Seaborn, statsmodels, yfinance
- **Gretl:** ARMA model selection and GARCH volatility analysis
- **Microsoft Excel:** Financial datasets and MACD trading strategy analysis
- **Git/GitHub:** Version control and project documentation

## Risk Analysis Results

The corrected Python implementation produced the following results from 1,013 daily log returns:

| Metric | Result |
|---|---:|
| Mean daily log return | 0.0437% |
| Daily standard deviation | 3.22% |
| Excess kurtosis | 3.89 |
| Historical VaR (99.9%) | -17.63% |
| Historical Expected Shortfall (99.9%) | -18.09% |
| Student-t VaR (99.9%) | -19.93% |
| Student-t Expected Shortfall (99.9%) | -29.35% |

The positive excess kurtosis indicates a heavy-tailed return distribution relative to a normal distribution.

The fitted Student-t model estimates more severe downside risk than the historical estimator at the 99.9% confidence level.

**Important:** The corrected Python implementation uses a fitted Student-t quantile and tail expectation. These results differ from the Student-t estimates in the original academic report. Historical 99.9% tail estimates are also subject to substantial sampling uncertainty.

## Econometric and Trading Analysis

### ARMA Return Modelling

ARMA specifications were investigated to evaluate short-term return predictability, including model selection using information criteria.

### GARCH Volatility Modelling

GARCH specifications were explored to analyze time-varying volatility, including comparisons involving outlier treatment.

### MACD Trading Strategy

An Excel-based MACD strategy was evaluated against a buy-and-hold benchmark. Strategy performance should be interpreted with attention to transaction costs, trading assumptions, and possible backtesting biases.

## Repository Contents

| File | Description |
|---|---|
| `Exc1y2.py` | Python return distribution and tail-risk analysis |
| `Final Report.pdf` | Original academic project report |
| `Final Report.docx` | Editable report |
| `MACDex3.xlsx` | MACD trading strategy workbook |
| `logreturnsexcel.xlsx` | Financial returns dataset |
| `Exerc22.gretl` | Gretl econometric analysis session |
| `codegretlselectingarma.inp` | ARMA model selection script |
| `GRETLeXC4.inp` | GARCH analysis script |
| `Exc 4 with outliers.xlsx` | Dataset with outliers |
| `Exc4 without outliers.gdt` | Dataset without outliers |
| `Exc4GarchModel.gdt` | GARCH model dataset |
| `Exc4GarchModelwithoutOutliers.gdt` | Alternative GARCH model dataset |

## Running the Python Analysis

Install the required libraries:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn statsmodels yfinance
```

Run:

```bash
python Exc1y2.py
```

The script downloads MELI historical prices using Yahoo Finance, calculates daily logarithmic returns, estimates VaR and Expected Shortfall, and displays diagnostic charts.

Results may vary if the underlying data provider revises historical prices.

## Limitations

- Extreme-tail estimates are uncertain with approximately 1,000 observations.
- Parametric tail estimates depend on the fitted distribution.
- ARMA and GARCH results require appropriate diagnostic validation.
- Trading strategy results may be affected by transaction costs and backtesting assumptions.
- The original report and corrected Python script use different Student-t risk calculations.

## Project Focus

Financial Risk Management | Market Risk | Time Series Analysis | Quantitative Finance | Python | Econometrics

---

## Reproducibility and validation roadmap
**Research question:** how do historical and fitted-distribution tail risk, time-series dynamics and MACD strategy outcomes compare for MELI?

**Evidence status:** the 1,013-return Python estimates above are corrected research outputs. The original report's Student-t calculations differ; do not treat the two sets of numbers as interchangeable.

**Reproduction:** use the Python installation and execution instructions above. Fix the January 2022–January 2026 date range, check adjusted-price conventions and data-provider revisions, and record package versions and fitted distribution parameters before comparing numbers.

**Validation checklist (not completed claims):** chronological train/test split; rolling VaR forecasts; Kupiec unconditional coverage and Christoffersen independence tests where sample size permits; GARCH standardized-residual diagnostics; strategy transaction costs, turnover, benchmark alignment and look-ahead checks.

**Important limitation:** with roughly 1,000 daily observations, a 99.9% empirical tail has extremely few expected exceedances. Out-of-sample backtesting at this level is statistically weak; model risk must be disclosed.

## Dependency manifest and execution notes

The repository now includes [`requirements.txt`](requirements.txt), derived from the imports in `Exc1y2.py`. To create an environment and run the existing analysis from the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python Exc1y2.py
```

The script requires internet access to download MELI data from Yahoo Finance through `yfinance` and a graphical session to display Matplotlib charts. Its price request starts on **2021-12-31** and ends at **2026-01-16 (exclusive)**. The script uses adjusted closing prices and logarithmic returns. The dependency versions are bounded ranges, **not a tested lockfile**; the exact results depend on data-provider revisions and the installed package versions.

**Verification status:** The script and its imports were inspected to prepare these instructions. A clean-environment end-to-end execution, numerical output comparison, automated test suite and rolling risk backtests have **not** been completed as part of this documentation update.

### Synthetic risk-measure unit tests (new)

The [tail-risk tests](tests/test_tail_risk.py) check historical quantile/ES conventions, the expected ordering of ES and VaR, Student-t location shifts and invalid-input handling using **synthetic return arrays**. Run them with:

```bash
python -m unittest discover -s tests -v
```

A [GitHub Actions workflow](.github/workflows/tail-risk-tests.yml) is configured to run these tests on relevant pushes, pull requests and manual dispatch. **These are formula-level checks, not MELI historical data validation, rolling VaR backtests or evidence of investment performance.** CI success has not yet been observed during this update.

## Out-of-sample rolling VaR backtesting

[`rolling_var_backtest.py`](rolling_var_backtest.py) is a **separate, new** reproducible backtest. It produces one-day-ahead rolling historical VaR forecasts, counts exceptions when the realized daily log return falls **below** the forecast VaR, and computes the Kupiec (1995) proportion-of-failures (POF) unconditional-coverage likelihood-ratio statistic.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python rolling_var_backtest.py --window 250 --alpha 0.01 --output backtest_results.csv
```

By default it downloads adjusted MELI closing prices from Yahoo Finance for 2021-12-31 through 2026-01-16 (exclusive). To avoid live downloads, provide a CSV with either `Close` or `log_return`:

```bash
python rolling_var_backtest.py --csv local_returns.csv --window 250 --alpha 0.01
```

The forecast at each observation uses **only the preceding 250 returns**, excluding the observed return being tested. With `alpha=0.01`, the model targets a 1% exception rate (99% VaR confidence). The Kupiec test reports the observed breach rate, expected count, LR statistic and approximate chi-square p-value. **A p-value at least 0.05 is not proof that the model is correct**; it means this test did not reject unconditional coverage at the 5% level.

**Limitations:** The test does not assess exception independence or clustering, conditional coverage, transaction costs or portfolio-level risk. At very high confidence levels such as 99.9%, there may be too few expected breaches for a reliable asymptotic test. Yahoo Finance historical data may change. The original academic results and original `Exc1y2.py` remain unchanged.

**Verification:** [Deterministic unit tests](tests/test_rolling_var.py) cover forecast timing, breach detection, sample counts, Kupiec boundary cases and invalid inputs. They are included in the [GitHub Actions workflow](.github/workflows/tail-risk-tests.yml). **No live-data backtest metrics are claimed here until a real run is reviewed.**

### Christoffersen exception-dependence diagnostics

The backtest now also prints **Christoffersen (1998) first-order independence** (LR-ind, asymptotic chi-square with 1 degree of freedom) and **conditional coverage** (LR-cc = Kupiec LR-uc + LR-ind, asymptotic chi-square with 2 degrees of freedom), including the 00, 01, 10 and 11 breach-transition counts. See [implementation](christoffersen_var.py) and [synthetic tests](tests/test_christoffersen.py).

Interpretation: independence assesses whether a breach is more likely after a previous breach; conditional coverage jointly examines the unconditional breach frequency and first-order dependence. **These tests do not establish forecasting skill or higher-order independence.** With relatively few VaR breaches, asymptotic p-values warrant caution.

**Observed MELI rolling 99% historical VaR backtest (GitHub Actions run reported by project owner, 2026-11-04):** 763 forecasts, 14 breaches (1.835%) versus 7.63 expected; Kupiec LR-uc = 4.3090, p = 0.03791. The 5% unconditional coverage null was rejected, suggesting underestimated downside tail risk during this sample. These figures are reported from the completed run's console output, not independently replicated from frozen price data. Christoffersen empirical results are **pending rerun**; do not interpret successful unit tests as empirical validation.

### Empirical Christoffersen backtest results (2026-11-04)

A completed manually triggered MELI backtest, using adjusted-price data downloaded during the run, reported **763** one-day forecasts with **14** breaches against **7.63** expected at the 99% VaR level (250-return rolling estimation window). The console output reported:

| Diagnostic | LR statistic | Asymptotic p-value | Decision at 5% |
| --- | ---: | ---: | --- |
| Kupiec unconditional coverage (1 df) | 4.3090 | 0.03791 | Reject |
| Christoffersen independence (1 df) | 1.3126 | 0.2519 | Do not reject |
| Christoffersen conditional coverage (2 df) | 5.6216 | 0.06016 | Do not reject |

Observed exception rate: **1.835%**, versus **1%** targeted. Transition counts: **n00=735, n01=13, n10=13, n11=1**. These results indicate evidence of excessive unconditional breach frequency but insufficient evidence of first-order breach dependence at the 5% level. The conditional coverage test does not reject at 5%, but its p-value is close to the threshold; non-rejection is not evidence of correct model calibration. The joint test and unconditional test can yield different decisions because they use different degrees of freedom.

**Reproducibility and limitations:** These figures are transcribed from the user's completed GitHub Actions log; the market-data download has not been independently frozen or reproduced here. With only 14 exceptions, chi-square approximations may be imperfect. No conclusion is drawn about higher-order clustering or forward-looking investment performance.

## Three-model VaR comparison: historical vs EWMA vs GARCH

[Comparison implementation](compare_var_models.py) | [Automated workflow](.github/workflows/meli-model-comparison.yml) | [Offline tests](tests/test_compare_var_models.py)

A separate out-of-sample comparison evaluates **historical rolling-quantile VaR**, **EWMA conditional-volatility normal VaR** (default decay 0.94), and **Gaussian GARCH(1,1) VaR** on exactly the same MELI forecast dates. All use one-day-ahead daily **log-return** VaR, a common 250-return initial window and default 99% confidence (alpha=0.01). EWMA estimates its mean and starting variance from training observations only, then updates variance after each realized return. GARCH(1,1) estimates mean and volatility parameters **once on the first 250 returns**, then updates conditional variance recursively without future leakage or parameter refitting. Historical VaR recalculates its quantile on the preceding 250 observations.

Each model reports observed breach frequency, Kupiec unconditional coverage, Christoffersen first-order independence and conditional coverage, plus transition counts. Results are **comparative diagnostics**, not a guarantee that any model is correctly calibrated. A fixed-parameter Gaussian GARCH is not a rolling-refitted or Student-t GARCH specification, and none of these models measures portfolio-level exposure.

```bash
python -m pip install numpy pandas scipy yfinance
python -m unittest discover -s tests -v
python compare_var_models.py --ticker MELI --start 2021-12-31 --end 2026-01-16 --window 250 --alpha 0.01
```

For reproducible offline input, use `--csv path/to/returns.csv` with a `log_return` or `Close` column. The command writes `model_comparison_results/forecasts.csv`, `summary.csv` and `configuration.json`. The GitHub Actions workflow uploads the same outputs as downloadable artifacts. **Do not treat the workflow as a validated empirical result until its actual run and metrics have been reviewed.** Live Yahoo Finance data can be revised; a fixed versioned input dataset would improve reproducibility. Low breach counts limit the accuracy of asymptotic chi-square tests, and comparisons should not be selected solely on p-values.

### Observed three-model MELI comparison (completed GitHub Actions run, 2026-11-04)

The project owner's completed [MELI model-comparison workflow](https://github.com/YASHHIINGORANI67/financial-time-series-risk-analysis/actions/runs/41751360867) reported the following results on **763 aligned daily forecasts**, with 250 initial training observations, 1% VaR tail probability and **7.63 expected breaches** per model:

| Model | Breaches | Observed rate | Kupiec p | Independence p | Conditional coverage p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical rolling VaR | 14 | 1.8349% | 0.037911 | 0.25192 | 0.060155 |
| EWMA-normal VaR | 12 | 1.5727% | 0.14229 | 0.17670 | 0.13682 |
| Fixed-parameter Gaussian GARCH(1,1) VaR | 3 | 0.39318% | 0.054823 | 0.87761 | 0.15636 |

Kupiec unconditional coverage rejects **historical VaR** at the 5% level. The independence and conditional coverage tests do **not** reject any model at 5%. GARCH produces the fewest exceptions but appears conservative relative to the nominal 1% breach frequency; its Kupiec p-value is only marginally above 0.05. EWMA is a promising candidate for further assessment, **not a proven winner**. The apparent discrepancy between historical VaR's Kupiec rejection and its conditional-coverage non-rejection reflects distinct chi-square reference distributions (1 versus 2 degrees of freedom), not a contradiction.

Transition counts: historical (n00=735, n01=13, n10=13, n11=1); EWMA (739, 11, 11, 1); GARCH (756, 3, 3, 0). Inference from only 3 GARCH exceptions is especially fragile. This comparison does not quantify average forecast conservatism, exceedance severity, model stability across periods, or trading/investment performance. Results are transcribed from the successful workflow console output; a versioned market-data snapshot and further validation would improve reproducibility. The initial-window-only GARCH fit is deliberately not a rolling parameter re-estimation.
