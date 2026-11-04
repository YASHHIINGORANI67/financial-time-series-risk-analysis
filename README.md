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
