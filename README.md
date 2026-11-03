
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
