
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import seaborn as sns

from scipy.stats import t
from statsmodels.graphics.tsaplots import plot_acf

# Download historical MELI prices
initial_date = "2021-12-31"
end_date = "2026-01-16"

ticker = yf.Ticker("MELI")
price_history = ticker.history(
    start=initial_date,
    end=end_date,
    auto_adjust=True
)

if price_history.empty:
    raise ValueError("No MELI price data downloaded.")

# Calculate daily logarithmic returns
meli_prices = price_history[["Close"]].copy()

meli_prices["log_return"] = np.log(
    meli_prices["Close"] / meli_prices["Close"].shift(1)
)

serie_logreturns = meli_prices["log_return"].dropna()
df_logreturns = serie_logreturns.to_frame(name="returns")

print("\nDescriptive statistics:")
print(df_logreturns.describe())

# Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df_logreturns, x="returns", bins=50)
plt.xlabel("Daily log returns")
plt.ylabel("Frequency")
plt.title("MELI Returns Distribution")
plt.tight_layout()

# Boxplot
plt.figure(figsize=(8, 3))
sns.boxplot(data=df_logreturns, x="returns")
plt.xlabel("Daily log returns")
plt.title("MELI Returns Boxplot")
plt.tight_layout()

# Excess kurtosis
kurtosis = serie_logreturns.kurtosis()
print("\nExcess kurtosis:", kurtosis)

# Historical Value at Risk at 99.9% confidence
alpha = 0.001
historical_var = np.quantile(serie_logreturns, alpha)

# Historical Expected Shortfall
historical_es = serie_logreturns[
    serie_logreturns <= historical_var
].mean()

print("\nHistorical 99.9% VaR:", historical_var)
print("Historical 99.9% ES:", historical_es)

# Fit Student's t distribution
degrees_freedom, location, scale = t.fit(serie_logreturns)

print("\nStudent-t parameters:")
print("Degrees of freedom:", degrees_freedom)
print("Location:", location)
print("Scale:", scale)

# Student-t Value at Risk
student_var = t.ppf(
    alpha,
    degrees_freedom,
    loc=location,
    scale=scale
)

# Student-t Expected Shortfall
student_es = t.expect(
    lambda x: x,
    args=(degrees_freedom,),
    loc=location,
    scale=scale,
    lb=-np.inf,
    ub=student_var
) / alpha

print("\nStudent-t 99.9% VaR:", student_var)
print("Student-t 99.9% ES:", student_es)

# Distribution with density curve
plt.figure(figsize=(8, 4))
sns.histplot(
    data=df_logreturns,
    x="returns",
    kde=True,
    bins=30
)
plt.xlabel("Daily log returns")
plt.ylabel("Frequency")
plt.title("MELI Returns Distribution with KDE")
plt.tight_layout()

# Lag plot
plt.figure(figsize=(6, 5))
pd.plotting.lag_plot(serie_logreturns, lag=1)
plt.title("MELI Returns Lag Plot")
plt.tight_layout()

# Autocorrelation
plot_acf(serie_logreturns, zero=False)
plt.title("MELI Returns Autocorrelation")
plt.tight_layout()

plt.show()
