"""
Real-World Data Project: Finance (Stock Price Analysis & Prediction)
========================================================================
End-to-end pipeline:
  1. Generate/load daily OHLCV stock data
  2. EDA: trend, returns, volatility, volume patterns
  3. Feature engineering (technical indicators)
  4. Predictive model: next-day price direction (up/down classification)
  5. Evaluation + visualizations

NOTE ON DATA: This sandbox has no internet access to live market data
providers (e.g. Yahoo Finance), so we simulate a realistic daily price
series using Geometric Brownian Motion (GBM) — the standard stochastic
process used to model stock prices — with added volatility clustering
so it behaves like a real, messy market series rather than a smooth
random walk.

To use REAL data instead: replace the "GENERATE DATA" section with
    df = pd.read_csv("your_prices.csv", parse_dates=["date"])
expecting columns: date, open, high, low, close, volume
(e.g. exported from yfinance, Alpha Vantage, or your broker)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, classification_report
)

sns.set_theme(style="whitegrid")
OUT = "/home/claude"
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. GENERATE DATA (simulated realistic daily OHLCV series, ~3 years)
# ----------------------------------------------------------------------
n_days = 756  # ~3 trading years
dates = pd.bdate_range(start="2023-01-02", periods=n_days)

mu = 0.0004       # slight daily upward drift
base_vol = 0.012  # baseline daily volatility

# Simulate volatility clustering (GARCH-like) so vol isn't constant
vol = np.zeros(n_days)
vol[0] = base_vol
for t in range(1, n_days):
    vol[t] = 0.05 * base_vol + 0.9 * vol[t - 1] + 0.05 * base_vol * np.random.rand()

returns = np.random.normal(mu, 1, n_days) * vol
close = 150 * np.exp(np.cumsum(returns))  # start price $150

# Build OHLC around the close using intraday noise
open_ = close * (1 + np.random.normal(0, 0.003, n_days))
high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.004, n_days)))
low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.004, n_days)))
volume = np.random.lognormal(mean=15, sigma=0.3, size=n_days).astype(int)
volume = volume + (np.abs(returns) * 5_000_000).astype(int)  # higher volume on big moves

df = pd.DataFrame({
    "date": dates, "open": open_, "high": high, "low": low,
    "close": close, "volume": volume
})

print(f"Generated {len(df)} trading days from {df['date'].min().date()} to {df['date'].max().date()}")
print(df.head())

# ----------------------------------------------------------------------
# 2. EDA — price trend with moving averages
# ----------------------------------------------------------------------
df["ma20"] = df["close"].rolling(20).mean()
df["ma50"] = df["close"].rolling(50).mean()
df["ma200"] = df["close"].rolling(200).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(df["date"], df["close"], label="Close", color="black", linewidth=1)
ax1.plot(df["date"], df["ma20"], label="20-day MA", alpha=0.8)
ax1.plot(df["date"], df["ma50"], label="50-day MA", alpha=0.8)
ax1.plot(df["date"], df["ma200"], label="200-day MA", alpha=0.8)
ax1.set_title("Price Trend with Moving Averages")
ax1.set_ylabel("Price ($)")
ax1.legend(loc="upper left")

ax2.bar(df["date"], df["volume"], color="steelblue", width=1.5)
ax2.set_title("Trading Volume")
ax2.set_ylabel("Volume")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.tight_layout()
plt.savefig(f"{OUT}/price_trend.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. EDA — daily returns distribution & volatility
# ----------------------------------------------------------------------
df["daily_return"] = df["close"].pct_change()
df["volatility_20d"] = df["daily_return"].rolling(20).std()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.histplot(df["daily_return"].dropna(), bins=50, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Daily Return Distribution")
axes[0].axvline(0, color="red", linestyle="--", linewidth=1)

axes[1].plot(df["date"], df["volatility_20d"], color="darkorange")
axes[1].set_title("20-Day Rolling Volatility")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

sns.boxplot(y=df["daily_return"].dropna(), ax=axes[2], color="steelblue")
axes[2].set_title("Return Outliers")

plt.tight_layout()
plt.savefig(f"{OUT}/returns_volatility.png", dpi=150)
plt.close()

print(f"\nMean daily return: {df['daily_return'].mean():.5f}")
print(f"Daily volatility (std): {df['daily_return'].std():.5f}")
print(f"Annualized volatility: {df['daily_return'].std() * np.sqrt(252):.3f}")
print(f"Max single-day gain: {df['daily_return'].max():.3%}")
print(f"Max single-day loss: {df['daily_return'].min():.3%}")

# ----------------------------------------------------------------------
# 4. FEATURE ENGINEERING (technical indicators for prediction)
# ----------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["rsi_14"] = compute_rsi(df["close"])
df["return_lag1"] = df["daily_return"].shift(1)
df["return_lag2"] = df["daily_return"].shift(2)
df["return_lag3"] = df["daily_return"].shift(3)
df["ma20_ma50_ratio"] = df["ma20"] / df["ma50"]
df["price_vs_ma20"] = df["close"] / df["ma20"] - 1
df["volume_change"] = df["volume"].pct_change()

# Target: will tomorrow's close be higher than today's? (1 = up, 0 = down)
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

feature_cols = [
    "return_lag1", "return_lag2", "return_lag3", "rsi_14",
    "volatility_20d", "ma20_ma50_ratio", "price_vs_ma20", "volume_change"
]

model_df = df.dropna(subset=feature_cols + ["target"]).copy()
print(f"\nModeling dataset: {len(model_df)} rows after dropping NaNs from rolling features")
print(f"Class balance (target): {model_df['target'].value_counts(normalize=True).round(3).to_dict()}")

# ----------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT (chronological — no shuffling for time series!)
# ----------------------------------------------------------------------
split_idx = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 6. TRAIN MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }

baseline_acc = max(y_test.mean(), 1 - y_test.mean())  # naive "always predict majority class"

print("\nModel results (test set, chronological holdout):")
for name, r in results.items():
    print(f"  {name}: accuracy={r['accuracy']:.3f}, AUC={r['auc']:.3f}")
print(f"  Naive baseline (always predict majority class): {baseline_acc:.3f}")

# ----------------------------------------------------------------------
# 7. VISUALIZE — confusion matrices + ROC curves
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.3f}")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Next-Day Direction Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest)
# ----------------------------------------------------------------------
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

print("\nSaved plots: price_trend.png, returns_volatility.png, confusion_matrices.png, roc_curves.png, feature_importance.png")
