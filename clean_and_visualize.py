"""
==========================================================
DATA CLEANING & VISUALIZATION PROJECT
Dataset: Raw E-Commerce Sales Data
==========================================================
This script walks through a full data cleaning pipeline:
  1. Load & inspect raw data
  2. Fix data types
  3. Standardize text formatting
  4. Handle missing values
  5. Handle duplicates
  6. Handle outliers
  7. Feature engineering
  8. Export cleaned dataset
  9. Visualize key insights
==========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

OUT = "/mnt/user-data/outputs"

# ----------------------------------------------------------------
# STEP 1: LOAD & INSPECT
# ----------------------------------------------------------------
print("="*60)
print("STEP 1: LOAD & INSPECT RAW DATA")
print("="*60)

df = pd.read_csv(f"{OUT}/raw_ecommerce_sales.csv")
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isna().sum()}")
print(f"\nDuplicate rows (exact, incl. OrderID): {df.duplicated().sum()}")
print(f"Duplicate rows (excl. OrderID): {df.drop(columns=['OrderID']).duplicated().sum()}")

n_raw = len(df)

# ----------------------------------------------------------------
# STEP 2: FIX DATA TYPES
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 2: FIX DATA TYPES")
print("="*60)

# UnitPrice: strip "$" and "," then convert to float
df["UnitPrice"] = (
    df["UnitPrice"].astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .replace("nan", np.nan)
    .astype(float)
)

# OrderDate: mixed formats -> parse robustly with dateutil via pandas
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce", format="mixed")

print("UnitPrice and OrderDate converted to proper numeric/datetime types.")
print(f"Unparseable dates after conversion: {df['OrderDate'].isna().sum()}")

# ----------------------------------------------------------------
# STEP 3: STANDARDIZE TEXT FORMATTING
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 3: STANDARDIZE TEXT FORMATTING")
print("="*60)

text_cols = ["CustomerName", "Category", "Region", "PaymentMethod"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", np.nan)

# Title-case categorical columns and collapse variant spellings
df["Category"] = df["Category"].str.title().str.strip()
# Map known messy variants (extra spaces, truncation, apostrophes) to a canonical label
category_map = {
    "Cloth Ing": "Clothing", "Clothing": "Clothing",
    "Electronic": "Electronics", "Electronics": "Electronics",
    "Home And Kitchen": "Home & Kitchen", "Home & Kitchen": "Home & Kitchen",
    "Toy'S": "Toys", "Toys": "Toys",
    "Beauty": "Beauty", "Sports": "Sports", "Books": "Books",
    "Unknown": "Unknown",
}
df["Category"] = df["Category"].map(category_map).fillna(df["Category"])
df["Region"] = df["Region"].str.title()
df["CustomerName"] = df["CustomerName"].str.title()

print("Category unique values after cleaning:", sorted(df["Category"].dropna().unique()))
print("Region unique values after cleaning:", sorted(df["Region"].dropna().unique()))

# ----------------------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 4: HANDLE MISSING VALUES")
print("="*60)
print(f"Missing before:\n{df.isna().sum()}\n")

# Categorical: fill with "Unknown" (preserves the record instead of dropping)
for col in ["Category", "Region", "PaymentMethod"]:
    df[col] = df[col].fillna("Unknown")

# CustomerName: drop rows with no name - can't attribute an order to anyone meaningfully,
# and it's a small fraction of rows
df = df.dropna(subset=["CustomerName"])

# Quantity: impute with median (robust to outliers), rounded to int
qty_median = df["Quantity"].median()
df["Quantity"] = df["Quantity"].fillna(qty_median)

# UnitPrice: impute with category-level median (more accurate than global median)
df["UnitPrice"] = df.groupby("Category")["UnitPrice"].transform(
    lambda x: x.fillna(x.median())
)
df["UnitPrice"] = df["UnitPrice"].fillna(df["UnitPrice"].median())

# Discount_%: missing likely means no discount was applied
df["Discount_%"] = df["Discount_%"].fillna(0)

print(f"Missing after:\n{df.isna().sum()}")

# ----------------------------------------------------------------
# STEP 5: HANDLE DUPLICATES
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 5: HANDLE DUPLICATES")
print("="*60)

before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "OrderID"])
after = len(df)
print(f"Removed {before - after} duplicate rows (same order details, different/duplicate OrderID).")

# ----------------------------------------------------------------
# STEP 6: HANDLE OUTLIERS & INVALID VALUES
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 6: HANDLE OUTLIERS & INVALID VALUES")
print("="*60)

# Invalid negatives -> take absolute value (data entry sign errors) is risky;
# safer treatment: negative price/qty are invalid records -> drop
invalid_mask = (df["UnitPrice"] <= 0) | (df["Quantity"] <= 0)
print(f"Dropping {invalid_mask.sum()} rows with non-positive price or quantity (invalid entries).")
df = df[~invalid_mask]

# Outliers via IQR method on UnitPrice and Quantity
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

price_low, price_high = iqr_bounds(df["UnitPrice"])
qty_low, qty_high = iqr_bounds(df["Quantity"])
print(f"UnitPrice IQR bounds: [{price_low:.2f}, {price_high:.2f}]")
print(f"Quantity IQR bounds: [{qty_low:.2f}, {qty_high:.2f}]")

price_outliers = ((df["UnitPrice"] < price_low) | (df["UnitPrice"] > price_high)).sum()
qty_outliers = ((df["Quantity"] < qty_low) | (df["Quantity"] > qty_high)).sum()
print(f"UnitPrice outliers found: {price_outliers}")
print(f"Quantity outliers found: {qty_outliers}")

# Cap (winsorize) rather than drop, to preserve sample size while limiting distortion
df["UnitPrice"] = df["UnitPrice"].clip(lower=price_low, upper=price_high)
df["Quantity"] = df["Quantity"].clip(lower=qty_low, upper=qty_high).round().astype(int)

print("Outliers capped (winsorized) at IQR bounds rather than dropped.")

# ----------------------------------------------------------------
# STEP 7: FEATURE ENGINEERING
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 7: FEATURE ENGINEERING")
print("="*60)

df["Revenue"] = df["UnitPrice"] * df["Quantity"] * (1 - df["Discount_%"] / 100)
df["Revenue"] = df["Revenue"].round(2)
df["OrderMonth"] = df["OrderDate"].dt.to_period("M").astype(str)
df["OrderYear"] = df["OrderDate"].dt.year

print("Added columns: Revenue, OrderMonth, OrderYear")

# ----------------------------------------------------------------
# STEP 8: EXPORT CLEANED DATASET
# ----------------------------------------------------------------
print("\n" + "="*60)
print("STEP 8: EXPORT CLEANED DATASET")
print("="*60)

df = df.sort_values("OrderDate").reset_index(drop=True)
df.to_csv(f"{OUT}/cleaned_ecommerce_sales.csv", index=False)

n_clean = len(df)
print(f"Raw rows: {n_raw}  ->  Clean rows: {n_clean}  (removed {n_raw - n_clean}, "
      f"{(n_raw-n_clean)/n_raw*100:.1f}%)")
print(f"Saved to {OUT}/cleaned_ecommerce_sales.csv")

# ==================================================================
# STEP 9: VISUALIZATIONS
# ==================================================================
print("\n" + "="*60)
print("STEP 9: GENERATE VISUALIZATIONS")
print("="*60)

palette = sns.color_palette("viridis", 8)

# --- Chart 1: Revenue by Category ---
fig, ax = plt.subplots(figsize=(9, 5.5))
rev_by_cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
sns.barplot(x=rev_by_cat.values, y=rev_by_cat.index, hue=rev_by_cat.index,
            palette="viridis", legend=False, ax=ax)
ax.set_xlabel("Total Revenue ($)")
ax.set_ylabel("")
ax.set_title("Total Revenue by Product Category", fontsize=14, fontweight="bold")
for i, v in enumerate(rev_by_cat.values):
    ax.text(v, i, f"  ${v:,.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/chart1_revenue_by_category.png")
plt.close()

# --- Chart 2: Monthly Revenue Trend ---
fig, ax = plt.subplots(figsize=(11, 5.5))
monthly = df.groupby("OrderMonth")["Revenue"].sum().sort_index()
ax.plot(monthly.index, monthly.values, marker="o", color="#2c7fb8", linewidth=2)
ax.set_title("Monthly Revenue Trend (2023–2024)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/chart2_monthly_revenue_trend.png")
plt.close()

# --- Chart 3: Revenue Share by Region ---
fig, ax = plt.subplots(figsize=(7, 7))
rev_by_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
colors = sns.color_palette("viridis", len(rev_by_region))
ax.pie(rev_by_region.values, labels=rev_by_region.index, autopct="%1.1f%%",
       colors=colors, startangle=90, textprops={"fontsize": 10})
ax.set_title("Revenue Share by Region", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/chart3_revenue_share_by_region.png")
plt.close()

# --- Chart 4: Payment Method Popularity ---
fig, ax = plt.subplots(figsize=(9, 5.5))
pay_counts = df["PaymentMethod"].value_counts()
sns.barplot(x=pay_counts.values, y=pay_counts.index, hue=pay_counts.index,
            palette="mako", legend=False, ax=ax)
ax.set_xlabel("Number of Orders")
ax.set_ylabel("")
ax.set_title("Order Volume by Payment Method", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/chart4_payment_method_popularity.png")
plt.close()

# --- Chart 5: Unit Price Distribution (before/after outlier handling illustration) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.histplot(df["UnitPrice"], bins=40, kde=True, color="#31688e", ax=ax)
ax.set_title("Distribution of Unit Price (after cleaning)", fontsize=14, fontweight="bold")
ax.set_xlabel("Unit Price ($)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/chart5_unitprice_distribution.png")
plt.close()

# --- Chart 6: Category x Region heatmap of revenue ---
fig, ax = plt.subplots(figsize=(9, 6))
pivot = df.pivot_table(values="Revenue", index="Category", columns="Region", aggfunc="sum", fill_value=0)
sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="viridis", ax=ax, cbar_kws={"label": "Revenue ($)"})
ax.set_title("Revenue Heatmap: Category vs Region", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/chart6_category_region_heatmap.png")
plt.close()

print("Saved 6 charts to", OUT)

# ----------------------------------------------------------------
# Summary stats for the walkthrough doc
# ----------------------------------------------------------------
summary = {
    "n_raw": n_raw,
    "n_clean": n_clean,
    "rows_removed": n_raw - n_clean,
    "total_revenue": df["Revenue"].sum(),
    "avg_order_value": df["Revenue"].mean(),
    "top_category": rev_by_cat.index[0],
    "top_category_revenue": rev_by_cat.values[0],
    "top_region": rev_by_region.index[0],
    "top_payment": pay_counts.index[0],
    "date_min": df["OrderDate"].min(),
    "date_max": df["OrderDate"].max(),
}
import json
with open(f"{OUT}/_summary_stats.json", "w") as f:
    json.dump({k: str(v) for k, v in summary.items()}, f, indent=2)

print("\nDone.")
