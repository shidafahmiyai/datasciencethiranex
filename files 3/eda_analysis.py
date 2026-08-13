"""
Exploratory Data Analysis (EDA) — Titanic Dataset
====================================================
Full EDA workflow:
  1. Load & inspect data (shape, types, missing values)
  2. Statistical summaries
  3. Univariate & bivariate visualizations
  4. Correlation analysis
  5. Identify key influencing factors (what drives survival?)

To use your own data: replace the "LOAD DATA" section with
    df = pd.read_csv("your_file.csv")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "/home/claude"

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = sns.load_dataset("titanic")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("Data types:")
print(df.dtypes)

# ----------------------------------------------------------------------
# 2. MISSING VALUES
# ----------------------------------------------------------------------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
print("\nMissing values:")
print(missing_summary)

# ----------------------------------------------------------------------
# 3. STATISTICAL SUMMARY
# ----------------------------------------------------------------------
numeric_summary = df.describe().round(2)
print("\nNumeric summary:")
print(numeric_summary)

categorical_cols = ["sex", "pclass", "embarked", "class", "who", "alone"]
print("\nCategorical value counts:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ----------------------------------------------------------------------
# 4. VISUALIZATION 1 — Survival rate by key categorical factors
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.barplot(data=df, x="pclass", y="survived", ax=axes[0, 0], palette="Blues_d")
axes[0, 0].set_title("Survival Rate by Passenger Class")
axes[0, 0].set_ylabel("Survival Rate")

sns.barplot(data=df, x="sex", y="survived", ax=axes[0, 1], palette="Blues_d")
axes[0, 1].set_title("Survival Rate by Sex")
axes[0, 1].set_ylabel("Survival Rate")

sns.barplot(data=df, x="embarked", y="survived", ax=axes[1, 0], palette="Blues_d")
axes[1, 0].set_title("Survival Rate by Embarkation Port")
axes[1, 0].set_ylabel("Survival Rate")

sns.barplot(data=df, x="alone", y="survived", ax=axes[1, 1], palette="Blues_d")
axes[1, 1].set_title("Survival Rate: Alone vs. With Family")
axes[1, 1].set_ylabel("Survival Rate")

plt.tight_layout()
plt.savefig(f"{OUT}/survival_by_category.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. VISUALIZATION 2 — Distributions of numeric variables
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.histplot(df["age"].dropna(), bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Age Distribution")

sns.histplot(df["fare"], bins=40, kde=True, ax=axes[1], color="steelblue")
axes[1].set_title("Fare Distribution")

sns.boxplot(data=df, x="pclass", y="age", ax=axes[2], palette="Blues")
axes[2].set_title("Age Distribution by Class")

plt.tight_layout()
plt.savefig(f"{OUT}/distributions.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. VISUALIZATION 3 — Correlation heatmap
# ----------------------------------------------------------------------
numeric_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].copy()
corr = numeric_df.corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 7. VISUALIZATION 4 — Age vs Fare, colored by survival
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, palette=["#d62728", "#2ca02c"])
plt.title("Age vs. Fare, Colored by Survival")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(f"{OUT}/age_vs_fare.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. KEY INFLUENCING FACTORS — correlation with target, sorted
# ----------------------------------------------------------------------
survival_corr = corr["survived"].drop("survived").sort_values(key=abs, ascending=False)
print("\nCorrelation with survival (sorted by strength):")
print(survival_corr)

print("\nSaved plots: survival_by_category.png, distributions.png, correlation_heatmap.png, age_vs_fare.png")
