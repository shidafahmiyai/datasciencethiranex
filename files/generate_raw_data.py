"""
Generates a realistic MESSY raw e-commerce sales dataset for the
Data Cleaning & Visualization project.

Deliberately injects:
- Missing values (various columns)
- Duplicate rows
- Outliers (price, quantity)
- Inconsistent text casing / whitespace
- Inconsistent date formats
- Wrong data types (numbers stored as strings with symbols)
- Invalid / negative values
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 2000

categories = ["Electronics", "Clothing", "Home & Kitchen", "Beauty", "Sports", "Toys", "Books"]
category_variants = {
    "Electronics": ["Electronics", "electronics", "ELECTRONICS", "Electronic ", " Electronics"],
    "Clothing": ["Clothing", "clothing", "CLOTHING", "Cloth ing"],
    "Home & Kitchen": ["Home & Kitchen", "home & kitchen", "Home and Kitchen", "HOME & KITCHEN"],
    "Beauty": ["Beauty", "beauty", "BEAUTY", " Beauty"],
    "Sports": ["Sports", "sports", "SPORTS"],
    "Toys": ["Toys", "toys", "TOYS", "Toy's"],
    "Books": ["Books", "books", "BOOKS"],
}

regions = ["North", "South", "East", "West"]
region_variants = {
    "North": ["North", "north", "NORTH", " North"],
    "South": ["South", "south", "SOUTH"],
    "East": ["East", "east", "EAST", "East "],
    "West": ["West", "west", "WEST"],
}

payment_methods = ["Credit Card", "Debit Card", "PayPal", "UPI", "Cash on Delivery"]

first_names = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda","David","Elizabeth",
               "William","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen",
               "Ananya","Rohan","Priya","Arjun","Meera","Vikram","Neha","Aditya","Kavya","Rahul"]
last_names = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
              "Sharma","Patel","Kumar","Singh","Gupta","Nair","Iyer","Reddy","Rao","Das"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=np.random.randint(0, delta.days))

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

rows = []
for i in range(N):
    cat = np.random.choice(categories)
    reg = np.random.choice(regions)
    order_id = 10000 + i

    # Base price per category (roughly realistic)
    base_price = {
        "Electronics": 250, "Clothing": 40, "Home & Kitchen": 60,
        "Beauty": 25, "Sports": 55, "Toys": 30, "Books": 15
    }[cat]
    price = round(np.random.normal(base_price, base_price * 0.3), 2)
    price = max(price, 3)

    qty = np.random.randint(1, 6)

    order_date = random_date(start_date, end_date)

    name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"

    row = {
        "OrderID": order_id,
        "CustomerName": name,
        "Category": np.random.choice(category_variants[cat]),
        "Region": np.random.choice(region_variants[reg]),
        "PaymentMethod": np.random.choice(payment_methods),
        "Quantity": qty,
        "UnitPrice": price,
        "OrderDate": order_date,
        "Discount_%": round(np.random.choice([0, 0, 0, 5, 10, 15, 20]), 1),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ---- Inject missing values ----
for col, frac in [("CustomerName", 0.03), ("Category", 0.02), ("Region", 0.02),
                   ("PaymentMethod", 0.05), ("UnitPrice", 0.03), ("Quantity", 0.02),
                   ("Discount_%", 0.10)]:
    idx = np.random.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

# ---- Inject outliers ----
outlier_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[outlier_idx, "UnitPrice"] = df.loc[outlier_idx, "UnitPrice"] * np.random.uniform(15, 40)

qty_outlier_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[qty_outlier_idx, "Quantity"] = np.random.randint(150, 500, size=10)

# A few negative / invalid values (data entry errors)
neg_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[neg_idx, "UnitPrice"] = -df.loc[neg_idx, "UnitPrice"]

neg_qty_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[neg_qty_idx, "Quantity"] = -np.random.randint(1, 5, size=5)

# ---- Duplicate rows ----
dup_rows = df.sample(n=40, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# ---- Format inconsistencies ----
# UnitPrice as string with currency symbol for a subset of rows
money_str_idx = np.random.choice(df.index, size=200, replace=False)
df["UnitPrice"] = df["UnitPrice"].astype(object)
for i in money_str_idx:
    val = df.loc[i, "UnitPrice"]
    if pd.notna(val):
        df.loc[i, "UnitPrice"] = f"${val:,.2f}"

# OrderDate in mixed formats
def fmt_date(d, style):
    if pd.isna(d):
        return d
    if style == 0:
        return d.strftime("%Y-%m-%d")
    elif style == 1:
        return d.strftime("%d/%m/%Y")
    elif style == 2:
        return d.strftime("%m-%d-%Y")
    else:
        return d.strftime("%B %d, %Y")

styles = np.random.randint(0, 4, size=len(df))
df["OrderDate"] = [fmt_date(d, s) for d, s in zip(df["OrderDate"], styles)]

# Add stray whitespace to CustomerName randomly
ws_idx = np.random.choice(df.index, size=100, replace=False)
df.loc[ws_idx, "CustomerName"] = df.loc[ws_idx, "CustomerName"].apply(
    lambda x: f"  {x}  " if pd.notna(x) else x
)

# Shuffle rows so duplicates aren't all at the end
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/mnt/user-data/outputs/raw_ecommerce_sales.csv", index=False)
print("Raw dataset created:", df.shape)
print(df.head(10))
print("\nMissing values per column:\n", df.isna().sum())
