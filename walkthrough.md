# Data Cleaning & Visualization Project — Walkthrough
### Dataset: Raw E-Commerce Sales Data (2023–2024)

This project takes a messy, realistic e-commerce sales dataset and walks through
the full pipeline: inspecting it, cleaning it, and turning it into visual insights.

**Files in this project:**
| File | Description |
|---|---|
| `raw_ecommerce_sales.csv` | The original, messy dataset (2,040 rows) |
| `clean_and_visualize.py` | The full cleaning + visualization script (well-commented, runnable) |
| `cleaned_ecommerce_sales.csv` | The final cleaned dataset (1,937 rows) |
| `chart1_revenue_by_category.png` … `chart6_category_region_heatmap.png` | Output visualizations |

---

## 1. The Problem With Raw Data

Before touching anything, we inspected the raw file. It had every classic issue you'll
hit in real-world data:

- **Missing values** scattered across 7 of 9 columns (e.g. 207 missing `Discount_%`, 100 missing `PaymentMethod`)
- **8 exact duplicate rows**, plus 40 near-duplicate rows re-inserted with new `OrderID`s to simulate re-submitted orders
- **Inconsistent text formatting** — `"Electronics"`, `"electronics"`, `"ELECTRONICS"`, `" Electronic "` all meant the same thing
- **Mixed date formats** in a single column — `2024-02-20`, `07/02/2024`, `November 13, 2023` all appearing side by side
- **Numbers stored as text** — some prices were stored as `"$45.99"` instead of `45.99`
- **Outliers and invalid entries** — a handful of prices 15–40x too high, quantities in the hundreds, and even negative prices/quantities from data-entry errors

## 2. Cleaning Steps Taken

The full logic lives in `clean_and_visualize.py`, but here's the summary of every decision made and *why*:

1. **Fixed data types** — stripped `$`/`,` from price strings and converted to `float`; parsed the mixed-format date strings into proper `datetime` objects.
2. **Standardized text** — trimmed whitespace, normalized casing, and mapped known spelling/casing variants (`"Cloth Ing"`, `"Toy'S"`, etc.) to one canonical label per category/region.
3. **Handled missing values** — different strategy per column, chosen deliberately rather than blanket-dropping:
   - Categorical fields (`Category`, `Region`, `PaymentMethod`) → filled with `"Unknown"` so the record isn't lost
   - `CustomerName` missing → row dropped (can't meaningfully attribute an order without a customer)
   - `Quantity` → filled with the column median (robust to outliers)
   - `UnitPrice` → filled with the **median price within that product category** (more accurate than a single global fill value)
   - `Discount_%` → filled with `0` (missing discount reasonably means "no discount applied")
4. **Removed duplicates** — dropped rows identical on every field except `OrderID` (33 rows removed).
5. **Handled outliers & invalid values**:
   - Rows with negative or zero price/quantity were **dropped** (they're data-entry errors, not real orders)
   - Remaining extreme values were detected using the **IQR method** and **capped (winsorized)** at the IQR bounds rather than deleted — this keeps the sample size intact while preventing a few extreme values from distorting totals and averages
6. **Feature engineering** — added `Revenue` (price × quantity × (1 − discount)), plus `OrderMonth`/`OrderYear` for trend analysis.

**Result:** 2,040 raw rows → **1,937 clean rows** (103 rows removed, ~5%), zero missing values remaining, consistent types and formatting throughout.

---

## 3. Key Insights From the Cleaned Data

- **Total revenue:** $264,350.37 across the two-year window (Jan 2023 – Dec 2024)
- **Average order value:** ~$136.47
- **Top category by revenue:** Electronics ($92,689 — more than double the next category)
- **Top region:** West
- **Most-used payment method:** UPI

### Chart 1 — Revenue by Category
Electronics dominates total revenue, generating more than double the next-highest category (Sports). This is expected given electronics naturally carry a higher price point per unit — but it's worth checking whether it's driven by *volume* or just *price* (see Chart 5/6 for more context).

### Chart 2 — Monthly Revenue Trend
Shows how revenue moved month over month across 2023–2024, useful for spotting seasonality (e.g. holiday-season spikes) or growth/decline trends over time.

### Chart 3 — Revenue Share by Region
A pie breakdown of how revenue splits across North/South/East/West — useful for spotting whether the business is regionally balanced or concentrated.

### Chart 4 — Payment Method Popularity
Ranks payment methods by order volume, showing customer payment preferences.

### Chart 5 — Unit Price Distribution
A histogram of cleaned unit prices — after outlier capping, this should look like a much more realistic, non-distorted distribution compared to the raw data (which had prices up to 40x too high).

### Chart 6 — Category × Region Heatmap
Cross-tabulates revenue by category and region simultaneously, making it easy to spot, for example, if Electronics is strong everywhere or concentrated in one region.

---

## 4. How to Reproduce / Extend This

```bash
# Re-run the full pipeline from scratch
python3 generate_raw_data.py      # (only needed if regenerating raw data)
python3 clean_and_visualize.py    # cleans data + produces all 6 charts
```

Ideas to take this further:
- Swap in your own raw CSV (just match the column names, or adjust the script)
- Add a customer-level lifetime value (LTV) analysis
- Build the charts into an interactive dashboard (Plotly Dash / Streamlit) instead of static PNGs
- Add cohort or retention analysis using `OrderDate` and `CustomerName`
