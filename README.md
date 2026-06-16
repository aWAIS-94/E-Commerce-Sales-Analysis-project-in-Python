# E-Commerce Sales Analysis

A production-ready Python analysis project covering data cleaning, KPI generation, trend analysis, customer segmentation, and interactive visualisations — all in a single `main.py`.

---

## Project Structure

```
ecom_analysis/
├── main.py           ← entire project lives here
├── requirements.txt
├── README.md
└── output/           ← auto-created on first run
    ├── 01_monthly_trend.png
    ├── 02_top_products.png
    ├── 03_category_analysis.png
    ├── 04_region_revenue.png
    ├── 05_customer_segments.png
    ├── 06_heatmap_weekday_month.png
    ├── 07_payment_methods.html      ← interactive
    ├── 08_interactive_trend.html    ← interactive
    └── 09_discount_vs_revenue.html  ← interactive
```

---

## Quick Start

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

All charts land in the `output/` folder automatically.

---

## Using Your Own CSV

Replace the `generate_dataset()` call inside `main()` with:

```python
df_raw = pd.read_csv("sales.csv", parse_dates=["order_date"])
```

### Expected Column Schema

| Column | Type | Example |
|---|---|---|
| `order_id` | string | ORD-000123 |
| `order_date` | date | 2024-03-15 |
| `customer_id` | string | CUST-4821 |
| `product` | string | Laptop |
| `category` | string | Electronics |
| `region` | string | North |
| `quantity` | integer | 2 |
| `unit_price` | float | 799.99 |
| `discount_pct` | float | 10 |
| `revenue` | float | 1439.98 |
| `payment_method` | string | Credit Card |

Columns with missing values are handled automatically:
- `revenue` — recalculated from `unit_price × quantity × (1 – discount/100)`
- `discount_pct` — filled with 0

---

## What the Script Does

| Section | Function | Output |
|---|---|---|
| Data Generation | `generate_dataset()` | 12,000-row synthetic dataset |
| Data Cleaning | `clean_data()` | Dedup, dtype fix, NaN fill |
| KPIs | `compute_kpis()` | Printed to terminal |
| Monthly Trend | `plot_monthly_trend()` | `01_monthly_trend.png` |
| Top Products | `plot_top_products()` | `02_top_products.png` |
| Categories | `plot_category_analysis()` | `03_category_analysis.png` |
| Regions | `plot_region_revenue()` | `04_region_revenue.png` |
| Customer Segments | `plot_customer_analysis()` | `05_customer_segments.png` |
| Weekday Heatmap | `plot_heatmap()` | `06_heatmap_weekday_month.png` |
| Payment Methods | `plot_payment_methods()` | `07_payment_methods.html` |
| Interactive Trend | `plot_interactive_trend()` | `08_interactive_trend.html` |
| Discount Impact | `plot_discount_impact()` | `09_discount_vs_revenue.html` |
| Business Insights | `print_insights()` | Printed to terminal |

---

## KPIs Generated

- **Total Revenue** — sum of all order revenues
- **Total Orders** — count of unique order IDs
- **Average Order Value** — mean revenue per order
- **Unique Customers** — distinct customer IDs
- **Total Units Sold** — sum of quantity column
- **Average Discount (%)** — mean discount applied
- **Best Month** — highest-revenue calendar month

---

## Sample Terminal Output

```
══════════════════════════════════════════
  KEY PERFORMANCE INDICATORS
══════════════════════════════════════════
  Total Revenue ($)          4,823,610.45
  Total Orders               11,950
  Avg Order Value ($)        403.64
  Unique Customers           8,712
  Total Units Sold           35,821
  Avg Discount (%)           5.2
  Best Month                 2023-11
══════════════════════════════════════════
```

---

## Tech Stack

| Library | Used For |
|---|---|
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numeric operations, random data generation |
| `matplotlib` | Static charts (PNG) |
| `seaborn` | Styled statistical plots |
| `plotly` | Interactive HTML charts |
