"""
E-Commerce Sales Analysis
Author : Senior Data Analyst
Purpose: End-to-end sales analysis — cleaning, KPIs, trends, and charts.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET  (swap with pd.read_csv("sales.csv"))
# ──────────────────────────────────────────────────────────────

def generate_dataset(n: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """
    Produces a realistic e-commerce dataset.
    Column schema mirrors common real-world exports (Shopify / WooCommerce style).
    """
    rng = np.random.default_rng(seed)

    categories = {
        "Electronics":   ["Laptop", "Headphones", "Smartwatch", "Tablet", "Keyboard",
                          "Monitor", "Webcam", "Speaker"],
        "Clothing":      ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress", "Hoodie"],
        "Home & Kitchen":["Coffee Maker", "Blender", "Air Fryer", "Cookware Set",
                          "Knife Set", "Vacuum Cleaner"],
        "Books":         ["Python Crash Course", "Data Science Handbook", "Clean Code",
                          "Atomic Habits", "Deep Work"],
        "Sports":        ["Yoga Mat", "Resistance Bands", "Dumbbells", "Running Shoes",
                          "Cycling Gloves"],
    }
    regions = ["North", "South", "East", "West", "Central"]

    # Build category / product columns
    cat_list, prod_list = [], []
    for cat, prods in categories.items():
        cat_list.extend([cat] * len(prods))
        prod_list.extend(prods)

    chosen_idx   = rng.integers(0, len(prod_list), n)
    category_col = [cat_list[i] for i in chosen_idx]
    product_col  = [prod_list[i] for i in chosen_idx]

    # Price ranges differ by category
    price_map = {
        "Electronics": (50, 1500), "Clothing": (15, 200),
        "Home & Kitchen": (25, 350), "Books": (10, 60), "Sports": (10, 250),
    }
    unit_prices = np.array([
        rng.uniform(*price_map[c]) for c in category_col
    ]).round(2)

    quantity     = rng.integers(1, 6, n)
    discount_pct = rng.choice([0, 5, 10, 15, 20], n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    revenue      = (unit_prices * quantity * (1 - discount_pct / 100)).round(2)

    # Dates spread across 2 full years with a seasonal boost in Nov-Dec
    base_dates  = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    weights     = np.array([
        1.5 if d.month in (11, 12) else (1.2 if d.month in (6, 7) else 1.0)
        for d in base_dates
    ])
    weights    /= weights.sum()
    order_dates = rng.choice(base_dates, n, p=weights, replace=True)

    customers = [f"CUST-{rng.integers(1000, 9999)}" for _ in range(n)]
    orders    = [f"ORD-{str(i).zfill(6)}" for i in range(n)]

    df = pd.DataFrame({
        "order_id":      orders,
        "order_date":    order_dates,
        "customer_id":   customers,
        "product":       product_col,
        "category":      category_col,
        "region":        rng.choice(regions, n),
        "quantity":      quantity,
        "unit_price":    unit_prices,
        "discount_pct":  discount_pct,
        "revenue":       revenue,
        "payment_method": rng.choice(["Credit Card", "PayPal", "Debit Card", "UPI"], n,
                                     p=[0.45, 0.25, 0.20, 0.10]),
    })

    # Inject noise: missing values + duplicate rows
    null_idx = rng.choice(df.index, 120, replace=False)
    df.loc[null_idx[:60], "revenue"]  = np.nan
    df.loc[null_idx[60:], "discount_pct"] = np.nan
    df = pd.concat([df, df.sample(50, random_state=1)], ignore_index=True)

    return df


# ──────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ──────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[clean] Raw rows      : {len(df):,}")

    df.drop_duplicates(subset="order_id", inplace=True)
    print(f"[clean] After dedup   : {len(df):,}")

    # Fix dtypes
    df["order_date"]   = pd.to_datetime(df["order_date"], errors="coerce")
    df["unit_price"]   = pd.to_numeric(df["unit_price"],  errors="coerce")
    df["revenue"]      = pd.to_numeric(df["revenue"],     errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"],errors="coerce")
    df["quantity"]     = pd.to_numeric(df["quantity"],    errors="coerce").astype("Int64")

    # Fill missing revenue from unit_price × quantity
    mask = df["revenue"].isna()
    df.loc[mask, "revenue"] = (
        df.loc[mask, "unit_price"] * df.loc[mask, "quantity"].astype(float)
        * (1 - df.loc[mask, "discount_pct"].fillna(0) / 100)
    ).round(2)

    df["discount_pct"].fillna(0, inplace=True)

    # Drop rows where we still can't compute revenue
    df.dropna(subset=["revenue", "order_date"], inplace=True)

    # Derived time columns used throughout the analysis
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["year_month"] = df["order_date"].dt.to_period("M")
    df["weekday"]    = df["order_date"].dt.day_name()

    print(f"[clean] Final rows    : {len(df):,}")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 3. KPI SUMMARY
# ──────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {
        "Total Revenue ($)":       round(df["revenue"].sum(), 2),
        "Total Orders":            df["order_id"].nunique(),
        "Avg Order Value ($)":     round(df["revenue"].mean(), 2),
        "Unique Customers":        df["customer_id"].nunique(),
        "Total Units Sold":        int(df["quantity"].sum()),
        "Avg Discount (%)":        round(df["discount_pct"].mean(), 1),
        "Best Month":              df.groupby("year_month")["revenue"].sum().idxmax(),
    }

    print("\n" + "═" * 42)
    print("  KEY PERFORMANCE INDICATORS")
    print("═" * 42)
    for k, v in kpis.items():
        print(f"  {k:<26} {v}")
    print("═" * 42 + "\n")
    return kpis


# ──────────────────────────────────────────────────────────────
# 4. VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────

def save_fig(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def fmt_millions(x, _):
    return f"${x/1_000:,.0f}K" if x < 1_000_000 else f"${x/1_000_000:.1f}M"


# ──────────────────────────────────────────────────────────────
# 5. MONTHLY SALES TREND  (line chart – matplotlib)
# ──────────────────────────────────────────────────────────────

def plot_monthly_trend(df: pd.DataFrame):
    monthly = (
        df.groupby(["year", "month"])["revenue"]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
    )
    monthly["label"] = monthly.apply(
        lambda r: pd.Timestamp(int(r.year), int(r.month), 1).strftime("%b %Y"), axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    for yr, grp in monthly.groupby("year"):
        ax.plot(grp["label"], grp["revenue"], marker="o", linewidth=2.2,
                markersize=5, label=str(yr))
        ax.fill_between(grp["label"], grp["revenue"], alpha=0.08)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Monthly Revenue Trend", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.set_xticklabels(monthly["label"].unique(), rotation=45, ha="right", fontsize=8)
    ax.legend(title="Year")
    fig.tight_layout()
    save_fig(fig, "01_monthly_trend.png")


# ──────────────────────────────────────────────────────────────
# 6. TOP 10 PRODUCTS  (horizontal bar – seaborn)
# ──────────────────────────────────────────────────────────────

def plot_top_products(df: pd.DataFrame):
    top10 = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(data=top10, y="product", x="revenue", ax=ax,
                       palette="Blues_r", edgecolor="white")

    for bar in bars.patches:
        ax.text(
            bar.get_width() + top10["revenue"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt_millions(bar.get_width(), None),
            va="center", fontsize=9,
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Top 10 Products by Revenue", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Total Revenue")
    ax.set_ylabel("")
    fig.tight_layout()
    save_fig(fig, "02_top_products.png")


# ──────────────────────────────────────────────────────────────
# 7. REVENUE BY CATEGORY  (pie + bar side-by-side)
# ──────────────────────────────────────────────────────────────

def plot_category_analysis(df: pd.DataFrame):
    cat_rev = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie
    explode = [0.04] * len(cat_rev)
    ax1.pie(
        cat_rev["revenue"],
        labels=cat_rev["category"],
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
        colors=sns.color_palette("Set2", len(cat_rev)),
        textprops={"fontsize": 10},
    )
    ax1.set_title("Revenue Share by Category", fontsize=13, weight="bold")

    # Bar
    sns.barplot(data=cat_rev, x="category", y="revenue", ax=ax2,
                palette="Set2", edgecolor="white")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax2.set_title("Revenue by Category", fontsize=13, weight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Total Revenue")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha="right")

    for p in ax2.patches:
        ax2.annotate(
            fmt_millions(p.get_height(), None),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    save_fig(fig, "03_category_analysis.png")


# ──────────────────────────────────────────────────────────────
# 8. REVENUE BY REGION  (bar chart)
# ──────────────────────────────────────────────────────────────

def plot_region_revenue(df: pd.DataFrame):
    region_rev = (
        df.groupby("region")
        .agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("coolwarm", len(region_rev))
    bars = sns.barplot(data=region_rev, x="region", y="revenue", ax=ax,
                       palette=palette, edgecolor="white")

    for p in bars.patches:
        ax.annotate(
            fmt_millions(p.get_height(), None),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=10, weight="bold",
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.set_title("Revenue by Region", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Revenue")
    fig.tight_layout()
    save_fig(fig, "04_region_revenue.png")


# ──────────────────────────────────────────────────────────────
# 9. CUSTOMER ANALYSIS  (order frequency distribution)
# ──────────────────────────────────────────────────────────────

def plot_customer_analysis(df: pd.DataFrame):
    cust = (
        df.groupby("customer_id")
        .agg(orders=("order_id", "nunique"), revenue=("revenue", "sum"))
        .reset_index()
    )

    # Segment by order count
    cust["segment"] = pd.cut(
        cust["orders"],
        bins=[0, 1, 3, 6, 999],
        labels=["One-time", "Occasional (2-3)", "Regular (4-6)", "Loyal (7+)"],
    )

    seg_summary = (
        cust.groupby("segment", observed=True)
        .agg(customers=("customer_id", "count"), revenue=("revenue", "sum"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Customer count per segment
    sns.barplot(data=seg_summary, x="segment", y="customers", ax=axes[0],
                palette="viridis", edgecolor="white")
    axes[0].set_title("Customers by Segment", fontsize=13, weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("# Customers")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9)

    # Revenue per segment
    sns.barplot(data=seg_summary, x="segment", y="revenue", ax=axes[1],
                palette="viridis", edgecolor="white")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    axes[1].set_title("Revenue by Customer Segment", fontsize=13, weight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total Revenue")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    save_fig(fig, "05_customer_segments.png")


# ──────────────────────────────────────────────────────────────
# 10. WEEKDAY × MONTH HEATMAP  (seaborn)
# ──────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame):
    day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    pivot = (
        df.groupby(["weekday", "month_name"])["revenue"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=day_order, columns=month_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        pivot / 1_000, ax=ax, cmap="YlOrRd", linewidths=0.5,
        annot=True, fmt=".0f", annot_kws={"size": 8},
        cbar_kws={"label": "Revenue ($K)"},
    )
    ax.set_title("Revenue Heatmap – Weekday × Month ($K)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    save_fig(fig, "06_heatmap_weekday_month.png")


# ──────────────────────────────────────────────────────────────
# 11. PAYMENT METHOD PIE  (plotly – saved as HTML)
# ──────────────────────────────────────────────────────────────

def plot_payment_methods(df: pd.DataFrame):
    pm = df["payment_method"].value_counts().reset_index()
    pm.columns = ["method", "count"]

    fig = px.pie(
        pm, names="method", values="count",
        title="Orders by Payment Method",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.35,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False, title_font_size=16)

    path = os.path.join(OUTPUT_DIR, "07_payment_methods.html")
    fig.write_html(path)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────
# 12. INTERACTIVE MONTHLY TREND  (plotly – saved as HTML)
# ──────────────────────────────────────────────────────────────

def plot_interactive_trend(df: pd.DataFrame):
    monthly = (
        df.groupby("year_month")
        .agg(revenue=("revenue","sum"), orders=("order_id","nunique"))
        .reset_index()
    )
    monthly["period"] = monthly["year_month"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=monthly["period"], y=monthly["revenue"],
                   name="Revenue", mode="lines+markers",
                   line=dict(color="#2196F3", width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=monthly["period"], y=monthly["orders"],
               name="Orders", opacity=0.35,
               marker_color="#FFC107"),
        secondary_y=True,
    )

    fig.update_layout(
        title="Monthly Revenue & Orders (Interactive)",
        xaxis_title="Month",
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified",
        title_font_size=16,
    )
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Order Count", secondary_y=True)

    path = os.path.join(OUTPUT_DIR, "08_interactive_trend.html")
    fig.write_html(path)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────
# 13. DISCOUNT vs REVENUE SCATTER  (plotly)
# ──────────────────────────────────────────────────────────────

def plot_discount_impact(df: pd.DataFrame):
    sample = df.sample(min(3000, len(df)), random_state=42)

    fig = px.scatter(
        sample, x="discount_pct", y="revenue", color="category",
        size="quantity", hover_data=["product", "region"],
        title="Discount % vs Revenue by Category",
        labels={"discount_pct": "Discount (%)", "revenue": "Revenue ($)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.65,
    )
    fig.update_layout(title_font_size=16)

    path = os.path.join(OUTPUT_DIR, "09_discount_vs_revenue.html")
    fig.write_html(path)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────
# 14. PRINT BUSINESS INSIGHTS
# ──────────────────────────────────────────────────────────────

def print_insights(df: pd.DataFrame):
    print("\n" + "═" * 42)
    print("  BUSINESS INSIGHTS")
    print("═" * 42)

    top_cat    = df.groupby("category")["revenue"].sum().idxmax()
    top_region = df.groupby("region")["revenue"].sum().idxmax()
    top_prod   = df.groupby("product")["revenue"].sum().idxmax()
    best_month = df.groupby("month_name")["revenue"].sum().idxmax()
    high_disc_rev = (
        df[df["discount_pct"] >= 15]["revenue"].sum()
        / df["revenue"].sum() * 100
    )
    repeat_pct = (
        df.groupby("customer_id")["order_id"].nunique()
        .gt(1).mean() * 100
    )

    insights = [
        f"Top category      : {top_cat}",
        f"Top region        : {top_region}",
        f"Best-selling item : {top_prod}",
        f"Peak sales month  : {best_month}",
        f"Revenue from ≥15% discount orders: {high_disc_rev:.1f}%",
        f"Repeat customer rate : {repeat_pct:.1f}%",
    ]

    for line in insights:
        print(f"  {line}")

    print("\n  Recommendations:")
    print(f"  1. Double down on {top_cat} — it drives the most revenue.")
    print(f"  2. Investigate low performance in non-{top_region} regions.")
    print(f"  3. {high_disc_rev:.0f}% revenue from heavy discounts — review margin impact.")
    print(f"  4. Repeat rate {repeat_pct:.0f}% — launch loyalty programme to grow it.")
    print("═" * 42 + "\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 42)
    print("  E-COMMERCE SALES ANALYSIS")
    print("═" * 42 + "\n")

    # ── Load data ──────────────────────────────────
    # To use your own CSV, replace the line below with:
    #   df_raw = pd.read_csv("sales.csv", parse_dates=["order_date"])
    print("[load] Generating synthetic dataset …")
    df_raw = generate_dataset(n=12_000)

    # ── Clean ──────────────────────────────────────
    df = clean_data(df_raw)

    # ── KPIs ───────────────────────────────────────
    compute_kpis(df)

    # ── Charts ─────────────────────────────────────
    print("[charts] Building visualisations …")
    plot_monthly_trend(df)
    plot_top_products(df)
    plot_category_analysis(df)
    plot_region_revenue(df)
    plot_customer_analysis(df)
    plot_heatmap(df)
    plot_payment_methods(df)
    plot_interactive_trend(df)
    plot_discount_impact(df)

    # ── Insights ───────────────────────────────────
    print_insights(df)

    print(f"✓ All outputs saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        raise
