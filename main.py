# main.py
# Car Price Analysis — meets Sections 1, 2.1–2.12, 3.1–3.7
# Usage: python main.py --csv car_prices.csv --out outputs

import os, math, argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CURRENT_YEAR = 2025


# -------------------- Utilities --------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create canonical columns used in the assignment by mirroring common variants.
    Canonical: selling_price, make, model, year, condition, odometer, fuel, state, color, interior
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    mapping: Dict[str, str] = {}

    def find_one(cands: List[str]) -> str | None:
        # exact
        for c in cands:
            if c in df.columns:
                return c
        # contains
        for c in cands:
            for col in df.columns:
                if c in col:
                    return col
        return None

    m = {
        "selling_price": find_one(["selling_price","price","sale_price","list_price","sellingprice"]),
        "make":          find_one(["make","brand","manufacturer","company"]),
        "model":         find_one(["model","car_model","variant"]),
        "year":          find_one(["year","model_year","manufacture_year","yr"]),
        "condition":     find_one(["condition","condition_score","score","vehicle_condition"]),
        "odometer":      find_one(["odometer","mileage","kms_driven","km_driven","miles"]),
        "fuel":          find_one(["fuel","fuel_type"]),
        "state":         find_one(["state","location_state","seller_state","region_state"]),
        "color":         find_one(["color","exterior_color","paint_color"]),
        "interior":      find_one(["interior","interior_color","trim","interior_type"]),
    }

    for canon, src in m.items():
        if src is not None and canon not in df.columns:
            df[canon] = df[src]
            mapping[canon] = src

    return df, mapping

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df

def iqr_filter_group(g: pd.DataFrame, target: str) -> pd.DataFrame:
    q1, q3 = g[target].quantile(0.25), g[target].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return g[(g[target] >= lo) & (g[target] <= hi)]


# -------------------- Stage 1: Load & Clean --------------------
def load_and_clean(csv_path: str, out_dir: str) -> pd.DataFrame:
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print("1.1 First 5 rows:\n", df.head(5))
    print("\n1.1 Dtypes:\n", df.dtypes)
    print("\n1.1 Record count:", len(df))

    # 1.2 Structure
    print("\n1.2 Shape (rows, cols):", df.shape)
    print("1.2 Columns:", list(df.columns))

    # Canonicalize names
    df, colmap = canonicalize_columns(df)
    if colmap:
        print("\nCanonical columns created from:", colmap)

    # 1.3 Missing & Anomaly Detection (counts + cleaning)
    nulls = df.isna().sum().sort_values(ascending=False)
    print("\n1.3 Null counts (top 20):\n", nulls.head(20))

    df = fill_nulls(df)

    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print("\n1.3 Duplicates removed:", dup_count)

    df = to_numeric(df, ["selling_price","odometer","condition","year"])
    if "year" in df.columns:
        df["car_age"] = CURRENT_YEAR - df["year"]

    ensure_dir(out_dir)
    cleaned_path = os.path.join(out_dir, "car_prices_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print("Saved cleaned CSV ->", cleaned_path)

    return df


# -------------------- Stage 2: Queries --------------------
def run_queries(df: pd.DataFrame, out_dir: str):
    def show(title, obj):
        print(f"\n--- {title} ---")
        print(obj)

    # 2.1
    if "selling_price" in df.columns:
        show("2.1 Avg/Min/Max Selling Price",
             df["selling_price"].agg(["mean","min","max"]).round(2))

    # 2.2
    if "color" in df.columns:
        colors = sorted(df["color"].astype(str).str.strip().str.title().unique())
        show("2.2 Unique Colors", pd.Series(colors, name="color"))

    # 2.3
    out = {}
    if "make" in df.columns:  out["unique_makes"]  = df["make"].nunique()
    if "model" in df.columns: out["unique_models"] = df["model"].nunique()
    if out:
        show("2.3 Unique Brands & Models", pd.Series(out))

    # 2.4
    if "selling_price" in df.columns:
        high = df[df["selling_price"] > 165000]
        show("2.4 Count of cars with price > 165,000", len(high))
        # Save ALL rows (plus a small sample)
        high.to_csv(os.path.join(out_dir, "cars_price_gt_165k.csv"), index=False)
        high.head(200).to_csv(os.path.join(out_dir, "price_gt_165k_sample.csv"), index=False)

    # 2.5
    if "model" in df.columns:
        top5 = (df["model"].astype(str).str.strip().str.upper()
                .value_counts().head(5))
        show("2.5 Top 5 most frequently sold models", top5)

    # 2.6
    if {"make","selling_price"}.issubset(df.columns):
        avg_by_make = (df.groupby("make")["selling_price"]
                       .mean().sort_values(ascending=False))
        show("2.6 Avg selling price by make (top 50 shown)", avg_by_make.head(50))

    # 2.7
    if {"interior","selling_price"}.issubset(df.columns):
        min_by_interior = (df.groupby("interior")["selling_price"]
                           .min().sort_values())
        show("2.7 Min selling price by interior (top 50 shown)", min_by_interior.head(50))

    # 2.8
    if {"odometer","year"}.issubset(df.columns):
        max_odo_per_year = (df.groupby("year")["odometer"]
                            .max().sort_values(ascending=False))
        show("2.8 Highest odometer per year (desc)", max_odo_per_year.head(50))

    # 2.9
    if "car_age" in df.columns:
        show("2.9 Car age preview", df[["year","car_age"]].head(10))

    # 2.10
    if {"condition","odometer"}.issubset(df.columns):
        count = int(((df["condition"] >= 48) & (df["odometer"] > 90000)).sum())
        show("2.10 Count (condition>=48 & odometer>90000)", count)

    # 2.11
    if {"state","selling_price","year"}.issubset(df.columns):
        newer = df[df["year"] > 2013]
        by_state = newer.groupby("state")["selling_price"].mean().sort_values(ascending=False)
        show("2.11 Avg price by state for newer cars (year>2013)", by_state)
        top_state = by_state.idxmax()
        print(f"\n>>> 2.11 Answer: Top state for newer cars is: {top_state}")
        by_state.head(15).to_csv(os.path.join(out_dir, "avg_price_by_state_newer_top15.csv"))

    # 2.12
    if {"condition","make","selling_price"}.issubset(df.columns):
        cutoff = df["condition"].quantile(0.80)
        top20 = df[df["condition"] >= cutoff]
        excellent_low = (top20.groupby("make")["selling_price"]
                         .mean().sort_values(ascending=True))
        show("2.12 Excellent condition (top 20%) — lowest avg price makes", excellent_low.head(20))


# -------------------- Stage 3: Visualizations --------------------
def make_charts(df: pd.DataFrame, charts_dir: str):
    print("\nCreating charts …")
    ensure_dir(charts_dir)

    # 1.3 Nulls per column (bar)
    nulls = df.isna().sum()
    if nulls.sum() > 0:
        plt.figure(figsize=(10,4))
        nulls[nulls > 0].sort_values(ascending=False).plot(kind="bar")
        plt.title("Nulls per Column")
        plt.ylabel("Count of Nulls")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "1.3_nulls_per_column.png"))
        plt.close()

    # 3.1 Correlation heatmap (numeric-only)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        plt.figure()
        im = plt.imshow(corr, interpolation='nearest')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar(im)
        plt.title("Correlation Heatmap (Numeric)")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.1_correlation_heatmap.png"))
        plt.close()

    # 3.2 Average selling price by year — scatter (per “bar or scatter?”)
    if {"year","selling_price"}.issubset(df.columns):
        avg_by_year = df.groupby("year")["selling_price"].mean().reset_index().sort_values("year")
        plt.figure()
        plt.scatter(avg_by_year["year"], avg_by_year["selling_price"])
        plt.xlabel("Year"); plt.ylabel("Average Selling Price")
        plt.title("Average Selling Price by Year")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.2_avg_price_by_year.png"))
        plt.close()

    # 3.3 Selling price vs odometer — scatter + averaged bins bar
    if {"odometer","selling_price"}.issubset(df.columns):
        # scatter
        sample = df.sample(min(5000, len(df)), random_state=42)
        plt.figure()
        plt.scatter(sample["odometer"], sample["selling_price"], alpha=0.5)
        plt.xlabel("Odometer"); plt.ylabel("Selling Price")
        plt.title("Selling Price vs Odometer")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.3_price_vs_odometer.png"))
        plt.close()

        # averaged by bins (bar)
        valid = df.dropna(subset=["odometer"])
        if not valid.empty:
            step = max(25000, int((valid["odometer"].max() - valid["odometer"].min()) // 15) or 25000)
            bins = np.arange(valid["odometer"].min(), valid["odometer"].max() + step, step)
            odobin = pd.cut(valid["odometer"], bins=bins, right=False, include_lowest=True)
            avg_price_by_odo = valid.groupby(odobin)["selling_price"].mean()
            plt.figure(figsize=(10,5))
            avg_price_by_odo.plot(kind="bar")
            plt.xlabel("Odometer Bins"); plt.ylabel("Average Selling Price")
            plt.title("Average Selling Price by Odometer (binned)")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "3.3_avg_price_by_odo_bins.png"))
            plt.close()

    # 3.4 Cars sold per state (bar) + print top 3
    if "state" in df.columns:
        counts = df["state"].astype(str).value_counts()
        plt.figure(figsize=(10,5))
        counts.plot(kind="bar")
        plt.xlabel("State"); plt.ylabel("Number of Cars Sold")
        plt.title("Cars Sold per State")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.4_cars_per_state.png"))
        plt.close()
        top3 = counts.head(3)
        print("\n>>> 3.4 Top 3 states by cars sold:\n", top3)
        top3.to_csv(os.path.join(charts_dir, "3.4_top3_states.csv"))

    # 3.5 Avg price by condition score ranges of size 5 (bar)
    if {"condition","selling_price"}.issubset(df.columns):
        cmin, cmax = df["condition"].min(), df["condition"].max()
        start = math.floor(cmin/5)*5
        end   = math.ceil(cmax/5)*5 + 5
        bins5 = np.arange(start, end+1, 5)
        cond5 = pd.cut(df["condition"], bins=bins5, right=False, include_lowest=True)
        avg5  = df.groupby(cond5)["selling_price"].mean()
        plt.figure(figsize=(10,5))
        avg5.plot(kind="bar")
        plt.xlabel("Condition Range (width=5)"); plt.ylabel("Average Selling Price")
        plt.title("Avg Selling Price by Condition (width=5)")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.5_avg_price_by_condition_w5.png"))
        plt.close()

        # 3.6 Count of cars by condition ranges of size 10 (bar)
        start10 = math.floor(cmin/10)*10
        end10   = math.ceil(cmax/10)*10 + 10
        bins10  = np.arange(start10, end10+1, 10)
        cond10  = pd.cut(df["condition"], bins=bins10, right=False, include_lowest=True)
        cnt10   = cond10.value_counts().sort_index()
        plt.figure(figsize=(10,5))
        cnt10.plot(kind="bar")
        plt.xlabel("Condition Range (width=10)"); plt.ylabel("Number of Cars")
        plt.title("Cars by Condition (width=10)")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.6_cars_by_condition_w10.png"))
        plt.close()

    # 3.7 Box plot by color (with & without outliers)
    if {"selling_price","color"}.issubset(df.columns):
        # with outliers
        plt.figure(figsize=(12,6))
        df.boxplot(column="selling_price", by="color", rot=90)
        plt.suptitle("")
        plt.title("Selling Price by Color (with outliers)")
        plt.xlabel("Color"); plt.ylabel("Selling Price")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "3.7_price_by_color_with_outliers.png"))
        plt.close()

        # remove outliers per color via IQR
        no_out = df.groupby("color", group_keys=False).apply(lambda g: iqr_filter_group(g, "selling_price"))
        if not no_out.empty:
            plt.figure(figsize=(12,6))
            no_out.boxplot(column="selling_price", by="color", rot=90)
            plt.suptitle("")
            plt.title("Selling Price by Color (outliers removed)")
            plt.xlabel("Color"); plt.ylabel("Selling Price")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "3.7_price_by_color_no_outliers.png"))
            plt.close()


# -------------------- Entrypoint --------------------
def main():
    parser = argparse.ArgumentParser(description="Car Price Analysis (complete)")
    parser.add_argument("--csv", default="car_prices.csv", help="Path to raw CSV")
    parser.add_argument("--out", default="outputs", help="Output folder")
    parser.add_argument("--charts", default=None, help="Charts folder (default: OUT/charts)")
    a = parser.parse_args()

    out_dir = a.out
    charts_dir = a.charts or os.path.join(out_dir, "charts")
    ensure_dir(out_dir); ensure_dir(charts_dir)

    df = load_and_clean(a.csv, out_dir)
    run_queries(df, out_dir)
    make_charts(df, charts_dir)

    print("\nDone ✅")
    print(f"- Cleaned CSV: {os.path.join(out_dir, 'car_prices_cleaned.csv')}")
    print(f"- Charts: {charts_dir}")

if __name__ == "__main__":
    main()
