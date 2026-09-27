"""Build a synthetic retail sales star schema in DuckDB.

Everything here is generated from a fixed random seed, so the warehouse (and
every number quoted in the README) is fully reproducible. No real company data
is used anywhere in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # DuckDB is imported lazily so the browser demo (no DuckDB) can use frames()
    import duckdb

SEED = 42
START, END = "2023-01-01", "2025-12-31"

REGIONS: dict[str, dict[str, list[str]]] = {
    "India": {
        "Haryana": ["Gurugram", "Faridabad"],
        "Delhi": ["New Delhi"],
        "Karnataka": ["Bengaluru", "Mysuru"],
        "Maharashtra": ["Mumbai", "Pune"],
        "Telangana": ["Hyderabad"],
        "Tamil Nadu": ["Chennai"],
        "West Bengal": ["Kolkata"],
    },
    "United Arab Emirates": {"Dubai": ["Dubai"], "Abu Dhabi": ["Abu Dhabi"]},
    "Singapore": {"Singapore": ["Singapore"]},
}

CATALOG: dict[str, dict[str, tuple[float, float]]] = {
    # category -> subcategory -> (min price, max price) in INR
    "Electronics": {"Mobiles": (8000, 60000), "Laptops": (35000, 120000), "Audio": (800, 15000)},
    "Home & Kitchen": {"Cookware": (500, 6000), "Appliances": (2500, 40000), "Decor": (300, 5000)},
    "Fashion": {"Men": (400, 5000), "Women": (400, 6000), "Footwear": (700, 8000)},
    "Grocery": {"Staples": (50, 1200), "Beverages": (40, 800), "Snacks": (20, 500)},
    "Sports": {"Fitness": (500, 20000), "Outdoor": (800, 15000)},
}
BRANDS = ["Aurora", "Nimbus", "Vertex", "Kestrel", "Lotus", "Zenith", "Orbit", "Saffron"]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]
CHANNELS = ["Store", "Online", "Marketplace"]


def _dim_date() -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})
    d["date_key"] = d["date"].dt.strftime("%Y%m%d").astype(int)
    d["year"] = d["date"].dt.year
    d["quarter"] = d["date"].dt.quarter
    d["month"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["day_of_week"] = d["date"].dt.strftime("%a")
    d["is_weekend"] = d["date"].dt.dayofweek >= 5
    # Indian financial year starts in April: FY2024 = Apr-2023..Mar-2024
    d["fiscal_year"] = np.where(d["month"] >= 4, d["year"] + 1, d["year"])
    d["date"] = d["date"].dt.date
    return d[
        [
            "date_key",
            "date",
            "year",
            "quarter",
            "month",
            "month_name",
            "day_of_week",
            "is_weekend",
            "fiscal_year",
        ]
    ]


def _dim_store(rng: np.random.Generator) -> pd.DataFrame:
    rows, key = [], 1
    for country, states in REGIONS.items():
        for state, cities in states.items():
            for city in cities:
                for n in range(1, int(rng.integers(1, 4)) + 1):
                    rows.append(
                        {
                            "store_key": key,
                            "store_name": f"{city} Store {n:02d}",
                            "city": city,
                            "state": state,
                            "country": country,
                            "region": "Domestic" if country == "India" else "International",
                            "opened_on": pd.Timestamp("2018-01-01")
                            + pd.Timedelta(days=int(rng.integers(0, 1500))),
                        }
                    )
                    key += 1
    df = pd.DataFrame(rows)
    df["opened_on"] = df["opened_on"].dt.date
    return df


def _dim_product(rng: np.random.Generator) -> pd.DataFrame:
    rows, key = [], 1
    for cat, subs in CATALOG.items():
        for sub, (lo, hi) in subs.items():
            for _ in range(12):
                price = round(float(rng.uniform(lo, hi)), -1)
                rows.append(
                    {
                        "product_key": key,
                        "sku": f"SKU-{key:05d}",
                        "product_name": f"{rng.choice(BRANDS)} {sub} {key:03d}",
                        "category": cat,
                        "subcategory": sub,
                        "brand": str(rng.choice(BRANDS)),
                        "list_price": price,
                        "unit_cost": round(price * float(rng.uniform(0.55, 0.8)), 2),
                    }
                )
                key += 1
    return pd.DataFrame(rows)


def _dim_customer(rng: np.random.Generator, stores: pd.DataFrame, n: int = 4000) -> pd.DataFrame:
    home = stores.sample(n=n, replace=True, random_state=SEED).reset_index(drop=True)
    first = pd.Timestamp(START)
    return pd.DataFrame(
        {
            "customer_key": np.arange(1, n + 1),
            "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
            "segment": rng.choice(SEGMENTS, n, p=[0.7, 0.18, 0.12]),
            "city": home["city"],
            "state": home["state"],
            "country": home["country"],
            "signup_date": [
                (first + pd.Timedelta(days=int(x))).date() for x in rng.integers(-400, 900, n)
            ],
        }
    )


def _fact_sales(
    rng: np.random.Generator,
    dates: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    n_orders: int = 30000,
) -> pd.DataFrame:
    # Demand grows ~12% a year and peaks in Oct-Nov (festive season).
    day_idx = np.arange(len(dates))
    season = 1 + 0.45 * np.isin(dates["month"], [10, 11]) + 0.15 * (dates["month"] == 12)
    trend = 1.12 ** (day_idx / 365)
    weekend = 1 + 0.25 * dates["is_weekend"].to_numpy()
    w = (season * trend * weekend).to_numpy(dtype=float)
    order_dates = rng.choice(dates["date_key"].to_numpy(), n_orders, p=w / w.sum())

    # 1-3 lines per order. intp keeps np.repeat happy on 32-bit WebAssembly (browser demo).
    lines = rng.integers(1, 4, n_orders).astype(np.intp)
    order_ids = np.repeat(np.arange(1, n_orders + 1), lines)
    n = len(order_ids)
    prod = products.sample(
        n=n, replace=True, random_state=SEED + 1, weights=1 / np.sqrt(products["list_price"])
    ).reset_index(drop=True)
    cust = rng.integers(1, len(customers) + 1, n_orders)
    store = rng.choice(stores["store_key"].to_numpy(), n_orders)
    channel = rng.choice(CHANNELS, n_orders, p=[0.55, 0.3, 0.15])
    discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], n)
    qty = rng.integers(1, 5, n)
    unit_price = prod["list_price"].to_numpy()
    df = pd.DataFrame(
        {
            "sales_line_id": np.arange(1, n + 1),
            "order_id": [f"SO-{o:07d}" for o in order_ids],
            "date_key": np.repeat(order_dates, lines),
            "customer_key": np.repeat(cust, lines),
            "store_key": np.repeat(store, lines),
            "channel": np.repeat(channel, lines),
            "product_key": prod["product_key"].to_numpy(),
            "quantity": qty,
            "unit_price": unit_price,
            "discount_pct": discount,
        }
    )
    df["net_revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"])).round(2)
    df["cost"] = (df["quantity"] * prod["unit_cost"].to_numpy()).round(2)
    # ~2% of lines are returned
    df["is_returned"] = rng.random(n) < 0.02
    return df


def build(db_path: str | Path = "data/warehouse.duckdb") -> Path:
    """Create (or overwrite) the DuckDB warehouse and return its path."""
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    load_into(con)
    con.close()
    return db_path


def frames() -> dict[str, pd.DataFrame]:
    """Generate every table of the star schema as DataFrames (same seed, same data)."""
    rng = np.random.default_rng(SEED)
    dates = _dim_date()
    stores = _dim_store(rng)
    products = _dim_product(rng)
    customers = _dim_customer(rng, stores)
    sales = _fact_sales(rng, dates, stores, products, customers)
    return {
        "dim_date": dates,
        "dim_store": stores,
        "dim_product": products,
        "dim_customer": customers,
        "fact_sales": sales,
    }


def load_into(con: duckdb.DuckDBPyConnection) -> None:
    """Create the star schema in an open DuckDB connection (file-backed or in-memory)."""
    for name, df in frames().items():
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.execute("""
        COMMENT ON TABLE fact_sales IS 'One row per order line. Grain: order_id + product.';
        COMMENT ON COLUMN fact_sales.net_revenue IS 'qty * unit_price * (1 - discount_pct), INR';
        COMMENT ON COLUMN fact_sales.cost IS 'quantity * unit_cost, INR';
        COMMENT ON COLUMN fact_sales.is_returned IS 'TRUE if returned; exclude for net sales';
        COMMENT ON COLUMN dim_date.fiscal_year IS 'Indian FY (Apr-Mar). FY2025 = Apr-24..Mar-25';
    """)


if __name__ == "__main__":
    print(f"Warehouse written to {build()}")
