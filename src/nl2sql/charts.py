"""Pick a sensible Plotly chart for a result set."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

TIME_HINTS = ("month", "date", "year", "quarter", "week", "day")


def auto_chart(df: pd.DataFrame, hint: str = "table") -> Figure | None:
    """Return a figure, or None when a table (or single KPI) is the better display."""
    if df is None or df.empty or df.shape[1] < 2 or hint in {"table", "kpi"} or len(df) > 200:
        return None
    x, y = df.columns[0], df.columns[-1]
    if not pd.api.types.is_numeric_dtype(df[y]):
        return None
    if hint == "line" or any(h in str(x).lower() for h in TIME_HINTS):
        return px.line(df, x=x, y=y, markers=True)
    if hint == "pie" and len(df) <= 8:
        return px.pie(df, names=x, values=y, hole=0.45)
    return px.bar(df, x=x, y=y)
