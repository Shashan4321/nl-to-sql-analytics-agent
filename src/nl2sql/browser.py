"""SQLite backend for the in-browser demo.

Pyodide (Python compiled to WebAssembly) ships SQLite but not DuckDB, so the browser demo
loads the same seeded tables into SQLite and uses sqlglot to transpile each validated
DuckDB query to the SQLite dialect. `tests/test_browser.py` proves all gold queries return
the same results on both engines.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import sqlglot
from sqlglot import exp

from . import warehouse


def load_sqlite(con: sqlite3.Connection) -> None:
    """Create the star schema in SQLite (dates stored as ISO text, booleans as 0/1)."""
    for name, df in warehouse.frames().items():
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(int)
            elif out[col].dtype == object and len(out) and hasattr(out[col].iloc[0], "isoformat"):
                out[col] = out[col].map(lambda d: d.isoformat())
        out.to_sql(name, con, index=False)


_STRFTIME = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}


def _strftime_int(fmt: str, value: exp.Expression) -> exp.Expression:
    call = exp.Anonymous(this="strftime", expressions=[exp.Literal.string(fmt), value.copy()])
    return exp.Cast(this=call, to=exp.DataType.build("INT"))


def _extract_to_strftime(node: exp.Expression) -> exp.Expression:
    """SQLite has no EXTRACT(): rewrite EXTRACT(YEAR FROM d) as CAST(strftime('%Y', d) AS INT)."""
    if isinstance(node, exp.Extract):
        part = node.this.name.upper()
        if part in _STRFTIME:
            return _strftime_int(_STRFTIME[part], node.expression)
        if part == "QUARTER":
            month = _strftime_int("%m", node.expression)
            # typed=True keeps integer division (sqlglot otherwise emits a REAL cast).
            return exp.Div(
                this=exp.Paren(this=exp.Add(this=month, expression=exp.Literal.number(2))),
                expression=exp.Literal.number(3),
                typed=True,
            )
    return node


def to_sqlite(sql: str) -> str:
    """Transpile DuckDB SQL (already checked by the guardrails) to SQLite."""
    tree = sqlglot.parse_one(sql, read="duckdb").transform(_extract_to_strftime)
    return tree.sql(dialect="sqlite")


def run_sqlite(con: sqlite3.Connection, duckdb_sql: str) -> pd.DataFrame:
    return pd.read_sql_query(to_sqlite(duckdb_sql), con)
