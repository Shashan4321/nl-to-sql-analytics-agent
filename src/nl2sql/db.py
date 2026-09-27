"""Database access: schema introspection for the prompt and read-only execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .guardrails import ValidatedSQL, validate_sql


@dataclass
class Warehouse:
    """A read-only handle on the DuckDB warehouse."""

    path: str | Path
    max_rows: int = 1000
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"{self.path} not found. Run `make data` first.")
        # read_only=True is the second line of defence after the SQL guardrails
        self._con = duckdb.connect(str(self.path), read_only=True)
        self._con.execute("SET enable_external_access = false")

    @property
    def tables(self) -> set[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        return {r[0] for r in rows}

    def schema_prompt(self, sample_values: int = 5) -> str:
        """Compact, LLM-friendly description of every table and column.

        Includes column comments and a few distinct values for low-cardinality text
        columns, which is what lets the model write ``WHERE category = 'Electronics'``
        instead of guessing spellings.
        """
        cols = self._con.execute("""
            SELECT c.table_name, c.column_name, c.data_type, col.comment
            FROM information_schema.columns c
            LEFT JOIN duckdb_columns() col
              ON col.table_name = c.table_name AND col.column_name = c.column_name
            WHERE c.table_schema = 'main'
            ORDER BY c.table_name, c.ordinal_position
        """).fetchall()
        tcomments = dict(
            self._con.execute(
                "SELECT table_name, comment FROM duckdb_tables() WHERE comment IS NOT NULL"
            ).fetchall()
        )
        out: list[str] = []
        current = None
        for table, col, dtype, comment in cols:
            if table != current:
                n = self._con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                out.append(
                    f"\nTABLE {table} ({n:,} rows)"
                    + (f" -- {tcomments[table]}" if table in tcomments else "")
                )
                current = table
            line = f"  {col} {dtype}"
            if dtype == "VARCHAR":
                vals = self._con.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL '
                    f"LIMIT {sample_values + 1}"
                ).fetchall()
                if len(vals) <= sample_values:
                    line += "  values: " + ", ".join(repr(v[0]) for v in vals)
            if comment:
                line += f"  -- {comment}"
            out.append(line)
        return "\n".join(out).strip()

    def validate(self, sql: str) -> ValidatedSQL:
        return validate_sql(sql, self.tables, max_rows=self.max_rows)

    def run(self, sql: str) -> tuple[ValidatedSQL, pd.DataFrame]:
        """Validate then execute; returns the rewritten SQL and a DataFrame."""
        checked = self.validate(sql)
        return checked, self._con.execute(checked.sql).df()

    def run_trusted(self, sql: str) -> pd.DataFrame:
        """Execute gold SQL from the evaluation set (still read-only)."""
        return self._con.execute(sql).df()
