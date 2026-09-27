"""Read-only SQL guardrails.

Every query produced by the LLM passes through :func:`validate_sql` before it
touches the database. The checks are structural (parsed with sqlglot), not
regex-based, so tricks like comments or mixed case do not bypass them.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Grant,
    exp.TruncateTable,
    exp.Set,
    exp.Pragma,
    exp.Use,
    exp.Attach,
    exp.Detach,
)
# Table functions that can read files or reach the network
FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "glob",
    "httpfs",
}


class UnsafeSQLError(ValueError):
    """Raised when a query fails a guardrail."""


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str
    tables: frozenset[str]
    limited: bool  # True if a LIMIT was injected


def validate_sql(
    sql: str, allowed_tables: set[str], dialect: str = "duckdb", max_rows: int = 1000
) -> ValidatedSQL:
    """Parse ``sql`` and reject anything that is not a single read-only SELECT.

    * exactly one statement
    * top-level node is SELECT / UNION / WITH ... SELECT
    * no DML/DDL/admin nodes anywhere in the tree
    * only tables from ``allowed_tables`` (CTE names are allowed)
    * no file/network table functions
    * a LIMIT is added when missing, so a bad question can't dump the warehouse
    """
    sql = sql.strip().rstrip(";")
    if not sql:
        raise UnsafeSQLError("Empty query.")
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError("Only a single statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed, got {tree.key.upper()}.")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"Forbidden operation: {node.key.upper()}.")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                raise UnsafeSQLError(f"Function not allowed: {name}.")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    unknown = tables - cte_names - {t.lower() for t in allowed_tables}
    if unknown:
        raise UnsafeSQLError(f"Unknown or disallowed table(s): {', '.join(sorted(unknown))}.")

    limited = False
    if isinstance(tree, exp.Select) and tree.args.get("limit") is None:
        tree = tree.limit(max_rows)
        limited = True
    return ValidatedSQL(
        sql=tree.sql(dialect=dialect), tables=frozenset(tables - cte_names), limited=limited
    )
