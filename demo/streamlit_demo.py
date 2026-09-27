"""Browser demo: runs entirely client-side (stlite / Pyodide), no server and no API key.

It uses the same seeded warehouse, guardrails and evaluation set as the full app:

* Sample questions: the 32 evaluation questions, answered with their verified gold SQL
  (the SQL the agent's answers are scored against), after passing the guardrails.
* Guardrails playground: type any SQL and see it validated, rewritten or blocked.
* Schema: the star schema the agent is prompted with.

Pyodide ships SQLite but not DuckDB, so validated DuckDB SQL is transpiled to SQLite with
sqlglot (``nl2sql.browser``; parity with DuckDB is tested in ``tests/test_browser.py``).
Free-form English questions need Claude, so they run in the full app (``make app``).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl2sql.browser import load_sqlite, run_sqlite, to_sqlite  # noqa: E402
from nl2sql.charts import auto_chart  # noqa: E402
from nl2sql.guardrails import UnsafeSQLError, validate_sql  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Shashan4321/nl-to-sql-analytics-agent"
ATTACKS = {
    "Drop a table": "DROP TABLE fact_sales",
    "Delete all sales": "DELETE FROM fact_sales WHERE 1 = 1",
    "Sneak a second statement": "SELECT 1; DROP TABLE dim_date",
    "Read a file from disk": "SELECT * FROM read_csv('/etc/passwd')",
    "Unknown table": "SELECT * FROM salaries",
    "Safe, but no LIMIT": "SELECT * FROM fact_sales",
}
EXAMPLE_SQL = """SELECT p.category, ROUND(SUM(f.net_revenue), 2) AS revenue
FROM fact_sales f JOIN dim_product p USING (product_key)
WHERE NOT f.is_returned
GROUP BY p.category ORDER BY revenue DESC"""

st.set_page_config(page_title="NL-to-SQL Agent: live demo", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Building the synthetic warehouse in your browser...")
def get_con() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    load_sqlite(con)
    return con


@st.cache_data
def load_questions() -> list[dict]:
    return yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text(encoding="utf-8"))


def table_names(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def show_result(df: pd.DataFrame, hint: str = "bar") -> None:
    if df.shape == (1, 1):
        value = df.iloc[0, 0]
        st.metric(df.columns[0], f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
    else:
        fig = auto_chart(df, hint)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_sql(duckdb_sql: str) -> None:
    left, right = st.columns(2)
    left.caption("Validated SQL (DuckDB dialect, as the agent writes it)")
    left.code(duckdb_sql, language="sql")
    right.caption("Transpiled by sqlglot for the in-browser SQLite engine")
    right.code(to_sqlite(duckdb_sql), language="sql")


con = get_con()
tables = table_names(con)

st.title("📊 NL-to-SQL Analytics Agent: live demo")
st.markdown(
    "Everything below runs **in your browser**, with no server and no API key: the seeded retail "
    "warehouse (60k order lines), the **read-only SQL guardrails** and the **evaluation set** the "
    f"agent is scored on. Free-form English questions need Claude, so they run in the "
    f"[full app]({REPO}) (`make app`)."
)

ask, playground, schema = st.tabs(["💬 Sample questions", "🛡️ Guardrails playground", "🗂️ Schema"])

with ask:
    questions = load_questions()
    labels = {f"{q['question']}  ·  {', '.join(q.get('tags', []))}": q for q in questions}
    choice = st.selectbox("Pick a business question", list(labels), index=6)
    q = labels[choice]
    st.caption(
        "Answered with the question's verified gold SQL, the same SQL the agent's own answers "
        "are compared against (execution accuracy)."
    )
    try:
        checked = validate_sql(q["sql"], tables)
        show_result(run_sqlite(con, checked.sql))
        with st.expander("SQL", expanded=False):
            show_sql(checked.sql)
    except UnsafeSQLError as e:  # gold SQL is trusted, but keep the path honest
        st.error(f"Blocked by guardrails: {e}")

with playground:
    st.markdown(
        "Type any SQL, or try an attack. Queries are **parsed into a syntax tree** with sqlglot "
        "and checked before they run: one statement, SELECT only, allowed tables, no file or "
        "network functions, and a LIMIT added when missing."
    )
    cols = st.columns(3)
    for i, (label, attack) in enumerate(ATTACKS.items()):
        if cols[i % 3].button(label, use_container_width=True):
            st.session_state["sql"] = attack
    st.session_state.setdefault("sql", EXAMPLE_SQL)
    sql = st.text_area("SQL", key="sql", height=140)
    if st.button("Run through guardrails", type="primary") and sql.strip():
        try:
            checked = validate_sql(sql, tables)
        except UnsafeSQLError as e:
            st.error(f"⛔ Blocked: {e}")
        else:
            note = " (LIMIT added automatically)" if checked.limited else ""
            used = ", ".join(sorted(checked.tables)) or "none"
            st.success(f"✅ Allowed{note}. Tables used: {used}")
            show_sql(checked.sql)
            try:
                show_result(run_sqlite(con, checked.sql), hint="table")
            except Exception as e:  # valid and safe, but e.g. a typo in a column name
                st.warning(f"Safe, but the database returned an error: {e}")

with schema:
    for table in sorted(tables):
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        with st.expander(f"{table}  ({n:,} rows)"):
            info = pd.read_sql_query(f'PRAGMA table_info("{table}")', con)
            st.dataframe(
                info[["name", "type"]].rename(columns={"name": "column"}),
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.caption(
    "Synthetic data from a fixed seed, no real company data. "
    f"Code: [nl-to-sql-analytics-agent]({REPO}) · "
    "Built by [Shashank Singh](https://shashan4321.github.io)"
)
