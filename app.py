"""Streamlit UI: ask the sales warehouse questions in plain English.

Run locally with ``streamlit run app.py``. On Streamlit Community Cloud, add
``ANTHROPIC_API_KEY`` under *Settings -> Secrets*.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nl2sql import warehouse  # noqa: E402
from nl2sql.agent import NL2SQLAgent  # noqa: E402
from nl2sql.charts import auto_chart  # noqa: E402
from nl2sql.db import Warehouse  # noqa: E402

DB = Path("data/warehouse.duckdb")
EXAMPLES = [
    "Top 5 cities by revenue in 2025",
    "Year-over-year revenue growth percentage for each year",
    "Gross margin % by category",
    "Monthly revenue trend for 2025",
    "Which category grew fastest from 2024 to 2025?",
]

st.set_page_config(page_title="NL-to-SQL Analytics Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_agent() -> NL2SQLAgent:
    if not DB.exists():
        warehouse.build(DB)  # first run on a fresh deployment
    if "ANTHROPIC_API_KEY" in st.secrets and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    return NL2SQLAgent(Warehouse(DB))


st.title("📊 Ask your sales data")
st.caption(
    "Plain English → validated, read-only SQL → answer + chart. "
    "Synthetic retail data (seeded, reproducible)."
)

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    st.divider()
    st.markdown(
        "**Guardrails:** single SELECT only · table allow-list · no file/network "
        "functions · auto LIMIT · read-only connection"
    )

question = st.chat_input("e.g. Which state had the highest revenue in 2025?")
question = question or st.session_state.pop("q", None)

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Writing and checking SQL..."):
        try:
            res = get_agent().ask(question)
        except Exception as e:  # missing key, network, etc.
            st.error(f"Agent error: {e}")
            st.stop()
        st.write(res.answer)
        if res.ok:
            fig = auto_chart(res.data, res.chart)
            if res.chart == "kpi" and res.data.shape == (1, 1):
                st.metric(res.data.columns[0], f"{res.data.iloc[0, 0]:,.2f}")
            elif fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(res.data, use_container_width=True, hide_index=True)
            with st.expander("SQL"):
                st.code(res.sql, language="sql")
        st.caption(
            f"{res.steps} step(s) · {res.input_tokens + res.output_tokens:,} tokens"
            + (f" · self-corrected {len(res.errors)} error(s)" if res.errors else "")
        )
