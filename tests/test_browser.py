"""The browser demo runs on SQLite: every gold query must give the same answer there."""

import sqlite3

import pytest

from nl2sql.browser import load_sqlite, run_sqlite
from nl2sql.evaluate import load_cases, results_match

CASES = [c for c in load_cases() if c.get("sql")]


@pytest.fixture(scope="module")
def lite() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    load_sqlite(con)
    return con


def test_extract_parts_are_rewritten(lite):
    sql = (
        "SELECT EXTRACT(YEAR FROM date) AS y, EXTRACT(QUARTER FROM date) AS q, "
        "EXTRACT(MONTH FROM date) AS m FROM dim_date WHERE date_key = 20250815"
    )
    assert run_sqlite(lite, sql).iloc[0].tolist() == [2025, 3, 8]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_gold_sql_matches_on_sqlite(case, wh, lite):
    gold = wh.run_trusted(case["sql"])
    browser = run_sqlite(lite, case["sql"])
    assert results_match(browser, gold, ordered=case.get("ordered", False))
