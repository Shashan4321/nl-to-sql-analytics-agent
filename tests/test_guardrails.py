import pytest

from nl2sql.guardrails import UnsafeSQLError, validate_sql

ALLOWED = {"fact_sales", "dim_date", "dim_product", "dim_store", "dim_customer"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select count(*) from fact_sales",
        "WITH t AS (SELECT * FROM fact_sales) SELECT count(*) FROM t",
        "SELECT category FROM dim_product UNION SELECT city FROM dim_store",
        "SELECT * FROM fact_sales f JOIN dim_date d USING (date_key) LIMIT 5;",
    ],
)
def test_accepts_read_only_queries(sql):
    assert validate_sql(sql, ALLOWED).sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM fact_sales",
        "DROP TABLE dim_customer",
        "UPDATE dim_product SET list_price = 0",
        "INSERT INTO dim_store VALUES (1)",
        "CREATE TABLE x AS SELECT 1",
        "ALTER TABLE fact_sales ADD COLUMN x INT",
        "SELECT 1; DROP TABLE fact_sales",
        "select * from fact_sales; delete from fact_sales",
        "COPY fact_sales TO 'out.csv'",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM secrets_table",
        "",
    ],
)
def test_rejects_unsafe_queries(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql, ALLOWED)


def test_comment_tricks_do_not_bypass():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT 1 /* harmless */; -- \n DROP TABLE fact_sales", ALLOWED)


def test_limit_is_injected_when_missing():
    v = validate_sql("SELECT * FROM fact_sales", ALLOWED, max_rows=50)
    assert v.limited and "LIMIT 50" in v.sql


def test_existing_limit_is_kept():
    v = validate_sql("SELECT * FROM fact_sales LIMIT 3", ALLOWED)
    assert not v.limited and "LIMIT 3" in v.sql


def test_cte_names_are_not_treated_as_tables():
    v = validate_sql("WITH my_cte AS (SELECT 1 AS a) SELECT a FROM my_cte", ALLOWED)
    assert v.tables == frozenset()
