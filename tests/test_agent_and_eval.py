"""Agent loop and evaluation logic, tested with a scripted fake Claude client."""

from types import SimpleNamespace

import pandas as pd
import pytest

from nl2sql.agent import NL2SQLAgent
from nl2sql.evaluate import load_cases, results_match


def tool_use(name, **inp):
    return SimpleNamespace(type="tool_use", name=name, id=f"tu_{name}", input=inp)


class FakeClient:
    """Replays a fixed list of responses, recording what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def create(self, **kwargs):
        # snapshot: the agent keeps appending to the same messages list
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return SimpleNamespace(
            content=self.turns.pop(0), usage=SimpleNamespace(input_tokens=100, output_tokens=20)
        )


def test_schema_prompt_mentions_tables_and_values(wh):
    s = wh.schema_prompt()
    for t in ["fact_sales", "dim_product", "dim_date", "net_revenue", "'Electronics'"]:
        assert t in s


def test_agent_self_corrects_after_error(wh):
    client = FakeClient(
        [
            [tool_use("run_sql", sql="SELECT revenue FROM fact_sales")],  # bad column
            [
                tool_use(
                    "final_answer",
                    sql="SELECT COUNT(DISTINCT order_id) AS orders FROM fact_sales",
                    answer="There are N orders.",
                    chart="kpi",
                )
            ],
        ]
    )
    res = NL2SQLAgent(wh, client=client).ask("How many orders?")
    assert res.ok and res.steps == 2 and len(res.errors) == 1
    assert res.data.iloc[0, 0] > 0
    # the error text was fed back to the model as a tool_result
    fed_back = client.calls[1]["messages"][-1]["content"][0]
    assert fed_back["is_error"] and "ERROR" in fed_back["content"]


def test_agent_blocks_destructive_sql(wh):
    client = FakeClient(
        [
            [tool_use("final_answer", sql="DELETE FROM fact_sales", answer="done", chart="table")],
            [SimpleNamespace(type="text", text="I can only read data, so I won't delete rows.")],
        ]
    )
    res = NL2SQLAgent(wh, client=client).ask("Delete all sales")
    assert not res.ok
    assert "REJECTED" in client.calls[1]["messages"][-1]["content"][0]["content"]
    assert wh.run_trusted("SELECT COUNT(*) FROM fact_sales").iloc[0, 0] > 0


def test_every_gold_query_runs(wh):
    for case in load_cases():
        if case.get("expect_refusal"):
            continue
        df = wh.run_trusted(case["sql"])
        assert len(df) > 0, case["id"]


def test_eval_set_size():
    cases = load_cases()
    assert len([c for c in cases if not c.get("expect_refusal")]) >= 30
    assert len({c["id"] for c in cases}) == len(cases)


@pytest.mark.parametrize(
    "pred,gold,ordered,expected",
    [
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"x": [2, 1]}), False, True),
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"x": [2, 1]}), True, False),
        (
            pd.DataFrame({"c": ["A", "B"], "r": [10.04, 20.0]}),
            pd.DataFrame({"cat": ["B", "A"], "rev": [20.0, 10.0]}),
            False,
            True,
        ),
        (
            pd.DataFrame({"c": ["A"], "extra": [5], "r": [10.0]}),
            pd.DataFrame({"cat": ["A"], "rev": [10.0]}),
            False,
            True,
        ),
        (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1], "b": [2]}), False, False),
    ],
)
def test_results_match(pred, gold, ordered, expected):
    assert results_match(pred, gold, ordered) is expected
