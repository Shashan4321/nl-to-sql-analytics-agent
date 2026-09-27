"""NL-to-SQL agent built on Claude tool use.

Flow per question:
1. The system prompt carries the schema (tables, columns, comments, sample values)
   and business rules (how revenue, margin and fiscal year are defined).
2. Claude calls the ``run_sql`` tool. The query is validated by the guardrails and
   executed read-only; Claude sees either a preview of the rows or the error text.
3. On an error Claude can fix the query and try again (bounded by ``max_steps``).
4. Claude finishes by calling ``final_answer`` with the SQL it stands behind, a
   one-paragraph answer and a chart suggestion.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from .db import Warehouse
from .guardrails import UnsafeSQLError

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

BUSINESS_RULES = """\
- Revenue means SUM(fact_sales.net_revenue) excluding returned lines (is_returned = FALSE),
  unless the user explicitly asks about returns.
- Gross margin = revenue - SUM(cost) over the same lines; margin % = margin / revenue.
- Orders are counted with COUNT(DISTINCT order_id).
- Average order value (AOV) = revenue / orders.
- "Financial year" / FY uses dim_date.fiscal_year (Indian FY, April-March).
- Currency is INR. Round money to 2 decimals.
- Always join facts to dimensions on the *_key columns.
"""

SYSTEM_PROMPT = """You are a senior analytics engineer answering business questions \
over a DuckDB retail sales warehouse (star schema). Write DuckDB SQL.

Rules:
- Only read data: a single SELECT (CTEs are fine). Never modify anything.
- Use only the tables and columns listed in the schema. Never guess column names.
- Prefer clear column aliases (e.g. total_revenue, orders, margin_pct).
- Sort results in the most useful order and keep them small (use LIMIT for top-N).
- If the question is ambiguous, pick the most common business interpretation and say
  which one you picked in the answer.
- Call run_sql to test your query. If it errors, read the error and fix it.
- When the result looks right, call final_answer. Never state numbers you did not
  see in a run_sql result.

Business definitions:
{rules}
Schema:
{schema}
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_sql",
        "description": "Execute one read-only DuckDB SELECT and return up to 20 rows, "
        "or the error message if the query is invalid.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT."}},
            "required": ["sql"],
        },
    },
    {
        "name": "final_answer",
        "description": "Return the final SQL, a short plain-English answer and a chart hint.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "answer": {"type": "string", "description": "2-4 sentences, business tone."},
                "chart": {"type": "string", "enum": ["bar", "line", "pie", "table", "kpi"]},
            },
            "required": ["sql", "answer", "chart"],
        },
    },
]


class MessagesClient(Protocol):
    """The subset of ``anthropic.Anthropic().messages`` the agent needs (eases testing)."""

    def create(self, **kwargs: Any) -> Any: ...


@dataclass
class AgentResult:
    question: str
    sql: str | None
    answer: str
    chart: str = "table"
    data: pd.DataFrame | None = None
    steps: int = 0
    errors: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.sql is not None and self.data is not None


class NL2SQLAgent:
    def __init__(
        self,
        warehouse: Warehouse,
        client: MessagesClient | None = None,
        model: str = DEFAULT_MODEL,
        max_steps: int = 5,
    ) -> None:
        self.wh = warehouse
        if client is None:
            import anthropic  # imported lazily so tests don't need an API key

            client = anthropic.Anthropic().messages
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.system = SYSTEM_PROMPT.format(rules=BUSINESS_RULES, schema=warehouse.schema_prompt())

    def _run_sql_tool(self, sql: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            checked, df = self.wh.run(sql)
        except UnsafeSQLError as e:
            return f"REJECTED by guardrails: {e}", None, str(e)
        except Exception as e:  # database error -> let the model self-correct
            return f"ERROR: {e}", None, str(e)
        preview = df.head(20).to_csv(index=False)
        note = " (LIMIT added automatically)" if checked.limited else ""
        return f"{len(df)} rows{note}.\n{preview}", df, None

    def ask(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        result = AgentResult(question=question, sql=None, answer="")
        for step in range(1, self.max_steps + 1):
            result.steps = step
            resp = self.client.create(
                model=self.model,
                max_tokens=1500,
                temperature=0,
                system=[
                    {"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}
                ],
                tools=TOOLS,
                messages=messages,
            )
            usage = getattr(resp, "usage", None)
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == "final_answer":
                    sql = block.input["sql"]
                    text, df, err = self._run_sql_tool(sql)
                    if err:
                        result.errors.append(err)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": text,
                                "is_error": True,
                            }
                        )
                        continue
                    result.sql, result.data = self.wh.validate(sql).sql, df
                    result.answer = block.input.get("answer", "")
                    result.chart = block.input.get("chart", "table")
                    return result
                if block.name == "run_sql":
                    text, _, err = self._run_sql_tool(block.input["sql"])
                    if err:
                        result.errors.append(err)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                            "is_error": err is not None,
                        }
                    )
            if not tool_results:  # model answered in prose without tools
                result.answer = " ".join(getattr(b, "text", "") for b in resp.content).strip()
                return result
            messages.append({"role": "user", "content": tool_results})
        result.answer = "I could not produce a valid query within the step limit."
        return result


def to_json(result: AgentResult) -> str:
    return json.dumps(
        {
            "question": result.question,
            "sql": result.sql,
            "answer": result.answer,
            "chart": result.chart,
            "steps": result.steps,
            "errors": result.errors,
        }
    )
