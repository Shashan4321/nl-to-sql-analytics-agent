"""Execution-accuracy evaluation for the NL-to-SQL agent.

Usage::

    python -m nl2sql.evaluate                 # runs the agent (needs ANTHROPIC_API_KEY)
    python -m nl2sql.evaluate --gold-only     # sanity-check the gold SQL, no API calls

Writes ``evals/results.json`` and ``evals/REPORT.md`` with accuracy overall and by tag.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .db import Warehouse

ROOT = Path(__file__).resolve().parents[2]


def load_cases(path: Path = ROOT / "evals" / "questions.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text())


def _normalise(df: pd.DataFrame, ordered: bool) -> list[tuple]:
    rows = []
    for rec in df.itertuples(index=False, name=None):
        norm = []
        for v in rec:
            if isinstance(v, (bool, np.bool_)):
                norm.append(int(v))
            elif isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                norm.append(round(float(v), 1))
            else:
                norm.append(None if pd.isna(v) else str(v))
        rows.append(tuple(norm))
    return rows if ordered else sorted(rows, key=repr)


def results_match(pred: pd.DataFrame, gold: pd.DataFrame, ordered: bool = False) -> bool:
    """True if both result sets contain the same values.

    Column names are ignored. Extra columns in the prediction are allowed as long as
    the gold columns can be found in it (models often add a helpful extra column).
    """
    if len(pred) != len(gold):
        return False
    g = _normalise(gold, ordered)
    if pred.shape[1] == gold.shape[1] and _normalise(pred, ordered) == g:
        return True
    # try to find the gold columns inside a wider prediction
    if pred.shape[1] > gold.shape[1]:
        chosen = []
        for gc in range(gold.shape[1]):
            gcol = _normalise(gold.iloc[:, [gc]], ordered)
            for pc in range(pred.shape[1]):
                if pc not in chosen and _normalise(pred.iloc[:, [pc]], ordered) == gcol:
                    chosen.append(pc)
                    break
            else:
                return False
        return _normalise(pred.iloc[:, chosen], ordered) == g
    return False


def run(gold_only: bool = False, db: str = "data/warehouse.duckdb") -> dict:
    wh = Warehouse(ROOT / db)
    cases = load_cases()
    agent = None
    if not gold_only:
        from .agent import NL2SQLAgent

        agent = NL2SQLAgent(wh)

    records, by_tag = [], defaultdict(lambda: [0, 0])
    for case in cases:
        rec = {"id": case["id"], "question": case["question"], "tags": case["tags"]}
        if case.get("expect_refusal"):
            if gold_only:
                continue
            res = agent.ask(case["question"])
            rec["passed"] = res.sql is None or not res.ok
            rec["answer"] = res.answer
        else:
            gold = wh.run_trusted(case["sql"])
            if gold_only:
                rec["passed"] = len(gold) > 0
                rec["gold_rows"] = len(gold)
            else:
                t0 = time.perf_counter()
                res = agent.ask(case["question"])
                rec.update(
                    sql=res.sql,
                    steps=res.steps,
                    errors=res.errors,
                    latency_s=round(time.perf_counter() - t0, 2),
                    tokens=res.input_tokens + res.output_tokens,
                )
                rec["passed"] = bool(
                    res.ok and results_match(res.data, gold, ordered=case.get("ordered", False))
                )
        records.append(rec)
        for tag in case["tags"]:
            by_tag[tag][0] += rec["passed"]
            by_tag[tag][1] += 1

    passed = sum(r["passed"] for r in records)
    summary = {
        "mode": "gold-only" if gold_only else "agent",
        "cases": len(records),
        "passed": passed,
        "accuracy_pct": round(100 * passed / max(len(records), 1), 1),
        "by_tag": {k: {"passed": v[0], "total": v[1]} for k, v in sorted(by_tag.items())},
        "records": records,
    }
    if not gold_only:
        out = ROOT / "evals"
        (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
        lines = [
            f"# Evaluation report\n\nExecution accuracy: **{summary['accuracy_pct']}%** "
            f"({passed}/{len(records)})\n",
            "| Tag | Passed | Total |",
            "|---|---|---|",
        ]
        lines += [f"| {k} | {v['passed']} | {v['total']} |" for k, v in summary["by_tag"].items()]
        failed = [r for r in records if not r["passed"]]
        if failed:
            lines += ["\n## Failed cases\n"] + [f"- `{r['id']}` {r['question']}" for r in failed]
        (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-only", action="store_true", help="validate gold SQL only")
    args = p.parse_args()
    s = run(gold_only=args.gold_only)
    print(f"{s['mode']}: {s['passed']}/{s['cases']} passed ({s['accuracy_pct']}%)")


if __name__ == "__main__":
    main()
