"""
Prediction Executor Agent.

Runs PQL queries via KumoRFM with explain=True for native explanations.
Retries failed queries once by asking the LLM to fix them based on the error.
"""

from __future__ import annotations

import logging

import kumoai.experimental.rfm as rfm
from langchain_core.messages import SystemMessage, HumanMessage

from tools.kumo_tools import run_prediction
from tools.llm import get_llm
from tools.pql_validator import validate_pql

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


def _ask_llm_to_fix(pql: str, error: str, schema_desc: str) -> str | None:
    """Ask LLM to fix a failed PQL query based on the error message."""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You fix broken PQL (Predictive Query Language) queries. Return ONLY the corrected query.\n\n"
            "PQL syntax: PREDICT AGG(table.column, start, end[, days]) FOR entity_table.pk=value\n"
            "Supported AGGs: SUM, COUNT, AVG, MAX, MIN, COUNT_DISTINCT, LIST_DISTINCT, FIRST\n"
            "FOR clause: must use the table's primary key column with a literal value\n"
            "LIST_DISTINCT requires RANK TOP k\n"
            "Optional: WHERE AGG(table.column, start, end) operator value\n\n"
            "AGGREGATION-COLUMN RULES (API rejects violations):\n"
            "- SUM/AVG/MAX/MIN: only on numerical columns\n"
            "- COUNT: use COUNT(table.*, ...) with wildcard only — never COUNT(table.id_column)\n"
            "- COUNT_DISTINCT: only on categorical columns, NEVER on ID/PK/FK columns\n"
            "- LIST_DISTINCT: only on FK or categorical columns\n"
            "- No aggregation on ID columns (PK or FK) except LIST_DISTINCT on FK\n\n"
            "PQL does NOT support: GROUP BY, ORDER BY, HAVING, LIMIT, subqueries (SELECT...FROM), "
            "or EVALUATE PREDICT. Entity IDs must be literal values, never expressions."
        )),
        HumanMessage(content=f"Broken query: {pql}\nError: {error}\n\nSchema (table pk [stype]: columns):\n{schema_desc}\n\nReturn ONLY the fixed PQL query:"),
    ])
    fixed = response.content.strip()
    if fixed.startswith("```"):
        fixed = fixed.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return fixed if fixed.upper().startswith("PREDICT") or fixed.upper().startswith("EVALUATE") else None


def _build_brief_schema(tables: dict) -> str:
    """Build a brief schema string with stype info for error-fix prompts."""
    parts = []
    for name, info in tables.items():
        cols = info.get("columns", {})
        pk = info.get("primary_key", "?")
        stype = info.get("semantic_type", "")
        header = f"{name} (pk={pk}"
        if stype:
            header += f", stype={stype}"
        header += ")"

        col_parts = []
        if isinstance(cols, dict):
            for col, meta in cols.items():
                if isinstance(meta, dict):
                    st = meta.get("semantic_type", "")
                    col_parts.append(f"{col}[{st}]" if st else col)
                else:
                    col_parts.append(col)
        else:
            col_parts = list(cols)

        parts.append(f"{header}: {', '.join(col_parts)}")
    return "\n".join(parts)


def execute_predictions(state: dict) -> dict:
    """
    Node function: Execute PQL queries with explain=True.
    Retries failed queries once via LLM fix.

    Reads from state: hypotheses, graph_schema, tables, anchor_time
    Writes to state: predictions, current_step
    """
    import pandas as pd

    hypotheses = state.get("hypotheses", [])
    graph_schema = state.get("graph_schema", {})
    graph = graph_schema.get("graph_object")
    tables = state.get("tables", {})

    anchor_time_raw = state.get("anchor_time")
    anchor_time = pd.Timestamp(anchor_time_raw) if anchor_time_raw else None
    if anchor_time:
        logger.info(f"  Using anchor_time: {anchor_time}")

    if not hypotheses or graph is None:
        return {
            "predictions": [],
            "errors": ["No hypotheses or graph object"],
            "current_step": "prediction_failed",
        }

    model = rfm.KumoRFM(graph)
    schema_desc = _build_brief_schema(tables)

    results = []
    for i, hyp in enumerate(hypotheses):
        pql = hyp.get("pql_query", "")
        logger.info(f"  Running H{i+1}: {pql}")

        result = run_prediction(model, pql, explain=True, anchor_time=anchor_time)

        if not result["success"] and MAX_RETRIES > 0:
            logger.info(f"  H{i+1} failed, asking LLM to fix: {result['error']}")
            fixed_pql = _ask_llm_to_fix(pql, result["error"], schema_desc)
            if fixed_pql and fixed_pql != pql:
                is_valid, val_errors = validate_pql(fixed_pql, {"tables": tables})
                if not is_valid:
                    logger.warning(f"  H{i+1} LLM fix failed validation: {val_errors}")
                    result["fix_validation_errors"] = val_errors
                else:
                    logger.info(f"  Retrying H{i+1}: {fixed_pql}")
                    result = run_prediction(model, fixed_pql, explain=True, anchor_time=anchor_time)
                    result["retried_query"] = fixed_pql

        result["query"] = pql
        result["hypothesis"] = hyp.get("hypothesis", f"Hypothesis {i+1}")
        result["prediction_type"] = hyp.get("prediction_type", "unknown")
        result["rationale"] = hyp.get("rationale", "")
        results.append(result)

        status = "OK" if result["success"] else f"FAIL: {result['error']}"
        logger.info(f"  H{i+1}: {status}")

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Predictions complete: {successful}/{len(results)} succeeded")

    return {
        "predictions": results,
        "current_step": "predictions_executed",
    }
