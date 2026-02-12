"""
Strategy Synthesizer Agent.

Combines predictions (with native KumoRFM explanations) and pre-computed
data analysis into an actionable business strategy report.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from tools.llm import get_llm

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are a senior business strategist. You've been given prediction results AND
a pre-computed data analysis from a Relational Foundation Model (KumoRFM).

Your job: Synthesize predictions, model explanations, AND the data analysis into an actionable strategy report.

### Report Structure:
1. **Executive Summary** (2-3 sentences answering the business question directly, with specific numbers from predictions)
2. **Key Findings** (what the predictions revealed — cite exact prediction values and data statistics)
3. **Data-Backed Recommended Actions** (each action must reference a specific prediction or data statistic that supports it)
4. **Risk Assessment** (base this on the data quality metrics and prediction variance provided, not generic statements)

### Rules:
- EVERY claim must be backed by a number from the predictions or data analysis provided
- Do NOT invent statistics or percentages not present in the data
- If model explanations identify key factors, cite them by name
- Risk assessment must reference actual data limitations (sample sizes, missing data, variance)
- Recommended actions must tie to specific prediction values
- Use markdown formatting
- Do NOT use dollar signs ($) for currency — write "USD" or just the number instead

Write the report in markdown. Be concise and data-driven.
"""


def _extract_prediction_values(result_data: Any) -> list[float]:
    """Extract numeric prediction values from a serialized result."""
    values = []
    if not isinstance(result_data, dict):
        return values
    for col, vals in result_data.items():
        if col.startswith("_"):
            continue
        if col.upper().startswith("TARGET") or col.upper().endswith("PRED"):
            if isinstance(vals, list):
                values.extend(v for v in vals if isinstance(v, (int, float)))
            elif isinstance(vals, (int, float)):
                values.append(vals)
    return values


def _compute_data_analysis(predictions: list[dict], tables: dict) -> str:
    """Pre-compute statistics from prediction results and table metadata."""
    sections = []

    pred_values_all = []
    for pred in predictions:
        if not pred.get("success"):
            continue
        vals = _extract_prediction_values(pred.get("result", {}))
        if vals:
            pred_values_all.extend(vals)
            sections.append(
                f"- {pred.get('hypothesis', 'N/A')}: "
                f"predicted={vals[0]:.4f}" + (f" (mean={statistics.mean(vals):.4f}, "
                f"min={min(vals):.4f}, max={max(vals):.4f}, n={len(vals)})" if len(vals) > 1 else "")
            )

    if pred_values_all:
        sections.insert(0, "### Prediction Value Summary")
        if len(pred_values_all) > 1:
            sections.append(
                f"\nAcross all predictions: mean={statistics.mean(pred_values_all):.4f}, "
                f"stdev={statistics.stdev(pred_values_all):.4f}, "
                f"range=[{min(pred_values_all):.4f}, {max(pred_values_all):.4f}]"
            )

    explanations_text = []
    for pred in predictions:
        if not pred.get("success"):
            continue
        result = pred.get("result", {})
        if isinstance(result, dict) and "_summary" in result:
            explanations_text.append(
                f"- {pred.get('hypothesis', 'N/A')}:\n  {result['_summary'][:400]}"
            )
        entity_expls = pred.get("entity_explanations", [])
        for ee in entity_expls:
            ee_result = ee.get("result", {})
            if isinstance(ee_result, dict) and "_summary" in ee_result:
                explanations_text.append(
                    f"- Entity {ee.get('entity_id')}: {ee_result['_summary'][:300]}"
                )
    if explanations_text:
        sections.append("\n### Model Explanation Factors")
        sections.extend(explanations_text)

    table_stats = []
    for name, info in tables.items():
        row_count = info.get("row_count", "?")
        cols = info.get("columns", {})
        col_count = len(cols) if isinstance(cols, (dict, list)) else "?"
        numeric_cols = []
        if isinstance(cols, dict):
            for col_name, col_info in cols.items():
                if isinstance(col_info, dict):
                    dtype = col_info.get("dtype", "").lower()
                    if any(t in dtype for t in ("int", "float", "numeric", "double")):
                        stats_parts = []
                        if "min" in col_info:
                            stats_parts.append(f"range=[{col_info['min']}, {col_info['max']}]")
                        if stats_parts:
                            numeric_cols.append(f"    {col_name}: {', '.join(stats_parts)}")
        line = f"- {name}: {row_count} rows, {col_count} columns"
        table_stats.append(line)
        table_stats.extend(numeric_cols)

    if table_stats:
        sections.append("\n### Data Quality & Coverage")
        sections.extend(table_stats)
        total_rows = sum(t.get("row_count", 0) for t in tables.values())
        sections.append(f"- Total data points: {total_rows:,}")

    return "\n".join(sections)


def synthesize_strategy(state: dict) -> dict:
    """
    Node function: Synthesize predictions into strategy report.
    Pre-computes data statistics and feeds them into the LLM.

    Reads from state: question, predictions, tables
    Writes to state: strategy_report, confidence_score, current_step
    """
    question = state["question"]
    predictions = state.get("predictions", [])
    tables = state.get("tables", {})

    logger.info("Synthesizing strategy report...")

    successful = [p for p in predictions if p.get("success")]
    failed = [p for p in predictions if not p.get("success")]
    confidence = len(successful) / len(predictions) if predictions else 0.0

    data_analysis = _compute_data_analysis(predictions, tables)
    logger.info(f"Data analysis computed ({len(data_analysis)} chars)")

    context = f'## Business Question\n"{question}"\n\n'
    context += f"## Pre-Computed Data Analysis\n{data_analysis}\n\n"
    context += "## Raw Prediction Results\n"

    for i, pred in enumerate(successful):
        context += f"\n### Prediction {i+1}: {pred.get('hypothesis', 'N/A')}\n"
        context += f"- Type: {pred.get('prediction_type', 'unknown')}\n"
        context += f"- Query: `{pred['query']}`\n"
        result_str = json.dumps(pred["result"], indent=2, default=str)
        context += f"- Result: {result_str[:800]}\n"
        context += f"- Rationale: {pred.get('rationale', 'N/A')}\n"
        if "explanation" in pred:
            expl_str = json.dumps(pred["explanation"], indent=2, default=str)
            context += f"- Model Explanation: {expl_str[:500]}\n"
        entity_expls = pred.get("entity_explanations", [])
        if entity_expls:
            context += f"- Entity-Level Explanations ({len(entity_expls)} entities):\n"
            for ee in entity_expls:
                ee_str = json.dumps(ee.get("result", {}), indent=2, default=str)
                context += f"  - Entity {ee.get('entity_id')}: {ee_str[:300]}\n"

    if failed:
        context += f"\n**Note:** {len(failed)} prediction(s) failed and are excluded.\n"

    context += f"\n**Prediction Success Rate: {confidence:.0%}**\n"

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=context),
    ])

    report = response.content.strip()
    report += f"\n\n---\n*Generated by PredictiveAgent using KumoRFM*\n"
    report += f"*Predictions: {len(successful)} successful, {len(failed)} failed*\n"
    report += f"*Confidence: {confidence:.0%}*\n"

    logger.info("Strategy report generated")

    return {
        "strategy_report": report,
        "confidence_score": confidence,
        "current_step": "complete",
    }
