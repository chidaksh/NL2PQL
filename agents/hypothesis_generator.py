"""
Hypothesis Generator Agent.

Takes a business question + discovered schema and generates multiple
testable prediction hypotheses as PQL queries.
Uses RAG retrieval from pql_knowledge_base, full PQL reference,
and static validation with retry.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from tools.llm import get_llm
from tools.pql_knowledge_base import retrieve_similar_pql, format_examples_for_prompt, load_pql_reference
from tools.pql_validator import validate_pql

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _build_schema_description(tables: dict[str, Any]) -> str:
    """Build a rich schema description from Phase 2 table summaries."""
    parts = []
    for name, info in tables.items():
        cols = info.get("columns", {})
        pk = info.get("primary_key", "?")
        row_count = info.get("row_count", "?")
        ttype = info.get("table_type", "unknown")
        time_col = info.get("time_column")

        header = f"Table: {name} ({row_count} rows, type={ttype}, pk={pk}"
        if time_col:
            header += f", time={time_col}"
        header += ")"
        parts.append(header)

        for col, ci in cols.items():
            dtype = ci.get("dtype", "?")
            role = ci.get("role", "?")
            ref = ci.get("references", "")
            line = f"  {col}: {dtype} ({role})"
            if ref:
                line += f" -> {ref}"
            if "min" in ci:
                line += f" [{ci['min']}..{ci['max']}]"
            parts.append(line)

        fk_refs = info.get("foreign_key_references", [])
        if fk_refs:
            parts.append(f"  Foreign keys: {', '.join(fk_refs)}")

        sample_ids = info.get("sample_ids", [])
        if sample_ids:
            parts.append(f"  Sample IDs ({pk}): {sample_ids}")

        sample_rows = info.get("sample_rows", [])
        if sample_rows:
            parts.append(f"  Sample rows: {json.dumps(sample_rows[:2], default=str)}")
        parts.append("")

    return "\n".join(parts)


def _build_system_prompt(pql_ref: str, rag_examples: str) -> str:
    """Build the system prompt with PQL reference and RAG examples."""
    return f"""You are a predictive analytics expert working with KumoRFM, a Relational Foundation Model.

Your job: Given a business question and a database schema, generate 4-6 testable prediction hypotheses.
Each hypothesis must be answerable by a PQL query against the given schema.

=== PQL REFERENCE ===
{pql_ref}

=== SIMILAR PQL EXAMPLES (from knowledge base) ===
{rag_examples}

=== RULES ===
1. Each hypothesis must address a different angle of the business question
2. Use ONLY table names and column names from the schema provided
3. Use sample IDs from the schema for entity IDs
4. Time window start must be >= 0 in PREDICT (negative allowed only in WHERE)
5. FOR clause must use the primary key column of the entity table
6. LIST_DISTINCT requires RANK TOP k
7. SUM/AVG/MAX/MIN only work on numeric columns
8. ALL hypotheses must directly address the business question. Do NOT include unrelated prediction types (e.g. no churn queries if the question is about product promotion).
9. HYPOTHESIS DIVERSITY — this is critical:
   - If the question specifies a time window (e.g. "next 30 days"), use THAT time window for most hypotheses. Do NOT generate 5 queries that are identical except for the time window.
   - Instead, explore DIFFERENT ANGLES: different metrics (SUM vs COUNT vs AVG), different aggregation targets, complementary analyses (e.g. revenue + order count + average order value).
   - Only vary time windows when the question is vague/analytical (e.g. "How is my business doing?") or when comparing short vs long term is explicitly needed.
   - Each hypothesis should provide UNIQUE insight, not a trivially different version of the same query.
10. Prefer single-entity predictions (FOR table.id=value) over multi-entity (FOR table.id IN (...)) because explain works only for single entities. Use IN only when comparing multiple entities is essential.
11. Never use IN with a single value — use = instead (e.g. FOR users.user_id=0, NOT FOR users.user_id IN (0))
12. Use ONLY PREDICT queries. Do NOT use EVALUATE PREDICT — the SDK does not support this combination.

=== OUTPUT FORMAT (strict JSON) ===
{{
  "hypotheses": [
    {{
      "hypothesis": "Natural language description",
      "pql_query": "The exact PQL query",
      "rationale": "Why this matters for the business question",
      "prediction_type": "churn|forecast|recommendation|imputation|classification"
    }}
  ]
}}

Respond with ONLY the JSON, no markdown fences, no preamble."""


def _parse_llm_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(content)
    return parsed.get("hypotheses", [])


def generate_hypotheses(state: dict) -> dict:
    """
    Node function: Generate prediction hypotheses from question + schema.
    Uses RAG retrieval, full PQL reference, and validation with retry.

    Reads from state: question, tables
    Writes to state: hypotheses, current_step
    """
    question = state["question"]
    tables = state["tables"]

    logger.info(f"Generating hypotheses for: '{question}'")

    schema_desc = _build_schema_description(tables)
    schema_for_validator = {"tables": tables}

    pql_ref = load_pql_reference()
    rag_results = retrieve_similar_pql(question, schema_summary=schema_desc[:200], top_k=6)
    rag_examples = format_examples_for_prompt(rag_results)
    system_prompt = _build_system_prompt(pql_ref, rag_examples)

    for r in rag_results:
        logger.info(f"  RAG: score={r['similarity_score']:.3f} | {r.get('natural_language', '')[:80]}")
        logger.info(f"       PQL: {r['pql']}")

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Business Question: \"{question}\"\n\n{schema_desc}\n\nGenerate 4-6 prediction hypotheses as PQL queries."),
    ]

    hypotheses = []
    for attempt in range(1 + MAX_RETRIES):
        response = llm.invoke(messages)
        hypotheses = _parse_llm_response(response.content)

        invalid = []
        valid = []
        for h in hypotheses:
            pql = h.get("pql_query", "")
            is_valid, errors = validate_pql(pql, schema_for_validator)
            if is_valid:
                valid.append(h)
            else:
                invalid.append({"pql": pql, "errors": errors})

        if not invalid or attempt == MAX_RETRIES:
            hypotheses = valid
            break

        error_feedback = "Some PQL queries had validation errors. Fix them:\n"
        for item in invalid:
            error_feedback += f"  Query: {item['pql']}\n  Errors: {item['errors']}\n\n"
        error_feedback += "Regenerate ALL hypotheses with corrected PQL queries."

        messages.append(HumanMessage(content=error_feedback))
        logger.info(f"Retry {attempt + 1}/{MAX_RETRIES}: {len(invalid)} invalid queries")

    logger.info(f"Generated {len(hypotheses)} valid hypotheses")
    for i, h in enumerate(hypotheses):
        logger.info(f"  H{i+1}: {h.get('pql_query', '')}")

    return {
        "hypotheses": hypotheses,
        "current_step": "hypotheses_generated",
    }
