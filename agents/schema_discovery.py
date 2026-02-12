"""
Schema Discovery Agent.

Inspects data files, uses LLM reasoning to infer the relational schema
(primary keys, time columns, foreign key links), and builds a KumoRFM graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from tools.kumo_tools import (
    load_tables, build_graph, build_graph_with_schema, get_graph_info, ensure_init,
)
from tools.llm import get_llm

logger = logging.getLogger(__name__)


def _llm_infer_schema(raw_tables: dict) -> dict:
    """Use LLM to infer primary keys, time columns, and FK links."""
    schema_parts = []
    for name, df in raw_tables.items():
        cols = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            total = len(df)
            samples = [str(v) for v in df[col].dropna().head(3).tolist()]
            cols.append(
                f"  {col}: dtype={dtype}, unique={nunique}/{total}, samples=[{', '.join(samples)}]"
            )
        schema_parts.append(f"Table: {name} ({total} rows)\n" + "\n".join(cols))

    schema_text = "\n\n".join(schema_parts)

    prompt = f"""Analyze this relational database and identify the schema structure.

{schema_text}

Think step by step:
1. For each table, find the column with unique={total}/{total} (or near-unique) that serves as the row identifier — this is the primary key. A PK does NOT need "id" in its name; any column with all unique values that logically identifies the entity is valid.
2. For each table, check if any column has a datetime/timestamp dtype — this is the time column. Set to null if none.
3. For relationships: find columns that appear in multiple tables. If table A has a non-PK column whose values match table B's primary key, that column is a foreign key from A to B.

Return JSON with this exact structure:
{{
  "tables": {{
    "<table_name>": {{
      "primary_key": "<column that uniquely identifies rows>",
      "time_column": "<datetime column for temporal ordering, or null>"
    }}
  }},
  "links": [
    {{"src_table": "<table containing FK>", "fkey": "<FK column name>", "dst_table": "<table whose PK is referenced>"}}
  ]
}}

Constraints:
- The fkey column in a link must NOT be the src_table's own primary key. A PK cannot double as a FK in the same table.
- Links go from child table (src, has the FK) to parent table (dst, has the matching PK).
- Every transaction/event table should link to its parent entity tables.
Return ONLY valid JSON, no explanation."""

    llm = get_llm(model_name='gpt-4o')
    resp = llm.invoke([
        SystemMessage(content="You are a database schema analyst. Output only valid JSON."),
        HumanMessage(content=prompt),
    ])
    content = resp.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(content)
    for tname, tconf in result.get("tables", {}).items():
        logger.info(f"  LLM schema: {tname} -> pk={tconf.get('primary_key')}, time={tconf.get('time_column')}")
    for link in result.get("links", []):
        logger.info(f"  LLM link: {link['src_table']}.{link['fkey']} -> {link['dst_table']}")
    return result


def _summarize_table(name: str, df, schema_info: dict) -> dict[str, Any]:
    """Create a rich summary of a DataFrame using LLM-inferred schema info."""
    table_conf = schema_info.get("tables", {}).get(name, {})
    primary_key = table_conf.get("primary_key")
    time_column = table_conf.get("time_column")

    fk_map = {}
    for link in schema_info.get("links", []):
        if link.get("src_table") == name:
            fk_map[link["fkey"]] = link["dst_table"]

    columns = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if col == primary_key:
            role = "primary_key"
        elif col == time_column:
            role = "time_column"
        elif col in fk_map:
            role = "foreign_key"
        elif "datetime" in dtype.lower() or "timestamp" in dtype.lower():
            role = "time_column"
        elif any(t in dtype.lower() for t in ("int", "float")):
            role = "numeric"
        else:
            role = "categorical"

        col_info = {"dtype": dtype, "role": role, "nunique": int(df[col].nunique())}
        if col in fk_map:
            col_info["references"] = f"{fk_map[col]}.{col}"
        if any(t in dtype.lower() for t in ("int", "float")):
            if not df[col].isna().all():
                col_info["min"] = float(df[col].min())
                col_info["max"] = float(df[col].max())
                col_info["mean"] = round(float(df[col].mean()), 2)
        columns[col] = col_info

    has_time = time_column is not None
    is_event = has_time and len(fk_map) > 0

    sample_ids = []
    if primary_key and primary_key in df.columns:
        sample_ids = df[primary_key].dropna().head(5).tolist()

    return {
        "name": name,
        "row_count": len(df),
        "columns": columns,
        "primary_key": primary_key,
        "has_time_column": has_time,
        "time_column": time_column,
        "is_event_table": is_event,
        "table_type": "event" if is_event else "entity",
        "foreign_key_references": [f"{dst}.{fk}" for fk, dst in fk_map.items()],
        "sample_ids": sample_ids,
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


def discover_schema(state: dict) -> dict:
    """
    Node function: Discover schema from data and build KumoRFM graph.
    Uses LLM to infer PKs, time columns, and FK links.
    Falls back to auto-detection if LLM inference fails.

    Reads from state: question, data_path, table_names
    Writes to state: tables, graph_schema, graph_built
    """
    data_path = state["data_path"]
    logger.info(f"Discovering schema from: {data_path}")
    ensure_init()

    s3_table_names = state.get("table_names")
    raw_tables = load_tables(data_path, table_names=s3_table_names)
    if not raw_tables:
        return {
            "tables": {},
            "graph_built": False,
            "errors": [f"No tables found at {data_path}"],
            "current_step": "schema_discovery_failed",
        }

    try:
        schema_info = _llm_infer_schema(raw_tables)
        graph = build_graph_with_schema(raw_tables, schema_info)
        logger.info("Graph built with LLM-inferred schema")
    except Exception as e:
        logger.warning(f"LLM schema inference failed, using auto-detection: {e}")
        schema_info = {"tables": {}, "links": []}
        graph = build_graph(raw_tables)

    table_summaries = {
        name: _summarize_table(name, df, schema_info)
        for name, df in raw_tables.items()
    }
    graph_info = get_graph_info(list(raw_tables.keys()))

    schema = {
        "tables": table_summaries,
        "relationships": graph_info,
        "graph_object": graph,
    }

    logger.info(
        f"Schema discovered: {len(raw_tables)} tables, "
        f"{sum(len(df) for df in raw_tables.values())} total rows"
    )

    return {
        "tables": table_summaries,
        "graph_schema": schema,
        "graph_built": True,
        "llm_schema": schema_info,
        "current_step": "schema_discovered",
    }
