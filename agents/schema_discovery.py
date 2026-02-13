"""
Table Inspection Agent.

Inspects data files and uses LLM reasoning to infer the relational schema
(primary keys, time columns, foreign key links). Does NOT build the graph —
that is deferred to the query-driven graph builder.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from tools.kumo_tools import load_tables, ensure_init, get_semantic_types
from tools.llm import get_llm

logger = logging.getLogger(__name__)


def _resolve_schema_conflicts(schema: dict, raw_tables: dict) -> dict:
    """
    Fix schema conflicts that would cause KumoRFM to reject the graph:
    1. PK/FK conflict: column is both PK and FK on the same table
    2. PK/time conflict: column is both PK and time_column on the same table
    """
    table_configs = schema.get("tables", {})
    links = schema.get("links", [])

    # --- PK/FK conflicts ---
    fk_cols_by_table: dict[str, set[str]] = {}
    for link in links:
        src = link.get("src_table")
        fkey = link.get("fkey")
        if src and fkey:
            fk_cols_by_table.setdefault(src, set()).add(fkey)

    for tname, tconf in table_configs.items():
        pk = tconf.get("primary_key")
        if not pk or pk not in fk_cols_by_table.get(tname, set()):
            continue

        logger.warning(f"  PK/FK conflict in {tname}: '{pk}' is both PK and FK")

        df = raw_tables.get(tname)
        if df is None:
            tconf["primary_key"] = None
            logger.warning(f"  No DataFrame for {tname}, setting PK to None")
            continue

        all_fks = fk_cols_by_table.get(tname, set())
        time_col = tconf.get("time_column")
        best_alt = None
        for col in df.columns:
            if col == pk or col in all_fks or col == time_col:
                continue
            if df[col].nunique() == len(df) and df[col].notna().all():
                best_alt = col
                break

        if best_alt:
            tconf["primary_key"] = best_alt
            logger.info(f"  Resolved: {tname} PK changed from '{pk}' to '{best_alt}'")
        else:
            tconf["primary_key"] = None
            logger.info(f"  No alternative PK found for {tname}, setting PK to None")

    # --- PK/time conflicts ---
    for tname, tconf in table_configs.items():
        pk = tconf.get("primary_key")
        time_col = tconf.get("time_column")
        if pk and time_col and pk == time_col:
            logger.warning(f"  PK/time conflict in {tname}: '{pk}' is both PK and time_column")
            tconf["primary_key"] = None
            logger.info(f"  Resolved: {tname} PK set to None (keeping time_column)")

    return schema


def _llm_infer_schema(raw_tables: dict) -> dict:
    """Use LLM to infer primary keys, time columns, and FK links."""
    schema_parts = []
    for name, df in raw_tables.items():
        cols = []
        total = len(df)
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            samples = [str(v) for v in df[col].dropna().head(3).tolist()]
            cols.append(
                f"  {col}: dtype={dtype}, unique={nunique}/{total}, samples=[{', '.join(samples)}]"
            )
        schema_parts.append(f"Table: {name} ({total} rows)\n" + "\n".join(cols))

    schema_text = "\n\n".join(schema_parts)

    prompt = f"""Analyze this relational database and identify the schema structure.

{schema_text}

Think step by step:
1. For each table, find the column that uniquely identifies each row (unique count = row count) — this is the primary key. A PK does NOT need "id" in its name; any column with all unique values that logically identifies the entity is valid. If multiple columns are fully unique, prefer the one that is NOT referenced as a foreign key in other tables.
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
- If a column is a foreign key referencing another table, it CANNOT also be the primary key of its own table. Choose a different unique column as PK.
- A primary key CANNOT also be the time column. They must be different columns. If no other unique column exists, set primary_key to null.
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
    result = _resolve_schema_conflicts(result, raw_tables)

    for tname, tconf in result.get("tables", {}).items():
        logger.info(f"  LLM schema: {tname} -> pk={tconf.get('primary_key')}, time={tconf.get('time_column')}")
    for link in result.get("links", []):
        logger.info(f"  LLM link: {link['src_table']}.{link['fkey']} -> {link['dst_table']}")
    return result


def _summarize_table(name: str, df, schema_info: dict, semantic_types: dict[str, str] | None = None) -> dict[str, Any]:
    """Create a rich summary of a DataFrame using LLM-inferred schema info and KumoRFM semantic types."""
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
        if semantic_types and col in semantic_types:
            col_info["semantic_type"] = semantic_types[col]
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


def inspect_tables(state: dict) -> dict:
    """
    Node function: Load data and infer relational schema via LLM.
    Does NOT build the graph — that happens after PQL query generation.

    Reads from state: data_path, table_names
    Writes to state: tables, raw_tables, llm_schema, tables_loaded
    """
    data_path = state["data_path"]
    logger.info(f"Inspecting tables from: {data_path}")
    ensure_init()

    s3_table_names = state.get("table_names")
    raw_tables = load_tables(data_path, table_names=s3_table_names)
    if not raw_tables:
        return {
            "tables": {},
            "raw_tables": {},
            "tables_loaded": False,
            "errors": [f"No tables found at {data_path}"],
            "current_step": "table_inspection_failed",
        }

    try:
        schema_info = _llm_infer_schema(raw_tables)
        logger.info("Schema inferred via LLM")
    except Exception as e:
        logger.warning(f"LLM schema inference failed: {e}")
        schema_info = {"tables": {}, "links": []}

    all_semantic_types = {}
    table_configs = schema_info.get("tables", {})
    for tname, df in raw_tables.items():
        tconf = table_configs.get(tname, {})
        stypes = get_semantic_types(df, tname, primary_key=tconf.get("primary_key"), time_column=tconf.get("time_column"))
        all_semantic_types[tname] = stypes
        logger.info(f"  Semantic types for {tname}: {stypes}")

    table_summaries = {
        name: _summarize_table(name, df, schema_info, semantic_types=all_semantic_types.get(name))
        for name, df in raw_tables.items()
    }

    logger.info(
        f"Tables inspected: {len(raw_tables)} tables, "
        f"{sum(len(df) for df in raw_tables.values())} total rows"
    )

    return {
        "tables": table_summaries,
        "raw_tables": raw_tables,
        "llm_schema": schema_info,
        "tables_loaded": True,
        "current_step": "tables_inspected",
    }
