"""
Query-Driven Graph Builder.

Analyzes generated PQL queries to determine required entity-target links,
performs denormalization when the target table is >1 hop from the entity,
and builds the KumoRFM graph that supports all queries.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from tools.kumo_tools import build_graph_with_schema, get_graph_info, ensure_init
from tools.llm import get_llm

logger = logging.getLogger(__name__)

MAX_SCHEMA_RETRIES = 1


def _parse_entity_from_pql(pql: str) -> tuple[str | None, str | None]:
    """Extract entity table and column from the FOR clause."""
    match = re.search(r'FOR\s+(\w+)\.(\w+)\s*(?:=|IN\b)', pql, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _parse_target_table_from_pql(pql: str) -> str | None:
    """Extract the target table from the PREDICT clause."""
    match = re.search(r'PREDICT\s+(?:\w+\s*\()?\s*(\w+)\.(?:\w+|\*)', pql, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _has_direct_link(target_table: str, entity_table: str, links: list[dict], table_configs: dict) -> bool:
    """
    Check if any direct FK link exists between target and entity (either direction).
    Also verifies the link won't be skipped due to PK/FK conflict.
    """
    for link in links:
        src, fkey, dst = link.get("src_table"), link.get("fkey"), link.get("dst_table")
        if (src == target_table and dst == entity_table) or \
           (src == entity_table and dst == target_table):
            src_pk = table_configs.get(src, {}).get("primary_key")
            if fkey and fkey == src_pk:
                logger.warning(f"  Link {src}.{fkey} -> {dst} exists but will be skipped (FK == PK)")
                continue
            return True
    return False


def _find_intermediate(
    target_table: str,
    entity_table: str,
    links: list[dict],
) -> dict | None:
    """
    Find a 2-hop path: target -> intermediate -> entity.
    Returns path info dict or None if no path exists.
    """
    outgoing: dict[str, list[tuple[str, str, str]]] = {}
    for link in links:
        src, fkey, dst = link["src_table"], link["fkey"], link["dst_table"]
        outgoing.setdefault(src, []).append((dst, fkey, "forward"))
        outgoing.setdefault(dst, []).append((src, fkey, "reverse"))

    for mid, fkey1, dir1 in outgoing.get(target_table, []):
        if mid == entity_table:
            continue
        for next_hop, fkey2, dir2 in outgoing.get(mid, []):
            if next_hop == entity_table:
                return {
                    "intermediate": mid,
                    "hop1": {"fkey": fkey1, "direction": dir1},
                    "hop2": {"fkey": fkey2, "direction": dir2},
                }
    return None


def _denormalize_for_link(
    raw_tables: dict,
    llm_schema: dict,
    tables_meta: dict,
    target_table: str,
    entity_table: str,
    path: dict,
) -> bool:
    """
    Copy entity FK from intermediate into target table via merge.
    Modifies raw_tables and llm_schema in place.
    Returns True on success, False if structural preconditions aren't met.
    """
    intermediate = path["intermediate"]
    hop1, hop2 = path["hop1"], path["hop2"]
    table_configs = llm_schema.get("tables", {})

    # Only support: intermediate has FK to entity (forward direction)
    if hop2["direction"] != "forward":
        logger.warning(
            f"Skipping denormalization {target_table}->{entity_table}: "
            f"reverse entity link direction not supported"
        )
        return False

    entity_fk_col = hop2["fkey"]

    # Determine join columns between target and intermediate
    if hop1["direction"] == "forward":
        # target.fkey -> intermediate.PK
        target_join_col = hop1["fkey"]
        inter_pk = table_configs.get(intermediate, {}).get("primary_key")
        if not inter_pk:
            logger.warning(f"Skipping denormalization: {intermediate} has no known PK")
            return False
        inter_join_col = inter_pk
    else:
        # intermediate.fkey -> target.PK
        inter_join_col = hop1["fkey"]
        target_pk = table_configs.get(target_table, {}).get("primary_key")
        if not target_pk:
            logger.warning(f"Skipping denormalization: {target_table} has no known PK")
            return False
        target_join_col = target_pk

    target_df = raw_tables[target_table]
    inter_df = raw_tables[intermediate]

    # If column already exists in target, just add the link
    if entity_fk_col in target_df.columns:
        logger.info(f"Column {entity_fk_col} already in {target_table}, adding link only")
        llm_schema["links"].append({
            "src_table": target_table, "fkey": entity_fk_col, "dst_table": entity_table,
        })
        return True

    # Verify required columns exist in intermediate
    if inter_join_col not in inter_df.columns:
        logger.warning(f"Skipping denormalization: column {inter_join_col} not in {intermediate}")
        return False
    if entity_fk_col not in inter_df.columns:
        logger.warning(f"Skipping denormalization: column {entity_fk_col} not in {intermediate}")
        return False

    # Merge: copy entity FK from intermediate into target
    subset = inter_df[[inter_join_col, entity_fk_col]].drop_duplicates(subset=[inter_join_col])
    merged = target_df.merge(subset, left_on=target_join_col, right_on=inter_join_col, how="left")

    if target_join_col != inter_join_col and inter_join_col in merged.columns:
        merged = merged.drop(columns=[inter_join_col])

    raw_tables[target_table] = merged

    # Handle PK/FK conflict
    target_cfg = table_configs.get(target_table, {})
    if target_cfg.get("primary_key") == entity_fk_col:
        logger.info(f"PK/FK conflict in {target_table}: setting PK to None")
        target_cfg["primary_key"] = None
        if target_table in tables_meta:
            tables_meta[target_table]["primary_key"] = None

    # Register new link
    llm_schema["links"].append({
        "src_table": target_table, "fkey": entity_fk_col, "dst_table": entity_table,
    })

    # Update table metadata
    if target_table in tables_meta:
        tables_meta[target_table]["columns"][entity_fk_col] = {
            "dtype": str(merged[entity_fk_col].dtype),
            "role": "foreign_key",
            "semantic_type": "ID",
            "nunique": int(merged[entity_fk_col].nunique()),
            "references": f"{entity_table}.{entity_fk_col}",
        }
        tables_meta[target_table]["row_count"] = len(merged)
        fk_refs = tables_meta[target_table].get("foreign_key_references", [])
        fk_refs.append(f"{entity_table}.{entity_fk_col}")

    logger.info(
        f"Denormalized: copied {entity_fk_col} from {intermediate} into {target_table}, "
        f"added link {target_table}.{entity_fk_col} -> {entity_table}"
    )
    return True


def _fix_schema_with_llm(llm_schema: dict, raw_tables: dict, error_msg: str) -> dict:
    """Ask LLM to fix schema based on graph build error."""
    table_info = []
    for name, df in raw_tables.items():
        cols = [f"  {c}: dtype={df[c].dtype}, unique={df[c].nunique()}/{len(df)}" for c in df.columns]
        table_info.append(f"Table: {name} ({len(df)} rows)\n" + "\n".join(cols))

    prompt = f"""The following schema caused a graph construction error.

ERROR: {error_msg}

CURRENT SCHEMA:
{json.dumps(llm_schema, indent=2)}

TABLE DATA:
{chr(10).join(table_info)}

Fix the schema to resolve this error. Rules:
- A primary key CANNOT also be a foreign key on the same table
- A primary key CANNOT also be the time column
- If no valid PK exists, set primary_key to null
- Preserve all valid links

Return the corrected schema as JSON with the same structure (tables + links). Return ONLY valid JSON."""

    llm = get_llm(model_name='gpt-4o')
    resp = llm.invoke([
        SystemMessage(content="You are a database schema analyst. Fix the schema error. Output only valid JSON."),
        HumanMessage(content=prompt),
    ])
    content = resp.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


def build_query_graph(state: dict) -> dict:
    """
    Node function: Build KumoRFM graph driven by PQL queries.

    Analyzes hypotheses to find required entity-target links.
    Denormalizes (copies FKs through intermediate tables) when PQL targets
    are more than 1 hop from entities. Then builds the graph.
    On graph build failure, sends error to LLM to fix schema and retries.

    Reads from state: hypotheses, raw_tables, llm_schema, tables
    Writes to state: graph_schema, graph_built, raw_tables, llm_schema, tables
    """
    hypotheses = state.get("hypotheses", [])
    raw_tables = state.get("raw_tables", {})
    llm_schema = state.get("llm_schema", {})
    tables_meta = state.get("tables", {})

    ensure_init()

    links = llm_schema.get("links", [])
    table_configs = llm_schema.get("tables", {})

    # Collect required entity-target pairs from all queries
    required_pairs: set[tuple[str, str, str]] = set()
    for hyp in hypotheses:
        pql = hyp.get("pql_query", "")
        entity_table, entity_col = _parse_entity_from_pql(pql)
        target_table = _parse_target_table_from_pql(pql)
        if not entity_table or not target_table or entity_table == target_table:
            continue
        required_pairs.add((target_table, entity_table, entity_col))

    # Check and fix links for each required pair
    denormalized = []
    for target_table, entity_table, entity_col in required_pairs:
        if _has_direct_link(target_table, entity_table, links, table_configs):
            logger.info(f"Direct link exists: {target_table} <-> {entity_table}")
            continue

        logger.info(f"No direct link: {target_table} -> {entity_table}, searching for path...")
        path = _find_intermediate(target_table, entity_table, links)
        if not path:
            logger.warning(f"No intermediate path found: {target_table} -> {entity_table}")
            continue

        logger.info(f"Found path via {path['intermediate']}, denormalizing...")
        ok = _denormalize_for_link(raw_tables, llm_schema, tables_meta, target_table, entity_table, path)
        if ok:
            denormalized.append(f"{target_table} -> {entity_table} (via {path['intermediate']})")

    if denormalized:
        logger.info(f"Denormalized {len(denormalized)} link(s): {denormalized}")

    # Build graph with retry on failure
    for attempt in range(1 + MAX_SCHEMA_RETRIES):
        try:
            graph = build_graph_with_schema(raw_tables, llm_schema)
            logger.info("Query-driven graph built successfully")
            break
        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_SCHEMA_RETRIES:
                logger.warning(f"Graph build failed (attempt {attempt + 1}): {error_msg}")
                logger.info("Sending error to LLM for schema correction...")
                fixed = _fix_schema_with_llm(llm_schema, raw_tables, error_msg)
                llm_schema = fixed
                logger.info(f"LLM corrected schema: {json.dumps(fixed, indent=2)}")
            else:
                raise

    return {
        "graph_schema": {
            "tables": tables_meta,
            "relationships": get_graph_info(list(raw_tables.keys())),
            "graph_object": graph,
        },
        "graph_built": True,
        "raw_tables": raw_tables,
        "llm_schema": llm_schema,
        "tables": tables_meta,
        "current_step": "graph_built",
    }
