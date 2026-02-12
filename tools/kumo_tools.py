"""
KumoRFM SDK wrapper tools.

Handles initialization, graph construction, and predictions.
"""

from __future__ import annotations

import os
import re
import logging
from typing import Any

import pandas as pd
import kumoai.experimental.rfm as rfm

logger = logging.getLogger(__name__)

_initialized = False

def ensure_init():
    """Initialize KumoRFM client (idempotent)."""
    global _initialized
    if not _initialized:
        rfm.init()
        _initialized = True

def load_tables(
    data_path: str,
    table_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load parquet/csv files from a local directory or S3 path.
    For S3, table_names must be provided (no listing support).
    For local dirs, auto-discovers all parquet/csv files.
    """
    tables = {}

    if data_path.startswith("s3://"):
        if not table_names:
            raise ValueError("table_names required for S3 paths (e.g. ['users','items','orders'])")
        for name in table_names:
            url = f"{data_path.rstrip('/')}/{name}.parquet"
            df = pd.read_parquet(url)
            tables[name] = df
            logger.info(f"Loaded {name}: {len(df)} rows, {list(df.columns)}")
    else:
        import glob
        for ext in ["*.parquet", "*.csv"]:
            for filepath in glob.glob(os.path.join(data_path, ext)):
                name = os.path.splitext(os.path.basename(filepath))[0]
                if filepath.endswith(".parquet"):
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)
                tables[name] = df
                logger.info(f"Loaded {name}: {len(df)} rows")

    return tables

def build_graph(tables: dict[str, pd.DataFrame]) -> rfm.LocalGraph:
    """Build a KumoRFM LocalGraph from DataFrames (auto-detection fallback)."""
    ensure_init()
    graph = rfm.LocalGraph.from_data(tables)
    return graph


def build_graph_with_schema(
    tables: dict[str, pd.DataFrame],
    schema_info: dict,
) -> rfm.LocalGraph:
    """Build a KumoRFM LocalGraph with LLM-inferred PK, time columns, and links."""
    ensure_init()
    table_configs = schema_info.get("tables", {})
    links = schema_info.get("links", [])

    local_tables = []
    for name, df in tables.items():
        config = table_configs.get(name, {})
        kwargs = {"name": name}
        pk = config.get("primary_key")
        if pk and pk in df.columns:
            kwargs["primary_key"] = pk
        time_col = config.get("time_column")
        if time_col and time_col in df.columns:
            kwargs["time_column"] = time_col
        lt = rfm.LocalTable(df, **kwargs)
        local_tables.append(lt)
        logger.info(f"  LocalTable {name}: pk={kwargs.get('primary_key', 'auto')}, time={kwargs.get('time_column', 'none')}")

    graph = rfm.LocalGraph(tables=local_tables)

    for link in links:
        src, fkey, dst = link.get("src_table"), link.get("fkey"), link.get("dst_table")
        if not (src and fkey and dst):
            continue
        src_pk = table_configs.get(src, {}).get("primary_key")
        if fkey == src_pk:
            logger.warning(f"  Skipping link {src}.{fkey} -> {dst}: FK is the src table's PK")
            continue
        try:
            graph.link(src_table=src, fkey=fkey, dst_table=dst)
            logger.info(f"  Link: {src}.{fkey} -> {dst}")
        except Exception as e:
            logger.warning(f"  Link failed {src}.{fkey} -> {dst}: {e}")

    return graph


def get_graph_info(table_names: list[str]) -> dict[str, Any]:
    """Build graph metadata from known table names."""
    return {
        "tables": table_names,
        "description": "Relational graph built from loaded tables",
    }

def _serialize_result(obj) -> Any:
    """Convert a KumoRFM result to a JSON-serializable structure."""
    if hasattr(obj, "prediction"):
        pred = obj.prediction
        if hasattr(pred, "to_dict"):
            data = {col: pred[col].tolist() for col in pred.columns}
            if hasattr(obj, "summary"):
                data["_summary"] = str(obj.summary)
            return data
    if hasattr(obj, "to_pandas"):
        df = obj.to_pandas()
        return {col: df[col].tolist() for col in df.columns}
    if hasattr(obj, "columns") and hasattr(obj, "to_dict"):
        return {col: obj[col].tolist() for col in obj.columns}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _is_multi_entity(pql_query: str) -> bool:
    """Check if a PQL query targets multiple entities (uses IN (...))."""
    return bool(re.search(r'\bFOR\b.+\bIN\s*\(', pql_query, re.IGNORECASE))


def _decompose_for_explain(
    model: rfm.KumoRFM,
    pql_query: str,
    anchor_time: pd.Timestamp | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    For multi-entity queries: run bulk without explain, then explain
    the most interesting entities individually.
    """
    kwargs = {}
    if anchor_time is not None:
        kwargs["anchor_time"] = anchor_time

    bulk = model.predict(pql_query, explain=False, **kwargs)
    bulk_data = _serialize_result(bulk)

    for_match = re.search(r'FOR\s+(\w+)\.(\w+)\s+IN\s*\(([^)]+)\)', pql_query, re.IGNORECASE)
    if not for_match:
        return {"query": pql_query, "result": bulk_data, "success": True, "error": None}

    table_name, col_name = for_match.group(1), for_match.group(2)
    pred_col = None
    if isinstance(bulk_data, dict):
        for c in bulk_data:
            if c.upper().startswith("TARGET") or c.upper().endswith("PRED"):
                pred_col = c
                break

    entity_col = None
    if isinstance(bulk_data, dict):
        for c in bulk_data:
            if c.upper() == "ENTITY" or col_name.lower() in c.lower():
                entity_col = c
                break

    explanations = []
    if entity_col and pred_col and isinstance(bulk_data.get(entity_col), list):
        entities = bulk_data[entity_col]
        preds = bulk_data[pred_col]
        paired = sorted(zip(entities, preds), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
        interesting = [eid for eid, _ in paired[:top_k]]

        base_query = re.sub(r'FOR\s+\w+\.\w+\s+IN\s*\([^)]+\)', '', pql_query, flags=re.IGNORECASE).strip()
        for eid in interesting:
            single_pql = f"{base_query} FOR {table_name}.{col_name}={eid}"
            logger.info(f"  Explaining entity {col_name}={eid}: {single_pql}")
            try:
                expl_result = model.predict(single_pql, explain=True, **kwargs)
                explanations.append({
                    "entity_id": eid,
                    "query": single_pql,
                    "result": _serialize_result(expl_result),
                })
            except Exception as ex:
                logger.warning(f"  Explain failed for {col_name}={eid}: {ex}")

    output = {
        "query": pql_query,
        "result": bulk_data,
        "success": True,
        "error": None,
    }
    if explanations:
        output["entity_explanations"] = explanations
    if anchor_time is not None:
        output["anchor_time"] = str(anchor_time)
    return output


def run_prediction(
    model: rfm.KumoRFM,
    pql_query: str,
    explain: bool = False,
    anchor_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Execute a single PQL query. Optionally include explanations.
    For multi-entity queries with explain=True, decomposes into bulk + individual explains.
    """
    if explain and _is_multi_entity(pql_query):
        try:
            return _decompose_for_explain(model, pql_query, anchor_time=anchor_time)
        except Exception as e:
            logger.warning(f"Decompose-explain failed, falling back: {e}")

    try:
        kwargs = {"explain": explain}
        if anchor_time is not None:
            kwargs["anchor_time"] = anchor_time
        result = model.predict(pql_query, **kwargs)
        output = {
            "query": pql_query,
            "result": _serialize_result(result),
            "success": True,
            "error": None,
        }
        if explain and hasattr(result, 'explanation'):
            expl = result.explanation
            output["explanation"] = _serialize_result(expl)
        if anchor_time is not None:
            output["anchor_time"] = str(anchor_time)
        return output
    except Exception as e:
        if explain and "single entity" in str(e).lower():
            logger.info(f"Explain not supported for multi-entity, retrying without explain")
            return run_prediction(model, pql_query, explain=False, anchor_time=anchor_time)
        logger.error(f"Prediction failed for '{pql_query}': {e}")
        return {
            "query": pql_query,
            "result": None,
            "success": False,
            "error": str(e),
        }
