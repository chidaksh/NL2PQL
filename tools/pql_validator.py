from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_AGGREGATIONS = {"SUM", "COUNT", "AVG", "MAX", "MIN", "COUNT_DISTINCT", "LIST_DISTINCT", "FIRST"}
NUMERIC_ONLY_AGGS = {"SUM", "AVG", "MAX", "MIN"}


def validate_pql(query: str, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a PQL query against a discovered schema.

    Args:
        query: PQL query string
        schema: Dict with structure:
            {
                "tables": {
                    "table_name": {
                        "columns": {
                            "col_name": {"dtype": str, "role": str, ...}
                        },
                        "sample_ids": [...],
                        "primary_key": str,
                        ...
                    }
                },
                "event_table": str,
                "entity_tables": [str],
            }

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []
    q = query.strip()

    is_evaluate = q.upper().startswith("EVALUATE")
    if is_evaluate:
        if re.match(r"^EVALUATE\s+PREDICT\b", q, re.IGNORECASE):
            errors.append("'EVALUATE PREDICT' is not supported by the SDK. Use PREDICT or EVALUATE separately, not combined.")
            return False, errors
        q = re.sub(r"^EVALUATE\s+", "", q, flags=re.IGNORECASE).strip()

    if not re.match(r"^PREDICT\s+", q, re.IGNORECASE):
        errors.append("Query must start with PREDICT (or EVALUATE PREDICT)")
        return False, errors

    if not re.search(r"\bFOR\b", q, re.IGNORECASE):
        errors.append("Query must contain a FOR clause specifying the entity")
        return False, errors

    if re.search(r"\bFOR\s+EACH\b", q, re.IGNORECASE):
        errors.append("FOR EACH is enterprise-only. Use 'FOR table.id = value' or 'FOR table.id IN (values)'")

    if re.search(r"\bSELECT\b", q, re.IGNORECASE):
        errors.append("SQL subqueries (SELECT) are not supported in PQL. Entity IDs must be literal values.")

    tables_in_schema = schema.get("tables", {})
    table_names = set(tables_in_schema.keys())

    table_col_refs = re.findall(r"(\w+)\.(\w+|\*)", q)
    for table_ref, col_ref in table_col_refs:
        if table_ref.upper() in {"RANK", "TOP"}:
            continue
        if table_ref not in table_names:
            errors.append(f"Table '{table_ref}' not found in schema. Available: {sorted(table_names)}")
            continue
        if col_ref == "*":
            continue
        table_info = tables_in_schema[table_ref]
        col_info = table_info.get("columns", {})
        if isinstance(col_info, dict):
            valid_cols = set(col_info.keys())
        elif isinstance(col_info, list):
            valid_cols = set(col_info)
        else:
            continue
        if col_ref not in valid_cols:
            errors.append(f"Column '{col_ref}' not found in table '{table_ref}'. Available: {sorted(valid_cols)}")

    agg_match = re.search(r"\b(SUM|COUNT|AVG|MAX|MIN|COUNT_DISTINCT|LIST_DISTINCT|FIRST)\s*\(", q, re.IGNORECASE)
    if agg_match:
        agg_name = agg_match.group(1).upper()
        if agg_name not in VALID_AGGREGATIONS:
            errors.append(f"Unknown aggregation '{agg_name}'. Valid: {sorted(VALID_AGGREGATIONS)}")

        inner = re.search(rf"{agg_name}\s*\((\w+)\.(\w+)", q, re.IGNORECASE)
        agg_table, agg_col = (inner.group(1), inner.group(2)) if inner else (None, None)

        if agg_table and agg_col and agg_col != "*" and agg_table in tables_in_schema:
            col_info = tables_in_schema[agg_table].get("columns", {})
            if isinstance(col_info, dict) and agg_col in col_info:
                col_meta = col_info[agg_col]
                dtype = col_meta.get("dtype", "").lower()
                stype = col_meta.get("semantic_type", "").lower()
                role = col_meta.get("role", "")

                if agg_name in NUMERIC_ONLY_AGGS:
                    if dtype and not any(nt in dtype for nt in ("int", "float", "numeric", "double")):
                        errors.append(f"{agg_name} requires a numeric column, but '{agg_table}.{agg_col}' has type '{dtype}'")

                is_id = stype == "id" or role in ("primary_key", "foreign_key")
                if is_id and not (agg_name == "LIST_DISTINCT" and role == "foreign_key"):
                    errors.append(
                        f"{agg_name} cannot be used on ID column '{agg_table}.{agg_col}'. "
                        f"Use COUNT({agg_table}.*, ...) for counting rows."
                    )

        if agg_name == "LIST_DISTINCT":
            if not re.search(r"RANK\s+TOP\s+\d+", q, re.IGNORECASE):
                errors.append("LIST_DISTINCT requires 'RANK TOP k' (e.g., RANK TOP 10)")

    for_match = re.search(r"FOR\s+(\w+)\.(\w+)\s*(?:=|IN\b)", q, re.IGNORECASE)
    if for_match:
        entity_table, entity_col = for_match.group(1), for_match.group(2)
        if entity_table not in table_names:
            pass  # already reported above
        elif entity_table in tables_in_schema:
            t_info = tables_in_schema[entity_table]
            col_info = t_info.get("columns", {})
            if isinstance(col_info, dict):
                valid_cols = set(col_info.keys())
            elif isinstance(col_info, list):
                valid_cols = set(col_info)
            else:
                valid_cols = set()
            if entity_col not in valid_cols:
                pass  # already reported above

            pk = t_info.get("primary_key")
            if pk and entity_col != pk:
                errors.append(f"FOR clause should use primary key '{pk}' of table '{entity_table}', not '{entity_col}'")

    where_match = re.search(r"\bWHERE\b(.+)$", q, re.IGNORECASE)
    predict_part = q[:where_match.start()] if where_match else q

    pred_time_args = re.findall(r"\(\s*\w+\.(?:\w+|\*)\s*,\s*(-?\d+)\s*,\s*(-?\d+)", predict_part)
    for start, end in pred_time_args:
        s, e = int(start), int(end)
        if s < 0:
            errors.append(f"PREDICT target time window start must be >= 0, got {s}. Negative starts are only for WHERE.")
        if e <= s:
            errors.append(f"Time window end ({e}) must be greater than start ({s})")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"PQL validation failed: {errors}")
    return is_valid, errors
