#!/usr/bin/env python3
"""
PredictiveAgent — CLI entry point.

Usage:
    python main.py --question "How do I reduce customer churn?"
    python main.py --question "What products should I promote?" --data ./my_data/
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PredictiveAgent: Autonomous Business Intelligence powered by KumoRFM"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="Business question to analyze",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="s3://kumo-sdk-public/rfm-datasets/online-shopping",
        help="Path to data directory (local or S3). Default: Kumo sample e-commerce data",
    )
    parser.add_argument(
        "--tables", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Table names for S3 datasets (e.g. users items orders)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory or file path for results. Default: auto-generated under outputs/",
    )
    parser.add_argument(
        "--anchor-time",
        type=str,
        default=None,
        help="Historical anchor time for predictions (e.g. 2024-09-01)",
    )
    args = parser.parse_args()

    if not os.environ.get("KUMO_API_KEY"):
        print("KUMO_API_KEY not set.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PredictiveAgent")
    print(f"{'='*60}")
    print(f"Question: {args.question}")
    print(f"Data:     {args.data}")
    if args.anchor_time:
        print(f"Anchor:   {args.anchor_time}")
    print(f"{'='*60}\n")

    from agents.graph import build_agent_graph

    DEFAULT_S3 = "s3://kumo-sdk-public/rfm-datasets/online-shopping"
    table_names = args.tables
    if args.data.startswith("s3://") and table_names is None:
        if args.data.rstrip("/") == DEFAULT_S3.rstrip("/"):
            table_names = ["users", "items", "orders"]
        else:
            print("ERROR: --tables is required for custom S3 paths (e.g. --tables users items orders)")
            sys.exit(1)

    graph = build_agent_graph()
    initial_state = {
        "question": args.question,
        "data_path": args.data,
        "table_names": table_names,
        "anchor_time": args.anchor_time,
        "tables": {},
        "graph_schema": {},
        "graph_built": False,
        "llm_schema": {},
        "hypotheses": [],
        "predictions": [],
        "strategy_report": "",
        "confidence_score": 0.0,
        "errors": [],
        "current_step": "starting",
    }

    result = dict(initial_state)
    step_timings = {}
    step_start = time.time()

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            elapsed = time.time() - step_start
            step_timings[node_name] = elapsed
            step_start = time.time()
            result.update(node_output)

    print(f"\n{'='*60}")
    print(f"STRATEGY REPORT")
    print(f"{'='*60}\n")
    print(result.get("strategy_report", "No report generated."))

    predictions = result.get("predictions", [])
    successful = sum(1 for p in predictions if p.get("success"))
    total_time = sum(step_timings.values())
    sep = '\u2500' * 60
    print(f"\n{sep}")
    print(f"Pipeline Summary:")
    print(f"  Tables discovered: {len(result.get('tables', {}))}")
    print(f"  Hypotheses generated: {len(result.get('hypotheses', []))}")
    print(f"  Predictions: {successful}/{len(predictions)} succeeded")
    print(f"  Confidence: {result.get('confidence_score', 0):.0%}")
    print(f"\nStep Timings:")
    for node_name, elapsed in step_timings.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        print(f"  {node_name}: {elapsed:.1f}s ({pct:.0f}%)")
    print(f"  Total: {total_time:.1f}s")
    if result.get("errors"):
        print(f"\n  Errors: {len(result['errors'])}")
        for e in result["errors"]:
            print(f"    - {e}")
    print(f"{sep}\n")

    slug = re.sub(r'[^a-z0-9]+', '_', args.question.lower())[:40].strip('_')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.output if args.output else os.path.join("outputs", f"{ts}_{slug}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "strategy_report.md"), "w") as f:
        f.write(result.get("strategy_report", ""))
    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(result.get("predictions", []), f, indent=2, default=str)
    with open(os.path.join(out_dir, "hypotheses.json"), "w") as f:
        json.dump(result.get("hypotheses", []), f, indent=2, default=str)
    with open(os.path.join(out_dir, "schema.json"), "w") as f:
        json.dump(result.get("llm_schema", {}), f, indent=2, default=str)
    with open(os.path.join(out_dir, "timings.json"), "w") as f:
        json.dump(step_timings, f, indent=2)
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
