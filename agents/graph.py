"""
LangGraph orchestration for PredictiveAgent.

5-node query-first pipeline:
  Table Inspection → Hypothesis Generation → Graph Building → Prediction Execution → Strategy Synthesis

The graph is built AFTER PQL queries are generated, so that entity-target
links can be created (via denormalization) to satisfy PQL's direct-link requirement.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    # Input
    question: str
    data_path: str
    table_names: list[str] | None
    anchor_time: str | None

    # Table Inspection
    tables: dict[str, Any]
    raw_tables: dict[str, Any]
    llm_schema: dict[str, Any]
    tables_loaded: bool

    # Hypothesis Generator
    hypotheses: list[dict[str, str]]

    # Graph Builder
    graph_schema: dict[str, Any]
    graph_built: bool

    # Prediction Executor
    predictions: Annotated[list[dict[str, Any]], operator.add]

    # Strategy Synthesizer
    strategy_report: str
    confidence_score: float

    # Metadata
    errors: Annotated[list[str], operator.add]
    current_step: str


from agents.schema_discovery import inspect_tables
from agents.hypothesis_generator import generate_hypotheses
from agents.graph_builder import build_query_graph
from agents.prediction_executor import execute_predictions
from agents.strategy_synthesizer import synthesize_strategy


def should_continue_after_inspect(state: AgentState) -> str:
    if state.get("tables_loaded"):
        return "generate_hypotheses"
    return "end_with_error"


def should_continue_after_hypotheses(state: AgentState) -> str:
    if state.get("hypotheses"):
        return "build_query_graph"
    return "end_with_error"


def should_continue_after_graph(state: AgentState) -> str:
    if state.get("graph_built"):
        return "execute_predictions"
    return "end_with_error"


def should_continue_after_predictions(state: AgentState) -> str:
    successful = [p for p in state.get("predictions", []) if p.get("success")]
    if successful:
        return "synthesize_strategy"
    return "end_with_error"


def end_with_error(state: AgentState) -> dict:
    errors = state.get("errors", [])
    return {
        "strategy_report": "## Pipeline Failed\n\nErrors:\n" + "\n".join(f"- {e}" for e in errors),
        "confidence_score": 0.0,
        "current_step": "failed",
    }


def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("inspect_tables", inspect_tables)
    workflow.add_node("generate_hypotheses", generate_hypotheses)
    workflow.add_node("build_query_graph", build_query_graph)
    workflow.add_node("execute_predictions", execute_predictions)
    workflow.add_node("synthesize_strategy", synthesize_strategy)
    workflow.add_node("end_with_error", end_with_error)

    workflow.set_entry_point("inspect_tables")

    workflow.add_conditional_edges(
        "inspect_tables",
        should_continue_after_inspect,
        {"generate_hypotheses": "generate_hypotheses", "end_with_error": "end_with_error"},
    )
    workflow.add_conditional_edges(
        "generate_hypotheses",
        should_continue_after_hypotheses,
        {"build_query_graph": "build_query_graph", "end_with_error": "end_with_error"},
    )
    workflow.add_conditional_edges(
        "build_query_graph",
        should_continue_after_graph,
        {"execute_predictions": "execute_predictions", "end_with_error": "end_with_error"},
    )
    workflow.add_conditional_edges(
        "execute_predictions",
        should_continue_after_predictions,
        {"synthesize_strategy": "synthesize_strategy", "end_with_error": "end_with_error"},
    )
    workflow.add_edge("synthesize_strategy", END)
    workflow.add_edge("end_with_error", END)

    return workflow.compile()


def run_predictive_agent(
    question: str,
    data_path: str = "s3://kumo-sdk-public/rfm-datasets/online-shopping",
    table_names: list[str] | None = None,
    anchor_time: str | None = None,
) -> AgentState:
    """Run the full pipeline and return final state."""
    DEFAULT_S3 = "s3://kumo-sdk-public/rfm-datasets/online-shopping"
    if data_path.startswith("s3://") and table_names is None:
        if data_path.rstrip("/") == DEFAULT_S3.rstrip("/"):
            table_names = ["users", "items", "orders"]
        else:
            raise ValueError("table_names is required for custom S3 paths")

    graph = build_agent_graph()
    initial_state: AgentState = {
        "question": question,
        "data_path": data_path,
        "table_names": table_names,
        "anchor_time": anchor_time,
        "tables": {},
        "raw_tables": {},
        "llm_schema": {},
        "tables_loaded": False,
        "hypotheses": [],
        "graph_schema": {},
        "graph_built": False,
        "predictions": [],
        "strategy_report": "",
        "confidence_score": 0.0,
        "errors": [],
        "current_step": "starting",
    }
    return graph.invoke(initial_state)
