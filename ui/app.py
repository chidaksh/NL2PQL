"""
PredictiveAgent — Streamlit UI.

Run: streamlit run ui/app.py
"""

import json
import os
import re
import sys
import time
import logging
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="PredictiveAgent", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")

    st.text_input("KumoRFM API Key", type="password", key="kumo_api_key")
    st.text_input("OpenAI API Key", type="password", key="openai_api_key")

    st.markdown("---")

    data_source = st.selectbox("Data Source", ["Kumo E-Commerce (default)", "Custom S3 path", "Custom local path"])

    if data_source == "Custom S3 path":
        data_path = st.text_input("S3 URL", value="s3://kumo-sdk-public/rfm-datasets/online-shopping")
        table_names_str = st.text_input("Table names (space-separated)", value="users items orders")
        table_names = table_names_str.split() if table_names_str.strip() else None
    elif data_source == "Custom local path":
        data_path = st.text_input("Path to data directory", value="./data/")
        table_names = None
    else:
        data_path = "s3://kumo-sdk-public/rfm-datasets/online-shopping"
        table_names = ["users", "items", "orders"]

    st.markdown("---")
    anchor_time_str = st.text_input(
        "Anchor time (optional)",
        value="",
        placeholder="e.g. 2024-09-01",
        help="Historical anchor time for predictions. Leave empty to use latest.",
    )
    anchor_time = anchor_time_str.strip() if anchor_time_str.strip() else None

    st.markdown("---")
    st.markdown("Built with [KumoRFM](https://kumo.ai) + [LangGraph](https://github.com/langchain-ai/langgraph)")

st.markdown('<p class="main-header">PredictiveAgent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Autonomous Business Intelligence powered by KumoRFM</p>',
    unsafe_allow_html=True,
)

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

st.markdown("**Try an example: (for e-commerce dataset)**")
example_cols = st.columns(3)
examples = [
    "Predict top 10 items user_id=0 is likely to buy in the next 30 days",
    "Will user_id=1 make a purchase in the next 14 days?",
    "Predict total spending for user_id=2 in the next 60 days",
]

for i, (col, ex) in enumerate(zip(example_cols, examples)):
    if col.button(ex, key=f"example_{i}", use_container_width=True):
        st.session_state["question_input"] = ex
        st.rerun()

question = st.text_area(
    "Your business question:",
    height=80,
    placeholder="e.g., Predict top 10 items user_id=0 is likely to buy in the next 30 days",
    key="question_input",
)

run_col, clear_col = st.columns([4, 1])
run_clicked = run_col.button("Run PredictiveAgent", type="primary", use_container_width=True)
if clear_col.button("Clear Results", use_container_width=True):
    st.session_state.pop("results", None)
    st.session_state.pop("run_history", None)
    st.rerun()

if run_clicked:
    if not question.strip():
        st.error("Please enter a business question.")
    elif not st.session_state.get("kumo_api_key"):
        st.error("Please set your KumoRFM API key in the sidebar.")
    elif not st.session_state.get("openai_api_key"):
        st.error("Please set your OpenAI API key in the sidebar.")
    else:
        st.session_state.pop("results", None)

        os.environ["KUMO_API_KEY"] = st.session_state["kumo_api_key"]
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]

        status_container = st.status("Running pipeline...", expanded=True)
        progress_bar = st.progress(0, text="Starting pipeline...")

        steps = [
            ("Table Inspection", "Inspecting tables and inferring schema..."),
            ("Hypothesis Generation", "Generating prediction hypotheses..."),
            ("Graph Building", "Building query-driven relational graph..."),
            ("Prediction Execution", "Running PQL queries via KumoRFM..."),
            ("Strategy Synthesis", "Producing actionable strategy report..."),
        ]

        from agents.graph import build_agent_graph, AgentState

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

        step_idx = 0
        final_state = None
        step_timings = {}
        step_start = time.time()

        try:
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    elapsed = time.time() - step_start
                    step_timings[node_name] = elapsed
                    step_start = time.time()

                    if node_name == "inspect_tables":
                        step_idx = 0
                        progress_bar.progress(20, text=f"Tables inspected ({elapsed:.1f}s)")
                    elif node_name == "generate_hypotheses":
                        step_idx = 1
                        progress_bar.progress(40, text=f"Hypotheses generated ({elapsed:.1f}s)")
                    elif node_name == "build_query_graph":
                        step_idx = 2
                        progress_bar.progress(60, text=f"Graph built ({elapsed:.1f}s)")
                    elif node_name == "execute_predictions":
                        step_idx = 3
                        progress_bar.progress(80, text=f"Predictions executed ({elapsed:.1f}s)")
                    elif node_name == "synthesize_strategy":
                        step_idx = 4
                        progress_bar.progress(100, text="Strategy complete")

                    if step_idx < len(steps):
                        name, desc = steps[step_idx]
                        status_container.write(f"**{name}** -- Complete ({elapsed:.1f}s)")

                    if final_state is None:
                        final_state = dict(initial_state)
                    final_state.update(node_output)

            status_container.update(label="Pipeline complete", state="complete", expanded=False)

        except Exception as e:
            status_container.update(label=f"Pipeline failed: {e}", state="error", expanded=True)
            st.error(f"Pipeline error: {e}")

        if final_state and final_state.get("strategy_report"):
            run_result = {
                "question": question,
                "tables": {k: {kk: vv for kk, vv in v.items() if kk != "graph_object"}
                           for k, v in final_state.get("tables", {}).items()
                           if isinstance(v, dict)},
                "hypotheses": final_state.get("hypotheses", []),
                "llm_schema": final_state.get("llm_schema", {}),
                "predictions": final_state.get("predictions", []),
                "strategy_report": final_state.get("strategy_report", ""),
                "confidence_score": final_state.get("confidence_score", 0),
                "errors": final_state.get("errors", []),
                "step_timings": step_timings,
                "timestamp": datetime.now().isoformat(),
            }
            st.session_state["results"] = run_result

            if "run_history" not in st.session_state:
                st.session_state["run_history"] = []
            st.session_state["run_history"].append(run_result)

            slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:40].strip('_')
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", f"{ts}_{slug}",
            )
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "strategy_report.md"), "w") as f:
                f.write(run_result["strategy_report"])
            with open(os.path.join(out_dir, "predictions.json"), "w") as f:
                json.dump(run_result["predictions"], f, indent=2, default=str)
            with open(os.path.join(out_dir, "hypotheses.json"), "w") as f:
                json.dump(run_result["hypotheses"], f, indent=2, default=str)
            with open(os.path.join(out_dir, "schema.json"), "w") as f:
                json.dump(run_result.get("llm_schema", {}), f, indent=2, default=str)


def _escape_dollars(text: str) -> str:
    """Escape $ signs so Streamlit doesn't treat them as LaTeX."""
    return text.replace("$", "\\$") if text else text


def _render_prediction_result(result_data):
    """Render a prediction result as a dataframe or formatted text."""
    if isinstance(result_data, dict):
        display_data = {k: v for k, v in result_data.items() if not k.startswith("_")}
        summary = result_data.get("_summary")
        first_val = next(iter(display_data.values()), None) if display_data else None
        if isinstance(first_val, list):
            try:
                df_display = pd.DataFrame(display_data)
                for col in df_display.columns:
                    if df_display[col].dtype == bool or df_display[col].dtype == "boolean":
                        df_display[col] = df_display[col].astype(str)
                st.dataframe(df_display, use_container_width=True)
                if summary:
                    st.markdown(_escape_dollars(summary))
                return
            except Exception:
                pass
        st.code(json.dumps(display_data, indent=2, default=str), language="json")
        if summary:
            st.markdown(_escape_dollars(summary))
    elif isinstance(result_data, str):
        st.text(result_data)
    else:
        st.code(json.dumps(result_data, indent=2, default=str), language="json")


if "results" in st.session_state:
    res = st.session_state["results"]
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    tables_out = res.get("tables", {})
    preds = res.get("predictions", [])
    successful = [p for p in preds if p.get("success")]
    confidence = res.get("confidence_score", 0)

    col1.metric("Tables Analyzed", len(tables_out))
    col2.metric("Hypotheses", len(res.get("hypotheses", [])))
    col3.metric("Predictions", f"{len(successful)}/{len(preds)}")
    col4.metric("Confidence", f"{confidence:.0%}")

    llm_schema = res.get("llm_schema", {})
    if llm_schema:
        with st.expander("LLM Schema Inference", expanded=False):
            for tname, tconf in llm_schema.get("tables", {}).items():
                st.markdown(f"**{tname}** — PK: `{tconf.get('primary_key', 'auto')}`, Time: `{tconf.get('time_column', 'none')}`")
            links = llm_schema.get("links", [])
            if links:
                st.markdown("**Links:**")
                for link in links:
                    st.markdown(f"- `{link['src_table']}.{link['fkey']}` → `{link['dst_table']}`")
            st.code(json.dumps(llm_schema, indent=2), language="json")

    with st.expander("Generated Hypotheses", expanded=False):
        for i, h in enumerate(res.get("hypotheses", [])):
            st.markdown(f"**H{i+1}: {h.get('hypothesis', '')}**")
            st.code(h.get("pql_query", ""), language="sql")
            st.caption(h.get("rationale", ""))

    with st.expander("Prediction Results", expanded=False):
        for i, pred in enumerate(preds):
            label = pred.get("hypothesis", f"Prediction {i+1}")
            query = pred.get("query", "")
            if pred.get("success"):
                st.markdown(f"**[OK] {label}**")
                st.code(query, language="sql")
                _render_prediction_result(pred.get("result", {}))
            else:
                st.markdown(f"**[FAIL] {label}**")
                st.code(query, language="sql")
                retried = pred.get("retried_query")
                if retried:
                    st.caption(f"Retried with: `{retried}`")
                st.error(pred.get("error", "Unknown error"))

    st.markdown("---")
    st.markdown("## Strategy Report")
    report = res.get("strategy_report", "No report generated.")
    st.markdown(_escape_dollars(report))

    st.download_button("Download Report", data=report, file_name="predictive_agent_report.md", mime="text/markdown")

    step_timings = res.get("step_timings", {})
    if step_timings:
        total_time = sum(step_timings.values())
        with st.expander("Step Timings", expanded=False):
            for node_name, elapsed in step_timings.items():
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                st.markdown(f"- **{node_name}**: {elapsed:.1f}s ({pct:.0f}%)")
            st.markdown(f"- **Total**: {total_time:.1f}s")

    errors = res.get("errors", [])
    if errors:
        with st.expander("Warnings and Errors", expanded=False):
            for e in errors:
                st.warning(e)

    history = st.session_state.get("run_history", [])
    if len(history) > 1:
        with st.expander(f"Run History ({len(history)} runs)", expanded=False):
            for i, run in enumerate(reversed(history)):
                preds_r = run.get("predictions", [])
                ok = sum(1 for p in preds_r if p.get("success"))
                st.markdown(
                    f"**Run {len(history)-i}** ({run.get('timestamp', '?')[:19]}): "
                    f"_{run.get('question', '')[:60]}_ -- {ok}/{len(preds_r)} predictions"
                )
