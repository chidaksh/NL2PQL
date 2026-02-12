# PredictiveAgent

An autonomous business intelligence system that converts natural language questions into KumoRFM predictions and delivers actionable strategy reports. Unlike existing KumoRFM cookbooks that hardcode single-task PQL queries, PredictiveAgent discovers schemas, generates PQL autonomously, executes predictions with native explanations, and synthesizes results into a strategy report -- all without human intervention.

## What It Does

Given a business question and a relational dataset (local or S3), PredictiveAgent:

1. **Schema Discovery** -- loads tables, infers primary/foreign keys, time columns, and entity/event classification automatically.
2. **Hypothesis Generation** -- uses RAG retrieval from a PQL knowledge base, the full PQL reference, and static validation with retry to generate 4-6 diverse PQL queries. If the question specifies a time window, hypotheses explore different angles (SUM, AVG, COUNT, MAX) rather than repeating the same metric across time windows.
3. **Prediction Execution** -- runs each PQL query via KumoRFM with `explain=True` for native model explanations. Multi-entity queries are automatically decomposed: bulk predict without explain, then individual explain calls for the top entities. Failed queries are retried once after LLM-based fix. Supports optional `anchor_time` for historical predictions.
4. **Strategy Synthesis** -- pre-computes data statistics (prediction value summaries, model explanation factors, data quality metrics) and feeds them to an LLM to produce a data-driven strategy report with executive summary, key findings, data-backed recommended actions, and risk assessment.

## Installation

```bash
git clone <repo-url>
cd kumo
pip install -r requirements.txt
```

Required Python >= 3.10.

## Setup

Set your API keys via environment variables or a `.env` file in the project root:

```bash
export KUMO_API_KEY="your_kumo_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

Both keys are required. KumoRFM keys are available at https://kumorfm.ai.

## Usage

### CLI

```bash
# Default dataset (Kumo online-shopping)
python main.py --question "Which customers are likely to churn?"

# Custom S3 dataset
python main.py --question "Your question" --data s3://bucket/path --tables table1 table2 table3

# Custom local dataset (auto-discovers .parquet and .csv files)
python main.py --question "Your question" --data /path/to/data/

# Historical predictions with anchor time
python main.py --question "Your question" --anchor-time "2024-09-01"

# Custom output directory (default: auto-timestamped under outputs/)
python main.py --question "Your question" --output outputs/my_run
```

### Streamlit UI

```bash
streamlit run ui/app.py
```

The UI provides API key input, data source selection (default S3, custom S3, or local path), optional anchor time for historical predictions, example questions, live pipeline progress, and a downloadable report. Each run saves outputs to a timestamped folder.

## Running on New Data

To run on your own dataset:

1. **Local data** -- place `.parquet` or `.csv` files in a directory. Each file becomes one table (filename = table name). Use `--data /path/to/dir/`.
2. **S3 data** -- provide the S3 URL and table names: `--data s3://bucket/prefix --tables t1 t2 t3`. Files must be named `{table_name}.parquet`.

The system automatically infers primary keys, foreign keys, time columns, numeric/categorical roles, and entity vs event tables from the data. No configuration files needed.

## Sample data arguments

To quickly run and check the system you can use the sample data arguments:

```bash
python main.py --question "Predict top 10 items user_id=0 is likely to buy in the next 30 days" --data s3://kumo-public-datasets/online-shopping --tables users items orders
```

```bash
python main.py --question "Predict the transaction volume (count of transactions) the merchant 'fraud_Abbott-Rogahn' will process in the next 30 days" --data s3://kumo-public-datasets/credit_card_fraud --tables customers merchants transactions fraud_reports
```

## Project Structure

```
kumo/
├── main.py                       # CLI entry point
├── agents/
│   ├── graph.py                  # LangGraph orchestration (4-node pipeline)
│   ├── schema_discovery.py       # Table loading, PK/FK inference, graph building
│   ├── hypothesis_generator.py   # RAG + PQL reference + LLM + static validation
│   ├── prediction_executor.py    # KumoRFM predict with explain + multi-entity decomposition + LLM retry
│   └── strategy_synthesizer.py   # Data-driven strategy report (pre-computed stats + LLM)
├── tools/
│   ├── llm.py                    # Centralized LLM initialization (gpt-4o-mini)
│   ├── kumo_tools.py             # KumoRFM SDK wrappers
│   ├── pql_validator.py          # Static PQL syntax/semantic validator
│   └── pql_knowledge_base.py     # RAG retrieval over 33 PQL examples
├── ui/
│   └── app.py                    # Streamlit interface
├── pql_reference.txt             # Full PQL language reference
├── pql_knowledge_base.json       # 33 annotated PQL examples for RAG
├── outputs/                      # Pipeline output files
└── requirements.txt
```

## Built With

- **KumoRFM** -- Relational Foundation Model for zero-shot predictions on relational data
- **LangGraph** -- Multi-agent state graph orchestration
- **LangChain + OpenAI** -- LLM calls (gpt-4o-mini)
- **sentence-transformers** -- Embedding model (all-MiniLM-L6-v2) for PQL RAG retrieval
- **Streamlit** -- Web UI

## Assumptions

1. **KumoRFM SDK** -- the `kumoai` package is installed and `KUMO_API_KEY` is valid. The graph is built with explicit `LocalTable` objects and `graph.link()` calls based on LLM-inferred schema.
2. **OpenAI API** -- `gpt-4o-mini` is used for all LLM calls (schema inference, hypothesis generation, query fixing, strategy synthesis). No Anthropic fallback.
3. **Schema inference** -- an LLM analyzes column names, dtypes, uniqueness counts, and sample values to determine primary keys, time columns, and foreign key links. Falls back to `rfm.LocalGraph.from_data()` auto-detection if LLM inference fails. PKs do not need "id" in the name (e.g. `cc_num`, `merchant`, `trans_num` are valid).
4. **Graph links** -- FK relationships are inferred by the LLM and validated: a FK column cannot be the source table's own PK. Links are created via `graph.link(src_table, fkey, dst_table)`.
5. **Time column inference** -- determined by the LLM based on datetime/timestamp dtypes and column semantics.
6. **Event vs entity tables** -- a table is classified as an event table if it has both a time column and at least one foreign key reference. Otherwise it is an entity table.
7. **S3 datasets** -- `--tables` is required for custom S3 paths (no S3 listing). Files are assumed to be `.parquet`. Only the default online-shopping dataset auto-defaults to `[users, items, orders]`.
8. **Local datasets** -- all `.parquet` and `.csv` files in the given directory are loaded. Filename (without extension) becomes the table name.
9. **PQL validation** -- the static validator checks table/column existence, aggregation types, numeric constraints, PK usage in FOR clauses, and time window bounds. It does not validate WHERE clause column values.
10. **explain=True** -- KumoRFM's explain only works for single-entity predictions. For multi-entity queries (`FOR x.id IN (...)`), the system automatically decomposes: runs bulk prediction without explain, then explains the top 3 most interesting entities individually.
11. **Prediction retry** -- if a PQL query fails at execution, the LLM is asked to fix it once based on the error message. If the fix also fails, the prediction is marked as failed.
12. **Confidence score** -- computed as the ratio of successful predictions to total predictions. Not a statistical confidence interval.
13. **Strategy report** -- generated by the LLM based on prediction results and is not independently verified. It should be treated as a starting point for analysis, not a final recommendation.
