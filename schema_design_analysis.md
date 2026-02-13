# Schema Design Strategy for PredictiveAgent

## The Core Problem

When the LLM builds a KumoRFM graph, **the order of operations matters enormously.**

### What went wrong with the fraud dataset:

1. We built the graph with natural relationships: `customers ↔ transactions ↔ fraud_reports`
2. We wrote a query: `PREDICT COUNT(fraud_reports.*, 0, 30, days) FOR customers.cc_num=...`
3. **It failed** — PQL requires a **direct link** between the entity (customers) and the target table (fraud_reports)
4. We had to go back, denormalize `cc_num` into `fraud_reports`, and add a new link

This is not a one-off edge case. This will happen **every time** the prediction target lives in a table that's more than 1 hop away from the entity.

---

## The Answer: Query-First Schema Design

**The schema MUST be designed after the query is known, not before.**

### Why Query-First?

PQL has an asymmetry that most people (and LLMs) don't expect:

| Capability | How it works |
|---|---|
| **Feature learning** (inputs) | Model auto-traverses the FULL graph. Multi-hop is automatic. Customer → Transaction → Merchant works for learning. |
| **Target definition** (output) | PQL requires a **direct FK link** from the target table to the entity table. No multi-hop allowed. |

This means:
- The **backbone graph** (for features) can be built naturally from the data
- But the **target link** (for prediction) might need an extra denormalization step

---

## Recommended Architecture for PredictiveAgent

### Two-Phase Schema Design

```
Phase 1: QUERY GENERATION (from user's natural language)
    ↓
Phase 2: SCHEMA CONSTRUCTION (driven by the query)
```

### Phase 1: Generate the PQL Query First

Given the user's question and the raw table/column metadata, the LLM should:

1. Identify the **entity** (WHO are we predicting for?) → determines the `FOR` clause
2. Identify the **target** (WHAT are we predicting?) → determines the `PREDICT` clause
3. Identify the **target table** (WHERE does the target data live?)
4. Check: Is there a **direct FK path** from target table → entity table?

### Phase 2: Build Schema to Support the Query

#### Step 1: Classify every table

```
For each table, assign a role:
- ENTITY TABLE: The table in the FOR clause (e.g., customers)
- TARGET TABLE: The table being aggregated in PREDICT (e.g., fraud_reports)  
- EVENT TABLE: Tables with timestamps that connect entities (e.g., transactions)
- DIMENSION TABLE: Static context tables (e.g., merchants, items)
```

#### Step 2: Ensure the "Target Link" exists

This is the critical check:

```
IF target_table has a direct FK to entity_table:
    → Use it as-is (simple case)
    
ELIF target_table links to entity_table through an intermediate event_table:
    → DENORMALIZE: Copy the entity's PK into the target table
    → ADD LINK: Create direct FK from target_table to entity_table
    → This is NOT a hack — it's the standard pattern
    
ELIF target_table IS the entity_table (e.g., PREDICT customers.age):
    → No link needed — it's an imputation query
    
ELIF target is an aggregation on the event_table itself:
    → Direct link likely already exists (e.g., transactions.cc_num → customers)
    → This is the simplest and most common case
```

#### Step 3: Build the rest of the graph naturally

For all other tables (dimensions, other event tables):
- Link them based on natural FK relationships
- The model will auto-traverse these for feature learning
- **No need to create direct links between every pair of tables**

#### Step 4: Validate the full graph

```
Checks:
□ Every table has a PK (or explicitly None for pure FK tables)
□ No column is both PK and FK on the same table
□ Event tables have time_column set
□ The target link exists: target_table.FK → entity_table.PK
□ FK links point in correct direction (FK table first, PK table second)
□ All referenced columns actually exist in the data
```

---

## Decision Tree for the LLM

```
User asks: "Will this customer's transactions be flagged as fraud?"

Step 1: Parse intent
  → Entity: customers (cc_num)
  → Target: fraud_reports.is_real_fraud
  → Target table: fraud_reports

Step 2: Check link path
  → fraud_reports has trans_num → transactions has cc_num → customers
  → This is a 2-hop path ❌ (PQL needs 1-hop for targets)

Step 3: Fix the schema
  → Add cc_num to fraud_reports_df (via merge with transactions)
  → Set fraud_reports PK to something else (report_id) or None
  → Add link: fraud_reports.cc_num → customers.cc_num

Step 4: Write the query
  → PREDICT COUNT(fraud_reports.* WHERE fraud_reports.is_real_fraud = 1, 0, 30, days) >= 1
     FOR customers.cc_num = 2703186189652095

Step 5: Build graph with all tables + the extra link
```

---

## Common Patterns and Their Schema Requirements

### Pattern 1: Direct event prediction (EASY — no schema changes)
```
Question: "How much will customer X spend in 30 days?"
Entity: customers | Target: transactions.amt | Link: transactions.cc_num → customers.cc_num ✅
Query: PREDICT SUM(transactions.amt, 0, 30, days) FOR customers.cc_num = X
```

### Pattern 2: Indirect target prediction (NEEDS DENORMALIZATION)
```
Question: "Will customer X have a fraud report in 30 days?"
Entity: customers | Target: fraud_reports | Link: NONE DIRECT ❌
Fix: Add cc_num to fraud_reports, link fraud_reports.cc_num → customers.cc_num
Query: PREDICT COUNT(fraud_reports.*, 0, 30, days) >= 1 FOR customers.cc_num = X
```

### Pattern 3: Reverse-direction prediction
```
Question: "How many transactions will merchant Y process?"
Entity: merchants | Target: transactions | Link: transactions.merchant → merchants.merchant ✅
Query: PREDICT COUNT(transactions.*, 0, 30, days) FOR merchants.merchant = Y
```

### Pattern 4: Imputation (NO LINKS NEEDED for target)
```
Question: "What is customer X's likely city_pop?"
Entity: customers | Target: customers.city_pop | Same table ✅
Query: PREDICT customers.city_pop FOR customers.cc_num = X
```

### Pattern 5: Link prediction / recommendation
```
Question: "What merchants will customer X transact with?"
Entity: customers | Target: transactions.merchant | Link: transactions.cc_num → customers.cc_num ✅
Query: PREDICT LIST_DISTINCT(transactions.merchant, 0, 30, days) RANK TOP 5
         FOR customers.cc_num = X
```

---

## LLM Prompt Template for Schema Design

```
You are building a KumoRFM graph schema. Follow these rules strictly:

GIVEN:
- Available tables: {table_names_and_columns}
- User's question: {natural_language_question}
- Generated PQL query: {pql_query}

TASK: Design the graph schema.

RULES:
1. IDENTIFY the entity table (appears in FOR clause) and target table 
   (appears in PREDICT clause).

2. CHECK if the target table has a DIRECT foreign key to the entity table.
   - If YES → proceed normally
   - If NO → you MUST denormalize:
     a. Find the intermediate path (e.g., target → event → entity)
     b. Copy the entity's PK column into the target table
     c. Add a direct link from target table to entity table

3. For EVERY link, the format is:
   [fk_table, fk_column, pk_table, pk_column]
   The FK table is listed FIRST. A column CANNOT be both PK and FK 
   on the same table.

4. If a table only has one meaningful column that's used as FK to another 
   table (and was also being used as PK), either:
   - Set primary_key=None, OR
   - Create a synthetic PK column (e.g., auto-increment ID)

5. Every event/interaction table MUST have time_column set.

6. Dimension tables (static attributes) do NOT need time_column.

7. You do NOT need to link every table to every other table. 
   Only link tables that share a natural FK relationship.
   The model traverses the full graph automatically for feature learning.
   Only the ENTITY ↔ TARGET link must be direct.

OUTPUT FORMAT:
{
  "tables": {
    "table_name": {
      "primary_key": "col_name" or null,
      "time_column": "col_name" or null,
      "role": "entity|target|event|dimension"
    }
  },
  "links": [
    ["fk_table", "fk_col", "pk_table", "pk_col"]
  ],
  "denormalization_needed": true/false,
  "denormalization_steps": [
    "description of what to merge/copy"
  ]
}
```

---

## Key Lessons Summary

1. **Query first, schema second.** Always know what you're predicting before designing the graph.

2. **The model is smart, PQL is strict.** The GNN can learn multi-hop patterns automatically, 
   but PQL target definitions demand a 1-hop direct link.

3. **Denormalization is standard, not a hack.** Copying a FK into a target table to create a 
   direct link is the expected workflow in KumoRFM.

4. **PK/FK conflicts are the #1 gotcha.** Never let a column be both PK and FK on the same table. 
   Use synthetic IDs or set PK=None.

5. **Link direction matters.** Format is always `[fk_table, fk_col, pk_table, pk_col]`. 
   Getting this backwards causes silent schema errors.

6. **Don't over-link.** Only the entity↔target path needs to be direct. 
   All other relationships can be multi-hop — the model handles it.
