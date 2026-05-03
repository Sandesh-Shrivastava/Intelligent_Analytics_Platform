"""
Text to SQL
Converts natural language questions to SQL queries,
runs them on Athena, and returns results.

Run:
    python ai/text_to_sql.py
"""

import os
import sys
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from notebooks.athena_helper import query

# ── Schema context for the LLM ────────────────────────────────────────────────

SCHEMA_CONTEXT = """
You are an expert SQL analyst. Convert natural language questions to Athena SQL queries.

Available tables in database 'dbt_dev':

fct_orders:
  - order_id, customer_id, customer_unique_id
  - status (delivered/shipped/canceled/processing/invoiced/approved)
  - item_count, order_value, total_payment
  - purchased_at, delivered_at (timestamps)
  - purchase_year, purchase_month, purchase_dow
  - delivery_days, delivery_delay_days, delivered_on_time (boolean)
  - customer_city, customer_state
  - review_score (1-5), is_positive_review (boolean)
  - primary_payment_type (credit_card/boleto/voucher/debit_card)

dim_customers:
  - customer_unique_id, customer_city, customer_state
  - total_orders, total_items, total_revenue, avg_order_value
  - first_order_at, last_order_at, recency_days
  - avg_review_score, positive_review_rate
  - avg_delivery_days, on_time_deliveries
  - recency_score, frequency_score, monetary_score (1-5)
  - customer_segment (Champion/Loyal/New Customer/At Risk/Lost/Potential)

mart_customer_metrics:
  - All dim_customers columns plus:
  - orders_per_month, active_last_90d, active_last_180d
  - positive_review_rate, on_time_rate
  - is_churned (1=churned, 0=active)

Rules:
- Always use dbt_dev. prefix for tables
- Use single quotes for strings
- For aggregations always include GROUP BY
- Limit results to 20 rows unless asked otherwise
- Return ONLY the SQL query, no explanation, no markdown, no backticks
"""


def generate_sql(question: str) -> str:
    """Convert natural language question to SQL using Llama via LangChain."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=500,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SCHEMA_CONTEXT),
        ("user", "Question: {question}"),
    ])

    # LCEL Chain
    chain = prompt | llm | StrOutputParser()

    sql = chain.invoke({"question": question})

    # clean up any accidental markdown
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def ask(question: str) -> None:
    """Full pipeline: question → SQL → Athena → results."""
    print(f"\n❓ Question: {question}")
    print("-" * 50)

    # generate SQL
    sql = generate_sql(question)
    print(f"🔍 Generated SQL:\n{sql}")
    print("-" * 50)

    # run on Athena
    try:
        df = query(sql)
        print(f"📊 Results ({len(df)} rows):")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"❌ Query failed: {e}")


# ── Demo questions ────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What are the top 5 states by total revenue?",
    "How many orders were delivered on time vs late?",
    "What is the average order value by payment type?",
    "Which customer segment has the highest average review score?",
    "What are the top 3 months by number of orders?",
]


if __name__ == "__main__":
    print("=" * 55)
    print("  Text to SQL — Powered by Llama 3.1 on Groq")
    print("=" * 55)

    # run demo questions
    for question in DEMO_QUESTIONS:
        ask(question)
        print()

    # interactive mode
    print("\n" + "=" * 55)
    print("  Interactive Mode — type your question")
    print("  Type 'exit' to quit")
    print("=" * 55)

    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if question:
            ask(question)