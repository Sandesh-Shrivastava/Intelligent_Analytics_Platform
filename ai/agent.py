"""
AI Agent
Routes questions between Text-to-SQL and RAG pipeline
based on question type.

- Specific data questions → Text to SQL (exact numbers)
- General business questions → RAG (contextual answers)

Run:
    python ai/agent.py
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
from ai.text_to_sql import generate_sql
from ai.rag_pipeline import get_vector_store, rag_query, generate_insights, build_vector_store
from notebooks.athena_helper import query


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a query router. Classify the user's question into one of two categories:

1. SQL - Questions that need exact data, counts, aggregations, filtering, rankings
   Examples: "how many orders", "top 5 states", "average revenue by segment"

2. RAG - Questions about business insights, explanations, summaries, recommendations
   Examples: "what is our performance", "explain our churn", "how are we doing"

Respond with ONLY one word: SQL or RAG
"""


def route_question(question: str) -> str:
    """Decide whether to use SQL or RAG via LangChain."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=10,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_PROMPT),
        ("user", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    route = chain.invoke({"question": question}).strip().upper()
    return "SQL" if "SQL" in route else "RAG"


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_agent(question: str, collection) -> None:
    """Route and answer a question."""
    print(f"\n❓ Question: {question}")

    # route
    route = route_question(question)
    print(f"🔀 Routing to: {route}")
    print("-" * 50)

    if route == "SQL":
        # text to SQL path
        sql = generate_sql(question)
        print(f"🔍 SQL:\n{sql}\n")
        try:
            df = query(sql)
            print(f"📊 Results ({len(df)} rows):")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"❌ Query error: {e}")
            # fallback to RAG
            print("↩️  Falling back to RAG...")
            answer = rag_query(question, collection)
            print(f"💡 {answer}")
    else:
        # RAG path
        answer = rag_query(question, collection)
        print(f"💡 {answer}")


# ── Main ──────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "How many orders did we get in 2018?",
    "What is our overall business performance?",
    "Which are the top 3 states by revenue?",
    "Why are customers churning?",
    "What is the average delivery time for each state?",
    "How should we improve customer retention?",
]


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Agent — SQL + RAG Router")
    print("  Powered by Llama 3.1 70B on Groq")
    print("=" * 55)

    # initialize RAG knowledge base
    print("\n📚 Initializing knowledge base...")
    try:
        collection = get_vector_store()
        print("  Loaded existing ChromaDB collection")
    except Exception:
        print("  Building new ChromaDB collection...")
        insights   = generate_insights()
        collection = build_vector_store(insights)

    # demo
    print("\n" + "=" * 55)
    print("  Demo Questions")
    print("=" * 55)

    for question in DEMO_QUESTIONS:
        run_agent(question, collection)
        print()

    # interactive
    print("\n" + "=" * 55)
    print("  Interactive Mode — type 'exit' to quit")
    print("=" * 55)

    while True:
        question = input("\n❓ Ask anything: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if question:
            run_agent(question, collection)