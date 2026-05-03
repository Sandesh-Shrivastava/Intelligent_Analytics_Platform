"""
RAG Pipeline
Generates business insights from mart tables,
embeds them into ChromaDB vector store,
and answers questions using retrieval + Llama.

Run:
    python ai/rag_pipeline.py
"""

import os
import sys
from pathlib import Path

import chromadb
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import hashlib
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from notebooks.athena_helper import query

CHROMA_DIR = "ai/chroma_db"


# ── Custom Embeddings ─────────────────────────────────────────────────────────

class SimpleHashEmbeddings(Embeddings):
    """Simple hash-based embedding — no model download needed."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = []
        for i in range(64):
            h = int(hashlib.md5(f"{text}{i}".encode()).hexdigest(), 16)
            vec.append((h % 10000) / 10000.0)
        return vec


# ── Generate insights from Athena ─────────────────────────────────────────────

def generate_insights() -> list[dict]:
    """Pull key metrics from Athena and format as text documents."""
    insights = []

    # 1. Overall KPIs
    df = query("""
        select
            count(distinct order_id)            as total_orders,
            count(distinct customer_unique_id)  as total_customers,
            round(sum(order_value), 2)          as total_revenue,
            round(avg(order_value), 2)          as avg_order_value,
            round(avg(review_score), 2)         as avg_review_score,
            round(sum(case when delivered_on_time then 1 else 0 end) * 100.0
                / count(*), 2)                  as on_time_pct
        from dbt_dev.fct_orders
        where status = 'delivered'
    """)
    r = df.iloc[0]
    insights.append({
        "id": "kpi_overview",
        "text": f"""Overall Business KPIs:
        Total delivered orders: {r['total_orders']}
        Total unique customers: {r['total_customers']}
        Total revenue: R$ {r['total_revenue']}
        Average order value: R$ {r['avg_order_value']}
        Average review score: {r['avg_review_score']} out of 5
        On-time delivery rate: {r['on_time_pct']}%""",
    })

    # 2. Customer segments
    df = query("""
        select
            customer_segment,
            count(*)                        as customers,
            round(avg(total_revenue), 2)    as avg_revenue,
            round(avg(total_orders), 2)     as avg_orders
        from dbt_dev.dim_customers
        group by customer_segment
        order by avg_revenue desc
    """)
    seg_text = "Customer Segments:\n"
    for _, row in df.iterrows():
        seg_text += f"  {row['customer_segment']}: {row['customers']} customers, avg revenue R$ {row['avg_revenue']}, avg {row['avg_orders']} orders\n"
    insights.append({"id": "customer_segments", "text": seg_text})

    # 3. Top categories
    df = query("""
        select
            product_category,
            count(distinct order_id)    as orders,
            round(sum(total_amount), 2) as revenue
        from dbt_dev.int_order_items_enriched
        group by product_category
        order by revenue desc
        limit 10
    """)
    cat_text = "Top 10 Product Categories by Revenue:\n"
    for _, row in df.iterrows():
        cat_text += f"  {row['product_category']}: R$ {row['revenue']} from {row['orders']} orders\n"
    insights.append({"id": "top_categories", "text": cat_text})

    # 4. Payment types
    df = query("""
        select
            primary_payment_type,
            count(*)                        as orders,
            round(avg(order_value), 2)      as avg_order_value
        from dbt_dev.fct_orders
        group by primary_payment_type
        order by orders desc
    """)
    pay_text = "Payment Type Breakdown:\n"
    for _, row in df.iterrows():
        pay_text += f"  {row['primary_payment_type']}: {row['orders']} orders, avg R$ {row['avg_order_value']}\n"
    insights.append({"id": "payment_types", "text": pay_text})

    # 5. State performance
    df = query("""
        select
            customer_state,
            count(distinct order_id)        as orders,
            round(sum(order_value), 2)      as revenue
        from dbt_dev.fct_orders
        where status = 'delivered'
        group by customer_state
        order by revenue desc
        limit 10
    """)
    state_text = "Top 10 States by Revenue:\n"
    for _, row in df.iterrows():
        state_text += f"  {row['customer_state']}: R$ {row['revenue']} from {row['orders']} orders\n"
    insights.append({"id": "state_performance", "text": state_text})

    # 6. Churn stats
    df = query("""
        select
            is_churned,
            count(*)                        as customers,
            round(avg(total_revenue), 2)    as avg_revenue
        from dbt_dev.mart_customer_metrics
        group by is_churned
    """)
    churn_text = "Churn Analysis:\n"
    for _, row in df.iterrows():
        status = "Churned" if row['is_churned'] == 1 else "Active"
        churn_text += f"  {status}: {row['customers']} customers, avg revenue R$ {row['avg_revenue']}\n"
    insights.append({"id": "churn_stats", "text": churn_text})

    print(f"  Generated {len(insights)} insight documents")
    return insights


# ── Build vector store ────────────────────────────────────────────────────────

def build_vector_store(insights: list[dict]) -> Chroma:
    """Store insights in ChromaDB using LangChain and simple embeddings."""
    embeddings = SimpleHashEmbeddings()

    # Clear existing if any
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_texts(
        texts=[i["text"] for i in insights],
        metadatas=[{"id": i["id"]} for i in insights],
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="insights"
    )

    print(f"  Stored {len(insights)} documents in ChromaDB")
    return vectorstore


def get_vector_store() -> Chroma:
    """Load existing ChromaDB collection via LangChain."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=SimpleHashEmbeddings(),
        collection_name="insights"
    )


# ── RAG Query ─────────────────────────────────────────────────────────────────

def rag_query(question: str, vectorstore: Chroma) -> str:
    """Retrieve relevant context and answer using Llama via LangChain."""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=500,
    )

    template = """You are a helpful business analyst assistant.
    Answer questions based on the provided data context.
    Be concise, specific, and use numbers from the context.
    If the answer is not in the context, say so.

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


# ── Main ──────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What is our total revenue and how many customers do we have?",
    "Which customer segment generates the most revenue?",
    "What are our top performing product categories?",
    "How is our on-time delivery performance?",
    "What percentage of customers have churned?",
]


if __name__ == "__main__":
    print("=" * 55)
    print("  RAG Pipeline — Powered by Llama 3.1 on Groq")
    print("=" * 55)

    # build knowledge base
    print("\n📚 Building knowledge base from Athena...")
    insights   = generate_insights()
    collection = build_vector_store(insights)

    # demo questions
    print("\n" + "=" * 55)
    print("  Demo Questions")
    print("=" * 55)

    for question in DEMO_QUESTIONS:
        print(f"\n❓ {question}")
        print("-" * 50)
        answer = rag_query(question, collection)
        print(f"💡 {answer}")

    # interactive mode
    print("\n" + "=" * 55)
    print("  Interactive Mode — type 'exit' to quit")
    print("=" * 55)

    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if question:
            answer = rag_query(question, collection)
            print(f"\n💡 {answer}")