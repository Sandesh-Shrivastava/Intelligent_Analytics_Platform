"""
Intelligent Analytics Platform — Streamlit Dashboard
Run:
    streamlit run dashboard/app.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from notebooks.athena_helper import query
from dashboard.theme import get_theme
from ai.text_to_sql import generate_sql
from ai.rag_pipeline import (
    generate_insights, build_vector_store,
    get_vector_store, rag_query
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Intelligent Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dynamic Theme & CSS ───────────────────────────────────────────────────────

if "is_dark_mode" not in st.session_state:
    st.session_state.is_dark_mode = True


CSS, CHART_LAYOUT, CHART_LAYOUT_ROTATED, TEXT_COLOR = get_theme(st.session_state.is_dark_mode)
st.markdown(CSS, unsafe_allow_html=True)



# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_kpis():
    return query("""
        select
            count(distinct order_id)            as total_orders,
            count(distinct customer_unique_id)  as total_customers,
            round(sum(order_value), 2)          as total_revenue,
            round(avg(order_value), 2)          as avg_order_value,
            round(avg(cast(review_score as double)), 2) as avg_review_score,
            round(sum(case when delivered_on_time = true then 1.0 else 0.0 end)
                * 100.0 / count(*), 2)          as on_time_pct
        from dbt_dev.fct_orders
        where status = 'delivered'
    """)

@st.cache_data(ttl=300)
def load_revenue_trend():
    return query("""
        select
            cast(purchase_year as varchar) || '-' ||
            lpad(cast(purchase_month as varchar), 2, '0') as period,
            round(sum(order_value), 2)  as revenue,
            count(*)                    as orders
        from dbt_dev.fct_orders
        where status = 'delivered'
        group by purchase_year, purchase_month
        order by purchase_year, purchase_month
    """)

@st.cache_data(ttl=300)
def load_segments():
    return query("""
        select
            customer_segment,
            count(*)                        as customers,
            round(avg(total_revenue), 2)    as avg_revenue,
            round(avg(total_orders), 2)     as avg_orders,
            round(avg(avg_review_score), 2) as avg_score
        from dbt_dev.dim_customers
        group by customer_segment
        order by avg_revenue desc
    """)

@st.cache_data(ttl=300)
def load_categories():
    return query("""
        select
            product_category,
            count(distinct order_id)        as orders,
            round(sum(total_amount), 2)     as revenue
        from dbt_dev.int_order_items_enriched
        group by product_category
        order by revenue desc
        limit 15
    """)

@st.cache_data(ttl=300)
def load_state_map():
    return query("""
        select
            customer_state,
            count(distinct order_id)        as orders,
            round(sum(order_value), 2)      as revenue
        from dbt_dev.fct_orders
        where status = 'delivered'
        group by customer_state
        order by revenue desc
    """)

@st.cache_data(ttl=300)
def load_churn_data():
    return query("""
        select
            is_churned,
            count(*)                        as customers,
            round(avg(total_revenue), 2)    as avg_revenue,
            round(avg(total_orders), 2)     as avg_orders,
            round(avg(recency_days), 0)     as avg_recency
        from dbt_dev.mart_customer_metrics
        group by is_churned
    """)

@st.cache_data(ttl=300)
def load_clv_data():
    return query("""
        select
            customer_segment,
            round(avg(total_revenue), 2)    as avg_clv,
            round(min(total_revenue), 2)    as min_clv,
            round(max(total_revenue), 2)    as max_clv,
            count(*)                        as customers
        from dbt_dev.mart_customer_metrics
        group by customer_segment
        order by avg_clv desc
    """)

@st.cache_data(ttl=300)
def load_recommender_data():
    return query("""
        select
            o.customer_unique_id,
            i.product_category,
            count(*) as purchases
        from dbt_dev.int_order_items_enriched i
        join dbt_dev.fct_orders o on i.order_id = o.order_id
        where o.status = 'delivered'
          and i.product_category is not null
        group by o.customer_unique_id, i.product_category
    """)

@st.cache_resource
def load_rag_collection():
    try:
        return get_vector_store()
    except Exception:
        insights = generate_insights()
        return build_vector_store(insights)





# ── Title ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 30px; margin-top: 0px; border-bottom: 1px solid var(--border-color); padding-bottom: 25px; width: 100%;'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <div style='font-size: 4rem; line-height: 1;'>📊</div>
        <div style='font-size: 4rem; font-weight: 900; letter-spacing: -0.02em; word-spacing: 0.35em; line-height: 1.0; 
                    background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(165,180,252,0.9) 40%, rgba(56,189,248,0.8) 100%); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-transform: uppercase; 
                    text-shadow: 0px 4px 30px rgba(99,102,241,0.25); white-space: nowrap;'>
            INTELLIGENT ANALYTICS PLATFORM
        </div>
    </div>
    <div style='font-size: 1.25rem; color: #64748b; margin-top: 10px; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase;'>
        Brazilian E-Commerce · 2017–2018
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🤖  Ask AI",
    "🎯  ML Insights",
    "🛒  Buying History",
])


# ══════════════════════════════════════════════════════
# TAB 1 — Business Overview
# ══════════════════════════════════════════════════════

with tab1:
    # Hero banner
    st.markdown("""
    <div style="background-color: var(--background-color); border-radius: 12px; padding: 20px; border: 1px solid var(--border-color); margin-bottom: 25px; overflow-x: auto; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <div style="font-size: 0.75rem; font-weight: 700; color: #6366f1; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 15px;">End-to-End Data Pipeline Architecture</div>
        <div style="display: flex; justify-content: space-between; align-items: center; min-width: 800px; padding: 10px 0;">
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">🐍</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">Fake Data Script</div></div>
            <div style="color: #94a3b8; font-size: 1.2rem; font-weight: bold;">→</div>
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">☁️</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">AWS S3 & Athena</div></div>
            <div style="color: #94a3b8; font-size: 1.2rem; font-weight: bold;">→</div>
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">🔄</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">dbt Transforms</div></div>
            <div style="color: #94a3b8; font-size: 1.2rem; font-weight: bold;">→</div>
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">🤖</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">XGBoost Models</div></div>
            <div style="color: #94a3b8; font-size: 1.2rem; font-weight: bold;">→</div>
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">🧠</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">Llama AI Agent</div></div>
            <div style="color: #94a3b8; font-size: 1.2rem; font-weight: bold;">→</div>
            <div style="text-align: center;"><div style="font-size: 2rem; margin-bottom: 5px;">📊</div><div style="font-size: 0.85rem; font-weight: 700; color: var(--text-color);">Live Dashboard</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    with st.spinner(""):
        kpis = load_kpis()

    if not kpis.empty:
        r = kpis.iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Orders",      f"{int(r['total_orders']):,}")
        c2.metric("Unique Customers",  f"{int(r['total_customers']):,}")
        c3.metric("Total Revenue",     f"$ {float(r['total_revenue']):,.0f}")
        c4.metric("Avg Order Value",   f"$ {float(r['avg_order_value']):.2f}")
        c5.metric("Avg Review Score",  f"⭐ {float(r['avg_review_score']):.2f}")
        c6.metric("On-Time Delivery",  f"{float(r['on_time_pct']):.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend + Category breakdown
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### 📈 Revenue Trend")
        trend = load_revenue_trend()
        if not trend.empty:
            trend["revenue"] = pd.to_numeric(trend["revenue"])
            fig = px.area(
                trend, x="period", y="revenue",
                labels={"period": "", "revenue": "Revenue ($)"},
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_traces(
                line=dict(width=2.5),
                fillcolor="rgba(99,102,241,0.12)",
            )
            fig.update_layout(**CHART_LAYOUT_ROTATED)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 🏆 Top Categories")
        cats = load_categories()
        if not cats.empty:
            cats["revenue"] = pd.to_numeric(cats["revenue"])
            fig = px.bar(
                cats.head(10), x="revenue", y="product_category",
                orientation="h",
                color="revenue",
                color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc"],
                labels={"revenue": "Revenue ($)", "product_category": ""},
            )
            fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Segments + State map
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👥 Customer Segments")
        segs = load_segments()
        if not segs.empty:
            segs["avg_revenue"] = pd.to_numeric(segs["avg_revenue"])
            fig = px.bar(
                segs, x="customer_segment", y="customers",
                color="avg_revenue",
                color_continuous_scale=["#1e1b4b", "#6366f1", "#c7d2fe"],
                labels={"customer_segment": "", "customers": "Customers", "avg_revenue": "Avg Revenue ($)"},
                text="customers",
            )
            fig.update_traces(textfont=dict(color=TEXT_COLOR), textposition="outside")
            fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 🗺️ Revenue by State")
        states = load_state_map()
        if not states.empty:
            states["revenue"] = pd.to_numeric(states["revenue"])
            fig = px.bar(
                states.head(10), x="customer_state", y="revenue",
                color="revenue",
                color_continuous_scale=["#083344", "#06b6d4", "#a5f3fc"],
                labels={"customer_state": "", "revenue": "Revenue ($)"},
                text=states.head(10)["revenue"].apply(lambda x: f"${x:,.0f}"),
            )
            fig.update_traces(textfont=dict(color=TEXT_COLOR, size=10), textposition="outside")
            fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 2 — Ask AI
# ══════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🤖 Ask AI</p>
        <p class="hero-subtitle">Ask anything in plain English — powered by Llama 3.3 70B on Groq</p>
        <span class="hero-pill">🧠 Llama 3.3 70B</span>
        <span class="hero-pill">🔍 Text-to-SQL</span>
        <span class="hero-pill">📚 RAG</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["💬 Natural Language (RAG)", "🔍 Generate SQL"],
        horizontal=True,
    )

    question = st.text_input(
        "",
        placeholder="e.g.  What are the top 5 states by revenue?",
        label_visibility="collapsed",
    )

    example_questions = [
        "What are the top 5 states by total revenue?",
        "How many orders were delivered on time vs late?",
        "Which customer segment has the highest review score?",
        "What is our churn rate?",
        "What is the average order value by payment type?",
    ]

    st.markdown("<div style='font-size:0.78rem; color:#475569; margin-bottom:8px;'>💡 Try an example:</div>", unsafe_allow_html=True)
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq[:28] + "…", key=f"eq_{i}", use_container_width=True):
            question = eq

    if question:
        if "RAG" in mode:
            with st.spinner("Thinking…"):
                collection = load_rag_collection()
                answer = rag_query(question, collection)
            st.success("💡  " + answer)
        else:
            with st.spinner("Generating SQL…"):
                sql = generate_sql(question)
            st.code(sql, language="sql")

            with st.spinner("Running query…"):
                try:
                    df = query(sql)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    if len(df) > 1 and df.select_dtypes(include="number").shape[1] > 0:
                        num_col = df.select_dtypes(include="number").columns[0]
                        cat_col = df.columns[0]
                        fig = px.bar(
                            df.head(15), x=cat_col, y=num_col,
                            color_discrete_sequence=["#6366f1"],
                        )
                        fig.update_layout(**CHART_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Query error: {e}")


# ══════════════════════════════════════════════════════
# TAB 3 — ML Insights
# ══════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🎯 ML Insights</p>
        <p class="hero-subtitle">Churn prediction and Customer Lifetime Value — powered by XGBoost</p>
        <span class="hero-pill">🤖 XGBoost</span>
        <span class="hero-pill">📊 SHAP Explainability</span>
        <span class="hero-pill">🔬 MLflow Tracking</span>
    </div>
    """, unsafe_allow_html=True)

    # Model score summary
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Churn Model AUC",  "~85%", "XGBoost + SHAP")
    mc2.metric("CLV Model R²",     "92.5%", "XGBoost Regression")
    mc3.metric("Recommender",      "20 Categories", "Cosine Similarity")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔴 Churn Analysis")
        churn = load_churn_data()
        if not churn.empty:
            churn["label"] = churn["is_churned"].apply(
                lambda x: "At Risk (1 order)" if str(x) == "1" else "Active (2+ orders)"
            )
            churn["avg_revenue"] = pd.to_numeric(churn["avg_revenue"])
            churn["customers"]   = pd.to_numeric(churn["customers"])

            fig = px.pie(
                churn, values="customers", names="label",
                color_discrete_sequence=["#ef4444", "#22c55e"],
                hole=0.55,
            )
            fig.update_traces(
                textfont=dict(color=TEXT_COLOR, size=13),
                marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
            )
            fig.update_layout(
                **CHART_LAYOUT,
                annotations=[dict(
                    text="Churn<br>Rate",
                    x=0.5, y=0.5,
                    font=dict(size=14, color="#94a3b8", family="Inter"),
                    showarrow=False,
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                churn[["label", "customers", "avg_revenue", "avg_orders", "avg_recency"]].rename(columns={
                    "label": "Segment", "customers": "Customers",
                    "avg_revenue": "Avg Revenue ($)", "avg_orders": "Avg Orders",
                    "avg_recency": "Avg Recency (days)"
                }),
                use_container_width=True, hide_index=True,
            )

    with col2:
        st.markdown("##### 💰 Customer Lifetime Value by Segment")
        clv = load_clv_data()
        if not clv.empty:
            clv["avg_clv"] = pd.to_numeric(clv["avg_clv"])

            fig = px.bar(
                clv, x="customer_segment", y="avg_clv",
                color="avg_clv",
                color_continuous_scale=["#064e3b", "#10b981", "#6ee7b7"],
                labels={"customer_segment": "", "avg_clv": "Avg CLV ($)"},
                text=clv["avg_clv"].apply(lambda x: f"${x:.0f}"),
            )
            fig.update_traces(textfont=dict(color=TEXT_COLOR), textposition="outside")
            fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                clv[["customer_segment", "customers", "avg_clv", "min_clv", "max_clv"]].rename(columns={
                    "customer_segment": "Segment", "customers": "Customers",
                    "avg_clv": "Avg CLV ($)", "min_clv": "Min ($)", "max_clv": "Max ($)"
                }),
                use_container_width=True, hide_index=True,
            )


# ══════════════════════════════════════════════════════
# TAB 4 — Recommender
# ══════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🛍️ Product Recommender</p>
        <p class="hero-subtitle">Collaborative filtering — find similar customers and recommend untried categories</p>
        <span class="hero-pill">🔁 Collaborative Filtering</span>
        <span class="hero-pill">📐 Cosine Similarity</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading purchase data…"):
        rec_df = load_recommender_data()

    if not rec_df.empty:
        rec_df["purchases"] = pd.to_numeric(rec_df["purchases"])

        matrix = rec_df.pivot_table(
            index="customer_unique_id",
            columns="product_category",
            values="purchases",
            fill_value=0,
        )

        customers = matrix.index.tolist()
        selected  = st.selectbox(
            "🔍 Select a Customer ID",
            customers,
            format_func=lambda x: f"Customer: {x[:24]}…"
        )

        if selected:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### ✅ Already Purchased")
                bought = matrix.columns[matrix.loc[selected] > 0].tolist()
                for cat in bought:
                    st.markdown(f"""
                    <div class="rec-card">
                        <span style='color:#22c55e; font-size:1rem;'>✓</span>
                        <span style='color:{TEXT_COLOR}; font-size:0.88rem; font-weight:500;'>{cat.replace("_"," ").title()}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("##### 🎯 Recommended For You")
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                sim     = cosine_similarity(matrix)
                sim_df  = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
                similar = sim_df[selected].drop(selected).sort_values(ascending=False).head(10)

                already_bought = set(matrix.columns[matrix.loc[selected] > 0])
                weighted = matrix.loc[similar.index].multiply(similar.values, axis=0).sum()
                recs = (
                    weighted
                    .drop(list(already_bought), errors="ignore")
                    .sort_values(ascending=False)
                    .head(5)
                )

                for rank, (cat, score) in enumerate(recs.items(), 1):
                    medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
                    st.markdown(f"""
                    <div class="rec-card">
                        <span style='font-size:1.1rem;'>{medal}</span>
                        <div style='flex:1;'>
                            <div style='color:{TEXT_COLOR}; font-size:0.88rem; font-weight:600;'>{cat.replace("_"," ").title()}</div>
                            <div style='color:#475569; font-size:0.75rem;'>Similarity score: {score:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()
        st.markdown("##### 📊 Category Popularity")
        cat_pop = rec_df.groupby("product_category")["purchases"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(
            cat_pop, x="product_category", y="purchases",
            color="purchases",
            color_continuous_scale=["#431407", "#f97316", "#fed7aa"],
            labels={"product_category": "", "purchases": "Total Purchases"},
        )
        fig.update_layout(**CHART_LAYOUT_ROTATED)
        st.plotly_chart(fig, use_container_width=True)