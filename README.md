# 🧠 Intelligent Analytics Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-FF694B?style=for-the-badge&logo=dbt&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

🔴 **Live Application:** [https://intelligent-analytics-platform-sc4bohsb5aaqcwqbj6n6oz.streamlit.app/](https://intelligent-analytics-platform-sc4bohsb5aaqcwqbj6n6oz.streamlit.app/)

An end-to-end, FAANG-ready data product built for an e-commerce business. This platform bridges the gap between raw data and actionable business intelligence by combining Data Engineering, Data Analytics, Machine Learning, and Generative AI into a single, cohesive web application.

---

## 🚀 Project Overview

The Intelligent Analytics Platform takes raw e-commerce data (using the Brazilian Olist dataset) and transforms it into a live, interactive dashboard. It allows business stakeholders to view live KPIs, understand machine learning predictions for customer behavior, and even interact with their data using plain English via an integrated LLM.

### ✨ Key Features & Frontend Architecture
1. **Custom Borderless Streamlit UI:** A heavily customized, edge-to-edge responsive dashboard. Streamlit's native sidebar was strategically removed and the layout overridden via advanced CSS to create a seamless, highly immersive HD interface.
2. **Fluid Typography & Glassmorphism:** Implements dynamic viewport width (`vw`) typography, allowing the platform's headers to mathematically scale edge-to-edge across any ultra-wide monitor. Features custom transparent "ghost" buttons and floating, borderless tabs.
3. **Interactive Data Pipeline Visualization:** A visually stunning, horizontal architecture flow diagram (`Fake Data → AWS S3 → dbt → XGBoost → Llama 3.3`) built directly into the main Overview tab UI.
4. **Live Business Operations:** Displays active revenue trends, top category volume, and geographical sales modeling pulling directly from AWS Athena.
5. **Machine Learning Insights:**
   * **Churn Prediction:** An XGBoost classification model predicting which customers are at risk of leaving, explained via SHAP values.
   * **Customer Lifetime Value (CLV):** An XGBoost regression model predicting the future monetary value of customer segments.
   * **Product Recommender:** A collaborative filtering system suggesting untried product categories to existing customers.
6. **Generative AI Assistant ("Ask AI"):**
   * **Text-to-SQL (LangChain LCEL):** Ask hard data questions (e.g., *"How many orders last month?"*) and the Llama 3.3 LLM will generate and execute an AWS Athena SQL query using a declarative LangChain chain.
   * **RAG (Retrieval-Augmented Generation):** Ask business context questions using a custom `SimpleHashEmbeddings` pipeline and ChromaDB to search for summarized insights.

---

## 🏗️ Architecture & Data Workflow

This project follows a modern data stack architecture:

1. **Ingestion (`/ingestion`):** Raw CSV data is uploaded to **AWS S3** and registered as external tables in **AWS Athena** using the `boto3` library.
2. **Transformation (`/dbt_project`):** **dbt (Data Build Tool)** is used to clean, model, and aggregate the raw Athena tables into optimized Data Marts (Fact and Dimension tables).
3. **Machine Learning (`/ml`):** Advanced feature engineering is performed on the dbt Marts to train XGBoost models. Experiments and metrics are tracked locally using **MLflow**.
4. **Artificial Intelligence (`/ai`):** Powered by **LangChain**, utilizing **LCEL (LangChain Expression Language)** for modular AI orchestration. Uses **Llama 3.3 70B** on Groq for ultra-fast LPU inference, featuring a custom `SimpleHashEmbeddings` class for model-free vector retrieval.
5. **Presentation (`/dashboard`):** A premium **Streamlit** dashboard featuring glassmorphism design, cached Athena querying, and interactive Plotly visualizations.

---

## 🛠️ Required Tech Stack

* **Cloud & Data Pipeline:** AWS S3, AWS Athena, dbt-athena-community, boto3, pyathena.
* **Machine Learning:** XGBoost, Scikit-Learn, SHAP, MLflow.
* **Generative AI:** LangChain (LCEL), Groq API (Llama 3.3 70B), ChromaDB (Vector DB).
* **Frontend:** Streamlit (Glassmorphism), Plotly Express.

---

## 💻 Setup & Installation Instructions

If you wish to run this project locally, follow these steps:

### 1. Requirements
Ensure you have Python 3.9+ installed. Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
You must provide your own AWS and Groq API keys. Create a file named `.env` in the root directory (use the provided `.env.example` as a template) and add your keys:
```env
AWS_ACCESS_KEY_ID='your_aws_key'
AWS_SECRET_ACCESS_KEY='your_aws_secret'
AWS_REGION='eu-north-1'
S3_BUCKET_RAW='your_bucket_name'
ATHENA_DATABASE='raw_olist'
ATHENA_RESULTS_BUCKET='s3://your-athena-results-bucket'
GROQ_API_KEY='your_groq_key'
```

### 3. Run the Dashboard Locally
Once your environment is set up and your AWS infrastructure contains the necessary data, you can launch the platform locally:
```bash
streamlit run dashboard/app.py
```
The dashboard will be available in your browser at `http://localhost:8501`.

---

## ☁️ Deployment (AWS EC2)

To deploy this platform on a professional AWS EC2 instance, follow these steps:

### 1. Launch an EC2 Instance
*   Use an **Amazon Linux 2023** or **Ubuntu** AMI.
*   Instance type: `t3.medium` (recommended) or higher.
*   **Security Group**: Ensure port `8501` is open for your IP or `0.0.0.0/0`.

### 2. Setup & Deploy
Once connected to your EC2 via SSH, run:
```bash
# Clone the repository
git clone https://github.com/Sandesh-Shrivastava/Intelligent_Analytics_Platform.git
cd Intelligent_Analytics_Platform

# Create your .env file
cp .env.example .env
nano .env  # Add your AWS and Groq keys

# Run the deployment script
chmod +x scripts/deploy_ec2.sh
./scripts/deploy_ec2.sh
```

The app will be live at `http://your-ec2-public-ip:8501`.

---

## 🔮 Future Enhancements

As the architecture and business requirements evolve, the following features are slated for future development:

1. **Multi-Format Data Upload Engine (Drag & Drop):** 
   A robust, real-time file upload component allowing users to securely drag-and-drop new datasets. Tabular formats (CSVs, Excel) will seamlessly auto-update the dbt transformations and Plotly charts, while unstructured files (PDFs, Word Docs) will be automatically vectorized and embedded into the ChromaDB RAG database for the Llama 3 Agent to analyze.
2. **Expanded Advanced Dataviz Dashboards:** 
   Scaling out the "Business Overview" tab by integrating real-time geospatial heatmaps, granular "Delivery Logistics Performance" visualizations, and strict "Payment Method" breakdowns to create a true 360-degree command center.
3. **Real-Time Data Streaming (Apache Kafka):** 
   Upgrading the current batch-processing architecture by injecting a live event stream. This will enable sub-second latency for transaction logging, meaning the Streamlit dashboard will update its KPIs precisely as orders happen live.
4. **Automated Anomaly Detection & Slack Alerts:** 
   Deploying an AWS Lambda cron job that leverages an Isolation Forest ML model against the live revenue data. If a mathematically abnormal dip in sales (or spike in churn) is detected, the Llama agent will automatically triage the issue and ping a detailed alert directly to the company's Slack/Microsoft Teams channel.
5. **Autonomous Multi-Agent AI System (LangGraph):** 
   Evolving the "Ask AI" router into a multi-agent framework where a designated "Data Engineer Agent" writes and pulls raw SQL, while a secure "Data Scientist Agent" writes sandbox Python code to proactively render completely custom Seaborn/Plotly charts on the fly.
