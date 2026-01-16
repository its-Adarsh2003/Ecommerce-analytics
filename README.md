# 🚀 E‑Commerce Revenue & Churn Analytics

End‑to‑end analytics project simulating **10,000+ customers** and **50,000+ transactions** to answer a CEO’s core questions:  
**“Where is our revenue coming from, which customers are at risk, and what should we do next?”**[web:40][web:39]

---

## 🔍 In One Glance

- **Business problem:** Improve revenue, profitability, and customer retention for an e‑commerce store.[web:39][web:40]  
- **What this project does:**
  - Builds a complete Python analytics pipeline (data → KPIs → segments → forecast → report)
  - Segments customers (RFM), flags churn risk, and forecasts revenue
  - Presents results in an executive‑ready report and an interactive dashboard  
- **Why it matters:** Shows how a data analyst can convert raw transactions into **clear, monetizable business actions**, not just pretty charts.[web:36][web:43][web:47]

---

## 💼 Business Impact (Simulated)

Using synthetic but realistic data, this project surfaces insights similar to a real e‑commerce business:[web:40][web:39]

- **Revenue growth:** Detects ~**35% year‑over‑year revenue growth** and highlights which categories and products drive it.  
- **Customer economics:** Confirms the classic pattern that the **top 20% of customers generate ≈50% of revenue**, motivating VIP/loyalty focus.[web:51][web:42]  
- **Churn risk & value at risk:** Flags customers inactive for **60+ days** and estimates the **total revenue at risk**, giving a target list for retention campaigns.[web:51][web:45]  
- **Forecasting:** Produces a **90‑day revenue forecast with confidence bands** to support inventory planning and marketing budgets.[web:40][web:42]

All of these are backed by code in `analytics_pipeline.py` and surfaced in `ANALYTICS_REPORT.txt` and the dashboard.

---

## 🧱 Project Structure

```text
ecommerce-analytics/
├── analytics_pipeline.py        # Main Python analytics pipeline
├── dashboard.html               # Interactive KPI dashboard (Chart.js, static HTML)
├── README.md                    # This documentation
│
├── data/
│   └── raw/
│       └── sample_ecommerce_data.csv   # Example raw data (schema example)
│
├── analytics_output/            # Created at runtime
│   └── analytics_dashboard.png  # Auto‑generated matplotlib dashboard
│
├── ANALYTICS_REPORT.txt         # Auto‑generated executive report
└── sql/
    └── analysis_queries.sql     # Example SQL for trends & RFM‑style analysis
