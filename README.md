# 🚀 E‑Commerce Revenue & Churn Analytics

End‑to‑end analytics project simulating **10,000+ customers** and **50,000+ transactions** to answer a CEO’s core questions:  
**“Where is our revenue coming from, which customers are at risk, and what should we do next?”**

---

## 🔍 In One Glance

- **Business problem:** Improve revenue, profitability, and customer retention for an e‑commerce store.
- **What this project does:**
  - Builds a complete Python analytics pipeline (data → KPIs → segments → forecast → report)
  - Segments customers (RFM), flags churn risk, and forecasts revenue
  - Presents results in an executive‑ready report and an interactive dashboard  
- **Why it matters:** Shows how a data analyst can convert raw transactions into **clear, monetizable business actions**, not just pretty charts.

---

## 📊 Dashboard Preview
[E-commerce Analytics Dashboard](analytics_output/analytics_dashboard.png)

---

## 🛠️ Tech Stack
Python, Pandas, NumPy, SQL, Matplotlib, Seaborn

---

## 📌 Key KPIs Tracked
- Total Revenue & Month-over-Month Growth
- Year-over-Year Revenue Growth
- Average Order Value (AOV)
- Customer Lifetime Value (CLV)
- Churn Rate & Revenue at Risk
- 90-Day Revenue Forecast

---

## 💼 Business Impact (Simulated)

Using synthetic but realistic data, this project surfaces insights similar to a real e‑commerce business:

- **Revenue growth:** Detects ~**35% year‑over‑year revenue growth** and highlights which categories and products drive it.  
- **Customer economics:** Confirms the classic pattern that the **top 20% of customers generate ≈50% of revenue**, motivating VIP/loyalty focus.  
- **Churn risk & value at risk:** Flags customers inactive for **60+ days** and estimates the **total revenue at risk**, giving a target list for retention campaigns. 
- **Forecasting:** Produces a **90‑day revenue forecast with confidence bands** to support inventory planning and marketing budgets.

All of these are backed by code in `analytics_pipeline.py` and surfaced in `ANALYTICS_REPORT.txt` and the dashboard.

---

## 🧱 Project Structure

```text
ecommerce-analytics/
├── analytics_pipeline.py        
├── dashboard.html                
├── README.md                    
│
├── data/
│   └── raw/
│       └── sample_ecommerce_data.csv   
│
├── analytics_output/            
│   └── analytics_dashboard.png  
│
├── ANALYTICS_REPORT.txt         
└── sql/
    └── analysis_queries.sql
```
## 📊 How to View the Dashboard

To view the dashboard, open `dashboard.html` in your browser after cloning the repo.

## 🧪 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/its-Adarsh2003/ecommerce-analytics.git
   cd ecommerce-analytics
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate    # Windows
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn
4. Run the analytics pipeline:
   ```bash
   python analytics_pipeline.py

## 📬 Contact

- LinkedIn: https://www.linkedin.com/in/adarsh-dubey-81881a2a5
- GitHub: https://github.com/its-Adarsh2003






