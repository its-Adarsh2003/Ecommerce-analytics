import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# E-COMMERCE SALES ANALYTICS - DATA GENERATION & ANALYSIS
# ============================================================================
# This is a self-contained script that generates realistic e-commerce data
# and performs comprehensive analytics

class EcommerceSalesAnalytics:
    """
    Production-ready e-commerce analytics pipeline
    Handles data generation, processing, and analysis
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.raw_data = None
        self.processed_data = None
        self.insights = {}
        
    # ========================================================================
    # 1. DATA GENERATION (Production-grade synthetic data)
    # ========================================================================
    
    def generate_synthetic_data(self, n_days=450, n_customers=10000, n_products=100):
        """
        Generate realistic e-commerce transactional data
        
        Parameters:
        - n_days: Number of days to simulate (15 months)
        - n_customers: Number of unique customers
        - n_products: Number of unique products
        """
        
        print("🔄 Generating synthetic e-commerce data...")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate transactions
        n_transactions = np.random.randint(n_days*30, n_days*50)
        
        transactions = {
            'transaction_id': range(1, n_transactions + 1),
            'date': np.random.choice(dates, n_transactions),
            'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
            'product_id': np.random.randint(1, n_products + 1, n_transactions),
            'quantity': np.random.randint(1, 5, n_transactions),
            'unit_price': np.random.uniform(10, 500, n_transactions).round(2),
            'discount_percent': np.random.choice(
                [0, 5, 10, 15, 20],
                n_transactions,
                p=[0.6, 0.15, 0.15, 0.07, 0.03]
            ),
            'payment_method': np.random.choice(
                ['Credit Card', 'Debit Card', 'PayPal', 'Wallet'],
                n_transactions
            ),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_transactions)
        }
        
        df_transactions = pd.DataFrame(transactions)
        
        # Calculate revenue metrics
        df_transactions['gross_amount'] = df_transactions['quantity'] * df_transactions['unit_price']
        df_transactions['discount_amount'] = df_transactions['gross_amount'] * (df_transactions['discount_percent'] / 100)
        df_transactions['net_amount'] = df_transactions['gross_amount'] - df_transactions['discount_amount']
        
        # Generate customer master data
        customers = {
            'customer_id': range(1, n_customers + 1),
            'customer_name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
            'signup_date': [
                start_date + timedelta(days=np.random.randint(0, n_days))
                for _ in range(n_customers)
            ],
            'customer_segment': np.random.choice(
                ['Premium', 'Standard', 'Budget'],
                n_customers,
                p=[0.2, 0.5, 0.3]
            ),
            'email_verified': np.random.choice([True, False], n_customers, p=[0.85, 0.15])
        }
        df_customers = pd.DataFrame(customers)
        
        # Generate product master data
        products = {
            'product_id': range(1, n_products + 1),
            'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
            'category': np.random.choice(
                ['Electronics', 'Fashion', 'Home', 'Sports', 'Books'],
                n_products
            ),
            'cost_price': np.random.uniform(5, 300, n_products).round(2),
            'stock_quantity': np.random.randint(10, 1000, n_products)
        }
        df_products = pd.DataFrame(products)
        
        # Merge all data
        df_full = df_transactions.merge(df_customers, on='customer_id', how='left')
        df_full = df_full.merge(df_products, on='product_id', how='left')
        
        self.raw_data = df_full
        print(f"✅ Generated {len(df_full):,} transactions from {n_customers:,} customers")
        return df_full
    
    # ========================================================================
    # 2. DATA CLEANING & TRANSFORMATION
    # ========================================================================
    
    def clean_and_transform(self):
        """
        Clean data and create analytical features
        """
        print("🧹 Cleaning and transforming data...")
        
        df = self.raw_data.copy()
        
        # Date parsing
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        # Remove nulls
        df = df.dropna()
        
        # Outlier handling (cap extreme prices)
        Q1 = df['unit_price'].quantile(0.25)
        Q3 = df['unit_price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df['unit_price'] >= Q1 - 1.5*IQR) &
            (df['unit_price'] <= Q3 + 1.5*IQR)
        ]
        
        # Calculate profit margin
        df['profit'] = df['net_amount'] - (df['quantity'] * df['cost_price'])
        df['profit_margin'] = (df['profit'] / df['net_amount'] * 100).round(2)
        
        # Customer age (days since signup)
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['customer_age_days'] = (df['date'] - df['signup_date']).dt.days
        
        self.processed_data = df
        print(f"✅ Cleaned data: {len(df):,} records, {df.shape[1]} features")
        return df
    
    # ========================================================================
    # 3. SALES ANALYSIS
    # ========================================================================
    
    def analyze_sales_trends(self):
        """
        Comprehensive sales trend analysis
        """
        print("\n📊 SALES ANALYSIS")
        print("=" * 60)
        
        df = self.processed_data
        
        # Key metrics
        total_revenue = df['net_amount'].sum()
        total_profit = df['profit'].sum()
        profit_margin = (total_profit / total_revenue * 100)
        avg_order_value = df.groupby('transaction_id')['net_amount'].sum().mean()
        
        insights = {
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'profit_margin': profit_margin,
            'avg_order_value': avg_order_value,
            'total_transactions': len(df),
            'unique_customers': df['customer_id'].nunique(),
            'unique_products': df['product_id'].nunique(),
            'repeat_purchase_rate': (
                (df.groupby('customer_id').size() > 1).sum()
                / df['customer_id'].nunique() * 100
            )
        }
        
        # Monthly trends
        monthly = df.groupby(df['date'].dt.to_period('M')).agg({
            'net_amount': 'sum',
            'profit': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).round(2)
        
        # Product performance
        top_products = df.groupby('product_name').agg({
            'net_amount': 'sum',
            'transaction_id': 'count',
            'profit': 'sum'
        }).sort_values('net_amount', ascending=False).head(10).round(2)
        
        # Category analysis
        category_sales = df.groupby('category').agg({
            'net_amount': 'sum',
            'profit': 'sum',
            'transaction_id': 'count'
        }).sort_values('net_amount', ascending=False)
        
        # Region analysis
        region_sales = df.groupby('region').agg({
            'net_amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).sort_values('net_amount', ascending=False)
        
        # Print results
        print(f"💰 Total Revenue: ₹{total_revenue:,.2f}")
        print(f"📈 Total Profit: ₹{total_profit:,.2f}")
        print(f"🎯 Profit Margin: {profit_margin:.2f}%")
        print(f"💳 Average Order Value: ₹{avg_order_value:,.2f}")
        print(f"👥 Unique Customers: {insights['unique_customers']:,}")
        print(f"🔄 Repeat Purchase Rate: {insights['repeat_purchase_rate']:.2f}%")
        
        print("\n📅 Top 5 Months by Revenue:")
        print(monthly[['net_amount', 'transaction_id']].tail(5))
        
        print("\n🏆 Top 5 Products by Revenue:")
        print(top_products[['net_amount', 'transaction_id']])
        
        self.insights['sales'] = insights
        self.insights['monthly_trends'] = monthly
        self.insights['top_products'] = top_products
        self.insights['category_sales'] = category_sales
        
        return insights
    
    # ========================================================================
    # 4. CUSTOMER SEGMENTATION (RFM ANALYSIS)
    # ========================================================================
    
    def customer_segmentation(self):
        """
        RFM (Recency, Frequency, Monetary) analysis for customer segmentation
        """
        print("\n👥 CUSTOMER SEGMENTATION (RFM ANALYSIS)")
        print("=" * 60)
        
        df = self.processed_data
        reference_date = df['date'].max() + timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',                         # Frequency
            'net_amount': 'sum'                                # Monetary
        }).round(2)
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Score each dimension (1-4, higher is better)
        rfm['r_score'] = pd.qcut(
            rfm['recency'],
            q=4,
            labels=[4, 3, 2, 1],
            duplicates='drop'
        )
        rfm['f_score'] = pd.qcut(
            rfm['frequency'].rank(method='first'),
            q=4,
            labels=[1, 2, 3, 4],
            duplicates='drop'
        )
        rfm['m_score'] = pd.qcut(
            rfm['monetary'],
            q=4,
            labels=[1, 2, 3, 4],
            duplicates='drop'
        )
        
        # Calculate RFM score
        rfm['rfm_score'] = (
            rfm['r_score'].astype(str)
            + rfm['f_score'].astype(str)
            + rfm['m_score'].astype(str)
        )
        
        # Segmentation logic
        def segment_customer(row):
            if row['r_score'] == 4 and row['f_score'] == 4 and row['m_score'] == 4:
                return 'Champions'
            elif row['r_score'] == 4 and (row['f_score'] == 4 or row['m_score'] == 4):
                return 'Loyal Customers'
            elif row['r_score'] == 4 and row['f_score'] == 1:
                return 'New Customers'
            elif row['r_score'] == 1 and (row['f_score'] == 4 or row['m_score'] == 4):
                return 'At Risk'
            elif row['r_score'] == 1:
                return 'Lost Customers'
            else:
                return 'Average Customers'
        
        rfm['segment'] = rfm.apply(segment_customer, axis=1)
        
        # Summary statistics
        segment_summary = rfm.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'rfm_score': 'count'
        }).round(2)
        segment_summary.columns = [
            'Avg Recency',
            'Avg Frequency',
            'Avg Monetary',
            'Customer Count'
        ]
        
        print("\n📊 RFM Segment Summary:")
        print(segment_summary)
        
        # Customer Lifetime Value
        rfm['clv'] = rfm['monetary'] * rfm['frequency']
        print("\n💎 Customer Lifetime Value (CLV) Insights:")
        print(f"Average CLV: ₹{rfm['clv'].mean():,.2f}")
        print(f"Top 20% CLV: ₹{rfm['clv'].quantile(0.8):,.2f}+")
        
        self.insights['rfm'] = rfm
        self.insights['segments'] = segment_summary
        
        return rfm
    
    # ========================================================================
    # 5. PREDICTIVE ANALYTICS
    # ========================================================================
    
    def forecast_revenue(self):
        """
        Time series forecasting for next 90 days
        """
        print("\n🔮 REVENUE FORECASTING (90-Day Outlook)")
        print("=" * 60)
        
        df = self.processed_data
        
        # Prepare daily revenue
        daily_revenue = df.groupby(df['date'].dt.date)['net_amount'].sum()
        
        # Basic statistics
        mean_revenue = daily_revenue.mean()
        std_revenue = daily_revenue.std()
        
        # Simple exponential smoothing style loop (toy example)
        alpha = 0.3
        forecast = []
        for i in range(90):
            if i == 0:
                pred = daily_revenue.iloc[-1]
            else:
                pred = alpha * daily_revenue.iloc[-(i)] + (1 - alpha) * forecast[-1]
            forecast.append(pred)
        
        ci_lower = [f - 1.96 * std_revenue for f in forecast]
        ci_upper = [f + 1.96 * std_revenue for f in forecast]
        
        total_forecasted_revenue = sum(forecast)
        avg_daily_forecast = np.mean(forecast)
        
        print(f"📈 90-Day Forecasted Revenue: ₹{total_forecasted_revenue:,.2f}")
        print(f"📊 Average Daily Revenue (Forecast): ₹{avg_daily_forecast:,.2f}")
        print(f"📌 Confidence Interval: ±₹{1.96*std_revenue:,.2f}")
        
        self.insights['forecast'] = {
            'revenue_forecast': forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'total_90_day': total_forecasted_revenue
        }
        
        return forecast
    
    # ========================================================================
    # 6. CHURN PREDICTION
    # ========================================================================
    
    def predict_churn(self):
        """
        Identify customers at risk of churning
        """
        print("\n⚠️ CHURN RISK ANALYSIS")
        print("=" * 60)
        
        df = self.processed_data
        reference_date = df['date'].max()
        
        customer_activity = df.groupby('customer_id').agg({
            'date': ['max', 'count'],
            'net_amount': ['sum', 'mean']
        }).round(2)
        
        customer_activity.columns = [
            'last_purchase',
            'purchase_count',
            'total_spent',
            'avg_spent'
        ]
        
        customer_activity['days_since_purchase'] = (
            reference_date - customer_activity['last_purchase']
        ).dt.days
        
        customer_activity['churn_risk'] = customer_activity['days_since_purchase'] >= 60
        
        churn_rate = customer_activity['churn_risk'].sum() / len(customer_activity) * 100
        high_risk_count = customer_activity['churn_risk'].sum()
        
        print(f"🚨 At-Risk Customers (60+ days inactive): {high_risk_count:,} ({churn_rate:.2f}%)")
        print(
            "📊 Average Days Since Purchase (Churned): "
            f"{customer_activity[customer_activity['churn_risk']]['days_since_purchase'].mean():.1f} days"
        )
        print(
            "💰 Total Value at Risk: ₹"
            f"{customer_activity[customer_activity['churn_risk']]['total_spent'].sum():,.2f}"
        )
        
        self.insights['churn'] = {
            'churn_rate': churn_rate,
            'at_risk_customers': high_risk_count,
            'customer_activity': customer_activity
        }
        
        return customer_activity
    
    # ========================================================================
    # 7. GENERATE VISUALIZATIONS
    # ========================================================================
    
    def create_visualizations(self, save_path='./analytics_output/'):
        """
        Create professional visualizations
        """
        print("\n📉 Creating visualizations...")
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        df = self.processed_data
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('E-Commerce Sales Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Daily Revenue Trend
        daily_rev = df.groupby(df['date'].dt.date)['net_amount'].sum()
        axes[0, 0].plot(daily_rev.index, daily_rev.values, color='#2ecc71', linewidth=2)
        axes[0, 0].fill_between(daily_rev.index, daily_rev.values, alpha=0.3, color='#2ecc71')
        axes[0, 0].set_title('Daily Revenue Trend', fontweight='bold')
        axes[0, 0].set_ylabel('Revenue (₹)')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Top 10 Products
        top_10 = df.groupby('product_name')['net_amount'].sum().nlargest(10)
        axes[0, 1].barh(range(len(top_10)), top_10.values, color='#3498db')
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels(top_10.index, fontsize=9)
        axes[0, 1].set_title('Top 10 Products by Revenue', fontweight='bold')
        axes[0, 1].set_xlabel('Revenue (₹)')
        
        # 3. Sales by Category
        cat_sales = df.groupby('category')['net_amount'].sum().sort_values(ascending=False)
        axes[1, 0].bar(
            cat_sales.index,
            cat_sales.values,
            color=['#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
        )
        axes[1, 0].set_title('Sales by Category', fontweight='bold')
        axes[1, 0].set_ylabel('Revenue (₹)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Profit Margin Distribution
        axes[1, 1].hist(
            df['profit_margin'],
            bins=30,
            color='#16a085',
            edgecolor='black',
            alpha=0.7
        )
        axes[1, 1].axvline(
            df['profit_margin'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Mean: {df['profit_margin'].mean():.1f}%"
        )
        axes[1, 1].set_title('Profit Margin Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Profit Margin (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}analytics_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}analytics_dashboard.png")
        
        return fig
    
    # ========================================================================
    # 8. GENERATE EXECUTIVE REPORT
    # ========================================================================
    
    def generate_report(self, output_path='./ANALYTICS_REPORT.txt'):
        """
        Generate comprehensive analytics report
        """
        print("\n📄 Generating comprehensive analytics report...")
        
        report = []
        report.append("=" * 80)
        report.append("E-COMMERCE SALES ANALYTICS - EXECUTIVE REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            "Analysis Period: "
            f"{self.processed_data['date'].min().date()} "
            f"to {self.processed_data['date'].max().date()}"
        )
        
        # Key Metrics
        sales = self.insights.get('sales', {})
        report.append("\n" + "=" * 80)
        report.append("KEY BUSINESS METRICS")
        report.append("=" * 80)
        report.append(f"Total Revenue: ₹{sales.get('total_revenue', 0):,.2f}")
        report.append(f"Total Profit: ₹{sales.get('total_profit', 0):,.2f}")
        report.append(f"Profit Margin: {sales.get('profit_margin', 0):.2f}%")
        report.append(f"Average Order Value: ₹{sales.get('avg_order_value', 0):,.2f}")
        report.append(f"Total Transactions: {sales.get('total_transactions', 0):,}")
        report.append(f"Unique Customers: {sales.get('unique_customers', 0):,}")
        report.append(
            "Repeat Purchase Rate: "
            f"{sales.get('repeat_purchase_rate', 0):.2f}%"
        )
        
        # Churn Insights
        churn = self.insights.get('churn', {})
        report.append("\n" + "=" * 80)
        report.append("CUSTOMER RETENTION INSIGHTS")
        report.append("=" * 80)
        report.append(f"Churn Rate: {churn.get('churn_rate', 0):.2f}%")
        report.append(f"At-Risk Customers: {churn.get('at_risk_customers', 0):,}")
        
        # Forecast
        forecast_data = self.insights.get('forecast', {})
        report.append("\n" + "=" * 80)
        report.append("90-DAY REVENUE FORECAST")
        report.append("=" * 80)
        report.append(f"Projected Revenue: ₹{forecast_data.get('total_90_day', 0):,.2f}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved to {output_path}")
        return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 E-COMMERCE SALES ANALYTICS PIPELINE")
    print("=" * 80)
    
    analytics = EcommerceSalesAnalytics(random_seed=42)
    
    analytics.generate_synthetic_data(n_days=450, n_customers=10000, n_products=100)
    analytics.clean_and_transform()
    analytics.analyze_sales_trends()
    analytics.customer_segmentation()
    analytics.forecast_revenue()
    analytics.predict_churn()
    analytics.create_visualizations()
    analytics.generate_report()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📊 Outputs generated:")
    print("   ✓ Data processed and cleaned")
    print("   ✓ Sales analysis complete")
    print("   ✓ Customer segmentation (RFM)")
    print("   ✓ Revenue forecasts (90-day)")
    print("   ✓ Churn prediction results")
    print("   ✓ Professional visualizations")
    print("   ✓ Executive report")
    print("\n💼 Ready to showcase on Contra and GitHub!")
