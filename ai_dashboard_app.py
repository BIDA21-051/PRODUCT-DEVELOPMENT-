import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# --- DATA LOADING AND CLEANING ---

@st.cache_data
def load_data():
    # Replace with your actual data file
    # Simulate data for demonstration
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='H'),
        'country': np.random.choice(['South Africa', 'Kenya', 'India', 'Germany', 'UK'], size=n),
        'product': np.random.choice(['AI virtual Assistant', 'AI prototyping solution', 'AI-driven customer support'], size=n),
        'traffic_source': np.random.choice(['Paid Ads', 'Organic Search', 'Referral'], size=n),
        'campaign_tag': np.random.choice(['Spring_Sale', 'AI_Webinar', 'None'], size=n),
        'session_duration': np.random.exponential(10, size=n),
        'engagement_score': np.random.uniform(0, 100, size=n),
        'demo_request': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'conversion_status': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        'bounce_rate': np.random.uniform(0, 1, size=n)
    })
    return df

df = load_data()

# --- FEATURE ENGINEERING ---

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# --- MODELING: PREDICTIVE, DESCRIPTIVE, TIME SERIES ---

# Predictive: Conversion prediction (classification)
features = ['session_duration', 'engagement_score', 'demo_request', 'hour', 'is_weekend']
X = df[features]
y = df['conversion_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Bias-variance analysis (simple)
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy
bias = 1 - train_acc
variance = train_acc - test_acc

# Time Series: Demo requests per day
df['date'] = df['timestamp'].dt.date
demo_ts = df.groupby('date')['demo_request'].sum()
model = ARIMA(demo_ts, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=7)

# --- DASHBOARD LAYOUT ---

st.set_page_config(page_title="AI-Solutions Dashboard", layout="wide")
st.title("AI-Solutions Live Dashboard")
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Executive Leadership", "Sales & Marketing", "Product Development & Management"])

# --- EXECUTIVE LEADERSHIP TAB ---
if tab == "Executive Leadership":
    st.header("Executive Leadership Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global Market Penetration", df[df['conversion_status']==1]['country'].nunique())
    with col2:
        st.metric("Overall Conversion Rate", f"{df['conversion_status'].mean()*100:.2f}%")
    with col3:
        st.metric("Year-over-Year Growth (sample)", "N/A (demo data)")

    st.subheader("Conversions by Country")
    fig = px.bar(df[df['conversion_status']==1].groupby('country').size().reset_index(name='Conversions'),
                 x='country', y='Conversions', color='country')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Campaign ROI (Conversions per Campaign)")
    fig2 = px.pie(df[df['conversion_status']==1], names='campaign_tag', title="Conversions by Campaign")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Objective: Expand market reach, optimize campaigns, reduce CAC, double conversions in emerging markets, ensure campaign ROI.")

# --- SALES & MARKETING TAB ---
elif tab == "Sales & Marketing":
    st.header("Sales & Marketing Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lead Generation Rate", df['demo_request'].sum())
    with col2:
        st.metric("Conversion Rate (Paid Ads)", f"{df[df['traffic_source']=='Paid Ads']['conversion_status'].mean()*100:.2f}%")
    with col3:
        st.metric("Avg. Time-to-Conversion", f"{df[df['conversion_status']==1]['session_duration'].mean():.2f} min")

    st.subheader("Demo Requests by Region")
    fig = px.bar(df[df['demo_request']==1].groupby('country').size().reset_index(name='Demo Requests'),
                 x='country', y='Demo Requests', color='country')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Conversions by Traffic Source")
    fig2 = px.bar(df[df['conversion_status']==1].groupby('traffic_source').size().reset_index(name='Conversions'),
                 x='traffic_source', y='Conversions', color='traffic_source')
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Objective: Increase demo requests, improve conversion for Paid Ads, boost underperforming regions, campaign effectiveness, reduce time-to-conversion.")

# --- PRODUCT DEVELOPMENT & MANAGEMENT TAB ---
elif tab == "Product Development & Management":
    st.header("Product Development & Management")
    product_list = df['product'].unique()
    product = st.selectbox("Select Product", product_list)
    prod_df = df[df['product'] == product]

    kpi1 = prod_df['conversion_status'].mean()*100
    kpi2 = prod_df['engagement_score'].mean()
    kpi3 = prod_df['demo_request'].sum() / (prod_df['conversion_status'].sum()+1)
    kpi4 = prod_df['session_duration'].mean()
    kpi5 = prod_df['bounce_rate'].mean()*100

    col1, col2, col3 = st.columns(3)
    col1.metric("Product Conversion Rate", f"{kpi1:.2f}%")
    col2.metric("Avg. Engagement Score", f"{kpi2:.2f}")
    col3.metric("Demo Request-to-Conversion", f"{kpi3:.2f}")

    col4, col5 = st.columns(2)
    col4.metric("Avg. Session Duration", f"{kpi4:.2f} min")
    col5.metric("Bounce Rate", f"{kpi5:.2f}%")

    st.subheader("Session Duration Distribution")
    fig = px.histogram(prod_df, x='session_duration', nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Objective: Increase conversion for prototyping, improve engagement, achieve demo-to-conversion for customer support, reduce bounce rate, prioritize updates by session duration.")

# --- TIME SERIES & PREDICTIVE ANALYTICS ---

st.sidebar.markdown("---")
st.sidebar.header("Advanced Analytics")
if st.sidebar.checkbox("Show Time Series Forecast (Demo Requests)", value=True):
    st.subheader("Demo Requests Forecast (Next 7 Days)")
    st.line_chart(pd.Series(forecast, index=pd.date_range(df['date'].max(), periods=7, freq='D')))

if st.sidebar.checkbox("Show Model Performance", value=True):
    st.subheader("Model Performance Metrics")
    st.write(f"Accuracy: {accuracy:.2f} (Target: 78%-88%)")
    st.write(f"Bias: {bias:.2f}, Variance: {variance:.2f}")
    st.write(pd.DataFrame(report).transpose())

# --- NATURAL LANGUAGE GENERATION (NLG) SUMMARY ---

def nlg_summary(df):
    demo_increase = (df[df['timestamp']>df['timestamp'].max()-pd.Timedelta(days=7)]['demo_request'].sum() /
                     df[df['timestamp']<=df['timestamp'].max()-pd.Timedelta(days=7)]['demo_request'].sum() - 1) * 100
    top_country = df[df['demo_request']==1]['country'].mode()[0]
    return f"Demo requests increased by {demo_increase:.1f}% this week, primarily from {top_country}."

st.sidebar.markdown("---")
st.sidebar.subheader("AI-Generated Insights")
st.sidebar.write(nlg_summary(df))

# --- UX FEATURES ---
st.sidebar.markdown("---")
st.sidebar.toggle("Dark Mode (Streamlit native)", value=False)
st.sidebar.info("Dashboard is mobile-friendly and role-customizable.")

# --- LIVE UPDATES (Auto-refresh) ---
st_autorefresh = st.experimental_rerun if st.sidebar.button("Refresh Data") else None
