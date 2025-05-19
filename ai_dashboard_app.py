import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.arima.model import ARIMA

# ---- SET PAGE CONFIG FIRST! ----
st.set_page_config(page_title="AI-Solutions Dashboard", layout="wide")

# --- DATA LOADING AND CLEANING ---

@st.cache_data
def load_data():
    # Your actual dataset path
    file_path = r"C:\Users\bida21-051\OneDrive - Botswana Accountancy College\Documents\YEAR 4\semester 2\Product Development Material\raw_synthetic_iis_logs.csv"
    df = pd.read_csv(file_path)
    # Basic cleaning (customize as needed)
    df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Parse timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # Feature engineering
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['date'] = df['timestamp'].dt.date
    return df

df = load_data()

# --- FEATURE ENGINEERING (example, adapt as needed) ---
if 'session_duration' in df.columns and 'engagement_score' in df.columns:
    df['engagement_per_minute'] = df['engagement_score'] / (df['session_duration']/60 + 0.001)

# --- MODELING: PREDICTIVE, DESCRIPTIVE, TIME SERIES ---

# Predictive: Conversion prediction (classification)
features = [col for col in ['session_duration', 'engagement_score', 'demo_request', 'hour', 'is_weekend'] if col in df.columns]
if 'conversion_status' in df.columns and all(f in df.columns for f in features):
    X = df[features]
    y = df['conversion_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy
    bias = 1 - train_acc
    variance = train_acc - test_acc
else:
    accuracy = None
    report = None
    bias = None
    variance = None

# Time Series: Demo requests per day
if 'date' in df.columns and 'demo_request' in df.columns:
    demo_ts = df.groupby('date')['demo_request'].sum()
    try:
        model = ARIMA(demo_ts, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
    except Exception:
        forecast = pd.Series([np.nan]*7, index=pd.date_range(df['date'].max(), periods=7, freq='D'))
else:
    forecast = None

# --- DASHBOARD LAYOUT ---

st.title("AI-Solutions Live Dashboard")
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Executive Leadership", "Sales & Marketing", "Product Development & Management"])

# --- EXECUTIVE LEADERSHIP TAB ---
if tab == "Executive Leadership":
    st.header("Executive Leadership Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global Market Penetration", df[df['conversion_status']==1]['country'].nunique() if 'conversion_status' in df.columns and 'country' in df.columns else "N/A")
    with col2:
        st.metric("Overall Conversion Rate", f"{df['conversion_status'].mean()*100:.2f}%" if 'conversion_status' in df.columns else "N/A")
    with col3:
        st.metric("Year-over-Year Growth (sample)", "N/A (demo data)")

    if 'conversion_status' in df.columns and 'country' in df.columns:
        st.subheader("Conversions by Country")
        fig = px.bar(df[df['conversion_status']==1].groupby('country').size().reset_index(name='Conversions'),
                     x='country', y='Conversions', color='country')
        st.plotly_chart(fig, use_container_width=True)

    if 'conversion_status' in df.columns and 'campaign_tag' in df.columns:
        st.subheader("Campaign ROI (Conversions per Campaign)")
        fig2 = px.pie(df[df['conversion_status']==1], names='campaign_tag', title="Conversions by Campaign")
        st.plotly_chart(fig2, use_container_width=True)

    st.info("Objective: Expand market reach, optimize campaigns, reduce CAC, double conversions in emerging markets, ensure campaign ROI.")

# --- SALES & MARKETING TAB ---
elif tab == "Sales & Marketing":
    st.header("Sales & Marketing Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lead Generation Rate", df['demo_request'].sum() if 'demo_request' in df.columns else "N/A")
    with col2:
        if 'traffic_source' in df.columns and 'conversion_status' in df.columns:
            paid_ads_conv = df[df['traffic_source']=='Paid Ads']['conversion_status'].mean()*100
            st.metric("Conversion Rate (Paid Ads)", f"{paid_ads_conv:.2f}%")
        else:
            st.metric("Conversion Rate (Paid Ads)", "N/A")
    with col3:
        if 'conversion_status' in df.columns and 'session_duration' in df.columns:
            avg_time = df[df['conversion_status']==1]['session_duration'].mean()
            st.metric("Avg. Time-to-Conversion", f"{avg_time:.2f} min")
        else:
            st.metric("Avg. Time-to-Conversion", "N/A")

    if 'demo_request' in df.columns and 'country' in df.columns:
        st.subheader("Demo Requests by Region")
        fig = px.bar(df[df['demo_request']==1].groupby('country').size().reset_index(name='Demo Requests'),
                     x='country', y='Demo Requests', color='country')
        st.plotly_chart(fig, use_container_width=True)

    if 'conversion_status' in df.columns and 'traffic_source' in df.columns:
        st.subheader("Conversions by Traffic Source")
        fig2 = px.bar(df[df['conversion_status']==1].groupby('traffic_source').size().reset_index(name='Conversions'),
                     x='traffic_source', y='Conversions', color='traffic_source')
        st.plotly_chart(fig2, use_container_width=True)

    st.info("Objective: Increase demo requests, improve conversion for Paid Ads, boost underperforming regions, campaign effectiveness, reduce time-to-conversion.")

# --- PRODUCT DEVELOPMENT & MANAGEMENT TAB ---
elif tab == "Product Development & Management":
    st.header("Product Development & Management")
    if 'product' in df.columns:
        product_list = df['product'].unique()
        product = st.selectbox("Select Product", product_list)
        prod_df = df[df['product'] == product]
    else:
        prod_df = df.copy()
        product = None

    kpi1 = prod_df['conversion_status'].mean()*100 if 'conversion_status' in prod_df.columns else np.nan
    kpi2 = prod_df['engagement_score'].mean() if 'engagement_score' in prod_df.columns else np.nan
    kpi3 = prod_df['demo_request'].sum() / (prod_df['conversion_status'].sum()+1) if 'demo_request' in prod_df.columns and 'conversion_status' in prod_df.columns else np.nan
    kpi4 = prod_df['session_duration'].mean() if 'session_duration' in prod_df.columns else np.nan
    kpi5 = prod_df['bounce_rate'].mean()*100 if 'bounce_rate' in prod_df.columns else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Product Conversion Rate", f"{kpi1:.2f}%" if not np.isnan(kpi1) else "N/A")
    col2.metric("Avg. Engagement Score", f"{kpi2:.2f}" if not np.isnan(kpi2) else "N/A")
    col3.metric("Demo Request-to-Conversion", f"{kpi3:.2f}" if not np.isnan(kpi3) else "N/A")

    col4, col5 = st.columns(2)
    col4.metric("Avg. Session Duration", f"{kpi4:.2f} min" if not np.isnan(kpi4) else "N/A")
    col5.metric("Bounce Rate", f"{kpi5:.2f}%" if not np.isnan(kpi5) else "N/A")

    if 'session_duration' in prod_df.columns:
        st.subheader("Session Duration Distribution")
        fig = px.histogram(prod_df, x='session_duration', nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    st.info("Objective: Increase conversion for prototyping, improve engagement, achieve demo-to-conversion for customer support, reduce bounce rate, prioritize updates by session duration.")

# --- TIME SERIES & PREDICTIVE ANALYTICS ---

st.sidebar.markdown("---")
st.sidebar.header("Advanced Analytics")
if forecast is not None and st.sidebar.checkbox("Show Time Series Forecast (Demo Requests)", value=True):
    st.subheader("Demo Requests Forecast (Next 7 Days)")
    st.line_chart(pd.Series(forecast, index=pd.date_range(df['date'].max(), periods=7, freq='D')))

if accuracy is not None and st.sidebar.checkbox("Show Model Performance", value=True):
    st.subheader("Model Performance Metrics")
    st.write(f"Accuracy: {accuracy:.2f} (Target: 78%-88%)")
    st.write(f"Bias: {bias:.2f}, Variance: {variance:.2f}")
    st.write(pd.DataFrame(report).transpose())

# --- NATURAL LANGUAGE GENERATION (NLG) SUMMARY ---

def nlg_summary(df):
    if 'timestamp' in df.columns and 'demo_request' in df.columns and 'country' in df.columns:
        recent = df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=7))
        prev = df['timestamp'] <= (df['timestamp'].max() - pd.Timedelta(days=7))
        if prev.sum() > 0:
            demo_increase = (df[recent]['demo_request'].sum() / (df[prev]['demo_request'].sum()+1) - 1) * 100
        else:
            demo_increase = 0
        if df[df['demo_request']==1]['country'].size > 0:
            top_country = df[df['demo_request']==1]['country'].mode()[0]
        else:
            top_country = "N/A"
        return f"Demo requests increased by {demo_increase:.1f}% this week, primarily from {top_country}."
    return "Not enough data for NLG summary."

st.sidebar.markdown("---")
st.sidebar.subheader("AI-Generated Insights")
st.sidebar.write(nlg_summary(df))

# --- UX FEATURES ---
st.sidebar.markdown("---")
st.sidebar.toggle("Dark Mode (Streamlit native)", value=False)
st.sidebar.info("Dashboard is mobile-friendly and role-customizable.")

# --- LIVE UPDATES (Auto-refresh) ---
if st.sidebar.button("Refresh Data"):
    st.experimental_rerun()
