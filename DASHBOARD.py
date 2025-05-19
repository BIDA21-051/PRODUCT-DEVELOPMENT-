import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

# Load data
def load_data():
    local_path = os.path.join(os.path.dirname(__file__), "cleaned_iis_logs.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    url = "1\OneDrive - Botswana Accountancy College\Documents\YEAR 4\semester 2\Product Development Material"
    return pd.read_csv(url)

# AI Model Development
def train_ai_model(data):
    # Predictive Model (Conversion Classification)
    X = data.drop(['conversion_status', 'timestamp', 'ip_address', 'city', 'session_id'], axis=1)
    y = data['conversion_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Classification Model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Anomaly Detection
    iso = IsolationForest(contamination=0.1)
    anomalies = iso.fit_predict(X)
    
    # Regression Model (Session Duration Prediction)
    reg = LinearRegression()
    reg.fit(X_train, X_train['session_duration'])
    
    return clf, iso, reg, accuracy

# Train or load models
try:
    model = joblib.load('ai_model.pkl')
    iso = joblib.load('anomaly_detector.pkl')
    reg = joblib.load('regression_model.pkl')
except:
    model, iso, reg, acc = train_ai_model(df)
    joblib.dump(model, 'ai_model.pkl')
    joblib.dump(iso, 'anomaly_detector.pkl')
    joblib.dump(reg, 'regression_model.pkl')

# Streamlit Dashboard
st.set_page_config(layout="wide")
st.title("AI Solutions Business Dashboard")

# Executive Leadership Section
st.header("Executive Leadership KPIs")
col1, col2, col3 = st.columns(3)
with col1:
    countries = df[[c for c in df.columns if 'country_' in c]].sum().sort_values(ascending=False)
    st.metric("Global Market Penetration", f"{len(countries)} Countries")
    
with col2:
    overall_cr = df.conversion_status.mean() * 100
    st.metric("Overall Conversion Rate", f"{overall_cr:.1f}%")
    
with col3:
    cac = df[df['traffic_source_Paid Ads'] == 1].shape[0] / df.conversion_status.sum()
    st.metric("CAC (Paid Ads)", f"${cac:.2f}")

# Sales & Marketing Section
st.header("Sales & Marketing KPIs")
col1, col2, col3 = st.columns(3)
with col1:
    demo_requests = df.demo_request.sum()
    st.metric("Total Demo Requests", demo_requests)
    
with col2:
    paid_ads_cr = df[df['traffic_source_Paid Ads'] == 1].conversion_status.mean() * 100
    st.metric("Paid Ads CR", f"{paid_ads_cr:.1f}%")
    
with col3:
    region_perf = df[[c for c in df.columns if 'region_' in c]].sum().idxmax().split('_')[1]
    st.metric("Top Performing Region", region_perf)

# Product Development Section
st.header("Product Development KPIs")
col1, col2, col3 = st.columns(3)
with col1:
    product_cr = df[[c for c in df.columns if 'product_' in c]].mean().idxmax().split('_')[1]
    st.metric("Best Converting Product", product_cr)
    
with col2:
    avg_engagement = df.engagement_score.mean()
    st.metric("Average Engagement Score", f"{avg_engagement:.2f}")
    
with col3:
    bounce_rate = df.bounce_rate.mean() * 100
    st.metric("Average Bounce Rate", f"{bounce_rate:.1f}%")

# Visualizations
st.header("Data Visualizations")

# Geographical Performance Map
country_data = df[[c for c in df.columns if 'country_' in c]].sum().reset_index()
country_data.columns = ['Country', 'Conversions']
country_data['Country'] = country_data['Country'].str.replace('country_', '')
fig = px.choropleth(country_data, 
                    locations="Country",
                    locationmode='country names',
                    color="Conversions",
                    title="Geographical Conversion Distribution")
st.plotly_chart(fig, use_container_width=True)

# Time Series Trends
df['timestamp'] = pd.to_datetime(df['timestamp'])
time_series = df.resample('D', on='timestamp').conversion_status.mean()
fig = px.line(time_series, title="Conversion Rate Trend Over Time")
st.plotly_chart(fig, use_container_width=True)

# Traffic Source Analysis
traffic = df[[c for c in df.columns if 'traffic_source_' in c]].sum()
fig = px.pie(traffic, values=traffic.values, names=traffic.index, title="Traffic Source Distribution")
st.plotly_chart(fig, use_container_width=True)

# Anomaly Detection
st.header("Anomaly Detection")
anomalies = iso.predict(df.drop(['conversion_status', 'timestamp', 'ip_address', 'city', 'session_id'], axis=1))
df['anomaly'] = anomalies
st.dataframe(df[df['anomaly'] == -1].head(), use_container_width=True)

# Model Performance
st.header("AI Model Performance")
st.write(f"Current Conversion Prediction Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")

# Real-time Prediction Interface
st.header("Real-time Prediction")
with st.form("prediction_form"):
    session_dur = st.number_input("Session Duration")
    page_views = st.number_input("Page Views")
    submitted = st.form_submit_button("Predict Conversion")
    if submitted:
        sample = pd.DataFrame([[session_dur, page_views]], 
                            columns=['session_duration', 'page_views'])
        prediction = model.predict(sample)
        st.write(f"Conversion Prediction: {'Likely' if prediction[0] else 'Unlikely'}")
