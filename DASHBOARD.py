import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import joblib

# --------------------------------
# 1) DATA UPLOAD & VALIDATION
# --------------------------------
st.header("📥 Load your IIS logs CSV")

# The user must upload the CSV each time (unless you host it publicly later)
uploaded_file = st.file_uploader(
    label="Choose raw_synthetic_iis_logs.csv", 
    type="csv",
    help="Drag & drop your raw_synthetic_iis_logs.csv file here."
)

if uploaded_file is None:
    st.warning("Please upload your `raw_synthetic_iis_logs.csv` to proceed.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error parsing CSV: {e}")
    st.stop()

if df.empty:
    st.error("❌ Uploaded file is empty. Please check your CSV.")
    st.stop()

st.success(f"✅ Loaded {len(df)} rows from `{uploaded_file.name}`")

# --------------------------------
# 2) AI MODEL DEVELOPMENT
# --------------------------------
def train_ai_model(data):
    X = data.drop(['conversion_status', 'timestamp', 'ip_address', 'city', 'session_id'], axis=1)
    y = data['conversion_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classification
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Anomaly detection
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)

    # Simple regression on session_duration
    reg = LinearRegression()
    reg.fit(X_train[['session_duration']], y_train)

    return clf, iso, reg, acc

# --------------------------------
# 3) TRAIN OR LOAD MODELS
# --------------------------------
try:
    model = joblib.load('ai_model.pkl')
    iso   = joblib.load('anomaly_detector.pkl')
    reg   = joblib.load('regression_model.pkl')
    acc   = None
except Exception:
    model, iso, reg, acc = train_ai_model(df)
    joblib.dump(model, 'ai_model.pkl')
    joblib.dump(iso,   'anomaly_detector.pkl')
    joblib.dump(reg,   'regression_model.pkl')

# --------------------------------
# 4) BUILD THE DASHBOARD
# --------------------------------
st.set_page_config(layout="wide")
st.title("AI Solutions Business Dashboard")

# Executive KPIs
st.header("Executive Leadership KPIs")
c1, c2, c3 = st.columns(3)
with c1:
    countries = df.filter(like='country_').sum()
    st.metric("Global Market Penetration", f"{(countries>0).sum()} countries")
with c2:
    st.metric("Overall Conversion Rate", f"{df.conversion_status.mean()*100:.1f}%")
with c3:
    paid = df[df['traffic_source_Paid Ads']==1]
    cac = paid.shape[0] / df.conversion_status.sum() if df.conversion_status.sum() else np.nan
    st.metric("CAC (Paid Ads)", f"${cac:.2f}")

# Sales & Marketing KPIs
st.header("Sales & Marketing KPIs")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Demo Requests", int(df.demo_request.sum()))
with c2:
    paid_cr = df[df['traffic_source_Paid Ads']==1].conversion_status.mean()*100
    st.metric("Paid Ads CR", f"{paid_cr:.1f}%")
with c3:
    top_region = df.filter(like='region_').sum().idxmax().split('_',1)[1]
    st.metric("Top Performing Region", top_region)

# Product Development KPIs
st.header("Product Development KPIs")
c1, c2, c3 = st.columns(3)
with c1:
    best_prod = df.filter(like='product_').mean().idxmax().split('_',1)[1]
    st.metric("Best Converting Product", best_prod)
with c2:
    st.metric("Average Engagement Score", f"{df.engagement_score.mean():.2f}")
with c3:
    st.metric("Average Bounce Rate", f"{df.bounce_rate.mean()*100:.1f}%")

# Visualizations
st.header("Data Visualizations")

# Geo map
country_data = (
    df.filter(like='country_')
      .sum()
      .reset_index()
      .rename(columns={'index':'Country', 0:'Conversions'})
)
country_data['Country'] = country_data['Country'].str.replace('country_', '')
fig_map = px.choropleth(
    country_data, locations='Country', locationmode='country names',
    color='Conversions', title='Geographical Conversion Distribution'
)
st.plotly_chart(fig_map, use_container_width=True)

# Time series
df['timestamp'] = pd.to_datetime(df['timestamp'])
ts = df.resample('D', on='timestamp').conversion_status.mean().fillna(0)
fig_ts = px.line(ts, title='Daily Conversion Rate Over Time')
st.plotly_chart(fig_ts, use_container_width=True)

# Traffic pie
traffic = df.filter(like='traffic_source_').sum()
fig_pie = px.pie(
    values=traffic.values, names=traffic.index,
    title='Traffic Source Distribution'
)
st.plotly_chart(fig_pie, use_container_width=True)

# Anomaly Detection
st.header("Anomaly Detection")
X_full = df.drop(['conversion_status','timestamp','ip_address','city','session_id'], axis=1)
df['anomaly'] = iso.predict(X_full)
st.dataframe(df[df['anomaly']==-1].head(), use_container_width=True)

# Model Performance
st.header("AI Model Performance")
if acc is not None:
    st.write(f"Conversion Prediction Accuracy (test set): **{acc*100:.1f}%**")
else:
    st.write("Model loaded from disk; accuracy not recomputed.")

# Real-time Prediction
st.header("Real-time Conversion Prediction")
with st.form("predict_form"):
    session_dur = st.number_input("Session Duration", min_value=0.0, value=0.0)
    page_views  = st.number_input("Page Views",      min_value=0,   value=1)
    submitted   = st.form_submit_button("Predict")
    if submitted:
        sample = pd.DataFrame(
            [[session_dur, page_views]],
            columns=['session_duration','page_views']
        )
        pred = model.predict(sample)[0]
        st.success(f"Conversion is **{'Likely' if pred else 'Unlikely'}**")
