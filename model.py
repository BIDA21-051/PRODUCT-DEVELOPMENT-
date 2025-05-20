# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import joblib

# Load dataset (replace with actual data path)
df = pd.read_csv('C:\\Users\\bida21-051\\AI_Dashboard\\raw_synthetic_iis_logs.csv')

# Data Cleaning
def clean_data(df):
    # Handle missing values
    df.fillna({'campaign_tag': 'Unknown', 'region': 'Unknown'}, inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

df = clean_data(df)

# Feature Engineering
def engineer_features(df):
    # Time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    # Interaction features
    df['page_views_per_hour'] = df['page_views'] / (df['session_duration'] + 1e-5)
    return df

df = engineer_features(df)

# Define features
numeric_features = ['session_duration', 'page_views', 'bounce_rate', 'engagement_score']
categorical_features = ['country', 'device', 'campaign_tag', 'traffic_source', 'product']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define target and features
y = df['conversion_status']  # Make sure this column exists
X = df.drop(columns=['conversion_status', 'timestamp'])  # Remove target and timestamp from features

# Time series data for ARIMA
demo_ts = df.resample('D', on='timestamp')['demo_request'].sum()  # Make sure 'demo_request' exists

class HybridModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.anomaly_detector = IsolationForest(contamination=0.01)
        self.ts_model = None
        
    def fit(self, X, y, demo_ts):
        # Classification model
        X_processed = preprocessor.fit_transform(X)
        self.classifier.fit(X_processed, y)
        
        # Time series model (ARIMA)
        self.ts_model = ARIMA(demo_ts, order=(1,1,1)).fit()
        
    def predict(self, X):
        X_processed = preprocessor.transform(X)
        return self.classifier.predict(X_processed)

# Train model
model = HybridModel()
model.fit(X, y, demo_ts)

# Save model
joblib.dump(model, 'ai_model.pkl')

# Evaluate performance
y_pred = model.predict(X)
print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
print(classification_report(y, y_pred))

# Continuous Learning (Example: Weekly retraining)
def update_model(new_data):
    global model
    new_data = clean_data(new_data)
    new_data = engineer_features(new_data)
    y_new = new_data['conversion_status']
    X_new = new_data.drop(columns=['conversion_status', 'timestamp'])
    updated_demo_ts = pd.concat([demo_ts, new_data.resample('D', on='timestamp')['demo_request'].sum()])
    model.fit(pd.concat([X, X_new]), pd.concat([y, y_new]), updated_demo_ts)
    joblib.dump(model, 'ai_model.pkl')
