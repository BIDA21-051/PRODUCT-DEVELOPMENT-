# models.py
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.compose import ColumnTransformer

class HybridModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.anomaly_detector = IsolationForest(contamination=0.01)
        self.ts_model = None

    def fit(self, X, y, demo_ts):
        from model import preprocessor  # Ensure preprocessor is importable
        X_processed = preprocessor.fit_transform(X)
        self.classifier.fit(X_processed, y)
        self.ts_model = ARIMA(demo_ts, order=(1, 1, 1)).fit()

    def predict(self, X):
        from model import preprocessor
        X_processed = preprocessor.transform(X)
        return self.classifier.predict(X_processed)
