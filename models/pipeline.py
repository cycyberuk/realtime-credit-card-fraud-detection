# models/pipeline.py

class FraudDetectionPipeline:
    """Custom pipeline class for fraud detection model"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        X_scaled = X.copy()
        X_scaled[['Time', 'Amount']] = self.scaler.transform(X[['Time', 'Amount']])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = X.copy()
        X_scaled[['Time', 'Amount']] = self.scaler.transform(X[['Time', 'Amount']])
        return self.model.predict_proba(X_scaled)