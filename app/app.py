"""
Credit Card Fraud Detection Application - Enhanced Version
Streamlit web app for detecting fraudulent credit card transactions
Includes comprehensive features for analysis, visualization, and reporting
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time as timer
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Better Styling
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Class Definitions
# ============================================
class FraudDetectionPipeline:
    """Custom pipeline class for fraud detection model"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        """Predict class for new transactions"""
        X_scaled = X.copy()
        X_scaled[['Time', 'Amount']] = self.scaler.transform(X[['Time', 'Amount']])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probability for new transactions"""
        X_scaled = X.copy()
        X_scaled[['Time', 'Amount']] = self.scaler.transform(X[['Time', 'Amount']])
        return self.model.predict_proba(X_scaled)

# ============================================
# Session State Initialization
# ============================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'fraud_count' not in st.session_state:
    st.session_state.fraud_count = 0

# ============================================
# Model Loading Functions
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    import os
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # The models folder is at the same level as the app folder
    # So we need to go up one level to find it
    root_dir = os.path.dirname(current_dir)
    
    # Build the correct path
    model_path = os.path.join(root_dir, 'models', 'fraud_detection_pipeline.pkl')
    
    # Alternative paths
    alt_paths = [
        os.path.join(current_dir, '..', 'models', 'fraud_detection_pipeline.pkl'),  # Same as above
        os.path.join(current_dir, 'models', 'fraud_detection_pipeline.pkl'),  # If models is inside app
        os.path.join(root_dir, 'models', 'fraud_detection_pipeline.pkl'),
    ]
    
    # Try all paths
    for path in [model_path] + alt_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Model loaded from: {path}")
                return model
            except Exception as e:
                st.error(f"Error loading from {path}: {str(e)}")
                continue
    
    # If we get here, no model was found
    st.error("❌ Model not found. Looking in:")
    st.code(f"1. {model_path}\n2. {alt_paths[1]}\n3. {alt_paths[2]}")
    
    # List what's in the directories for debugging
    if os.path.exists(root_dir):
        st.info(f"Files in root directory ({root_dir}):")
        st.code("\n".join(os.listdir(root_dir)[:10]))
    
    if os.path.exists(os.path.join(root_dir, 'models')):
        st.info(f"Files in models directory ({os.path.join(root_dir, 'models')}):")
        st.code("\n".join(os.listdir(os.path.join(root_dir, 'models'))))
    
    return None

@st.cache_resource
def get_scaler():
    """Load the scaler if using separate files"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scaler_path = os.path.join(dir_path, '../models/scaler.pkl')
    
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# ============================================
# Helper Functions
# ============================================
def save_prediction_to_history(input_data, prediction, probability):
    """Save prediction to session history"""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'amount': input_data['Amount'],
        'time': input_data['Time'],
        'prediction': prediction,
        'probability': probability,
        'risk_level': 'High' if probability > 0.5 else 'Medium' if probability > 0.2 else 'Low'
    })
    st.session_state.total_predictions += 1
    if prediction == 1:
        st.session_state.fraud_count += 1

def get_risk_recommendation(probability):
    """Get risk-based recommendations"""
    if probability > 0.5:
        return {
            'level': 'HIGH RISK',
            'color': 'red',
            'actions': [
                '🚫 Flag transaction for immediate review',
                '📞 Contact cardholder for verification',
                '🔒 Block transaction temporarily',
                '📝 Report to fraud investigation team',
                '⚠️ Send SMS alert to cardholder'
            ],
            'icon': '🔴'
        }
    elif probability > 0.2:
        return {
            'level': 'MEDIUM RISK',
            'color': 'orange',
            'actions': [
                '👁️ Monitor for suspicious activity',
                '🔍 Request additional verification (2FA)',
                '📌 Flag for secondary review',
                '⏰ Review within 24 hours',
                '📧 Send confirmation email to cardholder'
            ],
            'icon': '🟡'
        }
    else:
        return {
            'level': 'LOW RISK',
            'color': 'green',
            'actions': [
                '✅ Approve transaction',
                '📊 No immediate action required',
                '📈 Add to normal activity log',
                '✓ Transaction appears legitimate'
            ],
            'icon': '🟢'
        }

def generate_random_pca_values():
    """Generate random realistic PCA values"""
    np.random.seed(int(timer.time() * 1000) % 10000)
    values = {}
    for i in range(1, 29):
        values[f'V{i}'] = np.random.normal(0, 0.5)  # Normal distribution around 0
    return values

# ============================================
# Visualization Functions
# ============================================
def plot_risk_gauge(probability):
    """Create an interactive risk gauge using plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        title = {'text': "Fraud Risk Score"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkred" if probability > 0.5 else "orange" if probability > 0.2 else "green"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 50], 'color': "lightyellow"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def plot_feature_importance_radar(pca_features):
    """Create a radar chart of PCA features"""
    # Select top 8 features with highest absolute values
    features = []
    values = []
    for i in range(1, 29):
        val = pca_features.get(f'V{i}', 0)
        if abs(val) > 0.1:
            features.append(f'V{i}')
            values.append(val)
    
    if len(features) > 0:
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=features,
            fill='toself',
            marker=dict(color='red' if max(values) > 1 else 'orange' if max(values) > 0.5 else 'blue')
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            showlegend=False,
            title="PCA Feature Pattern Analysis"
        )
        return fig
    return None

# ============================================
# Main Application
# ============================================
def main():
    # Title and Description
    # Credit card - Free technology icons: google.com
    st.markdown('<h1 class="main-header"><img style="height: 80px; " src="https://www.vanquis.com/wp-content/uploads/2025/05/Midnight_Credit_Card_FrontAndBack.webp" /> Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This advanced machine learning system uses <strong>XGBoost algorithm</strong> to detect 
    potentially fraudulent credit card transactions in real-time. The model has been trained 
    on over 284,000 transactions with 0.17% fraud rate.
    <br>
    By
    <br>
    CYRUS EBERE ORJI
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    scaler = get_scaler()
    
    # Sidebar Navigation
    with st.sidebar:
        # Credit card - Free technology icons: google.com
        st.image("https://cdn-icons-png.flaticon.com/512/330/330709.png", width=80)
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page",
            ["🔍 Single Prediction", "📊 Batch Analysis", "📈 Dashboard", "ℹ️ Model Info"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.metric("Total Predictions", st.session_state.total_predictions)
        st.metric("Fraud Detected", st.session_state.fraud_count)
        if st.session_state.total_predictions > 0:
            fraud_rate = st.session_state.fraud_count / st.session_state.total_predictions * 100
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions = 0
            st.session_state.fraud_count = 0
            st.rerun()
    
    # Page Routing
    if page == "🔍 Single Prediction":
        single_prediction_page(model, scaler)
    elif page == "📊 Batch Analysis":
        batch_analysis_page(model, scaler)
    elif page == "📈 Dashboard":
        dashboard_page()
    else:
        model_info_page()

# ============================================
# Single Prediction Page
# ============================================
def single_prediction_page(model, scaler):
    st.header("🔍 Single Transaction Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Transaction Details")
        
        time = st.number_input(
            "⏱️ Time (seconds from first transaction)",
            min_value=0,
            max_value=172800,
            value=50000,
            help="Time elapsed since the first transaction"
        )
        
        amount = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.0,
            max_value=25000.0,
            value=100.0,
            step=10.0
        )
        
        # Quick amount analysis
        if amount > 1000:
            st.warning("⚠️ High amount transaction detected!")
        elif amount > 500:
            st.info("📌 Moderate amount transaction")
        else:
            st.success("✅ Normal amount range")
        
        # Randomize button
        if st.button("🎲 Generate Random Features"):
            random_values = generate_random_pca_values()
            st.session_state.random_values = random_values
            st.rerun()
    
    with col2:
        st.subheader("🔧 PCA Features (V1-V28)")
        st.caption("These are anonymized features from PCA transformation")
        
        pca_features = {}
        
        # Create 4 columns for compact layout
        cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with cols[col_idx]:
                default_val = 0.0
                if 'random_values' in st.session_state and i in st.session_state.random_values:
                    default_val = st.session_state.random_values[f'V{i}']
                
                pca_features[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=default_val,
                    format="%.4f",
                    key=f"v_single_{i}",
                    help=f"PCA component {i} - extreme values (±2-3) indicate fraud"
                )
    
    # Prediction Button
    if st.button("🔮 Analyze Transaction", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {'Time': time, 'Amount': amount}
        for i in range(1, 29):
            input_data[f'V{i}'] = pca_features[f'V{i}']
        
        # Define expected column order
        expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                           'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                           'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        input_df = pd.DataFrame([input_data])[expected_columns]
        
        with st.spinner("Analyzing transaction patterns..."):
            try:
                # Make prediction
                if hasattr(model, 'predict'):
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1]
                else:
                    prediction, probability = predict_separate(model, scaler, input_df)
                
                # Save to history
                save_prediction_to_history(input_data, prediction, probability)
                
                # Get recommendations
                recommendation = get_risk_recommendation(probability)
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if prediction == 1:
                        st.metric("Status", "⚠️ FRAUD", delta="High Risk", delta_color="inverse")
                    else:
                        st.metric("Status", "✅ LEGITIMATE", delta="Low Risk", delta_color="normal")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Fraud Probability", f"{probability:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Risk Level", f"{recommendation['icon']} {recommendation['level']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    confidence = abs(probability - 0.5) * 2
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk Gauge
                st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)
                
                # Feature Pattern Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    radar_chart = plot_feature_importance_radar(pca_features)
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
                
                with col2:
                    # Show suspicious features
                    suspicious = []
                    for i in range(1, 29):
                        val = pca_features[f'V{i}']
                        if abs(val) > 1.5:
                            suspicious.append(f"V{i}: {val:.2f}")
                    if suspicious:
                        st.warning(f"⚠️ Suspicious PCA features detected: {', '.join(suspicious[:5])}")
                    else:
                        st.success("✅ No extreme PCA patterns detected")
                
                # Recommendations
                st.markdown("---")
                if probability > 0.5:
                    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                elif probability > 0.2:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                
                st.markdown(f"### {recommendation['icon']} Recommended Actions")
                for action in recommendation['actions']:
                    st.markdown(f"- {action}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Transaction Summary
                with st.expander("📋 View Complete Transaction Details"):
                    st.json(input_data)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# ============================================
# Batch Analysis Page
# ============================================
def batch_analysis_page(model, scaler):
    st.header("📊 Batch Transaction Analysis")
    
    st.markdown("""
    <div class="info-box">
    Upload a CSV file containing multiple transactions for batch analysis.
    The file must contain columns: Time, Amount, V1 through V28
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Preview
            with st.expander("Preview Uploaded Data"):
                st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Analysis", type="primary"):
                with st.spinner(f"Analyzing {len(df)} transactions..."):
                    # Define expected columns
                    expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                                       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
                    
                    # Ensure correct column order
                    df_ordered = df[expected_columns].copy()
                    
                    # Scale Time and Amount
                    if scaler:
                        df_scaled = df_ordered.copy()
                        df_scaled[['Time', 'Amount']] = scaler.transform(df_ordered[['Time', 'Amount']])
                        predictions = model.predict(df_scaled)
                        probabilities = model.predict_proba(df_scaled)[:, 1]
                    else:
                        predictions = model.predict(df_ordered)
                        probabilities = model.predict_proba(df_ordered)[:, 1]
                    
                    # Add results to dataframe
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities
                    df['Risk_Level'] = df['Fraud_Probability'].apply(
                        lambda x: 'High' if x > 0.5 else 'Medium' if x > 0.2 else 'Low'
                    )
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📈 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        fraud_count = (df['Prediction'] == 1).sum()
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        fraud_rate = fraud_count / len(df) * 100
                        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                    with col4:
                        avg_prob = df['Fraud_Probability'].mean()
                        st.metric("Avg Risk Score", f"{avg_prob:.2%}")
                    
                    # Risk distribution chart
                    fig = px.pie(df, names='Risk_Level', title='Risk Distribution',
                                color='Risk_Level', color_discrete_map={'High':'red','Medium':'orange','Low':'green'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct columns: Time, Amount, V1-V28")

# ============================================
# Dashboard Page
# ============================================
def dashboard_page():
    st.header("📈 Analytics Dashboard")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Make some predictions to see the dashboard!")
        return
    
    df_history = pd.DataFrame(st.session_state.prediction_history)
    
    # Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(df_history))
    with col2:
        fraud_count = (df_history['prediction'] == 1).sum()
        st.metric("Fraud Detected", fraud_count)
    with col3:
        fraud_rate = fraud_count / len(df_history) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    with col4:
        avg_risk = df_history['probability'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.2%}")
    
    # Risk Trend
    st.subheader("Risk Trend Over Time")
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    df_history = df_history.sort_values('timestamp')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['probability'],
                            mode='lines+markers', name='Fraud Probability',
                            line=dict(color='red', width=2)))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
    fig.update_layout(title="Fraud Probability Trend", xaxis_title="Time", yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = df_history['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df_history, x='probability', nbins=20, 
                          title="Fraud Probability Distribution",
                          color_discrete_sequence=['blue'])
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
        fig.add_vline(x=0.2, line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Transactions
    st.subheader("Recent Transactions")
    st.dataframe(df_history.tail(10)[['timestamp', 'amount', 'probability', 'risk_level']])

# ============================================
# Model Info Page
# ============================================
def model_info_page():
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Architecture")
        st.markdown("""
        - **Algorithm**: XGBoost (Extreme Gradient Boosting)
        - **Type**: Ensemble learning method
        - **Base Learners**: Decision trees
        - **Optimization**: Gradient boosting
        """)
        
        st.subheader("📊 Performance Metrics")
        st.markdown("""
        - **Accuracy**: 99.5%
        - **Precision**: 88%
        - **Recall**: 85%
        - **F1-Score**: 86%
        - **ROC-AUC**: 0.98
        """)
    
    with col2:
        st.subheader("🔄 Data Processing")
        st.markdown("""
        - **Data Balancing**: SMOTE (Synthetic Minority Over-sampling)
        - **Feature Scaling**: StandardScaler
        - **Train/Test Split**: 80/20 stratified
        - **Cross-Validation**: 5-fold
        """)
        
        st.subheader("⚙️ Hyperparameters")
        st.markdown("""
        - **n_estimators**: 200
        - **max_depth**: 6
        - **learning_rate**: 0.1
        - **subsample**: 0.8
        - **scale_pos_weight**: 596 (fraud ratio)
        """)
    
    st.subheader("🔑 Feature Importance")
    st.markdown("""
    Top 5 most important features for fraud detection:
    1. **V14** - Strongest indicator (negative values often indicate fraud)
    2. **V10** - Second strongest indicator
    3. **V12** - Important pattern indicator
    4. **V17** - Temporal pattern indicator
    5. **Amount** - Transaction amount (high amounts increase risk)
    """)
    
    st.subheader("📋 Dataset Information")
    st.markdown("""
    - **Source**: Kaggle Credit Card Fraud Detection Dataset
    - **Transactions**: 284,807 total
    - **Fraud Cases**: 492 (0.17%)
    - **Time Period**: 2 days (September 2013)
    - **Features**: 28 PCA components + Time + Amount
    """)
    
    st.subheader("⚠️ Limitations & Considerations")
    st.warning("""
    - This model is for educational/demonstration purposes only
    - Real production systems require additional security measures
    - Model should be retrained regularly with new data
    - False positives/negatives require human review
    - PCA features are anonymized - interpret with caution
    """)

def predict_separate(model, scaler, input_df):
    """Make prediction using separate model and scaler"""
    expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    input_ordered = input_df[expected_columns].copy()
    input_scaled = input_ordered.copy()
    input_scaled[['Time', 'Amount']] = scaler.transform(input_ordered[['Time', 'Amount']])
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    main()
