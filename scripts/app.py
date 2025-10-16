# app.py - Fixed Version with Analytics Report Button Removed


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="PowerCo Churn Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .high-risk {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
    }
    .low-risk {
        background-color: #e6ffe6;
        border-left: 4px solid #44ff44;
    }
    .sidebar-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class PowerCoChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'channel_sales', 'cons_12m', 'cons_gas_12m', 'forecast_cons_12m',
            'forecast_discount_energy', 'forecast_meter_rent_12m', 'has_gas',
            'imp_cons', 'margin_net_pow_ele', 'net_margin', 'num_years_antig',
            'origin_up', 'pow_max', 'price_mid_peak_fix'
        ]
        self.optimal_threshold = 0.162
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load('powerco_churn_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            return True
        except:
            return False
    
    def preprocess_data(self, df):
        """Preprocess the input data"""
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Convert ID if present
        if 'id' in processed_df.columns:
            processed_df['id'] = processed_df['id'].astype(str).str[:7]
        
        # Handle datetime columns
        date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
        
        # Calculate tenure if date columns exist
        if 'date_activ' in processed_df.columns and 'date_end' in processed_df.columns:
            processed_df['tenure'] = processed_df['date_end'].dt.year - processed_df['date_activ'].dt.year
        
        # Channel sales mapping
        channel_mapping = {
            'foosdfpfkusacimwkcsosbicdxkicaua': 'channel_sales_1',
            'MISSING': 'not_specified',
            'lmkebamcaaclubfxadlmueccxoimlema': 'channel_sales_2',
            'usilxuppasemubllopkaafesmlibmsdf': 'channel_sales_3',
            'ewpakwlliwisiwduibdlfmalxowmwpci': 'channel_sales_4',
            'sddiedcslfslkckwlfkdpoeeailfpeds': 'channel_sales_5',
            'epumfxlbckeskwekxbiuasklxalciiuu': 'channel_sales_6',
            'fixdbufsefwooaasfcxdxadsiekoceaa': 'channel_sales_7'
        }
        if 'channel_sales' in processed_df.columns:
            processed_df['channel_sales'] = processed_df['channel_sales'].replace(channel_mapping)
        
        # Encode categorical variables
        categorical_columns = ['channel_sales', 'has_gas', 'origin_up']
        le = LabelEncoder()
        
        for col in categorical_columns:
            if col in processed_df.columns:
                # Handle missing values
                processed_df[col] = processed_df[col].fillna('not_specified')
                try:
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                except:
                    # If encoding fails, use simple integer encoding
                    processed_df[col] = pd.factorize(processed_df[col])[0]
        
        # Select only the features used in training
        available_features = [f for f in self.feature_names if f in processed_df.columns]
        return processed_df[available_features]
    
    def predict_churn(self, features_df):
        """Make churn predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please load model first.")
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Get probabilities
        probabilities = self.model.predict_proba(scaled_features)[:, 1]
        
        # Apply optimal threshold
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        
        return probabilities, predictions

def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"
    
    # Initialize predictor
    predictor = PowerCoChurnPredictor()
    
    # Sidebar with centered image
    st.sidebar.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/648/648096.png", width=100)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.title("PowerCo Analytics")
    st.sidebar.markdown("---")
    
    # Navigation with session state support
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["🏠 Dashboard", "📊 Batch Prediction", "👤 Single Customer", "ℹ️ About"],
        index=["🏠 Dashboard", "📊 Batch Prediction", "👤 Single Customer", "ℹ️ About"].index(st.session_state.current_page)
    )
    
    # Update session state when navigation changes
    if app_mode != st.session_state.current_page:
        st.session_state.current_page = app_mode
        st.rerun()
    
    # Main content routing
    if st.session_state.current_page == "🏠 Dashboard":
        show_dashboard(predictor)
    elif st.session_state.current_page == "📊 Batch Prediction":
        show_batch_prediction(predictor)
    elif st.session_state.current_page == "👤 Single Customer":
        show_single_prediction(predictor)
    else:
        show_about()

def show_dashboard(predictor):
    """Display the main dashboard"""
    st.markdown('<h1 class="main-header">⚡ PowerCo Customer Churn Analytics</h1>', unsafe_allow_html=True)
    
    # Header with image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1509391366360-2e959784a276?ixlib=rb-4.0.3&w=800", 
                use_column_width=True, caption="Energy Analytics Dashboard")
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3>Predict and Prevent Customer Churn with AI-Powered Insights</h3>
        <p>Identify at-risk customers and take proactive retention measures</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics cards
    st.markdown('<div class="sub-header">📊 Business Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">14,606</div>
            <div>Active SME clients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value">9.7%</div>
            <div>Historical average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">82.9%</div>
            <div>Balanced performance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Detection Rate</div>
            <div class="metric-value">38.7%</div>
            <div>Churn recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.markdown("---")
    st.markdown('<div class="sub-header">📈 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance chart
        features = ['margin_net_pow_ele', 'cons_12m', 'forecast_meter_rent_12m', 
                   'net_margin', 'forecast_cons_12m', 'pow_max', 'imp_cons']
        importance = [13.8, 13.3, 12.1, 11.1, 11.0, 6.5, 5.2]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            title="Top Feature Importance",
            xaxis_title="Importance Score (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance metrics
        metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        scores = [25.2, 38.7, 30.6, 68.1]
        
        fig = go.Figure(go.Bar(
            x=metrics,
            y=scores,
            marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ))
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions - UPDATED: Removed Analytics Report button
    st.markdown("---")
    st.markdown('<div class="sub-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload Customer Data", use_container_width=True, key="upload_btn"):
            st.session_state.current_page = "📊 Batch Prediction"
            st.rerun()
    
    with col2:
        if st.button("👤 Analyze Single Customer", use_container_width=True, key="single_btn"):
            st.session_state.current_page = "👤 Single Customer"
            st.rerun()

def show_batch_prediction(predictor):
    """Batch prediction interface"""
    st.markdown('<h1 class="main-header">📊 Batch Churn Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4>💡 Upload your customer data CSV file</h4>
        <p>Ensure your data includes the required features for accurate churn prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} customer records")
            
            # Show data preview
            with st.expander("🔍 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Load model and make predictions
            if st.button("🔮 Predict Churn Risk", type="primary"):
                with st.spinner("Analyzing customer data..."):
                    if predictor.load_model():
                        # Preprocess data
                        features_df = predictor.preprocess_data(df)
                        
                        # Check if required features are present
                        missing_features = set(predictor.feature_names) - set(features_df.columns)
                        if missing_features:
                            st.error(f"❌ Missing required features: {missing_features}")
                            return
                        
                        # Make predictions
                        probabilities, predictions = predictor.predict_churn(features_df)
                        
                        # Create results dataframe
                        results_df = df.copy()
                        results_df['Churn_Probability'] = probabilities
                        results_df['Churn_Risk'] = ['High Risk' if p >= predictor.optimal_threshold else 'Low Risk' 
                                                  for p in probabilities]
                        results_df['Retention_Priority'] = results_df['Churn_Probability'].rank(ascending=False)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown('<div class="sub-header">📋 Prediction Results</div>', unsafe_allow_html=True)
                        
                        # Summary statistics
                        high_risk_count = (results_df['Churn_Risk'] == 'High Risk').sum()
                        high_risk_percent = (high_risk_count / len(results_df)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        with col2:
                            st.metric("High Risk Customers", high_risk_count)
                        with col3:
                            st.metric("High Risk Percentage", f"{high_risk_percent:.1f}%")
                        
                        # Display results table
                        st.dataframe(results_df.sort_values('Retention_Priority').head(10), use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv,
                            file_name="powerco_churn_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        with col1:
                            risk_counts = results_df['Churn_Risk'].value_counts()
                            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                       title="Churn Risk Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(results_df, x='Churn_Probability', 
                                            title="Churn Probability Distribution",
                                            nbins=20)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error("❌ Model files not found. Please ensure 'powerco_churn_model.pkl' and 'scaler.pkl' are in the directory.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_single_prediction(predictor):
    """Single customer prediction interface"""
    st.markdown('<h1 class="main-header">👤 Single Customer Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h4>🔍 Analyze individual customer churn risk</h4>
        <p>Enter customer details to get instant churn probability and recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            channel_sales = st.selectbox("Sales Channel", [
                "channel_sales_1", "channel_sales_2", "channel_sales_3", 
                "channel_sales_4", "channel_sales_5", "channel_sales_6", 
                "channel_sales_7", "not_specified"
            ])
            has_gas = st.radio("Has Gas Service", ["true", "false"])
            origin_up = st.selectbox("Origin Code", ["code_1", "code_2", "code_3", "code_4", "code_5", "not_specified"])
            num_years_antig = st.slider("Years with PowerCo", 1, 10, 3)
        
        with col2:
            st.subheader("Consumption & Financials")
            cons_12m = st.number_input("12M Consumption (kWh)", min_value=0.0, value=5000.0)
            cons_gas_12m = st.number_input("12M Gas Consumption", min_value=0.0, value=0.0)
            forecast_cons_12m = st.number_input("Forecasted Consumption", min_value=0.0, value=5200.0)
            net_margin = st.number_input("Net Margin (€)", min_value=0.0, value=150.0)
            margin_net_pow_ele = st.number_input("Power Margin (€)", min_value=0.0, value=120.0)
            pow_max = st.number_input("Subscribed Power (kW)", min_value=0.0, value=15.0)
        
        # Additional features
        col3, col4 = st.columns(2)
        with col3:
            forecast_discount_energy = st.number_input("Forecast Discount", min_value=0.0, value=0.0)
            forecast_meter_rent_12m = st.number_input("Meter Rent Forecast", min_value=0.0, value=50.0)
        
        with col4:
            imp_cons = st.number_input("Current Consumption", min_value=0.0, value=450.0)
            price_mid_peak_fix = st.number_input("Mid-Peak Price", min_value=0.0, value=0.12)
        
        submitted = st.form_submit_button("🔍 Analyze Churn Risk", type="primary")
    
    if submitted:
        with st.spinner("Calculating churn risk..."):
            # Create input dataframe
            input_data = {
                'channel_sales': channel_sales,
                'cons_12m': cons_12m,
                'cons_gas_12m': cons_gas_12m,
                'forecast_cons_12m': forecast_cons_12m,
                'forecast_discount_energy': forecast_discount_energy,
                'forecast_meter_rent_12m': forecast_meter_rent_12m,
                'has_gas': has_gas,
                'imp_cons': imp_cons,
                'margin_net_pow_ele': margin_net_pow_ele,
                'net_margin': net_margin,
                'num_years_antig': num_years_antig,
                'origin_up': origin_up,
                'pow_max': pow_max,
                'price_mid_peak_fix': price_mid_peak_fix
            }
            
            input_df = pd.DataFrame([input_data])
            
            if predictor.load_model():
                features_df = predictor.preprocess_data(input_df)
                probabilities, predictions = predictor.predict_churn(features_df)
                
                churn_prob = probabilities[0]
                is_high_risk = predictions[0] == 1
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="sub-header">📊 Risk Analysis Results</div>', unsafe_allow_html=True)
                
                risk_class = "high-risk" if is_high_risk else "low-risk"
                risk_text = "HIGH RISK 🚨" if is_high_risk else "LOW RISK ✅"
                
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h2>Churn Probability: {churn_prob:.1%}</h2>
                    <h3>{risk_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown('<div class="sub-header">💡 Retention Recommendations</div>', unsafe_allow_html=True)
                
                if is_high_risk:
                    st.warning("""
                    **Immediate Action Required:**
                    - 📞 Proactive outreach from retention team
                    - 💰 Personalized discount offers
                    - 🔧 Service review and optimization
                    - ⭐ Loyalty program enrollment
                    - 📊 Monthly consumption monitoring
                    """)
                else:
                    st.success("""
                    **Maintenance Actions:**
                    - ✅ Regular customer satisfaction checks
                    - 📧 Quarterly newsletter with energy tips
                    - 🎯 Cross-selling opportunities
                    - ⭐ Continue excellent service delivery
                    """)
                
                # Feature impact analysis
                st.markdown("---")
                st.markdown('<div class="sub-header">🔍 Key Risk Factors</div>', unsafe_allow_html=True)
                
                # Simple feature impact (based on domain knowledge)
                risk_factors = []
                if net_margin < 100:
                    risk_factors.append("Low profit margin")
                if forecast_cons_12m < cons_12m * 0.9:
                    risk_factors.append("Decreasing consumption forecast")
                if num_years_antig < 2:
                    risk_factors.append("New customer (higher churn risk)")
                if pow_max > 20:
                    risk_factors.append("High power subscription (price sensitive)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("No significant risk factors identified.")

def show_about():
    """About page"""
    st.markdown('<h1 class="main-header">ℹ️ About PowerCo Analytics</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Project Overview
        
        PowerCo Customer Churn Analytics is an AI-powered solution designed to predict 
        and prevent customer attrition for PowerCo's SME energy clients.
        
        ### 🔬 Methodology
        
        - **Algorithm**: Random Forest Classifier with 1000 trees
        - **Data**: Historical consumption, pricing, and customer data
        - **Features**: 14 key predictors including margins, consumption patterns, and contract details
        - **Optimization**: Threshold tuning for optimal business impact
        
        ### 📊 Model Performance
        
        - **ROC AUC**: 0.681
        - **Precision**: 25.2%
        - **Recall**: 38.7%
        - **F1-Score**: 30.6%
        - **Optimal Threshold**: 0.162
        
        ### 💼 Business Impact
        
        - **38.7%** improvement in churn detection
        - **Proactive** retention strategies
        - **Data-driven** decision making
        - **Customer lifetime value** optimization
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&w=400", 
                caption="AI Analytics")
        
        st.markdown("""
        ### 🛠️ Technical Stack
        
        - Python 3.8+
        - Scikit-learn
        - Streamlit
        - Plotly
        - Pandas/Numpy
        
        ### 👨‍💻 Developed By
        
        **Rotimi Sheriff Omosewo**
        - Data Scientist & AI Engineer
        - [GitHub](https://github.com/rotimi2020)
        - [Portfolio](https://rotimi2020.github.io)
        - [LinkedIn](https://linkedin.com/in/rotimi-sheriff-omosewo-939a806b)
        """)

if __name__ == "__main__":
    main()