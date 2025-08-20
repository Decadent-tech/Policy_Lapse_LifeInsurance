import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insurance Risk Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data (you would load your actual data here)
@st.cache_data
def load_data():
    # Simulate your actual data
    np.random.seed(42)
    n_policies = 5000
    
    data = {
        'Policy_ID': [f'INS-2024-{i:06d}' for i in range(1, n_policies+1)],
        'Premium': np.random.normal(500, 200, n_policies),
        'Benefit': np.random.normal(100000, 50000, n_policies),
        'Policy_Age': np.random.randint(1, 15, n_policies),
        'Customer_Age': np.random.normal(45, 15, n_policies),
        'Risk_Score': np.random.beta(2, 5, n_policies) * 100,
        'Predicted_Lapse': np.random.choice([0, 1], n_policies, p=[0.85, 0.15]),
        'Actual_Lapse': np.random.choice([0, 1], n_policies, p=[0.83, 0.17]),
        'Last_Payment': pd.date_range('2024-01-01', periods=n_policies, freq='D'),
        'Policy_Type': np.random.choice(['Term Life', 'Whole Life', 'Universal Life'], n_policies),
        'Payment_Frequency': np.random.choice(['Monthly', 'Quarterly', 'Annually'], n_policies)
    }
    
    df = pd.DataFrame(data)
    df['Risk_Level'] = pd.cut(df['Risk_Score'], 
                             bins=[0, 30, 60, 100], 
                             labels=['Low', 'Medium', 'High'])
    return df

def load_model_metrics():
    return {
        'accuracy': 0.958,
        'precision': 0.915,
        'recall': 0.932,
        'f1_score': 0.923,
        'auc_roc': 0.967
    }

# Main dashboard function
def main():
    st.title("🛡️ Insurance Risk Analytics Dashboard")
    st.markdown("**AI-Powered Policy Lapse Prediction & Fraud Detection System**")
    
    # Load data
    df = load_data()
    model_metrics = load_model_metrics()
    
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Filters")
    
    # Risk level filter
    risk_levels = st.sidebar.multiselect(
        "Risk Level",
        options=df['Risk_Level'].unique(),
        default=df['Risk_Level'].unique()
    )
    
    # Policy type filter
    policy_types = st.sidebar.multiselect(
        "Policy Type",
        options=df['Policy_Type'].unique(),
        default=df['Policy_Type'].unique()
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[df['Last_Payment'].min(), df['Last_Payment'].max()],
        min_value=df['Last_Payment'].min(),
        max_value=df['Last_Payment'].max()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Risk_Level'].isin(risk_levels)) & 
        (df['Policy_Type'].isin(policy_types)) &
        (df['Last_Payment'].dt.date >= date_range[0]) &
        (df['Last_Payment'].dt.date <= date_range[1])
    ]
    
    # Key Performance Indicators
    st.header("📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Policies",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - 4500:+,}"
        )
    
    with col2:
        high_risk_count = len(filtered_df[filtered_df['Risk_Level'] == 'High'])
        st.metric(
            label="High Risk Policies",
            value=f"{high_risk_count:,}",
            delta=f"{high_risk_count - 450:+,}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value=f"{model_metrics['accuracy']:.1%}",
            delta="+2.3%"
        )
    
    with col4:
        predicted_savings = filtered_df['Benefit'].sum() * 0.15 * 0.85 / 1000000
        st.metric(
            label="Predicted Savings",
            value=f"${predicted_savings:.1f}M",
            delta="+12.8%"
        )
    
    with col5:
        avg_risk_score = filtered_df['Risk_Score'].mean()
        st.metric(
            label="Avg Risk Score",
            value=f"{avg_risk_score:.1f}%",
            delta=f"{avg_risk_score - 35:.1f}%",
            delta_color="inverse"
        )
    
    # Row 1: Risk Distribution and Model Performance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Risk Distribution Over Time")
        
        # Create monthly aggregation
        filtered_df['Month'] = filtered_df['Last_Payment'].dt.to_period('M').dt.to_timestamp()
        monthly_risk = filtered_df.groupby(['Month', 'Risk_Level']).size().unstack(fill_value=0)

        fig = px.bar(
        monthly_risk.reset_index(),
        x='Month',
        y=['Low', 'Medium', 'High'],
        title="Risk Distribution Trends",
        color_discrete_map={'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Current Risk Portfolio")
        
        risk_counts = filtered_df['Risk_Level'].value_counts()
        colors = ['#4CAF50', '#FF9800', '#F44336']
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_sequence=colors,
            title="Risk Level Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Model Performance and Prediction Accuracy
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': [model_metrics['accuracy'], model_metrics['precision'], 
                     model_metrics['recall'], model_metrics['f1_score'], model_metrics['auc_roc']]
        })
        
        fig = px.bar(
        metrics_df,
        x='Metric',
        y='Score',
        title="Model Performance Metrics",
        color='Score',
        color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        
    
    with col2:
        st.subheader("📈 Prediction vs Actual Comparison")
        
        # Simulate monthly prediction accuracy
        months = pd.date_range('2024-01-01', periods=6, freq='M')
        accuracy_data = {
            'Month': months,
            'Predicted_Lapse_Rate': np.random.uniform(0.14, 0.18, 6),
            'Actual_Lapse_Rate': np.random.uniform(0.15, 0.19, 6)
        }
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=accuracy_df['Month'],
            y=accuracy_df['Predicted_Lapse_Rate'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#2196F3', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=accuracy_df['Month'],
            y=accuracy_df['Actual_Lapse_Rate'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#4CAF50', width=3)
        ))
        
        fig.update_layout(
            title="Lapse Rate: Predicted vs Actual",
            xaxis_title="Month",
            yaxis_title="Lapse Rate",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Feature Importance and Risk Factors
    st.subheader("🔍 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance data
        features = ['Premium_Amount', 'Policy_Age', 'Benefit_Premium_Ratio', 
                   'Payment_Frequency', 'Customer_Age', 'Policy_Type',
                   'Payment_Delays', 'Claim_History', 'Credit_Score', 'Employment_Status']
        importance = np.array([0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01])
        
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='Plasma'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Premium vs Risk Analysis")
        # Ensure positive, normalized size for scatter plot
        filtered_df['Benefit_Size'] = filtered_df['Benefit'].clip(lower=0) / 1000
        fig = px.scatter(
            filtered_df.sample(500),  # Sample for better performance
            x='Premium',
            y='Risk_Score',
            color='Risk_Level',
            size='Benefit_Size',
            hover_data=['Policy_Age', 'Customer_Age'],
            title="Premium Amount vs Risk Score",
            color_discrete_map={'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # High-Risk Alerts Section
    st.header("🚨 High-Risk Policy Alerts")
    
    high_risk_policies = filtered_df[filtered_df['Risk_Level'] == 'High'].head(10)
    
    if not high_risk_policies.empty:
        for _, policy in high_risk_policies.iterrows():
            risk_class = "risk-high" if policy['Risk_Score'] > 80 else "risk-medium"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <strong>Policy ID:</strong> {policy['Policy_ID']} | 
                <strong>Risk Score:</strong> {policy['Risk_Score']:.1f}% | 
                <strong>Premium:</strong> ${policy['Premium']:.0f} | 
                <strong>Type:</strong> {policy['Policy_Type']}
                <br><small>Recommended Action: {'Immediate intervention required' if policy['Risk_Score'] > 80 else 'Monitor closely'}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-risk policies found with current filters.")
    
    # Data Export Section
    st.header("📥 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Risk Summary"):
            risk_summary = filtered_df.groupby('Risk_Level').agg({
                'Policy_ID': 'count',
                'Premium': 'mean',
                'Risk_Score': 'mean',
                'Benefit': 'sum'
            }).round(2)
            st.download_button(
                label="Download CSV",
                data=risk_summary.to_csv(),
                file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔍 Export High-Risk Policies"):
            high_risk_export = filtered_df[filtered_df['Risk_Level'] == 'High']
            st.download_button(
                label="Download CSV",
                data=high_risk_export.to_csv(index=False),
                file_name=f"high_risk_policies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Export Model Metrics"):
            metrics_export = pd.DataFrame([model_metrics])
            st.download_button(
                label="Download CSV",
                data=metrics_export.to_csv(index=False),
                file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Dashboard Info")
st.sidebar.info("""
**Last Updated:** Dec 2024  
**Data Source:** Insurance Database  
**Model Version:** XGBoost v2.1  
**Refresh Rate:** Real-time  
""")

st.sidebar.markdown("### 📞 Contact Support")
st.sidebar.markdown("**Data Science Team**  \nEmail: ds-team@insurance.com  \nPhone: (555) 123-4567")

if __name__ == "__main__":
    main()