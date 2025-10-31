"""
CA2 Mini Project - Water Quality Classification Web App
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AI Water Quality Classifier",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .potable {
        background-color: #4CAF50;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 24px;
        text-align: center;
    }
    .not-potable {
        background-color: #f44336;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 24px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, scaler, features, info

try:
    model, scaler, feature_names, model_info = load_models()
except Exception as e:
    st.error(f"⚠️ Model files not found! Please run CA2_Complete_Analysis.py first. Error: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/water.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Model Performance", "🔮 Predict Quality", "🧠 Feature Insights", "📈 Data Explorer"])

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">💧 AI Water Quality Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Water Potability Assessment</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{model_info['best_model_name']}</h3>
            <p>Best Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{model_info['accuracy']:.2%}</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{model_info['roc_auc']:.3f}</h3>
            <p>ROC-AUC Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(feature_names)}</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Problem Statement")
        st.write("""
        Water quality assessment is critical for public health. Traditional laboratory 
        testing is time-consuming (24-48 hours) and expensive. This AI system provides 
        **instant water potability classification** using physicochemical parameters.
        
        **Key Features:**
        - ⚡ Real-time predictions
        - 🎯 High accuracy classification
        - 📊 Explainable AI with SHAP
        - 🌍 Scalable solution
        """)
        
        st.subheader("🌊 Dataset Overview")
        st.write("""
        - **Source:** Kaggle Water Potability Dataset
        - **Samples:** 3,276 water samples
        - **Parameters:** 9 physicochemical measurements
        - **Target:** Binary classification (Potable/Not Potable)
        - **Class Distribution:** 61% Non-Potable, 39% Potable
        """)
    
    with col2:
        st.subheader("🔬 Water Quality Parameters")
        params_df = pd.DataFrame({
            'Parameter': ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                         'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'],
            'Description': [
                'Acidity/Alkalinity level',
                'Mineral content (mg/L)',
                'Total dissolved solids (ppm)',
                'Disinfectant amount (ppm)',
                'Sulfate concentration (mg/L)',
                'Electrical conductivity (μS/cm)',
                'Organic carbon content (ppm)',
                'THM compounds (μg/L)',
                'Cloudiness measure (NTU)'
            ]
        })
        st.dataframe(params_df, hide_index=True)
        
        st.subheader("🚀 Project Impact")
        st.success("""
        **Real-World Applications:**
        - Emergency disaster response
        - Rural water quality monitoring
        - Treatment plant automation
        - Public health surveillance
        - Cost reduction up to 70%
        """)

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance Analysis</p>', unsafe_allow_html=True)
    
    # Load comparison data
    try:
        comparison_df = pd.read_csv('model_comparison.csv', index_col=0)
        
        st.subheader("🏆 Model Comparison Table")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Interactive performance chart
        st.subheader("📈 Performance Metrics Comparison")
        
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for model in comparison_df.index:
            fig.add_trace(go.Scatterpolar(
                r=[comparison_df.loc[model, m] for m in metrics],
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_column_width=True)
        
    except FileNotFoundError:
        st.warning("Model comparison file not found. Please run the analysis script first.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 ROC Curves")
        try:
            img = Image.open('roc_curves.png')
            st.image(img, use_column_width=True)
        except:
            st.info("ROC curve image not found.")
    
    with col2:
        st.subheader("🎯 Confusion Matrices")
        try:
            img = Image.open('confusion_matrices.png')
            st.image(img, use_column_width=True)
        except:
            st.info("Confusion matrix image not found.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Precision-Recall Curve")
        try:
            img = Image.open('precision_recall_curve.png')
            st.image(img, use_column_width=True)
        except:
            st.info("Precision-Recall curve not found.")
    
    with col2:
        st.subheader("📈 Learning Curve")
        try:
            img = Image.open('learning_curve.png')
            st.image(img, use_column_width=True)
        except:
            st.info("Learning curve not found.")
    
    st.markdown("---")
    st.subheader("🔍 Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cross-Validation Mean", f"{model_info['cv_mean']:.4f}")
    with col2:
        st.metric("CV Std Deviation", f"{model_info['cv_std']:.4f}")
    with col3:
        improvement = (model_info['accuracy'] - 0.61) * 100
        st.metric("Improvement vs CA1", f"{improvement:.2f}%", delta=f"{improvement:.2f}%")

# ============================================================================
# PAGE 3: PREDICT QUALITY
# ============================================================================
elif page == "🔮 Predict Quality":
    st.markdown('<p class="main-header">🔮 Water Quality Prediction</p>', unsafe_allow_html=True)
    st.write("Enter water quality parameters to predict potability")
    
    st.markdown("---")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Parameters")
        ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1, help="Acidity/Alkalinity (WHO: 6.5-8.5)")
        hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 200.0, 10.0)
        solids = st.number_input("Total Solids (ppm)", 0.0, 70000.0, 20000.0, 1000.0)
    
    with col2:
        st.subheader("Chemical Content")
        chloramines = st.number_input("Chloramines (ppm)", 0.0, 15.0, 7.0, 0.5)
        sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 250.0, 10.0)
        conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 10.0)
    
    with col3:
        st.subheader("Organic & Physical")
        organic_carbon = st.number_input("Organic Carbon (ppm)", 0.0, 30.0, 14.0, 1.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, 66.0, 5.0)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 4.0, 0.5)
    
    st.markdown("---")
    
    if st.button("🔍 Predict Water Quality", type="primary"):
        # Create base features
        input_data = {
            'ph': ph,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }
        
        # Add engineered features
        input_data['pH_Hardness'] = ph * hardness
        input_data['Solids_Chloramines_ratio'] = solids / (chloramines + 1)
        input_data['pH_category'] = 0 if ph < 6.5 else (1 if ph <= 8.5 else 2)
        input_data['Total_dissolved'] = solids + sulfate
        input_data['Chlorine_total'] = chloramines + trihalomethanes
        input_data['Sulfate_Conductivity_ratio'] = sulfate / (conductivity + 1)
        input_data['pH_deviation'] = abs(ph - 7.0)
        input_data['Hardness_category'] = 0 if hardness < 75 else (1 if hardness <= 150 else 2)
        input_data['Turbidity_Organic_ratio'] = turbidity / (organic_carbon + 1)
        input_data['pH_squared'] = ph ** 2
        input_data['Solids_log'] = np.log1p(solids)
        
        # Prepare for prediction
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct order
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        # Display result
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="potable">
                    ✅ WATER IS POTABLE<br>
                    Safe for drinking
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="not-potable">
                    ❌ WATER IS NOT POTABLE<br>
                    Not safe for drinking
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{max(probability):.1%}")
            st.metric("Potable Probability", f"{probability[1]:.1%}")
            st.metric("Non-Potable Probability", f"{probability[0]:.1%}")
        
        st.markdown("---")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Potability Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_column_width=True)
        
        # SHAP explanation
        st.markdown("---")
        st.subheader("🧠 Prediction Explanation (SHAP)")
        
        try:
            if 'RandomForest' in type(model).__name__ or 'XGB' in type(model).__name__:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                st.info("SHAP explanation available for tree-based models only")
                shap_values = None
            
            if shap_values is not None:
                # Feature contribution
                feature_contrib = pd.DataFrame({
                    'Feature': feature_names[:9],  # Show only original features
                    'Value': input_df.iloc[0, :9].values,
                    'Impact': shap_values[0, :9]
                }).sort_values('Impact', key=abs, ascending=False)
                
                fig = px.bar(feature_contrib.head(8), x='Impact', y='Feature', 
                            orientation='h', color='Impact',
                            color_continuous_scale=['red', 'yellow', 'green'],
                            title='Top Feature Contributions to Prediction')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_column_width=True)
                
                st.dataframe(feature_contrib.head(10), hide_index=True)
        except Exception as e:
            st.warning(f"SHAP explanation temporarily unavailable: {str(e)}")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 0:
            issues = []
            if ph < 6.5 or ph > 8.5:
                issues.append(f"⚠️ pH ({ph:.2f}) is outside safe range (6.5-8.5)")
            if turbidity > 5:
                issues.append(f"⚠️ High turbidity ({turbidity:.2f} NTU) - exceeds WHO guideline")
            if chloramines > 4:
                issues.append(f"⚠️ High chloramine levels ({chloramines:.2f} ppm)")
            if solids > 50000:
                issues.append(f"⚠️ Very high total dissolved solids ({solids:.0f} ppm)")
            
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.info("Parameters within normal ranges, but model predicts non-potable. Consider comprehensive laboratory testing.")
        else:
            st.success("✅ All major parameters within acceptable ranges for drinking water.")

# ============================================================================
# PAGE 4: FEATURE INSIGHTS
# ============================================================================
elif page == "🧠 Feature Insights":
    st.markdown('<p class="main-header">🧠 Feature Insights & SHAP Analysis</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Feature Importance")
    try:
        img = Image.open('feature_importance.png')
        st.image(img, use_column_width=True)
    except:
        st.info("Feature importance chart not found.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 SHAP Summary Plot")
        try:
            img = Image.open('shap_summary.png')
            st.image(img, use_column_width=True)
            st.caption("Shows how each feature impacts predictions across all samples")
        except:
            st.info("SHAP summary plot not found.")
    
    with col2:
        st.subheader("📊 SHAP Feature Importance")
        try:
            img = Image.open('shap_bar.png')
            st.image(img, use_column_width=True)
            st.caption("Average absolute impact of each feature on predictions")
        except:
            st.info("SHAP bar plot not found.")
    
    st.markdown("---")
    
    st.subheader("🔗 SHAP Dependence Plots")
    st.write("Showing how individual features affect predictions")
    try:
        img = Image.open('shap_dependence.png')
        st.image(img, use_column_width=True)
    except:
        st.info("SHAP dependence plots not found.")
    
           st.markdown("---")
    
    st.subheader("🔍 Individual Sample Explanations")
    st.write("SHAP plots showing how features contribute to specific predictions")
    
    # Try both force plots and waterfall plots
    force_plot_indices = [0, 10, 50]
    
    for idx in force_plot_indices:
        st.markdown(f"**Sample {idx}**")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                img = Image.open(f'shap_force_{idx}.png')
                st.image(img, use_column_width=True, caption=f"Force Plot - Sample {idx}")
            except:
                st.info(f"Force plot for sample {idx} not available")
        
        with col2:
            try:
                img = Image.open(f'shap_waterfall_{idx}.png')
                st.image(img, use_column_width=True, caption=f"Waterfall Plot - Sample {idx}")
            except:
                st.info(f"Waterfall plot for sample {idx} not available")
        
        st.markdown("---")
    
    st.subheader("📖 Understanding SHAP Values")
    st.info("""
    **What are SHAP values?**
    
    SHAP (SHapley Additive exPlanations) values explain individual predictions by showing:
    - **Red**: Features pushing prediction toward "Potable"
    - **Blue**: Features pushing prediction toward "Not Potable"
    - **Size**: Magnitude of feature's impact
    
    This helps understand WHY the model makes specific predictions, making AI transparent and trustworthy.
    """)

# ============================================================================
# PAGE 5: DATA EXPLORER
# ============================================================================
elif page == "📈 Data Explorer":
    st.markdown('<p class="main-header">📈 Dataset Explorer</p>', unsafe_allow_html=True)
    
    # Load and prepare data
    try:
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        st.error("water_potability.csv not found!")
        st.stop()
    
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Potable Samples", int(df['Potability'].sum()))
    with col4:
        st.metric("Non-Potable Samples", len(df) - int(df['Potability'].sum()))
    
    st.markdown("---")
    
    # Distribution plots
    st.subheader("📊 Feature Distributions")
    
    feature_to_plot = st.selectbox("Select feature to visualize", 
                                    ['ph', 'Hardness', 'Solids', 'Chloramines', 
                                     'Sulfate', 'Conductivity', 'Organic_carbon', 
                                     'Trihalomethanes', 'Turbidity'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=feature_to_plot, color='Potability', 
                          marginal="box", nbins=50,
                          color_discrete_map={0: 'red', 1: 'green'},
                          labels={'Potability': 'Water Quality'},
                          title=f'Distribution of {feature_to_plot}')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_column_width=True)
    
    with col2:
        fig = px.box(df, x='Potability', y=feature_to_plot,
                    color='Potability',
                    color_discrete_map={0: 'red', 1: 'green'},
                    labels={'Potability': 'Water Quality'},
                    title=f'{feature_to_plot} by Potability')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_column_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_column_width=True)
    
    st.markdown("---")
    
    # Statistics table
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())
    
    st.markdown("---")
    
    # Sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(20))
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=csv,
        file_name="water_quality_data.csv",
        mime="text/csv",
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>CA2 Mini Project - Water Quality Classification</strong></p>
    <p>Group 2: Karan Uderani, Vibhuti Sawant, Tejasvini Bachhav</p>
    <p>Advanced Machine Learning | ECEI MDM2</p>
</div>
""", unsafe_allow_html=True)

