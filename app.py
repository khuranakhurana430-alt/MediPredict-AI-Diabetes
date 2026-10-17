import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('diabetes_model.pkl')
    threshold = joblib.load('diabetes_threshold.pkl')
    features = joblib.load('diabetes_features.pkl')
    return model, threshold, features

model, threshold, features = load_artifacts()

# Sidebar
st.sidebar.title("Diabetes Analytics")
st.sidebar.info("This app uses an optimized XGBoost Pipeline with Median Imputation and SMOTE to predict diabetes risk.")
st.sidebar.markdown("---")
st.sidebar.write(f"**Decision Threshold:** {threshold:.2f}")

st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter the patient's clinical measurements below.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0, 250, 120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age (years)", 21, 100, 30)

if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
    
    # Prepare Input (matching the features list exactly)
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose if glucose > 0 else np.nan, 
        'BloodPressure': blood_pressure if blood_pressure > 0 else np.nan,
        'SkinThickness': skin_thickness if skin_thickness > 0 else np.nan,
        'Insulin': insulin if insulin > 0 else np.nan,
        'BMI': bmi if bmi > 0 else np.nan,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    input_df = pd.DataFrame([input_data])
    
    # The pipeline handles Imputation, Scaling, and Prediction automatically
    proba = model.predict_proba(input_df)[0, 1]
    
    prediction = "DIABETES" if proba >= threshold else "NO DIABETES"

    st.divider()
    
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Prediction Result")
        if proba >= threshold:
            st.error(f"### ⚠️ {prediction}")
            st.write("This patient is **at high risk** for diabetes.")
        else:
            st.success(f"### ✅ {prediction}")
            st.write("This patient is **at low risk** for diabetes.")
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba * 100,
            number = {'suffix': "%"},
            title = {'text': "Diabetes Probability", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': proba * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.subheader("Feature Importance")
        st.write("Which clinical measurements matter most?")
        
        try:
            # Extract feature importance from the pipeline's classifier step
            xgb_model = model.named_steps['classifier']
            importances = xgb_model.feature_importances_
            
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(8)
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Teal',
                title="Feature Importance"
            )
            fig_fi.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load chart: {e}")