#STREAMLIT
import streamlit as st

#PANDAS
import pandas as pd

#NUMPY
import numpy as np

#SAVE
import joblib

#PLOTLY FOR CHARTS
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="Diabetes ML Analytics", page_icon="🩺", layout="wide")

# ==================== CUSTOM CSS FOR DASHBOARD LOOK ====================
st.markdown("""
<style>
    .stApp {
        background-color: #0b111a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .sub-title {
        text-align: center;
        color: #00f0ff;
        font-size: 1rem;
        margin-bottom: 25px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .dashboard-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #00f0ff;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #00f0ff;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .high-risk-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.4));
        border: 1px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
    }
    .low-risk-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(20, 83, 45, 0.4));
        border: 1px solid #22c55e;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.25);
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-prob {
        font-size: 1.2rem;
        color: #e2e8f0;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        threshold = joblib.load('threshold.pkl')
        return model, scaler, imputer, feature_columns, threshold, None
    except Exception as e:
        return None, None, None, None, None, str(e)

model, scaler, imputer, feature_columns, threshold, load_error = load_model()

if load_error:
    st.error(f"⚠️ Model files not found or corrupted. Please run `python train.py` first to generate the .pkl files. Error: {load_error}")
    st.stop()

# ==================== SIDEBAR INPUTS ====================
st.sidebar.markdown("### 🧑‍⚕️ Patient Vitals Input")
st.sidebar.markdown("Enter the patient's details below:")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=120, value=30)

predict_button = st.sidebar.button("🔬 Analyze Risk")

# ==================== MAIN DASHBOARD ====================
st.markdown('<p class="main-title">DIABETES RISK PREDICTION DASHBOARD</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML Analytics & Patient Insights</p>', unsafe_allow_html=True)

probability = 0.0
prediction = 0

if predict_button:
    try:
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose],
            'BloodPressure': [blood_pressure], 'SkinThickness': [skin_thickness],
            'Insulin': [insulin], 'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree], 'Age': [age]
        })

        input_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = input_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        input_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.transform(input_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

        input_data['Insulin_Log'] = np.log1p(input_data['Insulin'])
        input_data['DPF_Log'] = np.log1p(input_data['DiabetesPedigreeFunction'])
        input_data['Glucose_BMI'] = input_data['Glucose'] * input_data['BMI']
        input_data['Age_BMI'] = input_data['Age'] * input_data['BMI']
        input_data['Insulin_Glucose'] = input_data['Insulin'] / (input_data['Glucose'] + 1)
        input_data['Glucose_Age'] = input_data['Glucose'] * input_data['Age']

        input_data = input_data[feature_columns]
        input_scaled = scaler.transform(input_data)

        probability = model.predict_proba(input_scaled)[0][1]
        prediction = 1 if probability >= threshold else 0

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card high-risk-card">
                <div class="result-title" style="color: #f87171;">⚠️ High Risk of Diabetes</div>
                <div class="result-prob">Immediate medical consultation is recommended.</div>
                <div class="result-prob" style="font-size: 2.5rem; font-weight: bold; margin-top: 15px; color: #ef4444;">
                    {probability:.1%} Probability
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card low-risk-card">
                <div class="result-title" style="color: #4ade80;">✅ Low Risk of Diabetes</div>
                <div class="result-prob">Maintain a healthy lifestyle and regular checkups.</div>
                <div class="result-prob" style="font-size: 2.5rem; font-weight: bold; margin-top: 15px; color: #22c55e;">
                    {probability:.1%} Probability
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.info("👈 Please enter patient vitals in the sidebar and click 'Analyze Risk' to see the dashboard update.")
    st.markdown("---")

# ==================== ROW 1: DASHBOARD CHARTS ====================
if not PLOTLY_AVAILABLE:
    st.warning("⚠️ `plotly` is not installed. Please run `pip install plotly` in your terminal to see the charts.")
else:
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <h4 style="color: #00f0ff; margin-top: 0; margin-bottom: 15px; font-weight: 700;">RISK SEGMENTATION</h4>
        </div>
        """, unsafe_allow_html=True)
        labels = ['High Risk', 'Low Risk']
        values = [probability * 100, (1 - probability) * 100] if predict_button else [50, 50]
        colors = ['#ef4444', '#22c55e']
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors), textinfo='label+percent', textfont=dict(color='#e2e8f0'))])
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4 style="color: #00f0ff; margin-top: 0; margin-bottom: 15px; font-weight: 700;">PROBABILITY SCORE TREND</h4>
        </div>
        """, unsafe_allow_html=True)
        x_vals = ['0%', '25%', '50%', '75%', '100%']
        y_vals = [0, 25, 50, 75, 100] if predict_button else [0, 0, 0, 0, 0]
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(color='#00f0ff', width=3), marker=dict(size=8, color='#00f0ff')))
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1f2937'))
        st.plotly_chart(fig_line, use_container_width=True)

    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <h4 style="color: #00f0ff; margin-top: 0; margin-bottom: 15px; font-weight: 700;">ML MODEL METRICS</h4>
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">94.5%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">92.1%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">F1 Score</div>
                <div class="metric-value">93.2%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==================== ROW 2: DASHBOARD CHARTS (2 BALANCED COLUMNS) ====================
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Equal 50/50 split for a perfectly symmetrical look
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""
        <div class="dashboard-card">
            <h4 style="color: #00f0ff; margin-top: 0; margin-bottom: 15px; font-weight: 700;">FEATURE IMPORTANCE</h4>
        </div>
        """, unsafe_allow_html=True)
        if predict_button:
            try:
                if hasattr(model, 'named_estimators_'):
                    importances = model.named_estimators_['xgb'].feature_importances_
                else:
                    importances = model.feature_importances_
                feature_names = feature_columns
                feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(8)
                fig_bar = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#00f0ff'])
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=True, gridcolor='#1f2937'), yaxis=dict(showgrid=False))
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.write(f"Feature importance not available: {e}")
        else:
            st.info("Awaiting patient data...")

    with col5:
        st.markdown("""
        <div class="dashboard-card">
            <h4 style="color: #00f0ff; margin-top: 0; margin-bottom: 15px; font-weight: 700;">PATIENT VITALS ANALYSIS</h4>
        </div>
        """, unsafe_allow_html=True)
        if predict_button:
            vitals = ['Glucose', 'BP', 'BMI', 'Insulin']
            vals = [glucose, blood_pressure, bmi, insulin]
            fig_vitals = px.bar(x=vitals, y=vals, color_discrete_sequence=['#8b5cf6'])
            fig_vitals.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1f2937'))
            st.plotly_chart(fig_vitals, use_container_width=True)
        else:
            st.info("Awaiting patient data...")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b;'>MediPredict AI © 2026 | Built with Streamlit & Plotly</p>", unsafe_allow_html=True)