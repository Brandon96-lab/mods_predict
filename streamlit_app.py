
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="MODS Prediction in Trauma Patients with Sepsis", page_icon="🏥", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #0e4c92;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

model = load_model()

# Main header
st.title("🏥 Prediction of MODS in Trauma Patients with Sepsis")

# Disclaimer
st.warning("""
**DISCLAIMER:**
This model is intended for research purposes only. Its clinical applicability requires prospective validation. 
Use caution when considering this for clinical decision-making. This tool is intended to complement, not replace, the expertise and judgment of healthcare professionals. 
""")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Parameters")
    
    admission_age = st.slider("Age", 18, 100, 50)
    weight = st.slider("Weight (kg)", 40, 200, 70)
    invasive_line_1stday = st.selectbox("Invasive Line", ("No", "Yes"))
    vaso_1stday = st.selectbox("Vasopressor Used", ("No", "Yes"))
    sapsii_1stday = st.slider("SAPS II Score", 0, 163, 30)
    mxaisbr_Chest = st.slider("Max Chest AIS", 0, 6, 2)
    mechvent = st.selectbox("Mechanical Ventilation", ("No", "Yes"))
    platelets_min = st.slider("Platelet Count (x10^9/L)", 0, 1000, 200)

    if st.button("Predict", key="predict"):
        # Prepare the input data
        input_data = pd.DataFrame([[admission_age, weight, invasive_line_1stday, vaso_1stday, 
                                    sapsii_1stday, mxaisbr_Chest, mechvent, platelets_min]],
                                  columns=['admission_age', 'weight', 'invasive_line_1stday', 'vaso_1stday',
                                           'sapsii_1stday', 'mxaisbr_Chest', 'mechvent', 'platelets_min'])
        
        # Convert categorical variables
        input_data = input_data.replace({"No": 0, "Yes": 1})
        
        # Make prediction
        prediction = model.predict_proba(input_data)[0, 1]
        
        # Display prediction in the second column
        with col2:
            st.subheader("Prediction Result")
            st.markdown(f"<h3 style='text-align: center;'>Probability of MODS: <span style='color: {'red' if prediction >= 0.5 else 'green'};'>{prediction:.2%}</span></h3>", unsafe_allow_html=True)
            
            if prediction < 0.1:
                st.success("Low Risk of MODS")
            elif prediction < 0.5:
                st.warning("Moderate Risk of MODS")
            else:
                st.error("High Risk of MODS")
            
            # SHAP explanation
            st.subheader("Model Explanation")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
            shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
            plt.sca(ax1)
            plt.title("Feature Importance")
            shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_data.iloc[0], matplotlib=True, show=False)
            plt.sca(ax2)
            plt.title("SHAP Force Plot")
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2024 MODS Prediction Model | For Research Purposes Only</p>
</div>
""", unsafe_allow_html=True)
