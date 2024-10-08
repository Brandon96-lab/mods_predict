
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

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Parameters")
    
    age = st.slider("Age", 18, 100, 50)
    weight = st.slider("Weight (kg)", 40, 200, 70)
    invasive_line = st.selectbox("Invasive Line", ("No", "Yes"))
    vasopressor = st.selectbox("Vasopressor Used", ("No", "Yes"))
    saps_ii = st.slider("SAPS II Score", 0, 163, 30)
    chest_ais = st.slider("Max Chest AIS", 0, 6, 2)
    mech_ventilation = st.selectbox("Mechanical Ventilation", ("No", "Yes"))
    platelet_count = st.slider("Platelet Count (x10^9/L)", 0, 1000, 200)

    if st.button("Predict", key="predict"):
        # Prepare the input data
        input_data = pd.DataFrame([[age, weight, invasive_line, vasopressor, 
                                    saps_ii, chest_ais, mech_ventilation, platelet_count]],
                                  columns=['Age', 'Weight', 'Invasive Line', 'Vasopressor',
                                           'SAPS II', 'Chest AIS', 'Mechanical Ventilation', 'Platelet Count'])
        
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
            
            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            shap.summary_plot(shap_values[1], input_data, plot_type="bar", show=False)
            ax.set_xlabel("SHAP Value (impact on model output)")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
            plt.close(fig)

            # Create optimized SHAP force plot
            st.subheader("SHAP Force Plot")
            
            # Increase the figure size and DPI for better quality
            fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
            
            # Use shap.plots.force() for a more modern and customizable force plot
            shap.plots.force(shap_values[1][0], 
                             feature_names=input_data.columns, 
                             matplotlib=True, 
                             show=False, 
                             plot_cmap="RdBu")
            
            # Customize the plot
            plt.title("SHAP Force Plot", fontsize=14)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            plt.close(fig)

# Disclaimer (at the bottom)
st.markdown("---")
st.warning("""
**DISCLAIMER:**

This online calculator is freely accessible and utilizes an advanced random forest algorithm for predicting Multiple Organ Dysfunction Syndrome (MODS) in trauma patients with sepsis. While the model has demonstrated good performance in validation studies, it is crucial to emphasize that this tool was developed solely for research purposes.

Key points to consider:
- The model's predictions should not be the sole basis for clinical decisions.
- This tool is intended to complement, not replace, the expertise and judgment of healthcare professionals.
- Clinical applicability requires further prospective validation.
- Always consult with qualified medical practitioners for diagnosis, treatment, and care decisions.

Remember: This AI platform is an aid to clinical decision-making, not a substitute for professional medical advice, diagnosis, or treatment.
""")

# Footer
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2024 MODS Prediction Model | For Research Purposes Only</p>
</div>
""", unsafe_allow_html=True)
