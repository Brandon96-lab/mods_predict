import numpy as np  
import pandas as pd  
import streamlit as st  
import joblib  
import shap

# Load the model and training data
@st.cache_resource
def load_model():
    return joblib.load('gramtot_rf_model.pkl')

@st.cache_data
def load_training_data():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    return X_train, y_train

model = load_model()
X_train, y_train = load_training_data()

st.header("Prediction of Gram-Negative Sepsis in Trauma Patients")

st.sidebar.title("Patient Parameters")

# Create input fields for each feature
admission_age = st.sidebar.slider("Age", 18, 100)
weight = st.sidebar.slider("Weight (kg)", 40, 200)
invasive_line_1stday = st.sidebar.selectbox("Invasive Line", ("No", "Yes"))
vaso_1stday = st.sidebar.selectbox("Vasopressor Used", ("No", "Yes"))
sapsii_1stday = st.sidebar.slider("SAPS II Score", 0, 163)
mxaisbr_Chest = st.sidebar.slider("Max Chest AIS", 0, 6)
mechvent = st.sidebar.selectbox("Mechanical Ventilation", ("No", "Yes"))
platelets_min = st.sidebar.slider("Platelet Count (x10^3/µL)", 0, 1000)

if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame([[admission_age, weight, invasive_line_1stday, vaso_1stday, 
                                sapsii_1stday, mxaisbr_Chest, mechvent, platelets_min]],
                              columns=['admission_age', 'weight', 'invasive_line_1stday', 'vaso_1stday',
                                       'sapsii_1stday', 'mxaisbr_Chest', 'mechvent', 'platelets_min'])
    
    # Convert categorical variables
    input_data = input_data.replace({"No": 0, "Yes": 1})
    
    # Make prediction
    prediction = model.predict_proba(input_data)[0, 1]
    
    # Display prediction
    st.subheader("Prediction Result")
    st.text(f"Probability of Gram-Negative Sepsis: {prediction:.2%}")
    
    if prediction < 0.5:
        st.success("Low Risk of Gram-Negative Sepsis")
    else:
        st.error("High Risk of Gram-Negative Sepsis")
    
    # SHAP explanation
    st.subheader("Model Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    st.pyplot(shap.summary_plot(shap_values[1], input_data, plot_type="bar"))
    st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_data.iloc[0], matplotlib=True))

st.sidebar.markdown("---")
st.sidebar.text("Note: This app is for research purposes only.")
