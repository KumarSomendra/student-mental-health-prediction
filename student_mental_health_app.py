import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model components
model = joblib.load("mental_health_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title(" Student Mental Wellness Prediction")

st.write("""
This AI assistant predicts whether a student is likely experiencing mental health challenges 
based on academic, lifestyle, and psychological inputs.
""")

# Get label options from encoders
gender_options = list(label_encoders['Gender'].classes_)
academic_options = list(label_encoders['Academic Pressure'].classes_)
dietary_options = list(label_encoders['Dietary Habits'].classes_)

# Collect user input
gender = st.selectbox("Gender", gender_options)
age = st.number_input("Age", min_value=15, max_value=35, value=20)
academic_pressure = st.selectbox("Academic Pressure", academic_options)
work_pressure = st.slider("Work Pressure (1 - Low, 5 - High)", 1, 5, 3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
study_satisfaction = st.slider("Study Satisfaction (1 - Low, 5 - High)", 1, 5, 3)
job_satisfaction = st.slider("Job Satisfaction (1 - Low, 5 - High)", 1, 5, 3)
sleep_duration = st.slider("Sleep Duration (Hours)", 0.0, 12.0, 6.0)
dietary_habits = st.selectbox("Dietary Habits", dietary_options)
work_study_hours = st.slider("Work/Study Hours", 0.0, 18.0, 6.0)
financial_stress = st.slider("Financial Stress (1 - Low, 5 - High)", 1, 5, 3)

# Prepare input dictionary
input_dict = {
    'Gender': gender,
    'Age': age,
    'Academic Pressure': academic_pressure,
    'Work Pressure': work_pressure,
    'CGPA': cgpa,
    'Study Satisfaction': study_satisfaction,
    'Job Satisfaction': job_satisfaction,
    'Sleep Duration': sleep_duration,
    'Dietary Habits': dietary_habits,
    'Work/Study Hours': work_study_hours,
    'Financial Stress': financial_stress
}

input_df = pd.DataFrame([input_dict])

# Apply label encoders
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform([input_df[col][0]])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Mental Health Status"):
    prediction = model.predict(input_scaled)[0]
    result = "⚠️ At Risk of Suicidal Thoughts" if prediction == 1 else "✅ No Suicidal Thoughts Detected"

    st.subheader("Prediction Result:")
    st.markdown(f"### {result}")

    if prediction == 1:
        st.info("💡 Suggestions from AI Assistant:")
        st.markdown("""
        - Try to talk to a trusted friend, mentor, or counselor.
        - Consider taking small breaks and structuring your study schedule better.
        - Practice mindfulness and journaling your thoughts.
        - You are not alone — help is always available.
        """)
    else:
        st.success("Keep maintaining your wellness! 😊 Stay consistent with healthy habits.")

