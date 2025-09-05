# student-mental-health-prediction
An AI-powered project that predicts student mental health risk using academic, lifestyle, and psychological data. It trains a Random Forest model for classification and deploys a Streamlit web app for real-time predictions, offering insights and wellness suggestions.

This project uses Machine Learning to predict whether a student may be at risk of suicidal thoughts based on academic, lifestyle, and psychological inputs.

It has two main parts:
 1.Model Training – Preprocess dataset, train model, and save files.
 2.Web App – Load the saved model and make predictions using a Streamlit app.

Features
 1.Cleans and preprocesses student dataset.
 2.Trains a Random Forest Classifier to predict suicidal thought risk.
 3.Saves trained model, encoders, and scaler as .pkl files.
 4.Provides an interactive Streamlit app for user input and predictions.

Libraries Used
 1.pandas – Data handling
 2.numpy – Numerical computations
 3.scikit-learn – ML models & preprocessing
 3.joblib – Save & load model files
 4.streamlit – Web app framework

 Use the App
 Fill in your details (Gender, Age, Academic Pressure, CGPA, Sleep Hours, etc.).
 Click Predict Mental Health Status.
 The app will show if the student is at risk of suicidal thoughts or not at risk.
 If risk is detected, it also shows helpful suggestions.

 What is Done in This Project
   1.Data Cleaning → Dropping irrelevant columns, handling missing values, converting ranges into averages.
   2.Feature Encoding → Encoding categorical columns like Gender, Academic Pressure, Dietary Habits.
   3.Scaling → Standardizing numerical features.
   4.Model Training → Training a Random Forest Classifier.
   5.Saving Model → Exporting model, scaler, and encoders for reuse.
   6.Deployment → Building a Streamlit web app to interact with the model.

Disclaimer: This project is for educational purposes only and should not replace professional medical or mental health advice.
