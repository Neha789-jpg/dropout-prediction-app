# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.title("Student Dropout Prediction App")

# Load CatBoost model
model = joblib.load("catboost_dropout_model.pkl")

st.write("""
Upload a CSV file with the student data to predict dropouts.
The CSV should have the same columns/features used during training.
""")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file, sep=None, engine='python')  # Detect separator automatically
        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # ======================
        # Feature Engineering
        # ======================
        # Adjust these according to your training features
        # Example based on your previous CatBoost code:
        df["failed_1st"] = df["Curricular units 1st sem (evaluations)"] - df["Curricular units 1st sem (approved)"]
        df["failed_2nd"] = df["Curricular units 2nd sem (evaluations)"] - df["Curricular units 2nd sem (approved)"]

        df["total_enrolled"] = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
        df["total_approved"] = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
        df["total_failed"] = df["failed_1st"] + df["failed_2nd"]
        df["total_passed"] = df["total_approved"]
        df["avg_grade"] = (df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]) / 2

        df["1st_sem_pass_ratio"] = df["Curricular units 1st sem (approved)"] / (df["Curricular units 1st sem (enrolled)"] + 1)
        df["2nd_sem_pass_ratio"] = df["Curricular units 2nd sem (approved)"] / (df["Curricular units 2nd sem (enrolled)"] + 1)
        df["total_pass_ratio"] = df["total_passed"] / (df["total_enrolled"] + 1)
        df["fail_ratio"] = df["total_failed"] / (df["total_enrolled"] + 1)
        df["risk_score"] = df["fail_ratio"] * 3 + (1 - df["total_pass_ratio"]) * 2 + (5 - df["avg_grade"]) * 0.5

        # ======================
        # Make Predictions
        # ======================
        predictions = model.predict(df)

        # Map predictions if necessary
        # Example: 0 = Not Dropout, 1 = Dropout
        pred_labels = ["Not Dropout" if p == 0 else "Dropout" for p in predictions]
        df["Prediction"] = pred_labels

        st.subheader("Predictions")
        st.dataframe(df[["Prediction"] + [col for col in df.columns if col != "Prediction"]])

        # Simple chart
        st.subheader("Prediction Summary")
        st.bar_chart(df["Prediction"].value_counts())

    except Exception as e:
        st.error(f"Error processing file: {e}")
