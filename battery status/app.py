import streamlit as st
import numpy as np
import joblib
from PIL import Image

model = joblib.load("battery_life_predictor.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="EV Battery State Predictor", page_icon="🔋", layout="centered")


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["🔍 Predict", "ℹ️ About"])


if selection == "🔍 Predict":
    st.title("🔋 Predicting Battery States in Electric Vehicles")
    st.markdown("Enter the battery feature values to predict the **Cycle Life**.")

    with st.form("prediction_form"):
        st.markdown("🔧 Enter Battery State Features:")

        f1 = st.number_input("🔌 IR (Internal Resistance)", value=0.01)
        f2 = st.number_input("⚡ QC (Charge Capacity)", value=1.0)
        f3 = st.number_input("🔋 QD (Discharge Capacity)", value=1.0)
        f4 = st.number_input("🌡️ Tavg (Average Temperature)", value=30.0)
        f5 = st.number_input("🌡️ Tmin (Min Temperature)", value=28.0)
        f6 = st.number_input("🔁 cycle_no (Cycle Number)", value=1)
        f7 = st.number_input("⏱️ chargetime (Charge Time)", value=13.0)

        submitted = st.form_submit_button("Predict")


    if submitted:
        user_input = np.array([[f1, f2, f3, f4, f5, f6, f7]])

        if np.all(user_input == 0):
            st.warning("⚠️ Please provide meaningful values. All inputs are currently zero.")
        else:
            prediction = model.predict(user_input)[0]
            st.success(f"🔮 Predicted Cycle Life: **{prediction:.2f}**")


elif selection == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This application uses a **machine learning model** trained on battery data to estimate the **Cycle Life** of electric vehicle (EV) batteries.

    - ✅ Built with Random Forest / XGBoost / Gradient Boosting
    - 📈 Predicts battery lifespan based on input characteristics
    - 💡 Powered by data-driven digital twin concepts

    **Model:** battery_life_predictor.pkl
    """)



