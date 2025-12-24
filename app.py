import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import warnings
warnings.filterwarnings("ignore")

# ===== 1. LOAD MODEL =====
st.write(f"Using sklearn version: {sklearn.__version__}")

try:
    model = joblib.load("telco_churn_best_model.pkl")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# ===== 2. JUDUL HALAMAN =====
st.markdown(
    "<h1 style='text-align: center;'>📊 Telco Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.write("")
st.write(
    "Aplikasi ini memprediksi kemungkinan pelanggan berhenti (churn) "
    "berdasarkan data pelanggan Telco."
)

st.write("---")
st.subheader("🧾 Input Data Pelanggan")

# ===== 3. FORM INPUT (2 kolom) =====
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=1, step=1)
    monthly_charges = st.number_input(
        "MonthlyCharges", min_value=0.0, max_value=10000.0,
        value=70.0, step=1.0
    )
    # fitur layanan online
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ]
    )
    total_charges = st.number_input(
        "TotalCharges", min_value=0.0, max_value=100000.0,
        value=800.0, step=10.0
    )
    # fitur tambahan lain
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

st.write("")
predict_btn = st.button("🔮 Prediksi Churn")

# ===== 4. PREDIKSI =====
if predict_btn:
    # DataFrame 1 baris – nama kolom HARUS sama dengan saat training
    data = {
        "gender": [gender],
        "SeniorCitizen": [int(senior)],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [float(tenure)],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "PaperlessBilling": [paperless_billing],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)],
    }
    input_df = pd.DataFrame(data)

    try:
        prob_churn = model.predict_proba(input_df)[0][1]
        pred_class = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.stop()

    st.write("---")
    st.subheader("📌 Hasil Prediksi")

    if pred_class == 1:
        st.error("🚨 Pelanggan DIPREDIKSI **CHURN**")
    else:
        st.success("✅ Pelanggan DIPREDIKSI **TIDAK CHURN**")

    st.write("")
    st.write(f"**Probabilitas Churn:** {prob_churn*100:.2f}%")
    st.progress(int(prob_churn * 100))

    st.write("")
    st.subheader("📘 Penjelasan Fitur (singkat)")
    st.markdown(
        """
        - **Tenure**: Lama berlangganan (bulan).  
        - **MonthlyCharges**: Tagihan bulanan.  
        - **TotalCharges**: Total tagihan sejak awal berlangganan.  
        - **Contract**: Jenis kontrak pelanggan (bulanan / tahunan).
        """
    )
