import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AutoWorth", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("autoworth_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("clean_v3.csv")

model = load_model()
df = load_data()
TYPICAL_ERROR = 100000  # about the model's test RMSE (Rs 1 lakh)

st.title("AutoWorth: Used Car Fair-Price Estimator")
st.write("Enter the car details to get an estimated fair price.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(df["brand"].unique()))
    fuel = st.selectbox("Fuel", sorted(df["fuel"].unique()))
    transmission = st.selectbox("Transmission", sorted(df["transmission"].unique()))
    seller_type = st.selectbox("Seller type", sorted(df["seller_type"].unique()))
    owner = st.selectbox("Owner", sorted(df["owner"].unique()))
    seat_options = sorted(df["seats"].unique().astype(int))
    seats = st.selectbox("Seats", seat_options, index=seat_options.index(5))
with col2:
    year = st.number_input("Year", 1995, 2026, 2015)
    km = st.number_input("Km driven", 0, 500000, 60000, step=5000)
    engine = st.number_input("Engine (CC)", 600, 3500, 1200, step=50)
    power = st.number_input("Max power (bhp)", 30.0, 400.0, 80.0, step=1.0)
    mileage = st.number_input("Mileage (kmpl; km/kg for CNG/LPG)", 5.0, 40.0, 18.0, step=0.5)

def make_row(km_value):
    return {
        "car_age": 2026 - year, "km_driven": km_value, "mileage": mileage,
        "engine": engine, "max_power": power, "seats": float(seats),
        "fuel": fuel, "seller_type": seller_type, "transmission": transmission,
        "owner": owner, "brand": brand,
    }

if st.button("Estimate price"):
    price = float(model.predict(pd.DataFrame([make_row(km)]))[0])
    price = max(price, 0)
    st.subheader(f"Estimated price: Rs {price:,.0f}")
    st.write(
        f"Typical error is about Rs {TYPICAL_ERROR:,.0f}, so a likely range is "
        f"Rs {max(price - TYPICAL_ERROR, 0):,.0f} to Rs {price + TYPICAL_ERROR:,.0f}. "
        "This is a rough guide, not a guaranteed interval."
    )

    # Value lost per extra 25,000 km: average slope over the 0-200,000 km curve
    kms = np.arange(0, 200001, 5000)
    curve = model.predict(pd.DataFrame([make_row(k) for k in kms]))
    slope = np.polyfit(kms, curve, 1)[0]  # rupees per km
    drop = max(-slope * 25000, 0)
    st.info(f"Every extra 25,000 km lowers this car's value by about Rs {drop:,.0f} (average over 0 to 200,000 km).")

st.caption("Trained on CarDekho listings (2020 and earlier). Estimates are statistical guidance, not a valuation or a guarantee.")
