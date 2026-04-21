import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('my_first_production_model.pkl')

st.title("🚀 My First AI App")
st.write("Upload customer data → Get instant predictions")
st.sidebar.header("Customer Profile")
feature_names = [
    'Age Score', 'Monthly Spend', 'Logins/Week', 'Cart Abandons', 'Email Opens',
    'Website Time', 'Page Views', 'Product Views', 'Demo Requests', 'Trial Signups',
    'Social Shares', 'Content Downloads', 'Price Sensitivity', 'Location Score',
    'Referral Source', 'Job Title Score', 'Company Size', 'Past Purchases',
    'Engagement Score', 'Lead Velocity'
]
features = []
for name in feature_names:
    val = st.sidebar.slider(name, 0.0, 1.0, 0.5)
    features.append(val)
features = []
for i in range(20):
    val = st.sidebar.slider(f"Feature {i+1}", 0.0, 1.0, 0.5)
    features.append(val)

if st.button("🚀 Predict Customer Value"):
    customer_data = np.array([features])
    prediction = model.predict(customer_data)[0]
    confidence = model.predict_proba(customer_data).max()
    
    st.success("✅ HIGH VALUE" if prediction == 1 else "❌ LOW VALUE")
    st.metric("Confidence", f"{confidence:.1%}")
    st.balloons()