import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load('my_first_production_model.pkl')

st.title("🚀 AI Sales Lead Predictor")
st.markdown("**Prioritize customers most likely to buy**")

st.sidebar.header("👤 Customer Profile")
feature_names = [
    'Age Score', 'Monthly Spend', 'Logins/Week', 'Cart Abandons', 'Email Opens',
    'Website Time', 'Page Views', 'Product Views', 'Demo Requests', 'Trial Signups',
    'Social Shares', 'Content Downloads', 'Price Sensitivity', 'Location Score',
    'Referral Source', 'Job Title Score', 'Company Size', 'Past Purchases',
    'Engagement Score', 'Lead Velocity'
]

features = []
for name in feature_names:
    val = st.sidebar.slider(name, 0.0, 1.0, 0.5, 0.05)
    features.append(val)

if st.button("🚀 Predict Purchase Likelihood", type="primary"):
    customer_data = np.array([features])
    prediction = model.predict(customer_data)[0]
    confidence = model.predict_proba(customer_data).max()
    
    if prediction == 1:
        st.success("✅ **HIGH VALUE CUSTOMER**")
        st.balloons()
    else:
        st.warning("❌ **LOW VALUE** - Nurture later")
    
    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{confidence:.1%}")
    col2.metric("Recommendation", "CALL NOW" if prediction == 1 else "EMAIL")

st.markdown("---")
st.caption("💡 Slide traits → Get instant AI sales prioritization")