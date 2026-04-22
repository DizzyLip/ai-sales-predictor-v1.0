import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title='AI Sales Lead Predictor', initial_sidebar_state='expanded')
st.title('🚀 AI Sales Lead Predictor')
st.markdown('**Prioritize customers most likely to buy**')

@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline_small.pkl')

pipeline = load_pipeline()

st.subheader('Company Profile')
col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input('Company Name', placeholder='Apple')
    country = st.text_input('Country', placeholder='US')
    employee_count = st.number_input('Employee Count', min_value=1, step=1, value=5000)
    revenue = st.number_input('Revenue', min_value=0.0, step=1000.0, value=500000.0)
with col2:
    region = st.selectbox('Region', ['North America','Europe','Asia Pacific','Latin America','Middle East/Africa'])
    exchange = st.text_input('Exchange', placeholder='NASDAQ')
    industry = st.selectbox('Industry', ['Tech','Finance','Retail','Healthcare','Manufacturing','Energy','Telecom','Other'])
    company_size_band = st.selectbox('Company Size Band', ['Small','Mid','Large','Enterprise'])

website_visits = st.slider('Website Visits', 0.0, 5000.0, 500.0, 10.0)
engagement_score = st.slider('Engagement Score', 0.0, 1.0, 0.5, 0.01)
lead_velocity = st.slider('Lead Velocity', 0.0, 1.0, 0.5, 0.01)

if st.button('Predict Purchase Likelihood', type='primary'):
    row = pd.DataFrame([{
        'country': country,
        'region': region,
        'industry': industry,
        'company_size_band': company_size_band,
        'employee_count': employee_count,
        'revenue': revenue,
        'website_visits': website_visits,
        'engagement_score': engagement_score,
        'lead_velocity': lead_velocity,
    }])
    proba = pipeline.predict_proba(row)[0][1]
    pred = int(proba >= 0.5)

    st.subheader('Prediction')
    st.write(f'**Company:** {company_name}')
    st.write(f'**Country:** {country}')
    st.write(f'**Region:** {region}')
    st.write(f'**Industry:** {industry}')
    st.write(f'**Exchange:** {exchange}')

    if pred == 1:
        st.success(f'✅ HIGH VALUE CUSTOMER — {proba:.1%}')
        st.balloons()
    else:
        st.warning(f'❌ LOWER PRIORITY — {proba:.1%}')

    c1, c2 = st.columns(2)
    c1.metric('Confidence', f'{proba:.1%}')
    c2.metric('Recommendation', 'CALL NOW' if pred == 1 else 'EMAIL')

st.markdown('---')
st.caption('💡 Global-market pipeline with encoded categorical features')