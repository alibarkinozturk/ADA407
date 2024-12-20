# -*- coding: utf-8 -*-
"""
Bank Marketing Prediction Web App
"""
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load model and sclaer
loaded_model = pickle.load(open('bank_marketing_predictor.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Label encoding
def categorize_month(month):
    if month in ['January', 'February', 'March']:
        return 'q1'
    elif month in ['April', 'May', 'June']:
        return 'q2'
    elif month in ['July', 'August', 'September']:
        return 'q3'
    elif month in ['October', 'November', 'December']:
        return 'q4'

job_encoder = {
    'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 
    'management': 4, 'retired': 6, 'self-employed': 7, 'services': 8, 
    'student': 9, 'technician': 10, 'unemployed': 11, 'unknown': 5
}

marital_encoder = {
    'divorced': 0, 'married': 1, 'single': 3, 'unknown': 2
}

education_encoder = {
    'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 
    'illiterate': 4, 'unknown': 5, 'professional.course': 6, 'university.degree': 7
}

contact_encoder = {
    'cellular': 0, 'telephone': 1
}

quarter_encoder = {
    'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3
}

# Parameters send to Model for prediction
def bank_subscription_prediction(input_data):
    
    # Change input to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape data to 1D
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Parameters send to Model for prediction
    prediction = loaded_model.predict(input_data_reshaped)
    prediction_proba = loaded_model.predict_proba(input_data_reshaped)[0][1]  # Probability of subscribing
    return prediction, prediction_proba


def main():
    # Header
    st.title('Bank Marketing Prediction Web App')
    st.subheader('Predict whether customers will subscribe for fixed deposit account or not.')

    # User Input
    age = st.number_input('Age', min_value=18, max_value=100, value=30,step=1)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                               'management', 'retired', 'self-employed', 'services', 
                               'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital', ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox('Education', ['basic.9y', 'high.school', 'university.degree', 'professional.course', 'basic.6y', 'basic.4y', 'unknown', 'illiterate'])
    default = st.radio('Has credit in default?', ['No', 'Yes'])
    housing = st.radio('Has housing loan?', ['No', 'Yes'])
    loan = st.radio('Has personal loan?', ['No', 'Yes'])
    contact = st.selectbox('Contact communication type', ['cellular', 'telephone', 'unknown'])
    month = st.selectbox('Last contact month of year ', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    campaign = st.number_input('Number of contacts performed during this campaign and for this client', min_value=0, max_value=50, value=0)
    pdays = st.number_input('Number of days that passed by after the client was last contacted from a previous campaign ', min_value=-1, max_value=999, value=0)
    previous = st.number_input('Number of contacts performed before this campaign and for this client', min_value=0, max_value=50, value=0)
    emp_var_rate = st.number_input('emp_var_rate', min_value=-4.0, max_value=2.0, value=0.0, step=0.1)
    cons_price_idx = st.number_input('cons_price_idx', min_value=90.0, max_value=95.0, value=93.0, step=0.01)
    cons_conf_idx = st.number_input('cons_conf_idx', min_value=-50.0, max_value=-20.0, value=-30.0, step=0.1)
    euribor3m = st.number_input('euribor3m', min_value=0.0, max_value=6.0, value=1.0, step=0.001)
    nr_employed = st.number_input('nr_employed', min_value=4900.0, max_value=5300.0, value=5200.0, step=0.1)

    # Binary encoding binary features
    default = 1 if default == 'Yes' else 0
    housing = 1 if housing == 'Yes' else 0
    loan = 1 if loan == 'Yes' else 0 
    quarter = categorize_month(month)

    # Label encoding for categorical features
    job_encoded = job_encoder[job]
    marital_encoded = marital_encoder[marital]
    education_encoded = education_encoder[education]
    contact_encoded = contact_encoder[contact]
    quarter_encoded = quarter_encoder[quarter]

    # Numeric features add to numpy array for using in scaling
    numeric_features = np.array([
        float(age), 
        float(campaign),
        float(pdays),
        float(previous),
        float(emp_var_rate),
        float(cons_price_idx),
        float(cons_conf_idx),
        float(euribor3m),
        float(nr_employed)
    ], dtype=np.float64)


    numeric_features_reshaped = numeric_features.reshape(1, -1)
    st.write(numeric_features_reshaped)
    scaled_features = scaler.transform(numeric_features_reshaped)
    

    # input parameters of model: Combine scaled features and the additional features into a list
    input_features = [
        scaled_features[0, 0],  
        scaled_features[0, 1],
        scaled_features[0, 5],
        scaled_features[0, 6],
        scaled_features[0, 7],
        scaled_features[0, 8],
        float(default),
        float(housing),
        float(loan),
        float(contact_encoded)
    ]

    st.write(input_features)

    diagnosis = ''
    probability = 0.0

    if st.button('Predict'):
        # call predict function
        prediction, probability = bank_subscription_prediction(input_features)

        # change predict value to text
        if prediction == 1:
            diagnosis = "The customer will subscribe to a term deposit account."
        else:
            diagnosis = "The customer will NOT subscribe to a term deposit account."

    st.success(diagnosis)

# Start UI
if __name__ == '__main__':
    main()
