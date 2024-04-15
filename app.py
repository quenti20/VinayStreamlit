import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model('')

# Function to preprocess the user input
def preprocess_input(input_data): 
    # Convert categorical variables to binary encoding
    for var in categorical_variables:
        input_data[var] = input_data[var].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Encode categorical variables
    for var in categorical_variables:
        le = LabelEncoder()
        input_data[var] = le.fit_transform(input_data[var])
    
    return input_data

# Streamlit app
def main():
    st.title('Thyroid Prediction App')
    
    # Collect user input features
    st.sidebar.header('User Input Features')

    # Numerical variables
    st.sidebar.subheader('Numerical Variables')
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=50)
    tsh = st.sidebar.number_input('TSH', min_value=0.0, max_value=100.0, value=10.0)
    t3 = st.sidebar.number_input('T3', min_value=0.0, max_value=100.0, value=50.0)
    tt4 = st.sidebar.number_input('TT4', min_value=0.0, max_value=100.0, value=70.0)
    t4u = st.sidebar.number_input('T4U', min_value=0.0, max_value=100.0, value=20.0)
    fti = st.sidebar.number_input('FTI', min_value=0.0, max_value=100.0, value=50.0)

    # Categorical variables
    st.sidebar.subheader('Categorical Variables')
    on_thyroxine = st.sidebar.radio('On Thyroxine', ['Yes', 'No'])
    query_on_thyroxine = st.sidebar.radio('Query On Thyroxine', ['Yes', 'No'])
    on_antithyroid_medication = st.sidebar.radio('On Antithyroid Medication', ['Yes', 'No'])
    sick = st.sidebar.radio('Sick', ['Yes', 'No'])
    pregnant = st.sidebar.radio('Pregnant', ['Yes', 'No'])
    thyroid_surgery = st.sidebar.radio('Thyroid Surgery', ['Yes', 'No'])
    i131_treatment = st.sidebar.radio('I131 Treatment', ['Yes', 'No'])
    query_hypothyroid = st.sidebar.radio('Query Hypothyroid', ['Yes', 'No'])
    query_hyperthyroid = st.sidebar.radio('Query Hyperthyroid', ['Yes', 'No'])
    lithium = st.sidebar.radio('Lithium', ['Yes', 'No'])
    goitre = st.sidebar.radio('Goitre', ['Yes', 'No'])
    tumor = st.sidebar.radio('Tumor', ['Yes', 'No'])
    hypopituitary = st.sidebar.radio('Hypopituitary', ['Yes', 'No'])
    psych = st.sidebar.radio('Psych', ['Yes', 'No'])
    tsh_measured = st.sidebar.radio('TSH Measured', ['Yes', 'No'])
    t3_measured = st.sidebar.radio('T3 Measured', ['Yes', 'No'])
    tt4_measured = st.sidebar.radio('TT4 Measured', ['Yes', 'No'])
    t4u_measured = st.sidebar.radio('T4U Measured', ['Yes', 'No'])
    fti_measured = st.sidebar.radio('FTI Measured', ['Yes', 'No'])
    tbg_measured = st.sidebar.radio('TBG Measured', ['Yes', 'No'])

    # Create a DataFrame with user input features
    input_data = pd.DataFrame({
        'age': [age], 'TSH': [tsh], 'T3': [t3], 'TT4': [tt4], 'T4U': [t4u], 'FTI': [fti],
        'on thyroxine': [on_thyroxine], 'query on thyroxine': [query_on_thyroxine],
        'on antithyroid medication:': [on_antithyroid_medication], 'sick': [sick],
        'pregnant': [pregnant], 'thyroid surgery': [thyroid_surgery], 'I131 treatment': [i131_treatment],
        'query hypothyroid': [query_hypothyroid], 'query hyperthyroid': [query_hyperthyroid],
        'lithium': [lithium], 'goitre': [goitre], 'tumor': [tumor], 'hypopituitary': [hypopituitary],
        'psych': [psych], 'TSH measured': [tsh_measured], 'T3 measured': [t3_measured],
        'TT4 measured': [tt4_measured], 'T4u measured': [t4u_measured],
        'FTI measurred': [fti_measured], 'TBG measured': [tbg_measured]
    })

    # Preprocess user input
    input_data_processed = preprocess_input(input_data)

    # Display user input features
    st.subheader('User Input Features')
    st.write(input_data_processed)

    # Make prediction
    prediction = xgb_model.predict(input_data_processed)
    st.subheader('Prediction')
    st.write(prediction)

if __name__ == '__main__':
    main()
