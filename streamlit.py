import streamlit as st
import pandas as pd
from customer_churn_model import predict

st.title("Customer Churn Probablity Calculator")

add_selectbox = st.sidebar.title("Welcome to Customer Churn Probability Calculator")

with st.sidebar:
    add_radio = st.radio(
        "Choose mode",
        ("Single", "List")
    )

if add_radio ==  'List':
    file = st.file_uploader("Upload file (in CSV format)", type="csv")
    if st.button('Predict'):
        df = pd.read_csv(file)
        pred = predict(df)
        st.dataframe(pred)

        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")

        csv = convert_df(pred)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="churn_data.csv",
            mime="text/csv",
        )

else:
    left_column, right_column = st.columns(2)
    with left_column:
        customer_id = st.text_input("Enter Customer ID")

        tenure = st.number_input("Enter Tenure", step=1)

        monthly_charges = st.number_input("Enter Monthly Chargers" , step=10)

        senior_citizen = st.radio(
            "Senior Citizen",
            ("Yes", "No")
        )
        partner= st.radio(
            "Partner",
            ("Yes", "No")
        )
        phone_service = st.radio(
            "Phone Service",
            ("Yes", "No")
        )
        multiple_line = st.radio(
            "Multiple Lines",
            ("Yes", "No" , "No Phone Service")
        )
        internet_service = st.radio(
            "Internet Service",
            ("Yes", "No" , "Fiber Optic")
        )
        contract = st.radio(
            "Contract",
            ("Month-to-Month", "One Year", "Two Year")
        )
        paperless_billing = st.radio(
            "Paperless Billing",
            ("Yes", "No")
        )
    with right_column:
        gender = st.radio(
            "Gender",
            ("Male", "Female")
        )
        total_charges = st.number_input("Enter Total Charges", step=100)
        
        dependent = st.radio(
            "Dependent",
            ("Yes", "No")
        )
        online_security = st.radio(
            "Online Security",
            ("Yes", "No" , "No Internet Service")
        )
        online_backup = st.radio(
            "Online Backup",
            ("Yes", "No" , "No Internet Service")
        )
        device_protection = st.radio(
            "Device Protection",
            ("Yes", "No" , "No Internet Service")
        )
        tech_support = st.radio(
            "Tech Support",
            ("Yes", "No" , "No Internet Service")
        )
        streaming_tv = st.radio(
            "Streaming TV",
            ("Yes", "No" , "No Internet Service")
        )
        streaming_movies = st.radio(
            "Streaming Movies",
            ("Yes", "No" , "No Internet Service")
        )
        payment_method = st.radio(
            "Online Security",
            ("Electronic Check", "Mailed Check" , "Bank Transfer (Automatic)" , "Credit Card (Automatic)")
        )
    if st.button('Predict'):
        data = [[customer_id , gender , senior_citizen , partner , dependent , tenure , 
                phone_service , multiple_line , internet_service , online_security ,
                online_backup , device_protection , tech_support , streaming_tv , streaming_movies ,
                contract , paperless_billing , payment_method , monthly_charges , total_charges]]
        
        columns = ['customerID' , 'gender' , 'SeniorCitizen' , 'Partner' , 'Dependents' ,
                   'tenure' , 'PhoneService' , 'MultipleLines' , 'InternetService' ,
                   'OnlineSecurity' , 'OnlineBackup' , 'DeviceProtection' , 'TechSupport' ,
                   'StreamingTV' , 'StreamingMovies' , 'Contract' , 'PaperlessBilling' ,
                   'PaymentMethod' , 'MonthlyCharges' , 'TotalCharges']
        
        df = pd.DataFrame(data , columns= columns)

        pred = predict(df)
        st.write(pred['customer_id'].to_string(index=False)  + " has " + pred['probablity'].to_string(index=False) + " probablity to churn.")

