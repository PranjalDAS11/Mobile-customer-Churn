import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

def predict(customer_df):
    df = pd.read_csv('churn_data.csv')
    user_df = customer_df.drop(columns= (['customerID']))

    columns_to_encode = ['SeniorCitizen' , 'gender', 'Partner', 'Dependents', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                         'Contract', 'PaperlessBilling','PaymentMethod']

    encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
    dummy_df = encoder.fit_transform(df[columns_to_encode])
    one_hot_df = pd.DataFrame(dummy_df, columns=encoder.get_feature_names_out(columns_to_encode))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(columns_to_encode, axis=1)

    x = df_encoded.drop(columns=['Churn' , 'customerID']).replace({' ': 0})
    y = df_encoded['Churn'].replace({'Yes': 1, 'No': 0 , ' ': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    user_df['TotalCharges'] = pd.to_numeric(user_df['TotalCharges'], errors='coerce').fillna(0)

    est = GradientBoostingClassifier(n_estimators = 30 , max_depth = 5 ,
                                  learning_rate = 0.1 , min_samples_split = 20 ,
                                  min_samples_leaf= 8 , subsample= 0.6)
    est.fit(x,y)

    user_encode = df.drop(columns= (['Churn']))
    user_encoder = encoder.fit(user_encode[columns_to_encode])

    
    x_pred= user_df
    x_pred = user_encoder.transform(x_pred[columns_to_encode])
    one_hot_df = pd.DataFrame(x_pred, columns=encoder.get_feature_names_out(columns_to_encode))
    df_encoded = pd.concat([user_df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(columns_to_encode, axis=1).replace({' ': 0})
    pred = est.predict(df_encoded)
    pred_proba = est.predict_proba(df_encoded)
    probablity_df = pd.DataFrame(pred_proba, columns=['probability_0', 'probability_1'])
    
    pred_df = {'customer_id': customer_df['customerID'],'probablity' : probablity_df['probability_1'].round(2)}
    pred_df = pd.DataFrame(pred_df)
    return pred_df

