import streamlit as st
import pandas as pd
import joblib


# load the pre-trained model and scaler
model = joblib.load('svm_model.pkl')

def predict_purchase(age, salary, gender):
    # create a dataframe with the user input
    input_data = pd.DataFrame([[age, salary, gender]], 
                              columns=['Age','EstimatedSalary','Gender_Encoded'])
    # make predictions
    prediction = model.predict(input_data)
    return prediction[0]
