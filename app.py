import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.set_page_config(page_title = 'Customer Churn Predictor',
                  layout = "wide", #wide
                  initial_sidebar_state = "expanded",
                  menu_items = {
                      'About' : ' Churn Predictor '
                  })
image = Image.open('freestocks-_3Q3tsJ01nc-unsplash.jpg')


# load model
from tensorflow.keras.models import load_model
class columnDropperTransformer():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

pickled = open('preprocessor.pkl', 'rb')
preprocessor = pickle.load(pickled)
savedModel=load_model('gfgModel.h5')

def predict(inputs):
    df = pd.DataFrame(inputs, index=[0])
    df = preprocessor.transform(df)
    y_pred = savedModel.predict(df)
    y_pred = np.where(y_pred < 0.5, 0, 1).squeeze()
    print(y_pred)
    return y_pred.item()

columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
label = ['0', '1']

st.title("Customer Churn Predictor")
st.image(image)

customerID = 'abcdefghijklmn'
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.number_input("Senior Citizen")
Partner = st.selectbox("Marriage Status", ['Yes', 'No'])
Dependents = st.selectbox("Head of Family", ['Yes', 'No'])
tenure = st.number_input("Tenure Length")
PhoneService = '12345678'
MultipleLines = '910111213'
InternetService = st.selectbox("Which Internet Service You Use?", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Do You Use Online Security?", ['No', 'Yes', 'No internet service'])
OnlineBackup = st.selectbox("Do You Use Online Backup?", ['No', 'Yes', 'No internet service'])
DeviceProtection = st.selectbox("Do You Use Device Protection?", ['No', 'Yes', 'No internet service'])
TechSupport = st.selectbox("Do You Use Tech Support?", ['No', 'Yes', 'No internet service'])
StreamingTV = st.selectbox("Do You Use Streaming TV?", ['No', 'Yes', 'No internet service'])
StreamingMovies = st.selectbox("Do You Use Streaming Movies?", ['No', 'Yes', 'No internet service'])
Contract = st.selectbox("Which Contract You Use?", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Do You Use Paperless Billing?", ['Yes', 'No'])
PaymentMethod = st.selectbox("Which Payment Method You Use?", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

#inference
new_data = [customerID, gender, SeniorCitizen, Partner, Dependents, tenure, 
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, 
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, 
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
new_data = pd.DataFrame([new_data], columns = columns)
new_data = preprocessor.transform(new_data)
res = savedModel.predict(new_data).item()
res = 0 if res < 0.5 else 1
press = st.button('PREDICT!')
if press:
    st.title(label[res])
    '''Explanation :
- '0' is You Likely Not Churning,
- '1' is You Likely Going to Churn'''