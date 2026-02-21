import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
import numpy as np
import pickle


## load the trained model
model = tf.keras.models.load_model('model.h5')

## load the label encoders, one hot encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


##streamlit app
st.title("Customer Churn Prediction")

## create input fields for the user to enter customer data
gender = st.selectbox("Gender",label_encoder_gender.classes_)
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=5)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

# prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [balance]  # Assuming Estimated Salary is the same as Balance for this example
})

## one hot encode the geography
one_encoded_geo = one_hot_encoder_geo.transform([[geography]]).toarray()
one_encoded_geo_df = pd.DataFrame(one_encoded_geo, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# combine the one hot encoded geography with the rest of the input data
input_df = pd.concat([input_data, one_encoded_geo_df], axis=1)

## scale the input data
scaled_input = scaler.transform(input_df)

## make prediction
prediction = model.predict(scaled_input)
prediction_prob = prediction[0][0]

st.write(f"Predicted probability of churn: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

