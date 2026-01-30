import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pickle 


model = tf.keras.models.load_model('model2.h5')

with open('label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender = pickle.load(file)


with open('one_hot_encoder.pk1', 'rb') as file:
    one_hot_encoder = pickle.load(file)


with open('scaler.pk1', 'rb' ) as file:
    scaler = pickle.load(file)




st.title("What's the Salary?: Customer Salary Prediction App")

geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 19, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])



input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember" :[is_active_member],
    "Exited" :[exited]
}

)


geo_encoded = one_hot_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns = one_hot_encoder.get_feature_names_out(['Geography']))



#input_data = pd.DataFrame([input_data])
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]
st.write(f"Predicted Estimated Salary: ${predicted_salary: .2f}")


