#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

filename = 'random_forest_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Tip Prediction App')
st.subheader('Please enter your data:')

df = pd.read_csv('tips.csv')

model_features = loaded_model.feature_names_in_

uploaded_file = st.file_uploader("tips.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    label_encoders = {}
    categorical_features=["sex", "smoker", "day", "time"]
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    
    # Ensure the columns are in the correct order
    column = ['total_bill', 'time', 'size']
    df_preprocessed = df.reindex(columns=column, fill_value=0)

  
    
    df_preprocessed = df.reindex(columns=model_features,fill_value=0)

    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('Lifestyle Change:')
    st.write(prediction_text)
 

