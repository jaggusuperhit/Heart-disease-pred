import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')

# Preparing the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Web app
st.title('Heart Disease Prediction Model')

input_text = st.text_input('Provide comma separated features to predict heart disease')
split_input = input_text.split(',')

try:
    np_df = np.asarray(split_input, dtype=float)
    reshaped_df = np_df.reshape(1, -1)
    prediction = model.predict(reshaped_df)
    if prediction[0] == 0:
        st.write("This person doesn't have heart disease")
    else:
        st.write("This person has heart disease")

except ValueError:
    st.write('Please provide comma separated values')

st.subheader("About Data")
st.write(heart_data.head())  # Showing only the first few rows for better readability
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy)
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)