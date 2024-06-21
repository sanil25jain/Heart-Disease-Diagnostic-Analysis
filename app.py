from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

with open('lr.pkl', 'rb') as file:
    lr = pickle.load(file)
with open('sc.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = sc.transform(input_data_reshaped)
    prediction = lr.predict(std_data)
    result = 'Patiest Diagnosed With Heart Disease' if prediction[0] == 1 else 'CONGRATULATIONS! Patiest Not Diagnosed With Heart Disease'
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)