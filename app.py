from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the input values from form
        MedInc = float(request.form['MedInc'])
        HouseAge = float(request.form['HouseAge'])
        AveRooms = float(request.form['AveRooms'])
        AveBedrms = float(request.form['AveBedrms'])
        Population = float(request.form['Population'])
        AveOccup = float(request.form['AveOccup'])
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])

        features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


