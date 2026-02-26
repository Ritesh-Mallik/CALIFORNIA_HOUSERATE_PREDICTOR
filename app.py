from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask App
app = Flask(__name__)

# Load saved models and scaler
reg_model = pickle.load(open("regression_model.pkl", "rb"))
class_model = pickle.load(open("classification_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        MedInc = float(request.form["MedInc"])
        HouseAge = float(request.form["HouseAge"])
        AveRooms = float(request.form["AveRooms"])
        AveBedrms = float(request.form["AveBedrms"])
        Population = float(request.form["Population"])
        AveOccup = float(request.form["AveOccup"])
        Latitude = float(request.form["Latitude"])
        Longitude = float(request.form["Longitude"])

        # Create feature array
        features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                              Population, AveOccup, Latitude, Longitude]])

        # Scale input using trained scaler
        scaled_features = scaler.transform(features)

        # Regression prediction
        reg_prediction = reg_model.predict(scaled_features)[0]

        # Convert to Indian Rupees
        # (Dataset gives values in units of 100,000 USD)
        formatted_price = "₹ {:,.2f}".format(reg_prediction * 100000)

        # Classification prediction
        class_prediction = class_model.predict(scaled_features)[0]

        return render_template(
            "index.html",
            regression_result=formatted_price,
            classification_result=class_prediction
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)