from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_input = np.array([features])
        
        # Apply scaling
        final_input_scaled = scaler.transform(final_input)
        
        # Make prediction
        prediction = model.predict(final_input_scaled)[0]
        
        # Convert class number to label
        if prediction == 0:
            label = "Low Value Area"
        elif prediction == 1:
            label = "Medium Value Area"
        else:
            label = "High Value Area"
        
        return render_template("index.html",
                               prediction_text=f"Prediction: {label}")
    
    except:
        return render_template("index.html",
                               prediction_text="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)