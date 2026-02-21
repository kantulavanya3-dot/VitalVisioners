from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("Anemia_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = float(request.form["Gender"])
    hemoglobin = float(request.form["Hemoglobin"])
    mch = float(request.form["MCH"])
    mchc = float(request.form["MCHC"])
    mcv = float(request.form["MCV"])

    data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        result = "Positive for Anemia"
    else:
        result = "Negative for Anemia"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)