from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("exoplanet_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📩 Received data:", data)

        # ✅ Model expects these columns
        expected_features = ["radius_earth", "orbital_period", "temp_equil", "disc_year"]

        # Create a single-row DataFrame
        input_data = {f: float(data.get(f, 0)) for f in expected_features}
        X = pd.DataFrame([input_data])

        # Make prediction
        prediction = int(model.predict(X)[0])
        result = "Confirmed Exoplanet 🌍" if prediction == 1 else "Not an Exoplanet 🚫"

        print("✅ Prediction complete:", result)

        return jsonify({
            "input": input_data,
            "prediction": prediction,
            "result": result
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
