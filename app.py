from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and related files
model = joblib.load("model/xgboost_credit_risk_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
model_features = joblib.load("model/model_features.pkl")

# Debug: print expected features once
print("Model expects features:")
print(model_features)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict class
        pred_encoded = model.predict(input_df, validate_features=False)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        # Predict probability
        pred_proba = model.predict_proba(input_df, validate_features=False)
        confidence = float(max(pred_proba[0])) * 100

        return jsonify({
            "prediction": pred_label,
            "confidence_percent": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        data = request.form.to_dict()

        # Convert values to numeric
        for key in data:
            data[key] = float(data[key])

        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Prediction
        pred_encoded = model.predict(input_df, validate_features=False)
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]

        # Probability
        pred_proba = model.predict_proba(input_df, validate_features=False)
        confidence = float(max(pred_proba[0])) * 100

        return render_template(
            "index.html",
            prediction=pred_label,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
