# api/app.py
import sys
from flask import Flask, request, jsonify
import joblib
sys.path.append('../utils')  # Add utils directory to sys.path
from shared_preprocessing import preprocess_text

app = Flask(__name__)

# Load model and vectorizer from training directory
model = joblib.load("training/nb_model.joblib")
vectorizer = joblib.load("training/vectorizer.joblib")


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict_proba(text_vec)

    return jsonify({
        "prediction": prediction.tolist(),
        "processed_text": processed_text  # For debugging
    }), 200


if __name__ == "__main__":
    app.run(debug=True)