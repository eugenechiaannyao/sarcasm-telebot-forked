from flask import Flask, request, jsonify
import joblib
import string
import spacy
import nltk
from nltk.corpus import stopwords
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np  # Import numpy

app = Flask(__name__)

# Load models and tokenizers
models = {
    "nb": {
        "model": joblib.load("training/nb_model.joblib"),
        "vectorizer": joblib.load("training/vectorizer.joblib")
    },
    "dBert": {
        "model": torch.load("training/dBert_model.pth", map_location=torch.device('cpu')),
        "tokenizer": torch.load("training/dBert_tokenizer.pth", map_location=torch.device('cpu'))
    },
    "dBert_typos": {
        "model": torch.load("training/dBert_typos_model.pth", map_location=torch.device('cpu')),
        "tokenizer": torch.load("training/dBert_typos_tokenizer.pth", map_location=torch.device('cpu'))
    },
    "dBert_syns": {
        "model": torch.load("training/dBert_syns_model.pth", map_location=torch.device('cpu')),
        "tokenizer": torch.load("training/dBert_syns_tokenizer.pth", map_location=torch.device('cpu'))
    },
}

# Default model
default_model = "nb"

# Initialize resources once
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Shared preprocessing pipeline for non-transformer models"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return ' '.join(tokens)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    model_name = data.get("model", default_model)

    if model_name not in models:
        return jsonify({
            "error": f"Model '{model_name}' not found",
            "available_models": list(models.keys())
        }), 400

    selected_model_data = models[model_name]

    if model_name == "nb":
        # Naive Bayes model with TF-IDF vectorizer
        processed_text = preprocess_text(text)
        text_vec = selected_model_data["vectorizer"].transform([processed_text])
        prediction = selected_model_data["model"].predict_proba(text_vec)
        sarcasm_prob = prediction[0][1]  # Probability of sarcasm

        # Convert NumPy types to Python native types
        sarcasm_prob = float(sarcasm_prob)
        is_sarcasm = bool(prediction[0][1] > 0.5)
    else:
        # DistilBERT-based models
        processed_text = text
        tokenizer = selected_model_data["tokenizer"]
        model = selected_model_data["model"]

        # Tokenize text
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        # Move inputs to CPU and make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Calculate probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        sarcasm_prob = probabilities.squeeze()[1].item()  # Probability of the positive class (sarcasm)
        is_sarcasm = bool(sarcasm_prob > 0.5)

    # Calculate confidence and response
    confidence_score = sarcasm_prob * 100

    return jsonify({
        "processed_text": processed_text,
        "prediction": sarcasm_prob,
        "confidence": confidence_score,
        "is_sarcasm": is_sarcasm,
        "model_used": model_name
    }), 200


if __name__ == "__main__":
    app.run(debug=True)