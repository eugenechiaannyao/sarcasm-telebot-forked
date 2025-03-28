# api/app.py
from flask import Flask, request, jsonify
import joblib
import string
import spacy
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer from training directory
model = joblib.load("../training/nb_model.joblib")
vectorizer = joblib.load("../training/vectorizer.joblib")

# Initialize resources once
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Shared preprocessing pipeline"""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and lemmatize
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

    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict_proba(text_vec)

    return jsonify({
        "prediction": prediction.tolist(),
        "processed_text": processed_text  # For debugging
    }), 200


if __name__ == "__main__":
    app.run(debug=True)