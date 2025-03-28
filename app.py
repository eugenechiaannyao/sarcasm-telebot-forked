from flask import Flask, request, jsonify
from nltk.corpus import stopwords
import joblib
import spacy
import string
import nltk

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("./training/nb_model.joblib")

# Ensure you save the vectorizer during training
vectorizer = joblib.load("./training/vectorizer.joblib")

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text data


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text using spaCy
    tokens = nlp(text)

    # Remove stopwords
    tokens = [token for token in tokens if token.text not in stop_words]

    # Lemmatize the tokens
    tokens = [token.lemma_ for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    text_vec = vectorizer.transform([processed_text])

    # Make a prediction
    prediction_confidence_scores = model.predict_proba(text_vec)

    return jsonify({"prediction": prediction_confidence_scores.tolist()}), 200


if __name__ == "__main__":
    app.run(debug=True)
