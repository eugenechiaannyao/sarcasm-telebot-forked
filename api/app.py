from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import string
import spacy
import nltk
from nltk.corpus import stopwords
import torch
import gdown  # Add this import

app = Flask(__name__)

# Google Drive file IDs (replace with your actual file links)
MODEL_FILES = {
    "nb_model": "1TWuzcFguEiYwSykQmAPwDTF55qNx27ce",
    "vectorizer": "1a6_E545QQ2miuINCE1A7W1Wl0vihSGe4",
    "dBert_model": "1EmZ5DEmnyR0WAJSuk2Bk19qLoyyivDHR",
    "dBert_tokenizer": "1pqzj25bDJkM4ihWDvn8wIn-gFPJqWb6d",
    "dBert_typos_model": "1b86nXNUyv0sQwh9gBJfDxwf4V49nDJII",
    "dBert_typos_tokenizer": "1mlDBxEDjty7bZb077kfR8zo8IPGuJ3oQ",
    "dBert_syns_model": "1UcF7C2DkBfkqTZURn5XeXLPc-xvu0Rx9",
    "dBert_syns_tokenizer": "1ENC7DqeSWBpZEbKfAHjdRQ9N0rcFJp2w",
    "mBert_model": "1UfasKY9e-9Y0SUWMigEtgVx5Y01YHp9L",
    "mBert_tokenizer": "1jyA9bffvjqLASperi7yH4vFgSDtHs6GZ",
    "mBert_typos_model": "1-4q78Y-66Ite9sQ6b9D2utyasmVAAs2u",
    "mBert_typos_tokenizer": "1-EvDJ_MydeHu3Vs_C_0iqEpk3Zs3BMzo",
    "tBert_model": "1-KyHoGYOlb-B3S93Y_JZiJZZgsJVSYLw",
    "tBert_tokenizer": "1-MKncOBUjtVg2Xd1oLCcc3Qa8KkvlmOY",
}

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_models():
    """Download and load all models"""
    """Download models once and cache permanently"""
    models_loaded = {}

    for name, file_id in MODEL_FILES.items():
        ext = ".pth" if "dBert" in name or "mBert" in name or "tBert" in name else ".joblib"
        cache_path = CACHE_DIR / f"{name}{ext}"

        # Download only if missing
        if not cache_path.exists():
            print(f"⏳ Downloading {name}...")
            download_from_gdrive(file_id, str(cache_path))

        # Load from cache
        try:
            if ext == ".pth":
                models_loaded[name] = torch.load(cache_path, map_location='cpu')
            else:
                models_loaded[name] = joblib.load(cache_path)
        except Exception as e:
            print(f"❌ Corrupted {name}: {e}")
            cache_path.unlink()  # Delete bad file
            raise

        # Load models

    return {
            "nb": {
                "model": models_loaded["nb_model"],
                "vectorizer": models_loaded["vectorizer"]
            },
            "dBert": {
                "model": models_loaded["dBert_model"],
                "tokenizer": models_loaded["dBert_tokenizer"]
            },
            "dBert_typos": {
                "model": models_loaded["dBert_typos_model"],
                "tokenizer": models_loaded["dBert_typos_tokenizer"]
            },
            "dBert_syns": {
                "model": models_loaded["dBert_syns_model"],
                "tokenizer": models_loaded["dBert_syns_tokenizer"]
            },
            "mBert": {
                "model": models_loaded["mBert_model"],
                "tokenizer": models_loaded["mBert_tokenizer"]
            },
            "mBert_typos": {
                "model": models_loaded["mBert_typos_model"],
                "tokenizer": models_loaded["mBert_typos_tokenizer"]
            },
            "tBert": {
                "model": models_loaded["tBert_model"],
                "tokenizer": models_loaded["tBert_tokenizer"]
            }
    }

try:
    models = load_models()
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    raise

# # Load models and tokenizers
# models = {
#     "nb": {
#         "model": joblib.load("training/nb_model.joblib"),
#         "vectorizer": joblib.load("training/vectorizer.joblib")
#     },
#     "dBert": {
#         "model": torch.load("training/dBert_model.pth", map_location=torch.device('cpu')),
#         "tokenizer": torch.load("training/dBert_tokenizer.pth", map_location=torch.device('cpu'))
#     },
#     "dBert_typos": {
#         "model": torch.load("training/dBert_typos_model.pth", map_location=torch.device('cpu')),
#         "tokenizer": torch.load("training/dBert_typos_tokenizer.pth", map_location=torch.device('cpu'))
#     },
#     "dBert_syns": {
#         "model": torch.load("training/dBert_syns_model.pth", map_location=torch.device('cpu')),
#         "tokenizer": torch.load("training/dBert_syns_tokenizer.pth", map_location=torch.device('cpu'))
#     },
# }

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

        # Apply temperature scaling and clamping
        temperature = 2.5  # Higher values, less extreme
        min_prob = 0.01  # Minimum probability for any class

        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=1)
        probabilities = torch.clamp(probabilities, min=min_prob, max=1 - min_prob)

        # Renormalize to ensure probabilities sum to 1
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

        # Get final sarcasm probability
        sarcasm_prob = probabilities[0][1].item()

    return jsonify({
        "processed_text": processed_text,
        "prediction": sarcasm_prob,
        "model_used": model_name
    }), 200


if __name__ == "__main__":
    # Initialize models at startup
    app.run(debug=True)

