# utils/preprocessing.py
import string
import spacy
import nltk
from nltk.corpus import stopwords

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