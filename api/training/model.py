from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import spacy
import string
import nltk
import joblib

# File path to the dataset
file_path = "data/Sarcasm_Headlines_Dataset_v2.json"

# Read the JSON data into a DataFrame
df = pd.read_json(file_path, lines=True)
print("Data loaded successfully.")

# Download stopwords from NLTK
nltk.download('stopwords')
print("Stopwords downloaded.")

# Load the spaCy model for NLP tasks
nlp = spacy.load('en_core_web_sm')
print("spaCy model loaded.")

# Define stopwords
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

# Apply preprocessing to the 'headline' column
df['processed_headline'] = df['headline'].apply(preprocess_text)
print("Text preprocessing completed.")

# Vectorize the text data using TF-IDF
vectorizer_nb = TfidfVectorizer()
X_vec = vectorizer_nb.fit_transform(df['processed_headline'])
print("Text vectorization completed.")

# Define the target variable
y = df['is_sarcastic']

# Split the data into training and testing sets
X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
print("Data split into training and testing sets.")

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_vec_train, y_train)
print("Naive Bayes model training completed.")

# Make predictions on the test set
y_pred = nb_classifier.predict(X_vec_test)

# Calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, y_pred)
nb_f1 = f1_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred)
nb_recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print("===============NAIVE BAYES=================")
print(f"Accuracy: {nb_accuracy}")
print(f"F1 Score: {nb_f1}")
print(f"Precision: {nb_precision}")
print(f"Recall: {nb_recall}")
print(classification_report(y_test, y_pred))
print("===========================================")

# Save the trained model to a file
joblib.dump(nb_classifier, 'nb_model.joblib')
print("Model saved as 'nb_model.joblib'.")

# Save the vectorizer to a file
joblib.dump(vectorizer_nb, 'vectorizer.joblib')
print("Vectorizer saved as 'vectorizer.joblib'.")
