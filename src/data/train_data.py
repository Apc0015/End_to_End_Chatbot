import json
import re
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        text = re.sub(' +', ' ', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def prepare_training_data(self, intents_file=None):
        if intents_file is None:
            # Get absolute path relative to this file
            current_dir = Path(__file__).parent
            intents_file = current_dir / 'Data' / 'intents.json'
        
        with open(intents_file, 'r') as f:
            intents = json.load(f)
        
        patterns = []
        labels = []
        
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                patterns.append(self.clean_text(pattern))
                labels.append(intent['tag'])
        
        return patterns, labels

def create_training_data():
    preprocessor = DataPreprocessor()
    patterns, labels = preprocessor.prepare_training_data()
    
    # Save preprocessed data
    training_data = {
        'patterns': patterns,
        'labels': labels
    }
    
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Training data created: {len(patterns)} patterns, {len(set(labels))} intents")
    return patterns, labels

if __name__ == "__main__":
    create_training_data()