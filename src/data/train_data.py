import json
import re
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
        self.words = []
        self.classes = []
        self.documents = []
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        text = re.sub(' +', ' ', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
        
        return tokens
    
    def prepare_training_data(self, intents_file=None):
        if intents_file is None:
            # Get absolute path relative to this file
            current_dir = Path(__file__).parent
            intents_file = current_dir / 'Data' / 'intents.json'
        
        with open(intents_file, 'r') as f:
            intents = json.load(f)
        
        words = []
        classes = []
        documents = []
        
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize and clean the pattern
                word_tokens = self.clean_text(pattern)
                words.extend(word_tokens)
                documents.append((word_tokens, intent['tag']))
                
                # Add to classes
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
        
        # Remove duplicates and sort
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        
        self.words = words
        self.classes = classes
        self.documents = documents
        
        return words, classes, documents
    
    def create_training_data(self):
        # Create training data
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            bag = []
            word_patterns = doc[0]
            
            # Create bag of words
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            # Output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle and convert to numpy array
        np.random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Split into X and y
        train_x = np.array(list(training[:, 0]))
        train_y = np.array(list(training[:, 1]))
        
        return train_x, train_y
    
    def split_data(self, train_x, train_y, test_size=0.2, val_size=0.1):
        # First split: train + val, test
        x_temp, x_test, y_temp, y_test = train_test_split(
            train_x, train_y, test_size=test_size, random_state=42
        )
        
        # Second split: train, val
        val_size_adjusted = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, base_path):
        # Save words
        with open(f"{base_path}_words.pkl", 'wb') as f:
            pickle.dump(self.words, f)
        
        # Save classes
        with open(f"{base_path}_classes.pkl", 'wb') as f:
            pickle.dump(self.classes, f)
        
        print(f"Preprocessed data saved to {base_path}")
    
    def load_preprocessed_data(self, base_path):
        # Load words
        with open(f"{base_path}_words.pkl", 'rb') as f:
            self.words = pickle.load(f)
        
        # Load classes
        with open(f"{base_path}_classes.pkl", 'rb') as f:
            self.classes = pickle.load(f)
        
        print(f"Preprocessed data loaded from {base_path}")
        return self.words, self.classes

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