import json
import pickle
import random
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class Chatbot:
    def __init__(self):
        self.model = None
        self.intents = None
        self.responses = {}
        self.confidence_threshold = 0.15  # Lower threshold for better responses
        
    def load_intents(self, intents_file=None):
        if intents_file is None:
            # Get absolute path relative to this file
            current_dir = Path(__file__).parent
            intents_file = current_dir.parent / 'data' / 'Data' / 'intents.json'
        
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)
        
        # Create response mapping
        for intent in self.intents['intents']:
            self.responses[intent['tag']] = intent['responses']
    
    def load_model(self, model_file=None):
        if model_file is None:
            # Try different possible locations
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir.parent.parent / 'chatbot_model.pkl',
                current_dir.parent / 'models' / 'saved_models' / 'chatbot_model.pkl',
                Path('chatbot_model.pkl')
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_file = path
                    break
            else:
                print("No model file found. Please train the model first.")
                return False
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Please train the model first.")
            return False
    
    def predict_intent(self, user_input):
        if not self.model:
            return None, 0.0
        
        # Use raw input for TF-IDF pipeline (it handles its own preprocessing)
        try:
            prediction = self.model.predict([user_input])[0]
            confidence = max(self.model.predict_proba([user_input])[0])
            return prediction, confidence
        except:
            # Fallback if prediction fails
            return None, 0.0
    
    def get_response(self, user_input):
        # Predict intent
        intent, confidence = self.predict_intent(user_input)
        
        if intent and confidence > self.confidence_threshold:
            # Get random response from intent
            if intent in self.responses:
                response = random.choice(self.responses[intent])
                return response, intent, confidence
            else:
                return "I'm not sure how to respond to that.", "unknown", confidence
        else:
            # Fallback responses for low confidence
            fallback_responses = [
                "I'm not sure I understand. Can you rephrase that?",
                "Could you please clarify what you mean?",
                "I'm still learning. Can you try asking in a different way?",
                "I don't quite understand. Can you be more specific?"
            ]
            return random.choice(fallback_responses), "fallback", confidence
    
    def train_simple_model(self):
        """Train a simple sklearn model"""
        print("Creating simple model...")
        
        # Get raw training data from intents
        patterns = []
        labels = []
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)  # Use raw patterns
                labels.append(intent['tag'])
        
        # Create a simple pipeline with improved TF-IDF parameters
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True, 
                stop_words='english', 
                ngram_range=(1, 2),  # Include bigrams
                max_features=1000,   # Limit features
                min_df=1,           # Include rare words
                max_df=0.8          # Exclude very common words
            )),
            ('classifier', MultinomialNB(alpha=0.1))  # Lower smoothing
        ])
        
        # Train the model
        self.model.fit(patterns, labels)
        
        # Save the model
        current_dir = Path(__file__).parent
        model_path = current_dir.parent.parent / 'chatbot_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model trained and saved to {model_path}")
        print(f"Trained on {len(patterns)} patterns with {len(set(labels))} intents")
    
    def chat(self):
        print("Chatbot: Hello! I'm your friendly chatbot. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye! Have a great day!")
                break
            
            if user_input:
                response, intent, confidence = self.get_response(user_input)
                print(f"Chatbot: {response}")
                
                # Optional: Show debug info
                if confidence < 0.5:
                    print(f"(Low confidence: {confidence:.2f})")
    
    def initialize(self):
        # Load intents and model
        self.load_intents()
        
        if not self.load_model():
            print("Training model...")
            self.train_simple_model()
            self.load_model()
        
        return True

def main():
    bot = Chatbot()
    
    if bot.initialize():
        print("Chatbot initialized successfully!")
        bot.chat()
    else:
        print("Failed to initialize chatbot.")

if __name__ == "__main__":
    main()