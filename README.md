# Simple Chatbot

A basic chatbot built with Python, scikit-learn, and Streamlit.

## Features

- Intent recognition using machine learning
- Web interface with Streamlit
- Extensible design for adding new intents
- Simple conversation flow

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web app:
```bash
streamlit run app.py
```

3. Or run the console version:
```bash
python chatbot.py
```

## Testing

Run the test suite:
```bash
python test_chatbot.py
```

## Project Structure

- `intents.json` - Define conversation intents and responses
- `train_data.py` - Data preprocessing functions
- `train_model.py` - Model training script
- `chatbot.py` - Main chatbot class
- `app.py` - Streamlit web interface
- `test_chatbot.py` - Testing script

## Customization

1. Edit `intents.json` to add new intents
2. Run `python train_model.py` to retrain
3. Restart the app

## Built With

- Python 3.8+
- scikit-learn
- NLTK
- Streamlit