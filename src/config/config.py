"""
Configuration file for the chatbot application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "src" / "data"
INTENTS_FILE = DATA_DIR / "Data" / "intents.json"
TRAINING_DATA_FILE = DATA_DIR / "training_data.json"

# Model paths
MODELS_DIR = BASE_DIR / "src" / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
MODEL_FILE = SAVED_MODELS_DIR / "chatbot_model.h5"
PICKLE_MODEL_FILE = BASE_DIR / "chatbot_model.pkl"

# Training parameters
CONFIDENCE_THRESHOLD = 0.3
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.5
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Logging configuration
LOGGING_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Streamlit configuration
APP_TITLE = "Simple Chatbot"
APP_ICON = "🤖"
LAYOUT = "centered"