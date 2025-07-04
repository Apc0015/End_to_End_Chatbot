import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.train_data import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotTrainer:
    """
    Neural network trainer for chatbot intent classification.
    """
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.history = None
        
    def build_model(self, hidden_layers=[128, 64], dropout_rate=0.5):
        """Build the neural network model."""
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(hidden_layers[0], input_shape=(self.input_dim,), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(self.output_dim, activation='softmax'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
        logger.info(f"Model summary:\\n{self.model.summary()}")
        
        return self.model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
        """Train the model."""
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.0001
        )
        
        model_checkpoint = ModelCheckpoint(
            'src/models/saved_models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
        
        callbacks = [early_stopping, reduce_lr, model_checkpoint]
        
        # Train the model
        logger.info("Starting training...")
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def evaluate(self, x_test, y_test, classes):
        """Evaluate the model on test data."""
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        # Predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report - handle cases where not all classes are predicted
        unique_classes = sorted(list(set(y_true_classes) | set(y_pred_classes)))
        class_names = [classes[i] for i in unique_classes]
        class_report = classification_report(y_true_classes, y_pred_classes, 
                                          target_names=class_names, 
                                          labels=unique_classes, 
                                          zero_division=0)
        logger.info(f"Classification Report:\\n{class_report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_true': y_test
        }
    
    def plot_training_history(self, save_path="src/models/saved_models/training_history.png"):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")
    
    def plot_confusion_matrix(self, cm, classes, save_path="src/models/saved_models/confusion_matrix.png"):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    def save_model(self, filepath="src/models/saved_models/chatbot_model.h5"):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath="src/models/saved_models/chatbot_model.h5"):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model
    
    def predict_intent(self, input_bag, classes, threshold=0.7):
        """Predict intent from input bag of words."""
        prediction = self.model.predict(np.array([input_bag]))[0]
        
        # Get the highest probability
        max_prob = np.max(prediction)
        predicted_class_idx = np.argmax(prediction)
        
        # Only return prediction if confidence is above threshold
        if max_prob > threshold:
            return classes[predicted_class_idx], max_prob
        else:
            return "noanswer", max_prob

def main():
    """Main function to train the chatbot model."""
    # Load training data
    try:
        with open("src/models/saved_models/training_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        x_train = data['x_train']
        x_val = data['x_val']
        x_test = data['x_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Load classes
        with open("src/models/saved_models/preprocessed_data_classes.pkl", 'rb') as f:
            classes = pickle.load(f)
        
        logger.info("Training data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        logger.info("Running data preprocessing first...")
        
        # Run preprocessing
        preprocessor = DataPreprocessor()
        words, classes, documents = preprocessor.prepare_training_data("src/data/Data/intents.json")
        train_x, train_y = preprocessor.create_training_data()
        x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.split_data(train_x, train_y)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data("src/models/saved_models/preprocessed_data")
        
        # Save training data
        with open("src/models/saved_models/training_data.pkl", 'wb') as f:
            pickle.dump({
                'x_train': x_train,
                'x_val': x_val,
                'x_test': x_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }, f)
    
    # Initialize trainer
    input_dim = x_train.shape[1]
    output_dim = len(classes)
    trainer = ChatbotTrainer(input_dim, output_dim)
    
    # Build model
    model = trainer.build_model(hidden_layers=[128, 64], dropout_rate=0.3)
    
    # Train model
    history = trainer.train(x_train, y_train, x_val, y_val, epochs=200, batch_size=16)
    
    # Evaluate model
    evaluation_results = trainer.evaluate(x_test, y_test, classes)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(evaluation_results['confusion_matrix'], classes)
    
    # Save model
    trainer.save_model()
    
    # Save training history
    with open("src/models/saved_models/training_history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save evaluation results
    with open("src/models/saved_models/evaluation_results.pkl", 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    logger.info("Model training completed successfully!")
    logger.info(f"Final test accuracy: {evaluation_results['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()