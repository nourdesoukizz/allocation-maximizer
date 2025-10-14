"""
LSTM model wrapper for demand forecasting
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List
import warnings

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction

logger = logging.getLogger(__name__)

# Suppress tensorflow warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class LSTMWrapper(BaseModelWrapper):
    """LSTM model wrapper using TensorFlow/Keras"""
    
    def __init__(self, **kwargs):
        """
        Initialize LSTM wrapper
        
        Args:
            **kwargs: Model parameters
        """
        super().__init__(ModelType.LSTM, **kwargs)
        
        # Default LSTM parameters
        self.sequence_length = kwargs.get('sequence_length', 10)
        self.lstm_units = kwargs.get('lstm_units', 50)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Initialize model components
        self.scaler = None
        self.model = None
        
        # Try to import required libraries
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import required ML libraries"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            from sklearn.preprocessing import MinMaxScaler
            
            # Set random seeds for reproducibility
            tf.random.set_seed(42)
            np.random.seed(42)
            
            self.tf = tf
            self.keras = keras
            self.layers = layers
            self.MinMaxScaler = MinMaxScaler
            
            # Suppress TensorFlow logging
            tf.get_logger().setLevel('ERROR')
            
        except ImportError as e:
            logger.error(f"Required libraries not available for LSTM: {e}")
            raise ImportError(
                "TensorFlow and scikit-learn are required for LSTM model. "
                "Install with: pip install tensorflow scikit-learn"
            )
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> tuple:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data array
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        if len(data) <= sequence_length:
            logger.warning(f"Data length {len(data)} is too small for sequence length {sequence_length}")
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: tuple) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Input shape for the model
        """
        model = self.keras.Sequential([
            # First LSTM layer with return sequences
            self.layers.LSTM(
                self.lstm_units, 
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.dropout_rate
            ),
            
            # Second LSTM layer
            self.layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                dropout=self.dropout_rate
            ),
            
            # Dense layers
            self.layers.Dense(25, activation='relu'),
            self.layers.Dropout(self.dropout_rate),
            self.layers.Dense(1)
        ])
        
        # Compile model
        optimizer = self.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the LSTM model
        
        Args:
            X: Training features (will be converted to sequences)
            y: Training targets
            **kwargs: Additional training parameters
        """
        self.validate_input(X, y)
        
        if X.shape[0] < self.sequence_length + 1:
            raise ValueError(
                f"Not enough data points. Need at least {self.sequence_length + 1}, "
                f"got {X.shape[0]}"
            )
        
        logger.info(f"Training LSTM model with {X.shape[0]} samples")
        
        # Use first feature column for time series modeling
        # In a real scenario, you might want to use all features
        time_series_data = X[:, 0]  # Use first feature as time series
        
        # Scale the data
        self.scaler = self.MinMaxScaler()
        scaled_data = self.scaler.fit_transform(time_series_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(scaled_data, self.sequence_length)
        
        if X_sequences.size == 0:
            raise ValueError("Could not create sequences from the data")
        
        # Reshape for LSTM (samples, time steps, features)
        X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))
        
        # Build model
        self._build_model((self.sequence_length, 1))
        
        # Set up callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = self.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = self.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        history = self.model.fit(
            X_sequences, y_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.training_history = history.history
        self.is_fitted = True
        
        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the trained LSTM model
        
        Args:
            X: Features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelPrediction object
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        # Use first feature column for time series prediction
        time_series_data = X[:, 0]
        
        # Scale the data using fitted scaler
        scaled_data = self.scaler.transform(time_series_data.reshape(-1, 1)).flatten()
        
        # For prediction, we need to use the last sequence_length points
        if len(scaled_data) < self.sequence_length:
            # If not enough data, pad with the last available value
            padding_needed = self.sequence_length - len(scaled_data)
            if len(scaled_data) > 0:
                padding_value = scaled_data[-1]
            else:
                padding_value = 0.0
            scaled_data = np.concatenate([
                np.full(padding_needed, padding_value),
                scaled_data
            ])
        
        # Take the last sequence_length points
        last_sequence = scaled_data[-self.sequence_length:]
        
        # Reshape for LSTM
        X_pred = last_sequence.reshape((1, self.sequence_length, 1))
        
        # Make prediction
        scaled_predictions = self.model.predict(X_pred, verbose=0)
        
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(scaled_predictions).flatten()
        
        # For multi-step prediction, predict multiple steps ahead
        num_predictions = len(X)
        if num_predictions > 1:
            # Multi-step prediction
            all_predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(num_predictions):
                # Predict next value
                X_step = current_sequence.reshape((1, self.sequence_length, 1))
                next_pred = self.model.predict(X_step, verbose=0)[0, 0]
                
                # Store prediction
                all_predictions.append(next_pred)
                
                # Update sequence (remove first, add prediction)
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Inverse transform all predictions
            predictions = self.scaler.inverse_transform(
                np.array(all_predictions).reshape(-1, 1)
            ).flatten()
        
        # Ensure predictions are positive (demands can't be negative)
        predictions = np.maximum(predictions, 0)
        
        return ModelPrediction(
            model_type=self.model_type,
            predictions=predictions,
            confidence_intervals=None,  # LSTM doesn't provide built-in confidence intervals
            feature_importance=None,    # Not applicable for LSTM
            model_params=self.model_params
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (not directly applicable for LSTM)
        
        Returns:
            None (LSTM doesn't provide traditional feature importance)
        """
        return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save LSTM model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Save the Keras model
        self.model.save(f"{filepath}_lstm_model")
        
        # Save the scaler
        import joblib
        joblib.dump(self.scaler, f"{filepath}_lstm_scaler.pkl")
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load LSTM model from file
        
        Args:
            filepath: Path to load the model from
        """
        # Load the Keras model
        self.model = self.keras.models.load_model(f"{filepath}_lstm_model")
        
        # Load the scaler
        import joblib
        self.scaler = joblib.load(f"{filepath}_lstm_scaler.pkl")
        
        self.is_fitted = True
        logger.info(f"LSTM model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history
        
        Returns:
            Dictionary with training metrics history
        """
        return self.training_history if self.training_history else {}
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history (requires matplotlib)
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.training_history['loss'], label='Training Loss')
            if 'val_loss' in self.training_history:
                ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            
            # Plot MAE
            if 'mae' in self.training_history:
                ax2.plot(self.training_history['mae'], label='Training MAE')
            if 'val_mae' in self.training_history:
                ax2.plot(self.training_history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_ylabel('MAE')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")