"""
Base model wrapper interface for all ML models
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# Import will be added when needed - keeping models independent for now


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """ML Model types enumeration"""
    LSTM = "lstm"
    PROPHET = "prophet" 
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    SARIMA = "sarima"


@dataclass
class ModelPrediction:
    """Container for model predictions with metadata"""
    model_type: ModelType
    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_params: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_type': self.model_type.value,
            'predictions': self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            'confidence_intervals': self.confidence_intervals.tolist() if isinstance(self.confidence_intervals, np.ndarray) else self.confidence_intervals,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }


@dataclass 
class ModelPerformance:
    """Container for model performance metrics"""
    model_type: ModelType
    rmse: float
    mae: float
    mape: float
    r2_score: float
    execution_time: float
    training_samples: int
    test_samples: int
    cross_val_scores: Optional[List[float]] = None
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted performance score (lower is better)"""
        # Normalize metrics and combine with weights
        normalized_rmse = min(self.rmse / 100, 1.0)  # Cap at 1.0
        normalized_mae = min(self.mae / 50, 1.0)
        normalized_mape = min(self.mape / 100, 1.0)
        time_penalty = min(self.execution_time / 60, 0.3)  # Cap time penalty at 0.3
        
        # Weights: RMSE=0.4, MAE=0.3, MAPE=0.2, Time=0.1
        score = (0.4 * normalized_rmse + 
                0.3 * normalized_mae + 
                0.2 * normalized_mape + 
                0.1 * time_penalty)
        
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_type': self.model_type.value,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'r2_score': self.r2_score,
            'execution_time': self.execution_time,
            'training_samples': self.training_samples,
            'test_samples': self.test_samples,
            'weighted_score': self.weighted_score,
            'cross_val_scores': self.cross_val_scores
        }


class BaseModelWrapper(ABC):
    """Abstract base class for all ML model wrappers"""
    
    def __init__(self, model_type: ModelType, **kwargs):
        """
        Initialize model wrapper
        
        Args:
            model_type: Type of ML model
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.is_fitted = False
        self.training_history = []
        self.feature_names = []
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelPrediction object
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature names to importance scores
        """
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Fit model and make predictions in one step
        
        Args:
            X_train: Training features
            y_train: Training targets  
            X_test: Test features
            **kwargs: Additional parameters
            
        Returns:
            ModelPrediction object
        """
        start_time = time.time()
        
        # Fit the model
        self.fit(X_train, y_train, **kwargs)
        training_time = time.time() - start_time
        
        # Make predictions
        prediction_start = time.time()
        prediction = self.predict(X_test, **kwargs)
        prediction_time = time.time() - prediction_start
        
        # Update timing information
        prediction.training_time = training_time
        prediction.prediction_time = prediction_time
        
        return prediction
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if X.size == 0:
            raise ValueError("X cannot be empty")
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")
            
            if y.size == 0:
                raise ValueError("y cannot be empty")
            
            if len(y.shape) != 1:
                raise ValueError("y must be a 1D array")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same number of samples")
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_column: str = 'allocated_quantity',
                    feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data from DataFrame
        
        Args:
            df: DataFrame with allocation data
            target_column: Name of target column
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays
        """
        if df.empty:
            return np.array([]), np.array([])
        
        # Default feature columns
        if feature_columns is None:
            feature_columns = [
                'dc_priority',
                'current_inventory',
                'forecasted_demand', 
                'historical_demand',
                'revenue_per_unit',
                'cost_per_unit',
                'margin',
                'risk_score',
                'lead_time_days',
                'min_order_quantity',
                'safety_stock'
            ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No valid feature columns found in data")
        
        # Extract features and target
        X = df[available_features].values
        y = df[target_column].values if target_column in df.columns else np.array([])
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found NaN values in features, filling with column means")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        if y.size > 0 and np.isnan(y).any():
            logger.warning("Found NaN values in target, filling with mean")
            y = pd.Series(y).fillna(pd.Series(y).mean()).values
        
        self.feature_names = available_features
        
        return X, y
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and status
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type.value,
            'is_fitted': self.is_fitted,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'training_history_length': len(self.training_history)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file (to be implemented by subclasses)
        
        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("save_model method must be implemented by subclasses")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file (to be implemented by subclasses)
        
        Args:
            filepath: Path to load the model from
        """
        raise NotImplementedError("load_model method must be implemented by subclasses")


class ModelRegistry:
    """Registry for managing model wrappers"""
    
    def __init__(self):
        """Initialize model registry"""
        self._models: Dict[ModelType, BaseModelWrapper] = {}
        self._model_classes: Dict[ModelType, type] = {}
    
    def register_model(self, model_type: ModelType, model_class: type) -> None:
        """
        Register a model class
        
        Args:
            model_type: Type of model
            model_class: Model wrapper class
        """
        if not issubclass(model_class, BaseModelWrapper):
            raise ValueError("Model class must inherit from BaseModelWrapper")
        
        self._model_classes[model_type] = model_class
        logger.info(f"Registered model class: {model_type.value}")
    
    def create_model(self, model_type: ModelType, **kwargs) -> BaseModelWrapper:
        """
        Create model instance
        
        Args:
            model_type: Type of model to create
            **kwargs: Model parameters
            
        Returns:
            Model wrapper instance
        """
        if model_type not in self._model_classes:
            raise ValueError(f"Model type {model_type.value} not registered")
        
        model_class = self._model_classes[model_type]
        model_instance = model_class(model_type, **kwargs)
        
        self._models[model_type] = model_instance
        logger.info(f"Created model instance: {model_type.value}")
        
        return model_instance
    
    def get_model(self, model_type: ModelType) -> Optional[BaseModelWrapper]:
        """
        Get existing model instance
        
        Args:
            model_type: Type of model
            
        Returns:
            Model instance or None if not found
        """
        return self._models.get(model_type)
    
    def list_registered_models(self) -> List[ModelType]:
        """
        List all registered model types
        
        Returns:
            List of registered model types
        """
        return list(self._model_classes.keys())
    
    def list_created_models(self) -> List[ModelType]:
        """
        List all created model instances
        
        Returns:
            List of created model types
        """
        return list(self._models.keys())


# Global model registry instance
model_registry = ModelRegistry()