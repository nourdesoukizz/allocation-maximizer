"""
XGBoost model wrapper for demand forecasting
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List
import warnings

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction

logger = logging.getLogger(__name__)


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost model wrapper"""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost wrapper
        
        Args:
            **kwargs: Model parameters
        """
        super().__init__(ModelType.XGBOOST, **kwargs)
        
        # Default XGBoost parameters
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 6)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.subsample = kwargs.get('subsample', 1.0)
        self.colsample_bytree = kwargs.get('colsample_bytree', 1.0)
        self.reg_alpha = kwargs.get('reg_alpha', 0)
        self.reg_lambda = kwargs.get('reg_lambda', 1)
        self.random_state = kwargs.get('random_state', 42)
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        self.eval_metric = kwargs.get('eval_metric', 'rmse')
        
        # Initialize model
        self.model = None
        self.feature_importance_ = None
        
        # Try to import XGBoost
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import required ML libraries"""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            
            self.xgb = xgb
            self.train_test_split = train_test_split
            
        except ImportError as e:
            logger.error(f"XGBoost not available: {e}")
            raise ImportError(
                "XGBoost and scikit-learn are required for XGBoost model. "
                "Install with: pip install xgboost scikit-learn"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        self.validate_input(X, y)
        
        logger.info(f"Training XGBoost model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data for early stopping
        test_size = kwargs.get('test_size', 0.2)
        if X.shape[0] > 10:  # Only split if we have enough data
            X_train, X_val, y_train, y_val = self.train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Initialize XGBoost model
        self.model = self.xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds if X.shape[0] > 10 else None
        )
        
        # Fit the model with early stopping
        try:
            if X.shape[0] > 10:  # Use early stopping only with sufficient data
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
            
            self.is_fitted = True
            self.feature_importance_ = self.model.feature_importances_
            
            logger.info(f"XGBoost training completed. Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise ValueError(f"Failed to train XGBoost model: {e}")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the trained XGBoost model
        
        Args:
            X: Features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelPrediction object
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        try:
            # Make predictions
            predictions = self.model.predict(X)
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, 0)
            
            # Calculate prediction intervals if requested
            confidence_intervals = None
            if kwargs.get('return_confidence', False):
                confidence_intervals = self._calculate_prediction_intervals(X, predictions)
            
            return ModelPrediction(
                model_type=self.model_type,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=self.get_feature_importance(),
                model_params=self.model_params
            )
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            # Return zero predictions as fallback
            return ModelPrediction(
                model_type=self.model_type,
                predictions=np.zeros(X.shape[0]),
                confidence_intervals=None,
                feature_importance=self.get_feature_importance(),
                model_params=self.model_params
            )
    
    def _calculate_prediction_intervals(self, X: np.ndarray, predictions: np.ndarray, 
                                      alpha: float = 0.1) -> np.ndarray:
        """
        Calculate prediction intervals using quantile regression
        
        Args:
            X: Input features
            predictions: Point predictions
            alpha: Significance level (0.1 for 90% intervals)
            
        Returns:
            Array of shape (n_samples, 2) with lower and upper bounds
        """
        try:
            # Train quantile regressors for lower and upper bounds
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            # Use a simple approach: assume normal distribution around predictions
            # In practice, you might want to train separate quantile regression models
            residual_std = np.std(predictions) if len(predictions) > 1 else 0.1 * np.mean(predictions)
            
            from scipy import stats
            z_score = stats.norm.ppf(upper_quantile)
            
            lower_bounds = predictions - z_score * residual_std
            upper_bounds = predictions + z_score * residual_std
            
            # Ensure bounds are positive
            lower_bounds = np.maximum(lower_bounds, 0)
            upper_bounds = np.maximum(upper_bounds, lower_bounds)
            
            return np.column_stack([lower_bounds, upper_bounds])
            
        except Exception as e:
            logger.warning(f"Could not calculate prediction intervals: {e}")
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores from XGBoost
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if not self.is_fitted or self.feature_importance_ is None:
            return None
        
        if not self.feature_names:
            # Use generic feature names if specific names not available
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance_))]
        else:
            feature_names = self.feature_names
        
        # Ensure we have the right number of feature names
        n_features = len(self.feature_importance_)
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        elif len(feature_names) < n_features:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
        
        return dict(zip(feature_names, self.feature_importance_))
    
    def get_booster_feature_importance(self, importance_type: str = 'gain') -> Optional[Dict[str, float]]:
        """
        Get detailed feature importance from XGBoost booster
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted or self.model is None:
            return None
        
        try:
            booster = self.model.get_booster()
            importance = booster.get_score(importance_type=importance_type)
            
            # Map feature indices to names
            if self.feature_names:
                mapped_importance = {}
                for key, value in importance.items():
                    # XGBoost uses f0, f1, f2, ... as feature names
                    if key.startswith('f'):
                        try:
                            feature_idx = int(key[1:])
                            if feature_idx < len(self.feature_names):
                                feature_name = self.feature_names[feature_idx]
                                mapped_importance[feature_name] = value
                        except (ValueError, IndexError):
                            mapped_importance[key] = value
                    else:
                        mapped_importance[key] = value
                
                return mapped_importance
            else:
                return importance
                
        except Exception as e:
            logger.warning(f"Could not get booster feature importance: {e}")
            return self.get_feature_importance()
    
    def plot_feature_importance(self, max_features: int = 20, 
                               importance_type: str = 'gain',
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            max_features: Maximum number of features to plot
            importance_type: Type of importance to plot
            save_path: Path to save the plot (optional)
        """
        importance = self.get_booster_feature_importance(importance_type)
        
        if not importance:
            logger.warning("No feature importance available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            sorted_features = sorted_features[:max_features]
            
            features, scores = zip(*sorted_features)
            
            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel(f'Importance ({importance_type})')
            plt.title('XGBoost Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save XGBoost model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Save using XGBoost's native format
        self.model.save_model(f"{filepath}_xgboost_model.json")
        
        # Also save using joblib for compatibility
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params
        }, f"{filepath}_xgboost_complete.pkl")
        
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load XGBoost model from file
        
        Args:
            filepath: Path to load the model from
        """
        import os
        
        # Try loading from native format first
        native_path = f"{filepath}_xgboost_model.json"
        complete_path = f"{filepath}_xgboost_complete.pkl"
        
        try:
            if os.path.exists(complete_path):
                # Load complete model with metadata
                import joblib
                data = joblib.load(complete_path)
                self.model = data['model']
                self.feature_names = data.get('feature_names', [])
                self.model_params.update(data.get('model_params', {}))
                
            elif os.path.exists(native_path):
                # Load from native format
                self.model = self.xgb.XGBRegressor()
                self.model.load_model(native_path)
                
            else:
                raise FileNotFoundError(f"No XGBoost model found at {filepath}")
            
            self.is_fitted = True
            self.feature_importance_ = getattr(self.model, 'feature_importances_', None)
            
            logger.info(f"XGBoost model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise ValueError(f"Failed to load XGBoost model from {filepath}: {e}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return self.model_params
        
        try:
            return self.model.get_params()
        except Exception:
            return self.model_params
    
    def set_model_params(self, **params) -> None:
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        self.model_params.update(params)
        
        if self.model is not None:
            try:
                self.model.set_params(**params)
            except Exception as e:
                logger.warning(f"Could not set model parameters: {e}")
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history and evaluation results
        
        Returns:
            Dictionary with training information
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        try:
            eval_results = getattr(self.model, 'evals_result_', {})
            best_iteration = getattr(self.model, 'best_iteration', None)
            best_score = getattr(self.model, 'best_score', None)
            
            return {
                'eval_results': eval_results,
                'best_iteration': best_iteration,
                'best_score': best_score,
                'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
            }
            
        except Exception as e:
            logger.warning(f"Could not retrieve training history: {e}")
            return {}