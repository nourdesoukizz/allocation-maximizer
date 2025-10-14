"""
Random Forest model wrapper for demand forecasting
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List
import warnings

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction

logger = logging.getLogger(__name__)


class RandomForestWrapper(BaseModelWrapper):
    """Random Forest model wrapper"""
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest wrapper
        
        Args:
            **kwargs: Model parameters
        """
        super().__init__(ModelType.RANDOM_FOREST, **kwargs)
        
        # Default Random Forest parameters
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.max_features = kwargs.get('max_features', 'sqrt')
        self.bootstrap = kwargs.get('bootstrap', True)
        self.random_state = kwargs.get('random_state', 42)
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.oob_score = kwargs.get('oob_score', True)
        
        # Initialize model
        self.model = None
        self.feature_importance_ = None
        self.oob_score_ = None
        
        # Try to import scikit-learn
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import required ML libraries"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            self.RandomForestRegressor = RandomForestRegressor
            self.train_test_split = train_test_split
            self.mean_squared_error = mean_squared_error
            self.mean_absolute_error = mean_absolute_error
            
        except ImportError as e:
            logger.error(f"Scikit-learn not available: {e}")
            raise ImportError(
                "Scikit-learn is required for Random Forest model. "
                "Install with: pip install scikit-learn"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the Random Forest model
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        self.validate_input(X, y)
        
        logger.info(f"Training Random Forest model with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize Random Forest model
        self.model = self.RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=self.oob_score if self.bootstrap else False
        )
        
        # Fit the model
        try:
            self.model.fit(X, y)
            
            self.is_fitted = True
            self.feature_importance_ = self.model.feature_importances_
            self.oob_score_ = getattr(self.model, 'oob_score_', None)
            
            logger.info(f"Random Forest training completed successfully")
            if self.oob_score_ is not None:
                logger.info(f"Out-of-bag score: {self.oob_score_:.4f}")
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise ValueError(f"Failed to train Random Forest model: {e}")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the trained Random Forest model
        
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
            
            # Calculate prediction intervals using tree predictions
            confidence_intervals = None
            if kwargs.get('return_confidence', False):
                confidence_intervals = self._calculate_prediction_intervals(X)
            
            return ModelPrediction(
                model_type=self.model_type,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=self.get_feature_importance(),
                model_params=self.model_params
            )
            
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            # Return zero predictions as fallback
            return ModelPrediction(
                model_type=self.model_type,
                predictions=np.zeros(X.shape[0]),
                confidence_intervals=None,
                feature_importance=self.get_feature_importance(),
                model_params=self.model_params
            )
    
    def _calculate_prediction_intervals(self, X: np.ndarray, alpha: float = 0.1) -> Optional[np.ndarray]:
        """
        Calculate prediction intervals using individual tree predictions
        
        Args:
            X: Input features
            alpha: Significance level (0.1 for 90% intervals)
            
        Returns:
            Array of shape (n_samples, 2) with lower and upper bounds
        """
        try:
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            
            # Calculate percentiles
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bounds = np.percentile(tree_predictions, lower_percentile, axis=0)
            upper_bounds = np.percentile(tree_predictions, upper_percentile, axis=0)
            
            # Ensure bounds are positive
            lower_bounds = np.maximum(lower_bounds, 0)
            upper_bounds = np.maximum(upper_bounds, lower_bounds)
            
            return np.column_stack([lower_bounds, upper_bounds])
            
        except Exception as e:
            logger.warning(f"Could not calculate prediction intervals: {e}")
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores from Random Forest
        
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
    
    def get_tree_feature_importance_stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get detailed feature importance statistics across all trees
        
        Returns:
            Dictionary with mean, std, min, max importance for each feature
        """
        if not self.is_fitted or self.model is None:
            return None
        
        try:
            # Get feature importances from all trees
            tree_importances = np.array([
                tree.feature_importances_ for tree in self.model.estimators_
            ])
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(tree_importances.shape[1])]
            
            stats = {}
            for i, feature_name in enumerate(feature_names):
                if i < tree_importances.shape[1]:
                    feature_importances = tree_importances[:, i]
                    stats[feature_name] = {
                        'mean': float(np.mean(feature_importances)),
                        'std': float(np.std(feature_importances)),
                        'min': float(np.min(feature_importances)),
                        'max': float(np.max(feature_importances)),
                        'median': float(np.median(feature_importances))
                    }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Could not calculate tree importance statistics: {e}")
            return None
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score
        
        Returns:
            OOB score if available
        """
        return self.oob_score_
    
    def partial_dependence_plot(self, X: np.ndarray, feature_names: List[str], 
                               save_path: Optional[str] = None) -> None:
        """
        Create partial dependence plots for selected features
        
        Args:
            X: Sample data for plotting
            feature_names: Names of features to plot
            save_path: Path to save the plots (optional)
        """
        if not self.is_fitted or self.model is None:
            logger.warning("Model must be fitted before plotting")
            return
        
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt
            
            # Map feature names to indices
            available_features = self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            feature_indices = []
            
            for feature_name in feature_names:
                if feature_name in available_features:
                    feature_indices.append(available_features.index(feature_name))
                else:
                    logger.warning(f"Feature '{feature_name}' not found in available features")
            
            if not feature_indices:
                logger.warning("No valid features found for partial dependence plot")
                return
            
            # Create partial dependence plot
            fig, ax = plt.subplots(figsize=(12, 4))
            display = PartialDependenceDisplay.from_estimator(
                self.model, X, feature_indices, 
                feature_names=available_features,
                ax=ax
            )
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Partial dependence plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Required libraries not available for partial dependence plotting")
        except Exception as e:
            logger.error(f"Error creating partial dependence plot: {e}")
    
    def plot_feature_importance(self, max_features: int = 20, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance with error bars
        
        Args:
            max_features: Maximum number of features to plot
            save_path: Path to save the plot (optional)
        """
        importance = self.get_feature_importance()
        importance_stats = self.get_tree_feature_importance_stats()
        
        if not importance:
            logger.warning("No feature importance available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            sorted_features = sorted_features[:max_features]
            
            features, scores = zip(*sorted_features)
            
            # Get error bars if statistics are available
            yerr = None
            if importance_stats:
                yerr = [importance_stats[feature]['std'] for feature in features]
            
            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            bars = plt.barh(range(len(features)), scores, yerr=yerr if yerr else None, capsize=3)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            
            # Add OOB score to the plot if available
            if self.oob_score_ is not None:
                plt.figtext(0.02, 0.02, f'OOB Score: {self.oob_score_:.4f}', fontsize=10)
            
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
        Save Random Forest model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        
        # Save complete model with metadata
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'feature_importance_': self.feature_importance_,
            'oob_score_': self.oob_score_
        }, f"{filepath}_rf_model.pkl")
        
        logger.info(f"Random Forest model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load Random Forest model from file
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        try:
            data = joblib.load(f"{filepath}_rf_model.pkl")
            
            self.model = data['model']
            self.feature_names = data.get('feature_names', [])
            self.model_params.update(data.get('model_params', {}))
            self.feature_importance_ = data.get('feature_importance_')
            self.oob_score_ = data.get('oob_score_')
            
            self.is_fitted = True
            logger.info(f"Random Forest model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load Random Forest model: {e}")
            raise ValueError(f"Failed to load Random Forest model from {filepath}: {e}")
    
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
    
    def get_tree_count(self) -> Optional[int]:
        """
        Get the number of trees in the forest
        
        Returns:
            Number of trees
        """
        if not self.is_fitted or self.model is None:
            return None
        
        return len(self.model.estimators_)
    
    def get_model_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size and complexity
        
        Returns:
            Dictionary with model size information
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        try:
            n_trees = self.get_tree_count()
            n_features = getattr(self.model, 'n_features_in_', None)
            
            # Estimate memory usage (approximate)
            total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
            
            return {
                'n_trees': n_trees,
                'n_features': n_features,
                'total_nodes': total_nodes,
                'avg_nodes_per_tree': total_nodes / n_trees if n_trees > 0 else 0,
                'oob_score': self.oob_score_
            }
            
        except Exception as e:
            logger.warning(f"Could not get model size info: {e}")
            return {}