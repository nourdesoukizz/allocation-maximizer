"""
Prophet model wrapper for demand forecasting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import warnings

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction

logger = logging.getLogger(__name__)

# Suppress prophet warnings
warnings.filterwarnings('ignore', module='prophet')


class ProphetWrapper(BaseModelWrapper):
    """Prophet model wrapper using Facebook Prophet"""
    
    def __init__(self, **kwargs):
        """
        Initialize Prophet wrapper
        
        Args:
            **kwargs: Model parameters
        """
        super().__init__(ModelType.PROPHET, **kwargs)
        
        # Default Prophet parameters
        self.growth = kwargs.get('growth', 'linear')
        self.seasonality_mode = kwargs.get('seasonality_mode', 'additive')
        self.changepoint_prior_scale = kwargs.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = kwargs.get('seasonality_prior_scale', 10.0)
        self.holidays_prior_scale = kwargs.get('holidays_prior_scale', 10.0)
        self.daily_seasonality = kwargs.get('daily_seasonality', False)
        self.weekly_seasonality = kwargs.get('weekly_seasonality', True)
        self.yearly_seasonality = kwargs.get('yearly_seasonality', True)
        self.interval_width = kwargs.get('interval_width', 0.8)
        
        # Initialize model
        self.model = None
        self.forecast_df = None
        
        # Try to import Prophet
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import required ML libraries"""
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            
        except ImportError:
            try:
                from fbprophet import Prophet
                self.Prophet = Prophet
                logger.warning("Using deprecated fbprophet. Consider upgrading to prophet.")
                
            except ImportError as e:
                logger.error(f"Prophet library not available: {e}")
                raise ImportError(
                    "Prophet is required for Prophet model. "
                    "Install with: pip install prophet"
                )
    
    def _prepare_prophet_data(self, X: np.ndarray, y: np.ndarray, 
                             start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns)
        
        Args:
            X: Feature matrix (used for date generation if no dates provided)
            y: Target values
            start_date: Starting date for the time series
            
        Returns:
            DataFrame with 'ds' (datestamp) and 'y' (target) columns
        """
        n_points = len(y)
        
        # Generate dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=n_points-1)
        
        dates = [start_date + timedelta(days=i) for i in range(n_points)]
        
        df = pd.DataFrame({
            'ds': dates,
            'y': y
        })
        
        return df
    
    def _add_regressors(self, df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        """
        Add external regressors from feature matrix
        
        Args:
            df: Prophet dataframe with ds, y columns
            X: Feature matrix
            
        Returns:
            DataFrame with additional regressor columns
        """
        if X.shape[1] > 0 and self.feature_names:
            # Add features as regressors
            for i, feature_name in enumerate(self.feature_names):
                if i < X.shape[1]:
                    df[feature_name] = X[:, i]
        
        return df
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the Prophet model
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        self.validate_input(X, y)
        
        logger.info(f"Training Prophet model with {X.shape[0]} samples")
        
        # Prepare data for Prophet
        start_date = kwargs.get('start_date')
        df = self._prepare_prophet_data(X, y, start_date)
        
        # Initialize Prophet model
        self.model = self.Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width
        )
        
        # Add regressors if features are available
        if X.shape[1] > 0 and self.feature_names:
            df = self._add_regressors(df, X)
            
            # Add each feature as a regressor to the model
            for feature_name in self.feature_names:
                if feature_name in df.columns:
                    self.model.add_regressor(feature_name)
        
        # Fit the model
        try:
            self.model.fit(df)
            self.is_fitted = True
            logger.info("Prophet model training completed successfully")
            
        except Exception as e:
            logger.error(f"Prophet model training failed: {e}")
            raise ValueError(f"Failed to train Prophet model: {e}")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the trained Prophet model
        
        Args:
            X: Features for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelPrediction object
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        n_predictions = X.shape[0]
        
        # Create future dataframe
        last_date = kwargs.get('last_date', datetime.now())
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_predictions)]
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add regressors if features are available
        if X.shape[1] > 0 and self.feature_names:
            future_df = self._add_regressors(future_df, X)
        
        # Make predictions
        try:
            forecast = self.model.predict(future_df)
            self.forecast_df = forecast
            
            # Extract predictions and confidence intervals
            predictions = forecast['yhat'].values
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, 0)
            
            # Extract confidence intervals
            confidence_intervals = np.column_stack([
                forecast['yhat_lower'].values,
                forecast['yhat_upper'].values
            ])
            
            return ModelPrediction(
                model_type=self.model_type,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=self._get_component_importance(),
                model_params=self.model_params
            )
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            # Return zero predictions as fallback
            return ModelPrediction(
                model_type=self.model_type,
                predictions=np.zeros(n_predictions),
                confidence_intervals=None,
                feature_importance=None,
                model_params=self.model_params
            )
    
    def _get_component_importance(self) -> Optional[Dict[str, float]]:
        """
        Get component importance from Prophet model
        
        Returns:
            Dictionary of component importance scores
        """
        if self.forecast_df is None:
            return None
        
        importance = {}
        
        # Calculate importance based on component magnitudes
        components = ['trend', 'weekly', 'yearly']
        
        for component in components:
            if component in self.forecast_df.columns:
                # Use standard deviation as measure of importance
                importance[component] = float(self.forecast_df[component].std())
        
        # Add regressor importance if available
        if self.feature_names:
            for feature_name in self.feature_names:
                if feature_name in self.forecast_df.columns:
                    importance[feature_name] = float(self.forecast_df[feature_name].std())
        
        # Normalize importance scores
        if importance:
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from Prophet components
        
        Returns:
            Dictionary of feature importance scores
        """
        return self._get_component_importance()
    
    def get_forecast_components(self) -> Optional[pd.DataFrame]:
        """
        Get forecast components (trend, seasonality, etc.)
        
        Returns:
            DataFrame with forecast components
        """
        if not self.is_fitted or self.model is None:
            return None
        
        return self.forecast_df
    
    def plot_forecast(self, save_path: Optional[str] = None) -> None:
        """
        Plot Prophet forecast (requires matplotlib)
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.is_fitted or self.model is None or self.forecast_df is None:
            logger.warning("Model must be fitted and predictions made before plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            from prophet.plot import plot_plotly, plot_components_plotly
            
            # Plot forecast
            fig = self.model.plot(self.forecast_df)
            
            if save_path:
                plt.savefig(f"{save_path}_forecast.png")
                logger.info(f"Forecast plot saved to {save_path}_forecast.png")
            else:
                plt.show()
            
            # Plot components
            fig_comp = self.model.plot_components(self.forecast_df)
            
            if save_path:
                plt.savefig(f"{save_path}_components.png")
                logger.info(f"Components plot saved to {save_path}_components.png")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save Prophet model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        
        # Save the Prophet model
        joblib.dump(self.model, f"{filepath}_prophet_model.pkl")
        
        # Save forecast data if available
        if self.forecast_df is not None:
            self.forecast_df.to_csv(f"{filepath}_prophet_forecast.csv", index=False)
        
        logger.info(f"Prophet model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load Prophet model from file
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        import os
        
        # Load the Prophet model
        self.model = joblib.load(f"{filepath}_prophet_model.pkl")
        
        # Load forecast data if available
        forecast_path = f"{filepath}_prophet_forecast.csv"
        if os.path.exists(forecast_path):
            self.forecast_df = pd.read_csv(forecast_path)
        
        self.is_fitted = True
        logger.info(f"Prophet model loaded from {filepath}")
    
    def add_country_holidays(self, country: str) -> None:
        """
        Add country-specific holidays to the model
        
        Args:
            country: Country code (e.g., 'US', 'UK', 'CA')
        """
        if self.model is None:
            logger.warning("Model not initialized. Initialize before adding holidays.")
            return
        
        try:
            self.model.add_country_holidays(country_name=country)
            logger.info(f"Added {country} holidays to Prophet model")
            
        except Exception as e:
            logger.warning(f"Could not add holidays for {country}: {e}")
    
    def add_custom_seasonality(self, name: str, period: float, fourier_order: int) -> None:
        """
        Add custom seasonality to the model
        
        Args:
            name: Name of the seasonality
            period: Period of the seasonality (in days)
            fourier_order: Fourier order for the seasonality
        """
        if self.model is None:
            logger.warning("Model not initialized. Initialize before adding seasonality.")
            return
        
        try:
            self.model.add_seasonality(
                name=name,
                period=period,
                fourier_order=fourier_order
            )
            logger.info(f"Added custom seasonality '{name}' with period {period}")
            
        except Exception as e:
            logger.error(f"Could not add seasonality '{name}': {e}")
    
    def cross_validate(self, initial: str, period: str, horizon: str) -> Optional[pd.DataFrame]:
        """
        Perform cross-validation on the Prophet model
        
        Args:
            initial: Size of initial training period (e.g., '730 days')
            period: Spacing between cutoff dates (e.g., '180 days')
            horizon: Forecast horizon (e.g., '365 days')
            
        Returns:
            DataFrame with cross-validation results
        """
        if not self.is_fitted or self.model is None:
            logger.warning("Model must be fitted before cross-validation")
            return None
        
        try:
            from prophet.diagnostics import cross_validation
            
            cv_results = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon
            )
            
            logger.info("Cross-validation completed successfully")
            return cv_results
            
        except ImportError:
            logger.error("Prophet diagnostics not available")
            return None
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return None