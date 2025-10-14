"""
SARIMA model wrapper for demand forecasting
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import warnings

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction

logger = logging.getLogger(__name__)

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', module='statsmodels')


class SARIMAWrapper(BaseModelWrapper):
    """SARIMA model wrapper using statsmodels"""
    
    def __init__(self, **kwargs):
        """
        Initialize SARIMA wrapper
        
        Args:
            **kwargs: Model parameters
        """
        super().__init__(ModelType.SARIMA, **kwargs)
        
        # Default SARIMA parameters
        self.order = kwargs.get('order', (1, 1, 1))  # (p, d, q)
        self.seasonal_order = kwargs.get('seasonal_order', (1, 1, 1, 12))  # (P, D, Q, s)
        self.trend = kwargs.get('trend', 'c')  # constant trend
        self.enforce_stationarity = kwargs.get('enforce_stationarity', True)
        self.enforce_invertibility = kwargs.get('enforce_invertibility', True)
        self.concentrate_scale = kwargs.get('concentrate_scale', True)
        
        # Initialize model components
        self.model = None
        self.fitted_model = None
        self.residuals = None
        self.aic = None
        self.bic = None
        
        # Try to import required libraries
        self._import_dependencies()
    
    def _import_dependencies(self):
        """Import required ML libraries"""
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.stattools import adfuller
            from scipy import stats
            
            self.sm = sm
            self.SARIMAX = SARIMAX
            self.seasonal_decompose = seasonal_decompose
            self.adfuller = adfuller
            self.stats = stats
            
        except ImportError as e:
            logger.error(f"Required libraries not available for SARIMA: {e}")
            raise ImportError(
                "statsmodels and scipy are required for SARIMA model. "
                "Install with: pip install statsmodels scipy"
            )
    
    def _check_stationarity(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Check if time series is stationary using ADF test
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (is_stationary, p_value)
        """
        try:
            adf_result = self.adfuller(data)
            p_value = adf_result[1]
            is_stationary = p_value < 0.05
            
            logger.info(f"ADF test p-value: {p_value:.4f}, Stationary: {is_stationary}")
            return is_stationary, p_value
            
        except Exception as e:
            logger.warning(f"Could not perform stationarity test: {e}")
            return False, 1.0
    
    def _auto_select_order(self, data: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Auto-select SARIMA order using grid search with AIC
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        try:
            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_seasonal_order = (1, 1, 1, 12)
            
            # Limited grid search to avoid long computation
            p_values = [0, 1, 2]
            d_values = [0, 1]
            q_values = [0, 1, 2]
            
            P_values = [0, 1]
            D_values = [0, 1]
            Q_values = [0, 1]
            s_value = 12  # Monthly seasonality
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    try:
                                        model = self.SARIMAX(
                                            data,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s_value),
                                            trend=self.trend,
                                            enforce_stationarity=self.enforce_stationarity,
                                            enforce_invertibility=self.enforce_invertibility,
                                            concentrate_scale=self.concentrate_scale
                                        )
                                        
                                        fitted = model.fit(disp=False)
                                        
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, s_value)
                                            
                                    except Exception:
                                        continue
            
            logger.info(f"Auto-selected SARIMA order: {best_order}, seasonal: {best_seasonal_order}, AIC: {best_aic:.2f}")
            return best_order, best_seasonal_order
            
        except Exception as e:
            logger.warning(f"Auto-selection failed, using default parameters: {e}")
            return self.order, self.seasonal_order
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the SARIMA model
        
        Args:
            X: Training features (used for date information if available)
            y: Training targets (time series data)
            **kwargs: Additional training parameters
        """
        self.validate_input(X, y)
        
        if len(y) < 24:  # Need at least 2 years of monthly data for seasonality
            logger.warning(f"SARIMA requires more data points. Got {len(y)}, recommended minimum: 24")
        
        logger.info(f"Training SARIMA model with {len(y)} time points")
        
        # Use y as time series data
        time_series = y.copy()
        
        # Check stationarity
        is_stationary, p_value = self._check_stationarity(time_series)
        
        # Auto-select parameters if requested
        if kwargs.get('auto_order', False):
            self.order, self.seasonal_order = self._auto_select_order(time_series)
        
        # Initialize SARIMA model
        self.model = self.SARIMAX(
            time_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            concentrate_scale=self.concentrate_scale
        )
        
        # Fit the model
        try:
            self.fitted_model = self.model.fit(disp=False)
            
            self.is_fitted = True
            self.residuals = self.fitted_model.resid
            self.aic = self.fitted_model.aic
            self.bic = self.fitted_model.bic
            
            logger.info(f"SARIMA training completed successfully")
            logger.info(f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
            logger.info(f"Log-likelihood: {self.fitted_model.llf:.2f}")
            
        except Exception as e:
            logger.error(f"SARIMA training failed: {e}")
            raise ValueError(f"Failed to train SARIMA model: {e}")
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the trained SARIMA model
        
        Args:
            X: Features for prediction (number of steps to forecast)
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelPrediction object
        """
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        # Number of steps to forecast
        n_forecast = X.shape[0]
        
        try:
            # Make forecast
            forecast = self.fitted_model.get_forecast(steps=n_forecast)
            
            # Get predictions and confidence intervals
            predictions = forecast.predicted_mean.values
            confidence_intervals = forecast.conf_int().values
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, 0)
            confidence_intervals = np.maximum(confidence_intervals, 0)
            
            return ModelPrediction(
                model_type=self.model_type,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=self._get_component_importance(),
                model_params=self.model_params
            )
            
        except Exception as e:
            logger.error(f"SARIMA prediction failed: {e}")
            # Return zero predictions as fallback
            return ModelPrediction(
                model_type=self.model_type,
                predictions=np.zeros(n_forecast),
                confidence_intervals=None,
                feature_importance=None,
                model_params=self.model_params
            )
    
    def _get_component_importance(self) -> Optional[Dict[str, float]]:
        """
        Get component importance from SARIMA model parameters
        
        Returns:
            Dictionary of component importance scores
        """
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        try:
            params = self.fitted_model.params
            importance = {}
            
            # AR parameters
            ar_params = []
            seasonal_ar_params = []
            
            for param_name, param_value in params.items():
                if 'ar' in param_name.lower():
                    if 'seasonal' in param_name.lower():
                        seasonal_ar_params.append(abs(param_value))
                    else:
                        ar_params.append(abs(param_value))
            
            # Calculate importance scores
            if ar_params:
                importance['autoregressive'] = float(np.mean(ar_params))
            
            if seasonal_ar_params:
                importance['seasonal_autoregressive'] = float(np.mean(seasonal_ar_params))
            
            # Trend importance (if trend is included)
            if 'const' in params.index:
                importance['trend'] = float(abs(params['const']))
            
            # Normalize importance scores
            if importance:
                total_importance = sum(importance.values())
                if total_importance > 0:
                    importance = {k: v/total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.warning(f"Could not calculate component importance: {e}")
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from SARIMA components
        
        Returns:
            Dictionary of component importance scores
        """
        return self._get_component_importance()
    
    def get_residual_diagnostics(self) -> Optional[Dict[str, Any]]:
        """
        Get residual diagnostics for model evaluation
        
        Returns:
            Dictionary with residual statistics
        """
        if not self.is_fitted or self.residuals is None:
            return None
        
        try:
            # Basic residual statistics
            residual_mean = float(np.mean(self.residuals))
            residual_std = float(np.std(self.residuals))
            residual_min = float(np.min(self.residuals))
            residual_max = float(np.max(self.residuals))
            
            # Ljung-Box test for autocorrelation
            ljung_box = self.sm.stats.diagnostic.acorr_ljungbox(
                self.residuals, lags=min(10, len(self.residuals)//4), return_df=True
            )
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = self.stats.jarque_bera(self.residuals)
            
            return {
                'mean': residual_mean,
                'std': residual_std,
                'min': residual_min,
                'max': residual_max,
                'ljung_box_pvalue': float(ljung_box['lb_pvalue'].iloc[-1]) if not ljung_box.empty else None,
                'jarque_bera_pvalue': float(jb_pvalue),
                'normality_test_passed': jb_pvalue > 0.05,
                'autocorrelation_test_passed': ljung_box['lb_pvalue'].iloc[-1] > 0.05 if not ljung_box.empty else None
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate residual diagnostics: {e}")
            return None
    
    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """
        Plot SARIMA diagnostic plots
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.is_fitted or self.fitted_model is None:
            logger.warning("Model must be fitted before plotting diagnostics")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create diagnostic plots
            fig = self.fitted_model.plot_diagnostics(figsize=(12, 8))
            
            if save_path:
                plt.savefig(f"{save_path}_sarima_diagnostics.png")
                logger.info(f"SARIMA diagnostics saved to {save_path}_sarima_diagnostics.png")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {e}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save SARIMA model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Save the fitted model
        self.fitted_model.save(f"{filepath}_sarima_model.pkl")
        
        # Save additional metadata
        import joblib
        metadata = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'model_params': self.model_params,
            'aic': self.aic,
            'bic': self.bic
        }
        
        joblib.dump(metadata, f"{filepath}_sarima_metadata.pkl")
        logger.info(f"SARIMA model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load SARIMA model from file
        
        Args:
            filepath: Path to load the model from
        """
        import os
        
        model_path = f"{filepath}_sarima_model.pkl"
        metadata_path = f"{filepath}_sarima_metadata.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SARIMA model file not found: {model_path}")
        
        # Load the fitted model
        self.fitted_model = self.sm.load(model_path)
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            import joblib
            metadata = joblib.load(metadata_path)
            self.order = metadata.get('order', self.order)
            self.seasonal_order = metadata.get('seasonal_order', self.seasonal_order)
            self.model_params.update(metadata.get('model_params', {}))
            self.aic = metadata.get('aic')
            self.bic = metadata.get('bic')
        
        self.residuals = self.fitted_model.resid
        self.is_fitted = True
        
        logger.info(f"SARIMA model loaded from {filepath}")
    
    def get_model_summary(self) -> Optional[str]:
        """
        Get model summary statistics
        
        Returns:
            Model summary as string
        """
        if not self.is_fitted or self.fitted_model is None:
            return None
        
        try:
            return str(self.fitted_model.summary())
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")
            return None
    
    def get_information_criteria(self) -> Dict[str, float]:
        """
        Get model information criteria
        
        Returns:
            Dictionary with AIC, BIC, and other criteria
        """
        if not self.is_fitted or self.fitted_model is None:
            return {}
        
        try:
            return {
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'hqic': float(self.fitted_model.hqic),
                'log_likelihood': float(self.fitted_model.llf)
            }
        except Exception as e:
            logger.warning(f"Could not get information criteria: {e}")
            return {}