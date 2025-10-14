import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMAResults
import itertools
from scipy import stats
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class SARIMAForecasting:
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) for Spare Parts Forecasting
    
    Handles seasonal patterns in spare parts demand using advanced time series modeling
    with automatic parameter selection and comprehensive seasonal analysis.
    """
    
    def __init__(self, 
                 seasonal_period: int = 12,
                 max_p: int = 3,
                 max_d: int = 2,
                 max_q: int = 3,
                 max_P: int = 2,
                 max_D: int = 1,
                 max_Q: int = 2,
                 auto_arima: bool = True):
        """
        Initialize SARIMA forecasting model
        
        Args:
            seasonal_period: Seasonal period (12 for monthly data)
            max_p, max_d, max_q: Maximum non-seasonal parameters
            max_P, max_D, max_Q: Maximum seasonal parameters
            auto_arima: Whether to use automatic ARIMA parameter selection
        """
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.auto_arima = auto_arima
        
        self.models = {}
        self.model_params = {}
        self.seasonal_analysis = {}
        self.forecast_results = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load spare parts data from Excel file
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            DataFrame with processed data
        """
        try:
            # Read Excel file properly handling the structure
            xl_file = pd.ExcelFile(file_path)
            raw_data = pd.read_excel(xl_file, header=None)
            
            # Extract years and months
            years = raw_data.iloc[0, 3:].values
            months = raw_data.iloc[1, 3:].values
            
            # Create proper column names and date mapping
            time_columns = []
            date_mapping = {}
            
            for i, (year, month) in enumerate(zip(years, months)):
                if pd.notna(year) and pd.notna(month) and month != 'Total':
                    col_name = f"{int(year)}-{month}"
                    time_columns.append(col_name)
                    # Create actual date for time series
                    month_num = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }[month]
                    date_mapping[col_name] = datetime(int(year), month_num, 1)
            
            # Extract item information and data
            items_data = []
            for idx in range(2, len(raw_data)):
                row = raw_data.iloc[idx]
                if pd.notna(row.iloc[0]):  # If item ID exists
                    item_info = {
                        'item_id': row.iloc[0],
                        'item_name': row.iloc[1],
                        'category': row.iloc[2]
                    }
                    
                    # Extract demand data (skip Total columns)
                    demand_data = []
                    col_idx = 3
                    for year in [2021, 2022, 2023, 2024, 2025]:
                        for month in ['January', 'February', 'March', 'April', 'May', 'June',
                                    'July', 'August', 'September', 'October', 'November', 'December']:
                            col_name = f"{year}-{month}"
                            if col_idx < len(row) and col_name in time_columns:
                                demand_data.append(row.iloc[col_idx] if pd.notna(row.iloc[col_idx]) else 0)
                            col_idx += 1
                        col_idx += 1  # Skip Total column
                        if year == 2025:  # 2025 data is incomplete
                            break
            
                    item_info['demand_data'] = demand_data[:len(time_columns)]
                    items_data.append(item_info)
            
            # Create DataFrame with time series
            df = pd.DataFrame(items_data)
            
            # Create time series columns
            for i, col_name in enumerate(time_columns):
                df[col_name] = df['demand_data'].apply(lambda x: x[i] if i < len(x) else 0)
            
            # Drop temporary column
            df = df.drop('demand_data', axis=1)
            
            # Store time information
            self.time_columns = time_columns
            self.date_mapping = date_mapping
            self.dates = [date_mapping[col] for col in time_columns]
            
            print(f"Loaded data for {len(df)} items with {len(time_columns)} time periods")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def test_stationarity(self, timeseries: pd.Series, item_id: str) -> Dict:
        """
        Test stationarity of time series using ADF and KPSS tests
        
        Args:
            timeseries: Time series data
            item_id: Item identifier
            
        Returns:
            Dictionary with stationarity test results
        """
        results = {}
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(timeseries.dropna())
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_critical_values'] = adf_result[4]
            results['is_stationary_adf'] = adf_result[1] < 0.05
        except Exception as e:
            print(f"ADF test failed for {item_id}: {e}")
            results['is_stationary_adf'] = False
        
        # KPSS test
        try:
            kpss_result = kpss(timeseries.dropna())
            results['kpss_statistic'] = kpss_result[0]
            results['kpss_pvalue'] = kpss_result[1]
            results['kpss_critical_values'] = kpss_result[3]
            results['is_stationary_kpss'] = kpss_result[1] > 0.05
        except Exception as e:
            print(f"KPSS test failed for {item_id}: {e}")
            results['is_stationary_kpss'] = False
        
        # Combined assessment
        results['is_stationary'] = results.get('is_stationary_adf', False) and results.get('is_stationary_kpss', False)
        
        return results
    
    def analyze_seasonality(self, timeseries: pd.Series, item_id: str) -> Dict:
        """
        Analyze seasonal patterns in the time series
        
        Args:
            timeseries: Time series data
            item_id: Item identifier
            
        Returns:
            Dictionary with seasonal analysis results
        """
        analysis = {}
        
        try:
            # Only perform seasonal decomposition if we have enough data
            if len(timeseries) >= 2 * self.seasonal_period:
                # Seasonal decomposition
                decomposition = seasonal_decompose(
                    timeseries, 
                    model='additive', 
                    period=self.seasonal_period,
                    extrapolate_trend='freq'
                )
                
                # Store decomposition components
                analysis['trend'] = decomposition.trend.tolist()
                analysis['seasonal'] = decomposition.seasonal.tolist()
                analysis['residual'] = decomposition.resid.tolist()
                
                # Calculate seasonal strength
                seasonal_var = np.var(decomposition.seasonal.dropna())
                residual_var = np.var(decomposition.resid.dropna())
                
                if residual_var > 0:
                    analysis['seasonal_strength'] = seasonal_var / (seasonal_var + residual_var)
                else:
                    analysis['seasonal_strength'] = 0
                
                # Detect seasonal pattern
                analysis['has_seasonality'] = analysis['seasonal_strength'] > 0.1
                
                # Monthly seasonal indices
                seasonal_indices = {}
                seasonal_component = decomposition.seasonal.dropna()
                if len(seasonal_component) >= 12:
                    for month in range(1, 13):
                        month_indices = [seasonal_component.iloc[i] for i in range(len(seasonal_component)) 
                                       if (i % 12) == (month - 1)]
                        if month_indices:
                            seasonal_indices[month] = np.mean(month_indices)
                
                analysis['seasonal_indices'] = seasonal_indices
                
            else:
                # Insufficient data for seasonal decomposition
                analysis['has_seasonality'] = False
                analysis['seasonal_strength'] = 0
                analysis['seasonal_indices'] = {}
                print(f"Insufficient data for seasonal analysis of {item_id}")
                
        except Exception as e:
            print(f"Seasonal analysis failed for {item_id}: {e}")
            analysis['has_seasonality'] = False
            analysis['seasonal_strength'] = 0
            analysis['seasonal_indices'] = {}
        
        return analysis
    
    def auto_arima_selection(self, timeseries: pd.Series, item_id: str) -> Tuple[tuple, Dict]:
        """
        Automatically select SARIMA parameters using pmdarima
        
        Args:
            timeseries: Time series data
            item_id: Item identifier
            
        Returns:
            Tuple of (order, seasonal_order) and model information
        """
        try:
            # Use pmdarima for automatic ARIMA parameter selection
            auto_model = pm.auto_arima(
                timeseries,
                start_p=0, start_q=0,
                max_p=self.max_p, max_q=self.max_q, max_d=self.max_d,
                start_P=0, start_Q=0,
                max_P=self.max_P, max_Q=self.max_Q, max_D=self.max_D,
                seasonal=True, m=self.seasonal_period,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_fits=50
            )
            
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            
            model_info = {
                'aic': auto_model.aic(),
                'bic': auto_model.bic(),
                'method': 'auto_arima',
                'converged': True
            }
            
            return (order, seasonal_order), model_info
            
        except Exception as e:
            print(f"Auto ARIMA failed for {item_id}: {e}")
            # Fallback to simple parameters
            return ((1, 1, 1), (1, 1, 1, self.seasonal_period)), {'method': 'fallback', 'converged': False}
    
    def grid_search_sarima(self, timeseries: pd.Series, item_id: str) -> Tuple[tuple, Dict]:
        """
        Grid search for optimal SARIMA parameters
        
        Args:
            timeseries: Time series data
            item_id: Item identifier
            
        Returns:
            Tuple of (order, seasonal_order) and model information
        """
        best_aic = np.inf
        best_params = None
        best_seasonal_params = None
        best_model_info = {}
        
        # Parameter ranges
        p_range = range(0, self.max_p + 1)
        d_range = range(0, self.max_d + 1)
        q_range = range(0, self.max_q + 1)
        P_range = range(0, self.max_P + 1)
        D_range = range(0, self.max_D + 1)
        Q_range = range(0, self.max_Q + 1)
        
        # Grid search
        for params in itertools.product(p_range, d_range, q_range):
            for seasonal_params in itertools.product(P_range, D_range, Q_range):
                try:
                    seasonal_order = seasonal_params + (self.seasonal_period,)
                    
                    model = SARIMAX(
                        timeseries,
                        order=params,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    fitted_model = model.fit(disp=False)
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = params
                        best_seasonal_params = seasonal_order
                        best_model_info = {
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'method': 'grid_search',
                            'converged': fitted_model.mle_retvals['converged']
                        }
                        
                except Exception:
                    continue
        
        if best_params is None:
            # Fallback to simple parameters
            best_params = (1, 1, 1)
            best_seasonal_params = (1, 1, 1, self.seasonal_period)
            best_model_info = {'method': 'fallback', 'converged': False}
        
        return (best_params, best_seasonal_params), best_model_info
    
    def fit_sarima_model(self, timeseries: pd.Series, order: tuple, seasonal_order: tuple, item_id: str):
        """
        Fit SARIMA model with given parameters
        
        Args:
            timeseries: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            item_id: Item identifier
            
        Returns:
            Fitted SARIMAX model
        """
        try:
            model = SARIMAX(
                timeseries,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=100)
            return fitted_model
            
        except Exception as e:
            print(f"Model fitting failed for {item_id}: {e}")
            return None
    
    def generate_forecasts(self, fitted_model, steps: int = 12) -> Dict:
        """
        Generate forecasts using fitted SARIMA model
        
        Args:
            fitted_model: Fitted SARIMAX model
            steps: Number of forecast steps
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        try:
            forecast_result = fitted_model.get_forecast(steps=steps)
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int()
            
            return {
                'forecasts': forecast_values.tolist(),
                'lower_ci': confidence_intervals.iloc[:, 0].tolist(),
                'upper_ci': confidence_intervals.iloc[:, 1].tolist(),
                'forecast_successful': True
            }
            
        except Exception as e:
            print(f"Forecast generation failed: {e}")
            return {
                'forecasts': [0] * steps,
                'lower_ci': [0] * steps,
                'upper_ci': [0] * steps,
                'forecast_successful': False
            }
    
    def calculate_model_diagnostics(self, fitted_model, timeseries: pd.Series) -> Dict:
        """
        Calculate model diagnostic statistics
        
        Args:
            fitted_model: Fitted SARIMAX model
            timeseries: Original time series
            
        Returns:
            Dictionary with diagnostic statistics
        """
        try:
            diagnostics = {}
            
            # Residuals
            residuals = fitted_model.resid
            
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
            diagnostics['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]
            diagnostics['residuals_autocorrelated'] = diagnostics['ljung_box_pvalue'] < 0.05
            
            # Residual statistics
            diagnostics['residual_mean'] = np.mean(residuals)
            diagnostics['residual_std'] = np.std(residuals)
            diagnostics['residual_skewness'] = stats.skew(residuals)
            diagnostics['residual_kurtosis'] = stats.kurtosis(residuals)
            
            # Model fit statistics
            diagnostics['aic'] = fitted_model.aic
            diagnostics['bic'] = fitted_model.bic
            diagnostics['log_likelihood'] = fitted_model.llf
            
            # In-sample predictions for accuracy
            in_sample_pred = fitted_model.fittedvalues
            diagnostics['mae_in_sample'] = mean_absolute_error(timeseries, in_sample_pred)
            diagnostics['rmse_in_sample'] = np.sqrt(mean_squared_error(timeseries, in_sample_pred))
            
            return diagnostics
            
        except Exception as e:
            print(f"Diagnostic calculation failed: {e}")
            return {}
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit SARIMA models and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'model_diagnostics': {},
            'seasonal_analysis': {},
            'model_parameters': {}
        }
        
        print("Starting SARIMA forecasting...")
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            print(f"Processing item {idx + 1}/{len(df)}: {item_id}")
            
            # Extract demand series
            demand_values = [row[col] for col in self.time_columns]
            
            # Create time series with proper date index
            timeseries = pd.Series(
                demand_values,
                index=pd.DatetimeIndex(self.dates),
                name=item_id
            )
            
            # Skip items with no demand or insufficient data
            if np.sum(timeseries) == 0 or len(timeseries) < 24:  # Need at least 2 years
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': [0] * forecast_periods,
                    'lower_ci': [0] * forecast_periods,
                    'upper_ci': [0] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': True
                }
                continue
            
            # Test stationarity
            stationarity_results = self.test_stationarity(timeseries, item_id)
            
            # Analyze seasonality
            seasonal_analysis = self.analyze_seasonality(timeseries, item_id)
            results['seasonal_analysis'][item_id] = seasonal_analysis
            
            # Select model parameters
            if self.auto_arima:
                (order, seasonal_order), model_info = self.auto_arima_selection(timeseries, item_id)
            else:
                (order, seasonal_order), model_info = self.grid_search_sarima(timeseries, item_id)
            
            results['model_parameters'][item_id] = {
                'order': order,
                'seasonal_order': seasonal_order,
                'model_info': model_info,
                'stationarity': stationarity_results
            }
            
            # Fit SARIMA model
            fitted_model = self.fit_sarima_model(timeseries, order, seasonal_order, item_id)
            
            if fitted_model is not None:
                # Generate forecasts
                forecast_results = self.generate_forecasts(fitted_model, forecast_periods)
                
                # Calculate diagnostics
                diagnostics = self.calculate_model_diagnostics(fitted_model, timeseries)
                results['model_diagnostics'][item_id] = diagnostics
                
                # Store model
                self.models[item_id] = fitted_model
                
                # Store results
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': [max(0, f) for f in forecast_results['forecasts']],  # Ensure non-negative
                    'lower_ci': forecast_results['lower_ci'],
                    'upper_ci': forecast_results['upper_ci'],
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': True,
                    'insufficient_data': False,
                    'forecast_successful': forecast_results['forecast_successful']
                }
                
            else:
                # Model fitting failed - use simple average
                avg_demand = np.mean(timeseries[timeseries > 0]) if np.sum(timeseries > 0) > 0 else 0
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': [avg_demand] * forecast_periods,
                    'lower_ci': [avg_demand * 0.8] * forecast_periods,
                    'upper_ci': [avg_demand * 1.2] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': False,
                    'forecast_successful': False
                }
        
        print(f"Completed SARIMA forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of SARIMA forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for item_id, forecast_data in results['item_forecasts'].items():
            # Get seasonal analysis if available
            seasonal_info = results['seasonal_analysis'].get(item_id, {})
            model_params = results['model_parameters'].get(item_id, {})
            diagnostics = results['model_diagnostics'].get(item_id, {})
            
            summary_data.append({
                'Item_ID': item_id,
                'Item_Name': forecast_data['item_name'][:50] + '...' if len(forecast_data['item_name']) > 50 else forecast_data['item_name'],
                'Category': forecast_data['category'],
                'Model_Fitted': forecast_data.get('model_fitted', False),
                'Has_Seasonality': seasonal_info.get('has_seasonality', False),
                'Seasonal_Strength': round(seasonal_info.get('seasonal_strength', 0), 3),
                'Historical_Total': sum(forecast_data['historical_demand']),
                'Avg_Monthly_Historical': round(np.mean(forecast_data['historical_demand']), 2),
                'Monthly_Forecast_Avg': round(np.mean(forecast_data['monthly_forecasts']), 2),
                'Annual_Forecast': round(sum(forecast_data['monthly_forecasts']), 2),
                'ARIMA_Order': str(model_params.get('order', 'N/A')),
                'Seasonal_Order': str(model_params.get('seasonal_order', 'N/A')),
                'AIC': round(diagnostics.get('aic', 0), 2),
                'BIC': round(diagnostics.get('bic', 0), 2),
                'MAE_In_Sample': round(diagnostics.get('mae_in_sample', 0), 3)
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create comprehensive visualization of SARIMA forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to analyze
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Seasonality Distribution
        seasonal_strengths = [results['seasonal_analysis'][item_id].get('seasonal_strength', 0) 
                             for item_id in results['seasonal_analysis'].keys()]
        seasonal_items = [s for s in seasonal_strengths if s > 0.1]
        
        axes[0, 0].hist(seasonal_strengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='Seasonality Threshold')
        axes[0, 0].set_xlabel('Seasonal Strength')
        axes[0, 0].set_ylabel('Number of Items')
        axes[0, 0].set_title(f'Seasonality Distribution\n({len(seasonal_items)} items with seasonality)')
        axes[0, 0].legend()
        
        # Plot 2: Model Performance (AIC distribution)
        aic_values = [results['model_diagnostics'][item_id].get('aic', 0) 
                     for item_id in results['model_diagnostics'].keys()]
        aic_values = [aic for aic in aic_values if aic > 0]
        
        if aic_values:
            axes[0, 1].hist(aic_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('AIC Score')
            axes[0, 1].set_ylabel('Number of Models')
            axes[0, 1].set_title('Model AIC Score Distribution')
        
        # Plot 3: Top Item with Seasonal Decomposition
        if top_items and results['seasonal_analysis']:
            top_item_id = top_items[0][0]
            seasonal_analysis = results['seasonal_analysis'].get(top_item_id, {})
            
            if 'seasonal' in seasonal_analysis:
                seasonal_component = seasonal_analysis['seasonal'][:12]  # First year
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                axes[1, 0].plot(months, seasonal_component, 'g-o', linewidth=2)
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('Seasonal Component')
                axes[1, 0].set_title(f'Seasonal Pattern: {top_item_id}')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Historical vs Forecast with Confidence Intervals
        if top_items:
            top_item_id = top_items[0][0]
            top_item_data = results['item_forecasts'][top_item_id]
            
            historical = top_item_data['historical_demand']
            forecasts = top_item_data['monthly_forecasts']
            lower_ci = top_item_data.get('lower_ci', forecasts)
            upper_ci = top_item_data.get('upper_ci', forecasts)
            
            # Plot historical data
            hist_months = list(range(1, len(historical) + 1))
            axes[1, 1].plot(hist_months, historical, 'b-o', alpha=0.7, label='Historical', markersize=4)
            
            # Plot forecasts with confidence intervals
            forecast_months = list(range(len(historical) + 1, len(historical) + 13))
            axes[1, 1].plot(forecast_months, forecasts, 'r--s', alpha=0.7, label='SARIMA Forecast', markersize=4)
            axes[1, 1].fill_between(forecast_months, lower_ci, upper_ci, alpha=0.2, color='red', label='95% CI')
            
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].set_title(f'SARIMA Forecast with CI: {top_item_id}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Model Parameter Distribution
        orders = [results['model_parameters'][item_id].get('order', (0, 0, 0))[0] 
                 for item_id in results['model_parameters'].keys()]
        seasonal_orders = [results['model_parameters'][item_id].get('seasonal_order', (0, 0, 0, 12))[0] 
                          for item_id in results['model_parameters'].keys()]
        
        order_counts = pd.Series(orders).value_counts().sort_index()
        seasonal_counts = pd.Series(seasonal_orders).value_counts().sort_index()
        
        axes[2, 0].bar(order_counts.index, order_counts.values, alpha=0.7)
        axes[2, 0].set_xlabel('AR Order (p)')
        axes[2, 0].set_ylabel('Number of Models')
        axes[2, 0].set_title('Distribution of AR Parameters')
        
        # Plot 6: Forecast Distribution
        all_forecasts = []
        for data in results['item_forecasts'].values():
            all_forecasts.extend(data['monthly_forecasts'])
        
        positive_forecasts = [f for f in all_forecasts if f > 0]
        if positive_forecasts:
            axes[2, 1].hist(positive_forecasts, bins=30, alpha=0.7, edgecolor='black')
            axes[2, 1].set_xlabel('Monthly Forecast Value')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].set_title('Distribution of SARIMA Forecasts')
        
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_analysis(self, results: Dict, item_id: str):
        """
        Create detailed seasonal analysis plot for a specific item
        
        Args:
            results: Results dictionary
            item_id: Item to analyze
        """
        if item_id not in results['seasonal_analysis']:
            print(f"No seasonal analysis available for {item_id}")
            return
        
        seasonal_analysis = results['seasonal_analysis'][item_id]
        item_data = results['item_forecasts'][item_id]
        
        if 'trend' not in seasonal_analysis:
            print(f"Insufficient data for seasonal decomposition of {item_id}")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Original time series
        dates = pd.date_range(start=self.dates[0], periods=len(item_data['historical_demand']), freq='M')
        axes[0].plot(dates, item_data['historical_demand'], 'b-', linewidth=2)
        axes[0].set_title(f'Original Time Series: {item_id}')
        axes[0].set_ylabel('Demand')
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        trend = seasonal_analysis['trend']
        axes[1].plot(dates, trend, 'g-', linewidth=2)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        seasonal = seasonal_analysis['seasonal']
        axes[2].plot(dates, seasonal, 'r-', linewidth=2)
        axes[2].set_title(f'Seasonal Component (Strength: {seasonal_analysis["seasonal_strength"]:.3f})')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        residual = seasonal_analysis['residual']
        axes[3].plot(dates, residual, 'purple', linewidth=1)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate SARIMA forecasting
    """
    print("SARIMA Forecasting for Spare Parts Demand")
    print("=" * 45)
    
    # Initialize SARIMA model
    sarima_forecaster = SARIMAForecasting(
        seasonal_period=12,
        max_p=3, max_d=2, max_q=3,
        max_P=2, max_D=1, max_Q=2,
        auto_arima=True  # Set to False for grid search
    )
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = sarima_forecaster.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit models and generate forecasts
    print("\nRunning SARIMA forecasting...")
    results = sarima_forecaster.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = sarima_forecaster.create_forecast_summary(results)
    
    print(f"\nSARIMA Forecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Seasonal analysis summary
    seasonal_items = summary_df[summary_df['Has_Seasonality'] == True]
    fitted_models = summary_df[summary_df['Model_Fitted'] == True]
    
    print(f"\nSeasonal Analysis Summary:")
    print(f"  Items with seasonal patterns: {len(seasonal_items)} ({len(seasonal_items)/len(summary_df)*100:.1f}%)")
    print(f"  Successfully fitted models: {len(fitted_models)} ({len(fitted_models)/len(summary_df)*100:.1f}%)")
    
    if len(fitted_models) > 0:
        print(f"  Average AIC score: {fitted_models['AIC'].mean():.2f}")
        print(f"  Average in-sample MAE: {fitted_models['MAE_In_Sample'].mean():.3f}")
    
    # Create visualizations
    print("\nGenerating SARIMA forecast visualizations...")
    sarima_forecaster.plot_forecast_analysis(results, top_n=5)
    
    # Show detailed seasonal analysis for top item
    top_item = summary_df.nlargest(1, 'Historical_Total')['Item_ID'].iloc[0]
    if summary_df[summary_df['Item_ID'] == top_item]['Has_Seasonality'].iloc[0]:
        print(f"\nDetailed seasonal analysis for top item: {top_item}")
        sarima_forecaster.plot_seasonal_analysis(results, top_item)
    
    # Save results to Excel
    output_file = 'SARIMA_Forecast_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Forecast_Summary', index=False)
        
        # Detailed forecasts with confidence intervals
        detailed_data = []
        top_items = summary_df.nlargest(10, 'Historical_Total')['Item_ID'].tolist()
        
        for item_id in top_items:
            item_data = results['item_forecasts'][item_id]
            for month in range(12):
                detailed_data.append({
                    'Item_ID': item_id,
                    'Item_Name': item_data['item_name'],
                    'Forecast_Month': month + 1,
                    'Forecast_Value': round(item_data['monthly_forecasts'][month], 2),
                    'Lower_CI': round(item_data.get('lower_ci', [0]*12)[month], 2),
                    'Upper_CI': round(item_data.get('upper_ci', [0]*12)[month], 2),
                    'Model_Fitted': item_data.get('model_fitted', False)
                })
        
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed_Forecasts', index=False)
        
        # Seasonal analysis results
        seasonal_data = []
        for item_id, analysis in results['seasonal_analysis'].items():
            seasonal_indices = analysis.get('seasonal_indices', {})
            for month, index in seasonal_indices.items():
                seasonal_data.append({
                    'Item_ID': item_id,
                    'Month': month,
                    'Seasonal_Index': round(index, 4),
                    'Seasonal_Strength': round(analysis.get('seasonal_strength', 0), 4)
                })
        
        if seasonal_data:
            pd.DataFrame(seasonal_data).to_excel(writer, sheet_name='Seasonal_Analysis', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("SARIMA forecasting completed successfully!")

if __name__ == "__main__":
    main()
