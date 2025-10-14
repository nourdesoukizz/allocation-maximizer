import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

warnings.filterwarnings('ignore')

class ProphetForecasting:
    """
    Prophet-based forecasting for spare parts demand
    
    Uses Facebook's Prophet algorithm for robust time series forecasting
    with automatic seasonality detection, trend changepoints, and
    uncertainty quantification.
    """
    
    def __init__(self, 
                 growth: str = 'linear',
                 yearly_seasonality: str = 'auto',
                 weekly_seasonality: bool = False,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 interval_width: float = 0.8):
        """
        Initialize Prophet forecasting model
        
        Args:
            growth: 'linear' or 'logistic' growth
            yearly_seasonality: 'auto', True, False, or int
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality  
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            interval_width: Width of uncertainty intervals
        """
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.interval_width = interval_width
        
        self.models = {}
        self.model_components = {}
        self.forecast_results = {}
        self.cv_results = {}
        
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
    
    def prepare_prophet_data(self, demand_values: List[float], item_id: str) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format (ds, y columns)
        
        Args:
            demand_values: Historical demand values
            item_id: Item identifier
            
        Returns:
            DataFrame with 'ds' (datestamp) and 'y' (value) columns
        """
        # Create DataFrame with proper date index
        prophet_df = pd.DataFrame({
            'ds': pd.DatetimeIndex(self.dates[:len(demand_values)]),
            'y': demand_values
        })
        
        # Handle missing values - Prophet can handle NaN but zeros might be better
        prophet_df['y'] = prophet_df['y'].fillna(0)
        
        return prophet_df
    
    def create_prophet_model(self, item_id: str, demand_stats: Dict) -> Prophet:
        """
        Create and configure Prophet model based on item characteristics
        
        Args:
            item_id: Item identifier
            demand_stats: Statistics about the demand pattern
            
        Returns:
            Configured Prophet model
        """
        # Adjust parameters based on demand characteristics
        changepoint_scale = self.changepoint_prior_scale
        seasonality_scale = self.seasonality_prior_scale
        
        # For intermittent demand, reduce changepoint sensitivity
        if demand_stats['demand_frequency'] < 0.3:
            changepoint_scale *= 0.5
        
        # For highly variable demand, increase seasonality flexibility
        if demand_stats['cv'] > 2.0:
            seasonality_scale *= 1.5
        
        # Create Prophet model
        model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=changepoint_scale,
            seasonality_prior_scale=seasonality_scale,
            interval_width=self.interval_width,
            uncertainty_samples=1000
        )
        
        # Add custom seasonalities for spare parts
        # Monthly seasonality (different from yearly)
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=3,
            mode=self.seasonality_mode
        )
        
        # Quarterly seasonality for business cycles
        model.add_seasonality(
            name='quarterly',
            period=365.25/4,
            fourier_order=2,
            mode=self.seasonality_mode
        )
        
        return model
    
    def calculate_demand_statistics(self, demand_values: List[float]) -> Dict:
        """
        Calculate statistical properties of demand pattern
        
        Args:
            demand_values: Historical demand values
            
        Returns:
            Dictionary with demand statistics
        """
        demand_array = np.array(demand_values)
        non_zero_demand = demand_array[demand_array > 0]
        
        stats = {
            'total_demand': np.sum(demand_array),
            'mean_demand': np.mean(demand_array),
            'std_demand': np.std(demand_array),
            'cv': np.std(demand_array) / np.mean(demand_array) if np.mean(demand_array) > 0 else 0,
            'demand_frequency': len(non_zero_demand) / len(demand_array),
            'mean_when_positive': np.mean(non_zero_demand) if len(non_zero_demand) > 0 else 0,
            'max_demand': np.max(demand_array),
            'min_demand': np.min(demand_array),
            'zero_periods': np.sum(demand_array == 0),
            'non_zero_periods': len(non_zero_demand)
        }
        
        return stats
    
    def detect_changepoints(self, model: Prophet, forecast_df: pd.DataFrame) -> Dict:
        """
        Analyze trend changepoints detected by Prophet
        
        Args:
            model: Fitted Prophet model
            forecast_df: Forecast DataFrame from Prophet
            
        Returns:
            Dictionary with changepoint analysis
        """
        changepoints_info = {}
        
        try:
            # Get changepoints
            changepoints = model.changepoints
            changepoints_info['changepoint_dates'] = changepoints.dt.strftime('%Y-%m-%d').tolist()
            changepoints_info['num_changepoints'] = len(changepoints)
            
            # Get changepoint effects
            if len(changepoints) > 0:
                # Calculate rate changes at changepoints
                deltas = model.params['delta'].mean(axis=0) if hasattr(model, 'params') else []
                changepoints_info['rate_changes'] = deltas.tolist() if len(deltas) > 0 else []
                
                # Identify significant changepoints (those with notable rate changes)
                if len(deltas) > 0:
                    significant_threshold = np.std(deltas) * 1.5
                    significant_changepoints = []
                    for i, (date, delta) in enumerate(zip(changepoints, deltas)):
                        if abs(delta) > significant_threshold:
                            significant_changepoints.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'rate_change': delta,
                                'magnitude': 'increase' if delta > 0 else 'decrease'
                            })
                    changepoints_info['significant_changepoints'] = significant_changepoints
            
        except Exception as e:
            print(f"Changepoint analysis failed: {e}")
            changepoints_info = {'num_changepoints': 0, 'changepoint_dates': []}
        
        return changepoints_info
    
    def perform_cross_validation(self, model: Prophet, prophet_df: pd.DataFrame) -> Dict:
        """
        Perform time series cross-validation
        
        Args:
            model: Fitted Prophet model
            prophet_df: Prophet format DataFrame
            
        Returns:
            Cross-validation results
        """
        try:
            # Only perform CV if we have sufficient data
            if len(prophet_df) < 24:  # Need at least 2 years
                return {'performed': False, 'reason': 'insufficient_data'}
            
            # Set up cross-validation parameters
            initial_days = max(365, len(prophet_df) * 30 // 2)  # At least 1 year or half the data
            period_days = 90  # 3 months
            horizon_days = 180  # 6 months
            
            # Perform cross-validation
            cv_results = cross_validation(
                model, 
                initial=f'{initial_days} days',
                period=f'{period_days} days',
                horizon=f'{horizon_days} days'
            )
            
            # Calculate performance metrics
            performance = performance_metrics(cv_results)
            
            return {
                'performed': True,
                'cv_results': cv_results,
                'performance_metrics': {
                    'mae': performance['mae'].mean(),
                    'mape': performance['mape'].mean(),
                    'rmse': performance['rmse'].mean(),
                    'coverage': performance['coverage'].mean() if 'coverage' in performance else None
                }
            }
            
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            return {'performed': False, 'reason': 'cv_failed'}
    
    def analyze_seasonality_components(self, model: Prophet, forecast_df: pd.DataFrame) -> Dict:
        """
        Analyze seasonal components from Prophet model
        
        Args:
            model: Fitted Prophet model
            forecast_df: Forecast DataFrame
            
        Returns:
            Dictionary with seasonality analysis
        """
        seasonality_info = {}
        
        try:
            # Yearly seasonality
            if 'yearly' in forecast_df.columns:
                yearly_component = forecast_df['yearly'].dropna()
                seasonality_info['yearly'] = {
                    'strength': np.std(yearly_component),
                    'peak_month': yearly_component.idxmax() % 12 + 1 if len(yearly_component) > 0 else None,
                    'low_month': yearly_component.idxmin() % 12 + 1 if len(yearly_component) > 0 else None,
                    'range': yearly_component.max() - yearly_component.min() if len(yearly_component) > 0 else 0
                }
            
            # Custom monthly seasonality
            if 'monthly' in forecast_df.columns:
                monthly_component = forecast_df['monthly'].dropna()
                seasonality_info['monthly'] = {
                    'strength': np.std(monthly_component),
                    'range': monthly_component.max() - monthly_component.min() if len(monthly_component) > 0 else 0
                }
            
            # Quarterly seasonality
            if 'quarterly' in forecast_df.columns:
                quarterly_component = forecast_df['quarterly'].dropna()
                seasonality_info['quarterly'] = {
                    'strength': np.std(quarterly_component),
                    'range': quarterly_component.max() - quarterly_component.min() if len(quarterly_component) > 0 else 0
                }
            
            # Overall seasonality strength
            total_seasonal_variance = 0
            residual_variance = np.var(forecast_df['yhat'] - forecast_df['trend']) if 'trend' in forecast_df.columns else 0
            
            for component in ['yearly', 'monthly', 'quarterly']:
                if component in forecast_df.columns:
                    total_seasonal_variance += np.var(forecast_df[component].dropna())
            
            if total_seasonal_variance + residual_variance > 0:
                seasonality_info['overall_seasonal_strength'] = total_seasonal_variance / (total_seasonal_variance + residual_variance)
            else:
                seasonality_info['overall_seasonal_strength'] = 0
                
        except Exception as e:
            print(f"Seasonality analysis failed: {e}")
            seasonality_info = {'overall_seasonal_strength': 0}
        
        return seasonality_info
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit Prophet models and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'model_components': {},
            'changepoint_analysis': {},
            'seasonality_analysis': {},
            'cross_validation': {}
        }
        
        print("Starting Prophet forecasting...")
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            print(f"Processing item {idx + 1}/{len(df)}: {item_id}")
            
            # Extract demand series
            demand_values = [row[col] for col in self.time_columns]
            
            # Skip items with no demand or insufficient data
            if np.sum(demand_values) == 0 or len(demand_values) < 12:
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': [0] * forecast_periods,
                    'lower_bound': [0] * forecast_periods,
                    'upper_bound': [0] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': True
                }
                continue
            
            # Calculate demand statistics
            demand_stats = self.calculate_demand_statistics(demand_values)
            
            # Prepare data for Prophet
            prophet_df = self.prepare_prophet_data(demand_values, item_id)
            
            # Create and configure Prophet model
            model = self.create_prophet_model(item_id, demand_stats)
            
            try:
                # Fit the model
                model.fit(prophet_df)
                
                # Create future dataframe for forecasting
                future = model.make_future_dataframe(periods=forecast_periods, freq='M')
                
                # Generate forecast
                forecast = model.predict(future)
                
                # Extract forecast values (ensure non-negative)
                forecast_values = forecast.tail(forecast_periods)['yhat'].tolist()
                forecast_values = [max(0, f) for f in forecast_values]
                
                # Extract confidence intervals
                lower_bound = forecast.tail(forecast_periods)['yhat_lower'].tolist()
                upper_bound = forecast.tail(forecast_periods)['yhat_upper'].tolist()
                lower_bound = [max(0, f) for f in lower_bound]
                upper_bound = [max(0, f) for f in upper_bound]
                
                # Store model and forecast results
                self.models[item_id] = model
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': forecast_values,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': True,
                    'insufficient_data': False,
                    'demand_statistics': demand_stats
                }
                
                # Analyze model components
                results['model_components'][item_id] = {
                    'trend': forecast['trend'].tolist(),
                    'seasonal_components': {}
                }
                
                # Store seasonal components if they exist
                for component in ['yearly', 'monthly', 'quarterly']:
                    if component in forecast.columns:
                        results['model_components'][item_id]['seasonal_components'][component] = forecast[component].tolist()
                
                # Analyze changepoints
                results['changepoint_analysis'][item_id] = self.detect_changepoints(model, forecast)
                
                # Analyze seasonality
                results['seasonality_analysis'][item_id] = self.analyze_seasonality_components(model, forecast)
                
                # Perform cross-validation (only for items with sufficient data and not too many to save time)
                if idx < 5 and len(prophet_df) >= 24:  # Only first 5 items for demo
                    print(f"  Performing cross-validation for {item_id}...")
                    results['cross_validation'][item_id] = self.perform_cross_validation(model, prophet_df)
                
            except Exception as e:
                print(f"  Prophet modeling failed for {item_id}: {e}")
                
                # Fallback to simple average
                avg_demand = np.mean([d for d in demand_values if d > 0]) if any(d > 0 for d in demand_values) else 0
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_values,
                    'monthly_forecasts': [avg_demand] * forecast_periods,
                    'lower_bound': [avg_demand * 0.8] * forecast_periods,
                    'upper_bound': [avg_demand * 1.2] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': False,
                    'demand_statistics': demand_stats
                }
        
        print(f"Completed Prophet forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of Prophet forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for item_id, forecast_data in results['item_forecasts'].items():
            # Get additional analysis if available
            seasonality_info = results['seasonality_analysis'].get(item_id, {})
            changepoint_info = results['changepoint_analysis'].get(item_id, {})
            cv_info = results['cross_validation'].get(item_id, {})
            demand_stats = forecast_data.get('demand_statistics', {})
            
            summary_data.append({
                'Item_ID': item_id,
                'Item_Name': forecast_data['item_name'][:50] + '...' if len(forecast_data['item_name']) > 50 else forecast_data['item_name'],
                'Category': forecast_data['category'],
                'Model_Fitted': forecast_data.get('model_fitted', False),
                'Historical_Total': sum(forecast_data['historical_demand']),
                'Avg_Monthly_Historical': round(np.mean(forecast_data['historical_demand']), 2),
                'Monthly_Forecast_Avg': round(np.mean(forecast_data['monthly_forecasts']), 2),
                'Annual_Forecast': round(sum(forecast_data['monthly_forecasts']), 2),
                'Demand_Frequency': round(demand_stats.get('demand_frequency', 0), 3),
                'CV': round(demand_stats.get('cv', 0), 2),
                'Seasonal_Strength': round(seasonality_info.get('overall_seasonal_strength', 0), 3),
                'Num_Changepoints': changepoint_info.get('num_changepoints', 0),
                'CV_MAE': round(cv_info.get('performance_metrics', {}).get('mae', 0), 3) if cv_info.get('performed', False) else None,
                'CV_MAPE': round(cv_info.get('performance_metrics', {}).get('mape', 0), 3) if cv_info.get('performed', False) else None
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create comprehensive visualization of Prophet forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to analyze
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Seasonality Strength Distribution
        seasonal_strengths = [results['seasonality_analysis'][item_id].get('overall_seasonal_strength', 0) 
                             for item_id in results['seasonality_analysis'].keys()]
        seasonal_items = [s for s in seasonal_strengths if s > 0.1]
        
        axes[0, 0].hist(seasonal_strengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='Seasonality Threshold')
        axes[0, 0].set_xlabel('Seasonal Strength')
        axes[0, 0].set_ylabel('Number of Items')
        axes[0, 0].set_title(f'Prophet Seasonality Distribution\n({len(seasonal_items)} items with strong seasonality)')
        axes[0, 0].legend()
        
        # Plot 2: Demand Pattern Characteristics
        fitted_items = {item_id: data for item_id, data in results['item_forecasts'].items() 
                       if data.get('model_fitted', False)}
        
        if fitted_items:
            demand_frequencies = [data['demand_statistics']['demand_frequency'] 
                                for data in fitted_items.values()]
            cvs = [data['demand_statistics']['cv'] for data in fitted_items.values()]
            
            scatter = axes[0, 1].scatter(demand_frequencies, cvs, alpha=0.6, c='blue')
            axes[0, 1].set_xlabel('Demand Frequency')
            axes[0, 1].set_ylabel('Coefficient of Variation')
            axes[0, 1].set_title('Demand Pattern Characteristics')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Changepoint Analysis
        changepoint_counts = [results['changepoint_analysis'][item_id].get('num_changepoints', 0) 
                             for item_id in results['changepoint_analysis'].keys()]
        
        if changepoint_counts:
            cp_counts = pd.Series(changepoint_counts).value_counts().sort_index()
            axes[1, 0].bar(cp_counts.index, cp_counts.values, alpha=0.7)
            axes[1, 0].set_xlabel('Number of Changepoints')
            axes[1, 0].set_ylabel('Number of Items')
            axes[1, 0].set_title('Trend Changepoints Distribution')
        
        # Plot 4: Top Item Forecast with Components
        if top_items and results['model_components']:
            top_item_id = top_items[0][0]
            top_item_data = results['item_forecasts'][top_item_id]
            
            historical = top_item_data['historical_demand']
            forecasts = top_item_data['monthly_forecasts']
            lower_bound = top_item_data['lower_bound']
            upper_bound = top_item_data['upper_bound']
            
            # Plot historical data
            hist_months = list(range(1, len(historical) + 1))
            axes[1, 1].plot(hist_months, historical, 'b-o', alpha=0.7, label='Historical', markersize=4)
            
            # Plot forecasts with confidence intervals
            forecast_months = list(range(len(historical) + 1, len(historical) + 13))
            axes[1, 1].plot(forecast_months, forecasts, 'r--s', alpha=0.7, label='Prophet Forecast', markersize=4)
            axes[1, 1].fill_between(forecast_months, lower_bound, upper_bound, 
                                   alpha=0.2, color='red', label='Confidence Interval')
            
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].set_title(f'Prophet Forecast: {top_item_id}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Forecast vs Historical Comparison
        if top_items:
            top_item_ids = [item[0] for item in top_items]
            historical_avgs = [np.mean(results['item_forecasts'][item_id]['historical_demand']) 
                              for item_id in top_item_ids]
            forecast_avgs = [np.mean(results['item_forecasts'][item_id]['monthly_forecasts']) 
                            for item_id in top_item_ids]
            
            x_pos = np.arange(len(top_item_ids))
            width = 0.35
            
            axes[2, 0].bar(x_pos - width/2, historical_avgs, width, label='Historical Avg', alpha=0.7)
            axes[2, 0].bar(x_pos + width/2, forecast_avgs, width, label='Prophet Forecast', alpha=0.7)
            axes[2, 0].set_xlabel('Items')
            axes[2, 0].set_ylabel('Average Monthly Demand')
            axes[2, 0].set_title('Top Items: Historical vs Prophet Forecast')
            axes[2, 0].set_xticks(x_pos)
            axes[2, 0].set_xticklabels([f"Item_{i+1}" for i in range(len(top_item_ids))], rotation=45)
            axes[2, 0].legend()
        
        # Plot 6: Cross-validation Results (if available)
        cv_results = [results['cross_validation'][item_id] for item_id in results['cross_validation'].keys() 
                     if results['cross_validation'][item_id].get('performed', False)]
        
        if cv_results:
            mae_values = [cv['performance_metrics']['mae'] for cv in cv_results]
            mape_values = [cv['performance_metrics']['mape'] for cv in cv_results]
            
            axes[2, 1].scatter(mae_values, mape_values, alpha=0.7, s=100)
            axes[2, 1].set_xlabel('Cross-validation MAE')
            axes[2, 1].set_ylabel('Cross-validation MAPE')
            axes[2, 1].set_title('Prophet Model Performance (CV)')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            # Show forecast distribution instead
            all_forecasts = []
            for data in results['item_forecasts'].values():
                all_forecasts.extend(data['monthly_forecasts'])
            
            positive_forecasts = [f for f in all_forecasts if f > 0]
            if positive_forecasts:
                axes[2, 1].hist(positive_forecasts, bins=30, alpha=0.7, edgecolor='black')
                axes[2, 1].set_xlabel('Monthly Forecast Value')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].set_title('Distribution of Prophet Forecasts')
        
        plt.tight_layout()
        plt.show()
    
    def plot_item_components(self, results: Dict, item_id: str):
        """
        Create detailed component analysis plot for a specific item
        
        Args:
            results: Results dictionary
            item_id: Item to analyze
        """
        if item_id not in self.models:
            print(f"No Prophet model available for {item_id}")
            return
        
        model = self.models[item_id]
        item_data = results['item_forecasts'][item_id]
        
        # Create Prophet's built-in plots
        try:
            # Forecast plot
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            
            fig1 = model.plot(forecast, figsize=(12, 6))
            plt.title(f'Prophet Forecast: {item_id}')
            plt.show()
            
            # Components plot
            fig2 = model.plot_components(forecast, figsize=(12, 10))
            plt.suptitle(f'Prophet Components: {item_id}', y=1.02)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create Prophet plots for {item_id}: {e}")

def main():
    """
    Main function to demonstrate Prophet forecasting
    """
    print("Prophet Forecasting for Spare Parts Demand")
    print("=" * 45)
    
    # Initialize Prophet model
    prophet_forecaster = ProphetForecasting(
        growth='linear',
        yearly_seasonality='auto',
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.8
    )
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = prophet_forecaster.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit models and generate forecasts
    print("\nRunning Prophet forecasting...")
    results = prophet_forecaster.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = prophet_forecaster.create_forecast_summary(results)
    
    print(f"\nProphet Forecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Analysis summary
    fitted_models = summary_df[summary_df['Model_Fitted'] == True]
    seasonal_items = summary_df[summary_df['Seasonal_Strength'] > 0.1]
    changepoint_items = summary_df[summary_df['Num_Changepoints'] > 0]
    
    print(f"\nProphet Analysis Summary:")
    print(f"  Successfully fitted models: {len(fitted_models)} ({len(fitted_models)/len(summary_df)*100:.1f}%)")
    print(f"  Items with strong seasonality: {len(seasonal_items)} ({len(seasonal_items)/len(summary_df)*100:.1f}%)")
    print(f"  Items with trend changes: {len(changepoint_items)} ({len(changepoint_items)/len(summary_df)*100:.1f}%)")
    
    if len(fitted_models) > 0:
        print(f"  Average seasonal strength: {fitted_models['Seasonal_Strength'].mean():.3f}")
        print(f"  Average changepoints per item: {fitted_models['Num_Changepoints'].mean():.1f}")
    
    # Cross-validation summary
    cv_items = summary_df.dropna(subset=['CV_MAE'])
    if len(cv_items) > 0:
        print(f"  Cross-validation results ({len(cv_items)} items):")
        print(f"    Average MAE: {cv_items['CV_MAE'].mean():.3f}")
        print(f"    Average MAPE: {cv_items['CV_MAPE'].mean():.3f}")
    
    # Create visualizations
    print("\nGenerating Prophet forecast visualizations...")
    prophet_forecaster.plot_forecast_analysis(results, top_n=5)
    
    # Show detailed component analysis for top item
    top_item = summary_df.nlargest(1, 'Historical_Total')['Item_ID'].iloc[0]
    if summary_df[summary_df['Item_ID'] == top_item]['Model_Fitted'].iloc[0]:
        print(f"\nDetailed component analysis for top item: {top_item}")
        prophet_forecaster.plot_item_components(results, top_item)
    
    # Save results to Excel
    output_file = 'Prophet_Forecast_Results.xlsx'
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
                    'Lower_Bound': round(item_data['lower_bound'][month], 2),
                    'Upper_Bound': round(item_data['upper_bound'][month], 2),
                    'Model_Fitted': item_data.get('model_fitted', False)
                })
        
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed_Forecasts', index=False)
        
        # Seasonality analysis
        seasonality_data = []
        for item_id, analysis in results['seasonality_analysis'].items():
            seasonality_data.append({
                'Item_ID': item_id,
                'Overall_Seasonal_Strength': round(analysis.get('overall_seasonal_strength', 0), 4),
                'Yearly_Strength': round(analysis.get('yearly', {}).get('strength', 0), 4),
                'Monthly_Strength': round(analysis.get('monthly', {}).get('strength', 0), 4),
                'Quarterly_Strength': round(analysis.get('quarterly', {}).get('strength', 0), 4)
            })
        
        pd.DataFrame(seasonality_data).to_excel(writer, sheet_name='Seasonality_Analysis', index=False)
        
        # Changepoint analysis
        changepoint_data = []
        for item_id, analysis in results['changepoint_analysis'].items():
            changepoint_data.append({
                'Item_ID': item_id,
                'Num_Changepoints': analysis.get('num_changepoints', 0),
                'Changepoint_Dates': ', '.join(analysis.get('changepoint_dates', [])),
                'Significant_Changes': len(analysis.get('significant_changepoints', []))
            })
        
        pd.DataFrame(changepoint_data).to_excel(writer, sheet_name='Changepoint_Analysis', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("Prophet forecasting completed successfully!")

if __name__ == "__main__":
    main()
