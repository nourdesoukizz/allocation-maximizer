import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from scipy import stats

warnings.filterwarnings('ignore')

class XGBoostForecasting:
    """
    XGBoost-based forecasting for intermittent spare parts demand
    
    Uses gradient boosting with advanced feature engineering to capture
    complex patterns in spare parts demand, including seasonality,
    trend, and intermittent patterns.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 optimize_hyperparams: bool = True):
        """
        Initialize XGBoost forecasting model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            optimize_hyperparams: Whether to optimize hyperparameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.optimize_hyperparams = optimize_hyperparams
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
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
            
            # Create proper column names
            time_columns = []
            date_mapping = {}
            
            for i, (year, month) in enumerate(zip(years, months)):
                if pd.notna(year) and pd.notna(month) and month != 'Total':
                    col_name = f"{int(year)}-{month}"
                    time_columns.append(col_name)
                    # Create actual date for better processing
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
            
            # Add time series columns
            for i, col_name in enumerate(time_columns):
                df[col_name] = df['demand_data'].apply(lambda x: x[i] if i < len(x) else 0)
            
            # Drop temporary column
            df = df.drop('demand_data', axis=1)
            
            # Store time information
            self.time_columns = time_columns
            self.date_mapping = date_mapping
            
            print(f"Loaded data for {len(df)} items with {len(time_columns)} time periods")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, demand_series: np.array, item_info: Dict) -> pd.DataFrame:
        """
        Create advanced features for XGBoost model
        
        Args:
            demand_series: Historical demand values
            item_info: Item metadata
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        # Create time index
        dates = [self.date_mapping[col] for col in self.time_columns[:len(demand_series)]]
        
        for i in range(len(demand_series)):
            features = {}
            current_date = dates[i]
            
            # Time-based features
            features['month'] = current_date.month
            features['quarter'] = (current_date.month - 1) // 3 + 1
            features['year'] = current_date.year
            features['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
            features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
            features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
            
            # Time index (trend)
            features['time_index'] = i
            features['time_index_squared'] = i ** 2
            
            # Lag features (previous demand values)
            for lag in [1, 2, 3, 6, 12]:
                if i >= lag:
                    features[f'lag_{lag}'] = demand_series[i - lag]
                else:
                    features[f'lag_{lag}'] = 0
            
            # Rolling statistics features
            for window in [3, 6, 12]:
                if i >= window - 1:
                    window_data = demand_series[max(0, i - window + 1):i + 1]
                    features[f'rolling_mean_{window}'] = np.mean(window_data)
                    features[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0
                    features[f'rolling_max_{window}'] = np.max(window_data)
                    features[f'rolling_min_{window}'] = np.min(window_data)
                    features[f'rolling_sum_{window}'] = np.sum(window_data)
                else:
                    features[f'rolling_mean_{window}'] = 0
                    features[f'rolling_std_{window}'] = 0
                    features[f'rolling_max_{window}'] = 0
                    features[f'rolling_min_{window}'] = 0
                    features[f'rolling_sum_{window}'] = 0
            
            # Exponential smoothing features
            if i > 0:
                alpha = 0.3
                exp_smooth = demand_series[0]
                for j in range(1, i + 1):
                    exp_smooth = alpha * demand_series[j] + (1 - alpha) * exp_smooth
                features['exp_smooth'] = exp_smooth
            else:
                features['exp_smooth'] = demand_series[0] if len(demand_series) > 0 else 0
            
            # Intermittency features
            if i > 0:
                recent_data = demand_series[:i + 1]
                features['demand_frequency'] = np.sum(recent_data > 0) / len(recent_data)
                features['avg_demand_when_positive'] = np.mean(recent_data[recent_data > 0]) if np.sum(recent_data > 0) > 0 else 0
                features['periods_since_last_demand'] = 0
                for j in range(i, -1, -1):
                    if demand_series[j] > 0:
                        break
                    features['periods_since_last_demand'] += 1
            else:
                features['demand_frequency'] = 1 if demand_series[0] > 0 else 0
                features['avg_demand_when_positive'] = demand_series[0] if demand_series[0] > 0 else 0
                features['periods_since_last_demand'] = 0 if demand_series[0] > 0 else 1
            
            # Statistical features
            if i >= 2:
                recent_data = demand_series[:i + 1]
                features['cv'] = np.std(recent_data) / np.mean(recent_data) if np.mean(recent_data) > 0 else 0
                features['skewness'] = stats.skew(recent_data)
                features['kurtosis'] = stats.kurtosis(recent_data)
            else:
                features['cv'] = 0
                features['skewness'] = 0
                features['kurtosis'] = 0
            
            # Item-specific features (encoded)
            features['category_encoded'] = hash(item_info['category']) % 1000
            
            # Target variable
            features['target'] = demand_series[i]
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Optimize XGBoost hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params, random_state=42)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                score = mean_absolute_error(y_fold_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        return study.best_params
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit XGBoost models and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'accuracy_metrics': {},
            'feature_importance': {},
            'model_performance': {}
        }
        
        print("Starting XGBoost forecasting...")
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            print(f"Processing item {idx + 1}/{len(df)}: {item_id}")
            
            # Extract demand series
            demand_series = np.array([row[col] for col in self.time_columns])
            
            # Skip items with no demand
            if np.sum(demand_series) == 0:
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': [0] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_trained': False
                }
                continue
            
            # Create features
            item_info = {
                'category': row['category'],
                'item_name': row['item_name']
            }
            
            feature_df = self.create_features(demand_series, item_info)
            
            # Prepare training data (use 80% for training, 20% for validation)
            split_point = int(len(feature_df) * 0.8)
            
            if split_point < 12:  # Need minimum data for training
                # Use simple average for items with insufficient data
                avg_demand = np.mean(demand_series[demand_series > 0]) if np.sum(demand_series > 0) > 0 else 0
                monthly_forecasts = [avg_demand] * forecast_periods
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': monthly_forecasts,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_trained': False
                }
                continue
            
            # Split data
            train_features = feature_df[:split_point]
            val_features = feature_df[split_point:]
            
            X_train = train_features.drop('target', axis=1)
            y_train = train_features['target']
            X_val = val_features.drop('target', axis=1)
            y_val = val_features['target']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            
            # Optimize hyperparameters for first few items, then use best params
            if self.optimize_hyperparams and idx < 3:
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
                self.best_params = best_params
            elif hasattr(self, 'best_params'):
                best_params = self.best_params
            else:
                best_params = {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate
                }
            
            # Train model
            model = xgb.XGBRegressor(**best_params, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Store model and scaler
            self.models[item_id] = model
            self.scalers[item_id] = scaler
            
            # Validation predictions
            val_pred = model.predict(X_val_scaled)
            val_pred = np.maximum(val_pred, 0)  # Ensure non-negative predictions
            
            # Calculate accuracy metrics
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            # Calculate MAPE
            mape_values = []
            for actual, pred in zip(y_val, val_pred):
                if actual != 0:
                    mape_values.append(abs((actual - pred) / actual))
            mape = np.mean(mape_values) * 100 if mape_values else 0
            
            results['accuracy_metrics'][item_id] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
            
            # Feature importance
            importance = model.feature_importances_
            feature_names = X_train_scaled.columns
            results['feature_importance'][item_id] = dict(zip(feature_names, importance))
            
            # Generate forecasts
            monthly_forecasts = self.generate_multi_step_forecast(
                model, scaler, feature_df, forecast_periods, item_info
            )
            
            results['item_forecasts'][item_id] = {
                'historical_demand': demand_series.tolist(),
                'monthly_forecasts': monthly_forecasts,
                'validation_predictions': val_pred.tolist(),
                'validation_actual': y_val.tolist(),
                'item_name': row['item_name'],
                'category': row['category'],
                'model_trained': True
            }
        
        print(f"Completed XGBoost forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def generate_multi_step_forecast(self, 
                                   model: xgb.XGBRegressor,
                                   scaler: StandardScaler,
                                   feature_df: pd.DataFrame,
                                   forecast_periods: int,
                                   item_info: Dict) -> List[float]:
        """
        Generate multi-step ahead forecasts
        
        Args:
            model: Trained XGBoost model
            scaler: Fitted scaler
            feature_df: Historical feature DataFrame
            forecast_periods: Number of periods to forecast
            item_info: Item information
            
        Returns:
            List of forecast values
        """
        forecasts = []
        
        # Get the last known features as starting point
        last_features = feature_df.iloc[-1].copy()
        last_date = list(self.date_mapping.values())[-1]
        
        # Historical demand for lag calculations
        historical_demand = feature_df['target'].values.tolist()
        
        for step in range(forecast_periods):
            # Update time-based features
            forecast_date = last_date + timedelta(days=30 * (step + 1))  # Approximate monthly
            
            last_features['month'] = forecast_date.month
            last_features['quarter'] = (forecast_date.month - 1) // 3 + 1
            last_features['year'] = forecast_date.year
            last_features['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
            last_features['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
            last_features['quarter_sin'] = np.sin(2 * np.pi * last_features['quarter'] / 4)
            last_features['quarter_cos'] = np.cos(2 * np.pi * last_features['quarter'] / 4)
            
            # Update time index
            last_features['time_index'] = len(feature_df) + step
            last_features['time_index_squared'] = (len(feature_df) + step) ** 2
            
            # Update lag features with recent predictions
            extended_demand = historical_demand + forecasts
            for lag in [1, 2, 3, 6, 12]:
                if len(extended_demand) >= lag:
                    last_features[f'lag_{lag}'] = extended_demand[-lag]
                else:
                    last_features[f'lag_{lag}'] = 0
            
            # Update rolling statistics
            for window in [3, 6, 12]:
                if len(extended_demand) >= window:
                    window_data = extended_demand[-window:]
                    last_features[f'rolling_mean_{window}'] = np.mean(window_data)
                    last_features[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0
                    last_features[f'rolling_max_{window}'] = np.max(window_data)
                    last_features[f'rolling_min_{window}'] = np.min(window_data)
                    last_features[f'rolling_sum_{window}'] = np.sum(window_data)
                else:
                    last_features[f'rolling_mean_{window}'] = 0
                    last_features[f'rolling_std_{window}'] = 0
                    last_features[f'rolling_max_{window}'] = 0
                    last_features[f'rolling_min_{window}'] = 0
                    last_features[f'rolling_sum_{window}'] = 0
            
            # Update exponential smoothing
            if len(extended_demand) > 0:
                alpha = 0.3
                exp_smooth = extended_demand[0]
                for val in extended_demand[1:]:
                    exp_smooth = alpha * val + (1 - alpha) * exp_smooth
                last_features['exp_smooth'] = exp_smooth
            
            # Update intermittency features
            if len(extended_demand) > 0:
                last_features['demand_frequency'] = np.sum(np.array(extended_demand) > 0) / len(extended_demand)
                positive_demands = [d for d in extended_demand if d > 0]
                last_features['avg_demand_when_positive'] = np.mean(positive_demands) if positive_demands else 0
                
                # Periods since last demand
                periods_since = 0
                for i in range(len(extended_demand) - 1, -1, -1):
                    if extended_demand[i] > 0:
                        break
                    periods_since += 1
                last_features['periods_since_last_demand'] = periods_since
            
            # Statistical features
            if len(extended_demand) >= 2:
                data_array = np.array(extended_demand)
                last_features['cv'] = np.std(data_array) / np.mean(data_array) if np.mean(data_array) > 0 else 0
                last_features['skewness'] = stats.skew(data_array)
                last_features['kurtosis'] = stats.kurtosis(data_array)
            
            # Prepare features for prediction (exclude target)
            feature_vector = last_features.drop('target').values.reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Make prediction
            prediction = model.predict(feature_vector_scaled)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
            forecasts.append(prediction)
        
        return forecasts
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of XGBoost forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for item_id, forecast_data in results['item_forecasts'].items():
            # Get accuracy metrics if available
            metrics = results['accuracy_metrics'].get(item_id, {})
            
            summary_data.append({
                'Item_ID': item_id,
                'Item_Name': forecast_data['item_name'][:50] + '...' if len(forecast_data['item_name']) > 50 else forecast_data['item_name'],
                'Category': forecast_data['category'],
                'Model_Trained': forecast_data['model_trained'],
                'Historical_Total': sum(forecast_data['historical_demand']),
                'Avg_Monthly_Historical': np.mean(forecast_data['historical_demand']),
                'Monthly_Forecast_Avg': round(np.mean(forecast_data['monthly_forecasts']), 2),
                'Annual_Forecast': round(sum(forecast_data['monthly_forecasts']), 2),
                'MAE': round(metrics.get('MAE', 0), 3),
                'RMSE': round(metrics.get('RMSE', 0), 3),
                'MAPE': round(metrics.get('MAPE', 0), 2)
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create comprehensive visualization of XGBoost forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to analyze
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Model Performance Distribution
        trained_items = [item_id for item_id, data in results['item_forecasts'].items() 
                        if data['model_trained']]
        
        if results['accuracy_metrics']:
            mae_values = [results['accuracy_metrics'][item_id]['MAE'] 
                         for item_id in trained_items if item_id in results['accuracy_metrics']]
            axes[0, 0].hist(mae_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Mean Absolute Error')
            axes[0, 0].set_ylabel('Number of Items')
            axes[0, 0].set_title('Distribution of Model MAE')
        
        # Plot 2: Feature Importance (Average across all models)
        if results['feature_importance']:
            # Calculate average feature importance
            all_features = {}
            for item_id, importance_dict in results['feature_importance'].items():
                for feature, importance in importance_dict.items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            avg_importance = {feature: np.mean(values) for feature, values in all_features.items()}
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            features, importances = zip(*top_features)
            axes[0, 1].barh(range(len(features)), importances, alpha=0.7)
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels(features)
            axes[0, 1].set_xlabel('Average Feature Importance')
            axes[0, 1].set_title('Top 15 Feature Importances')
        
        # Plot 3: Historical vs Forecast Comparison
        if top_items:
            top_item_ids = [item[0] for item in top_items]
            historical_avgs = [np.mean(results['item_forecasts'][item_id]['historical_demand']) 
                              for item_id in top_item_ids]
            forecast_avgs = [np.mean(results['item_forecasts'][item_id]['monthly_forecasts']) 
                            for item_id in top_item_ids]
            
            x_pos = np.arange(len(top_item_ids))
            width = 0.35
            
            axes[0, 2].bar(x_pos - width/2, historical_avgs, width, label='Historical Avg', alpha=0.7)
            axes[0, 2].bar(x_pos + width/2, forecast_avgs, width, label='XGBoost Forecast', alpha=0.7)
            axes[0, 2].set_xlabel('Items')
            axes[0, 2].set_ylabel('Average Monthly Demand')
            axes[0, 2].set_title('Top Items: Historical vs XGBoost Forecast')
            axes[0, 2].set_xticks(x_pos)
            axes[0, 2].set_xticklabels([f"Item_{i+1}" for i in range(len(top_item_ids))], rotation=45)
            axes[0, 2].legend()
        
        # Plot 4: Time Series for Top Item with Validation
        if top_items:
            top_item_id = top_items[0][0]
            top_item_data = results['item_forecasts'][top_item_id]
            
            historical = top_item_data['historical_demand']
            forecasts = top_item_data['monthly_forecasts']
            
            # Plot historical data
            hist_months = list(range(1, len(historical) + 1))
            axes[1, 0].plot(hist_months, historical, 'b-o', alpha=0.7, label='Historical', markersize=4)
            
            # Plot validation if available
            if 'validation_actual' in top_item_data and top_item_data['model_trained']:
                val_actual = top_item_data['validation_actual']
                val_pred = top_item_data['validation_predictions']
                split_point = len(historical) - len(val_actual)
                val_months = list(range(split_point + 1, len(historical) + 1))
                
                axes[1, 0].plot(val_months, val_actual, 'g-s', alpha=0.7, label='Validation Actual', markersize=4)
                axes[1, 0].plot(val_months, val_pred, 'orange', linestyle='--', marker='s', 
                               alpha=0.7, label='Validation Predicted', markersize=4)
            
            # Plot forecasts
            forecast_months = list(range(len(historical) + 1, len(historical) + 13))
            axes[1, 0].plot(forecast_months, forecasts, 'r--^', alpha=0.7, label='XGBoost Forecast', markersize=4)
            
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Demand')
            axes[1, 0].set_title(f'Top Item Time Series: {top_item_id}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Forecast Distribution
        all_forecasts = []
        for data in results['item_forecasts'].values():
            all_forecasts.extend(data['monthly_forecasts'])
        
        positive_forecasts = [f for f in all_forecasts if f > 0]
        if positive_forecasts:
            axes[1, 1].hist(positive_forecasts, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Monthly Forecast Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of XGBoost Forecasts')
        
        # Plot 6: Model Performance Scatter
        if results['accuracy_metrics']:
            mae_values = []
            rmse_values = []
            item_totals_list = []
            
            for item_id in trained_items:
                if item_id in results['accuracy_metrics']:
                    mae_values.append(results['accuracy_metrics'][item_id]['MAE'])
                    rmse_values.append(results['accuracy_metrics'][item_id]['RMSE'])
                    item_totals_list.append(sum(results['item_forecasts'][item_id]['historical_demand']))
            
            scatter = axes[1, 2].scatter(mae_values, rmse_values, c=item_totals_list, 
                                       alpha=0.6, cmap='viridis')
            axes[1, 2].set_xlabel('MAE')
            axes[1, 2].set_ylabel('RMSE')
            axes[1, 2].set_title('Model Performance: MAE vs RMSE')
            plt.colorbar(scatter, ax=axes[1, 2], label='Historical Total Demand')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate XGBoost forecasting
    """
    print("XGBoost Forecasting for Spare Parts Demand")
    print("=" * 50)
    
    # Initialize XGBoost model
    xgb_forecaster = XGBoostForecasting(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        optimize_hyperparams=True  # Set to False for faster execution
    )
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = xgb_forecaster.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit models and generate forecasts
    print("\nRunning XGBoost forecasting...")
    results = xgb_forecaster.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = xgb_forecaster.create_forecast_summary(results)
    
    print(f"\nXGBoost Forecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Model performance summary
    trained_models = summary_df[summary_df['Model_Trained'] == True]
    if len(trained_models) > 0:
        print(f"\nModel Performance Summary:")
        print(f"  Items with trained models: {len(trained_models)}")
        print(f"  Average MAE: {trained_models['MAE'].mean():.3f}")
        print(f"  Average RMSE: {trained_models['RMSE'].mean():.3f}")
        print(f"  Average MAPE: {trained_models['MAPE'].mean():.2f}%")
    
    # Create visualizations
    print("\nGenerating XGBoost forecast visualizations...")
    xgb_forecaster.plot_forecast_analysis(results, top_n=5)
    
    # Save results to Excel
    output_file = 'XGBoost_Forecast_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Forecast_Summary', index=False)
        
        # Detailed forecasts for top items
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
                    'Model_Trained': item_data['model_trained']
                })
        
        pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed_Forecasts', index=False)
        
        # Feature importance summary
        if results['feature_importance']:
            importance_data = []
            for item_id, importance_dict in results['feature_importance'].items():
                for feature, importance in importance_dict.items():
                    importance_data.append({
                        'Item_ID': item_id,
                        'Feature': feature,
                        'Importance': importance
                    })
            
            pd.DataFrame(importance_data).to_excel(writer, sheet_name='Feature_Importance', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("XGBoost forecasting completed successfully!")

if __name__ == "__main__":
    main()
