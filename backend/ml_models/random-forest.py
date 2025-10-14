import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import optuna
from scipy import stats
from joblib import Parallel, delayed
import itertools

warnings.filterwarnings('ignore')

class RandomForestForecasting:
    """
    Random Forest-based forecasting for spare parts demand
    
    Uses ensemble of decision trees with advanced feature engineering
    to capture complex patterns in spare parts demand. Excellent for
    12-month forecasting horizons with robust performance.
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 n_jobs: int = -1,
                 optimize_hyperparams: bool = True,
                 forecast_strategy: str = 'recursive'):
        """
        Initialize Random Forest forecasting model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            n_jobs: Number of parallel jobs
            optimize_hyperparams: Whether to optimize hyperparameters
            forecast_strategy: 'recursive' or 'direct' forecasting
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.optimize_hyperparams = optimize_hyperparams
        self.forecast_strategy = forecast_strategy
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.forecast_results = {}
        self.feature_names = []
        
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
    
    def create_comprehensive_features(self, demand_series: np.array, item_info: Dict) -> pd.DataFrame:
        """
        Create comprehensive feature set for Random Forest
        
        Args:
            demand_series: Historical demand values
            item_info: Item metadata
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        dates = [self.date_mapping[col] for col in self.time_columns[:len(demand_series)]]
        
        for i in range(len(demand_series)):
            features = {}
            current_date = dates[i]
            
            # === Time-based features ===
            features['month'] = current_date.month
            features['quarter'] = (current_date.month - 1) // 3 + 1
            features['year'] = current_date.year
            features['day_of_year'] = current_date.timetuple().tm_yday
            features['week_of_year'] = current_date.isocalendar()[1]
            
            # Cyclical time features
            features['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
            features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
            features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
            features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
            
            # Time index features
            features['time_index'] = i
            features['time_index_squared'] = i ** 2
            features['time_index_log'] = np.log(i + 1)
            features['time_index_sqrt'] = np.sqrt(i)
            
            # === Lag features ===
            lag_periods = [1, 2, 3, 6, 12, 18, 24]
            for lag in lag_periods:
                if i >= lag:
                    features[f'lag_{lag}'] = demand_series[i - lag]
                else:
                    features[f'lag_{lag}'] = 0
            
            # === Rolling window features ===
            window_sizes = [3, 6, 12, 24]
            
            for window in window_sizes:
                if i >= window - 1:
                    window_data = demand_series[max(0, i - window + 1):i + 1]
                    
                    # Basic statistics
                    features[f'rolling_mean_{window}'] = np.mean(window_data)
                    features[f'rolling_std_{window}'] = np.std(window_data)
                    features[f'rolling_median_{window}'] = np.median(window_data)
                    features[f'rolling_max_{window}'] = np.max(window_data)
                    features[f'rolling_min_{window}'] = np.min(window_data)
                    features[f'rolling_sum_{window}'] = np.sum(window_data)
                    features[f'rolling_range_{window}'] = np.max(window_data) - np.min(window_data)
                    
                    # Advanced statistics
                    features[f'rolling_skew_{window}'] = stats.skew(window_data) if len(window_data) > 2 else 0
                    features[f'rolling_kurtosis_{window}'] = stats.kurtosis(window_data) if len(window_data) > 2 else 0
                    
                    # Percentiles
                    features[f'rolling_q25_{window}'] = np.percentile(window_data, 25)
                    features[f'rolling_q75_{window}'] = np.percentile(window_data, 75)
                    features[f'rolling_iqr_{window}'] = np.percentile(window_data, 75) - np.percentile(window_data, 25)
                    
                    # Trend indicators
                    if len(window_data) > 1:
                        x = np.arange(len(window_data))
                        slope, _, _, _, _ = stats.linregress(x, window_data)
                        features[f'rolling_trend_{window}'] = slope
                    else:
                        features[f'rolling_trend_{window}'] = 0
                        
                else:
                    # Fill with zeros for insufficient data
                    fill_features = [
                        f'rolling_mean_{window}', f'rolling_std_{window}', f'rolling_median_{window}',
                        f'rolling_max_{window}', f'rolling_min_{window}', f'rolling_sum_{window}',
                        f'rolling_range_{window}', f'rolling_skew_{window}', f'rolling_kurtosis_{window}',
                        f'rolling_q25_{window}', f'rolling_q75_{window}', f'rolling_iqr_{window}',
                        f'rolling_trend_{window}'
                    ]
                    for feat in fill_features:
                        features[feat] = 0
            
            # === Exponential smoothing features ===
            alpha_values = [0.1, 0.3, 0.5, 0.7]
            for alpha in alpha_values:
                if i > 0:
                    exp_smooth = demand_series[0]
                    for j in range(1, i + 1):
                        exp_smooth = alpha * demand_series[j] + (1 - alpha) * exp_smooth
                    features[f'exp_smooth_{int(alpha*10)}'] = exp_smooth
                else:
                    features[f'exp_smooth_{int(alpha*10)}'] = demand_series[0] if len(demand_series) > 0 else 0
            
            # === Intermittency and demand pattern features ===
            if i > 0:
                recent_data = demand_series[:i + 1]
                
                # Demand frequency and patterns
                features['demand_frequency'] = np.sum(recent_data > 0) / len(recent_data)
                features['avg_demand_when_positive'] = np.mean(recent_data[recent_data > 0]) if np.sum(recent_data > 0) > 0 else 0
                features['max_demand_so_far'] = np.max(recent_data)
                features['min_demand_so_far'] = np.min(recent_data)
                
                # Periods since/until demand
                features['periods_since_last_demand'] = 0
                for j in range(i, -1, -1):
                    if demand_series[j] > 0:
                        break
                    features['periods_since_last_demand'] += 1
                
                # Demand variability
                if len(recent_data) > 1:
                    features['cv'] = np.std(recent_data) / np.mean(recent_data) if np.mean(recent_data) > 0 else 0
                    features['demand_volatility'] = np.std(np.diff(recent_data))
                else:
                    features['cv'] = 0
                    features['demand_volatility'] = 0
                    
                # Demand concentration (Gini coefficient approximation)
                if np.sum(recent_data) > 0:
                    sorted_demand = np.sort(recent_data)
                    n = len(sorted_demand)
                    cumsum = np.cumsum(sorted_demand)
                    features['demand_concentration'] = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
                else:
                    features['demand_concentration'] = 0
                    
            else:
                # Initialize for first period
                features['demand_frequency'] = 1 if demand_series[0] > 0 else 0
                features['avg_demand_when_positive'] = demand_series[0] if demand_series[0] > 0 else 0
                features['max_demand_so_far'] = demand_series[0]
                features['min_demand_so_far'] = demand_series[0]
                features['periods_since_last_demand'] = 0 if demand_series[0] > 0 else 1
                features['cv'] = 0
                features['demand_volatility'] = 0
                features['demand_concentration'] = 0
            
            # === Seasonal and cyclical features ===
            # Moving seasonal averages
            for season_lag in [12, 24]:  # 1 and 2 years ago
                if i >= season_lag:
                    features[f'seasonal_lag_{season_lag}'] = demand_series[i - season_lag]
                    
                    # Seasonal growth rate
                    if demand_series[i - season_lag] > 0:
                        current_val = demand_series[i] if i < len(demand_series) else 0
                        features[f'seasonal_growth_{season_lag}'] = (current_val - demand_series[i - season_lag]) / demand_series[i - season_lag]
                    else:
                        features[f'seasonal_growth_{season_lag}'] = 0
                else:
                    features[f'seasonal_lag_{season_lag}'] = 0
                    features[f'seasonal_growth_{season_lag}'] = 0
            
            # === Item-specific features ===
            # Encode categorical features
            features['category_encoded'] = hash(item_info['category']) % 1000
            
            # Item name length (proxy for complexity)
            features['item_name_length'] = len(item_info['item_name'])
            
            # === Interaction features ===
            # Time-demand interactions
            features['month_x_avg_demand'] = features['month'] * features.get('rolling_mean_12', 0)
            features['quarter_x_trend'] = features['quarter'] * features.get('rolling_trend_12', 0)
            
            # Lag interactions
            if features.get('lag_1', 0) > 0 and features.get('lag_12', 0) > 0:
                features['lag1_x_lag12'] = features['lag_1'] * features['lag_12']
            else:
                features['lag1_x_lag12'] = 0
            
            # === Target variable ===
            features['target'] = demand_series[i]
            
            features_list.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features_list)
        
        # Store feature names for later use
        self.feature_names = [col for col in feature_df.columns if col != 'target']
        
        return feature_df
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, item_id: str) -> Dict:
        """
        Optimize Random Forest hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training targets
            item_id: Item identifier
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params['bootstrap'] else None
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                score = mean_absolute_error(y_fold_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study with reduced trials for speed
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        return study.best_params
    
    def create_ensemble_forecast(self, model: RandomForestRegressor, X_last: pd.DataFrame, 
                               forecast_periods: int, historical_data: np.array) -> Tuple[List[float], List[float]]:
        """
        Create ensemble forecasts with uncertainty estimation
        
        Args:
            model: Trained Random Forest model
            X_last: Last feature vector
            forecast_periods: Number of periods to forecast
            historical_data: Historical demand data
            
        Returns:
            Tuple of (forecasts, uncertainties)
        """
        forecasts = []
        uncertainties = []
        
        # Get individual tree predictions for uncertainty estimation
        tree_predictions = []
        
        # Recursive forecasting approach
        current_features = X_last.copy()
        extended_history = list(historical_data)
        
        for step in range(forecast_periods):
            # Get predictions from all trees
            tree_preds = [tree.predict(current_features.values.reshape(1, -1))[0] 
                         for tree in model.estimators_]
            tree_predictions.append(tree_preds)
            
            # Calculate forecast and uncertainty
            forecast_mean = np.mean(tree_preds)
            forecast_std = np.std(tree_preds)
            
            # Ensure non-negative forecast
            forecast_mean = max(0, forecast_mean)
            forecasts.append(forecast_mean)
            uncertainties.append(forecast_std)
            
            # Update features for next step
            extended_history.append(forecast_mean)
            
            # Update lag features
            for lag in [1, 2, 3, 6, 12, 18, 24]:
                if len(extended_history) > lag:
                    current_features[f'lag_{lag}'] = extended_history[-lag-1]
            
            # Update rolling features
            for window in [3, 6, 12, 24]:
                if len(extended_history) >= window:
                    window_data = extended_history[-window:]
                    current_features[f'rolling_mean_{window}'] = np.mean(window_data)
                    current_features[f'rolling_std_{window}'] = np.std(window_data)
                    current_features[f'rolling_max_{window}'] = np.max(window_data)
                    current_features[f'rolling_min_{window}'] = np.min(window_data)
                    current_features[f'rolling_sum_{window}'] = np.sum(window_data)
            
            # Update time index
            current_features['time_index'] = current_features['time_index'] + 1
            current_features['time_index_squared'] = current_features['time_index'] ** 2
            current_features['time_index_log'] = np.log(current_features['time_index'] + 1)
            current_features['time_index_sqrt'] = np.sqrt(current_features['time_index'])
            
            # Update exponential smoothing
            for alpha_int in [1, 3, 5, 7]:
                alpha = alpha_int / 10
                if step == 0:
                    current_features[f'exp_smooth_{alpha_int}'] = alpha * forecast_mean + (1 - alpha) * current_features[f'exp_smooth_{alpha_int}']
                else:
                    current_features[f'exp_smooth_{alpha_int}'] = alpha * forecast_mean + (1 - alpha) * current_features[f'exp_smooth_{alpha_int}']
        
        return forecasts, uncertainties
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit Random Forest models and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'feature_importance': {},
            'model_performance': {},
            'ensemble_uncertainty': {}
        }
        
        print("Starting Random Forest forecasting...")
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            print(f"Processing item {idx + 1}/{len(df)}: {item_id}")
            
            # Extract demand series
            demand_series = np.array([row[col] for col in self.time_columns])
            
            # Skip items with no demand or insufficient data
            if np.sum(demand_series) == 0 or len(demand_series) < 18:  # Need at least 18 months
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': [0] * forecast_periods,
                    'forecast_uncertainty': [0] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': True
                }
                continue
            
            # Create features
            item_info = {
                'category': row['category'],
                'item_name': row['item_name']
            }
            
            feature_df = self.create_comprehensive_features(demand_series, item_info)
            
            # Prepare training data (use 80% for training, 20% for validation)
            split_point = int(len(feature_df) * 0.8)
            
            if split_point < 12:  # Need minimum data for training
                # Use simple average for items with insufficient data
                avg_demand = np.mean(demand_series[demand_series > 0]) if np.sum(demand_series > 0) > 0 else 0
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': [avg_demand] * forecast_periods,
                    'forecast_uncertainty': [avg_demand * 0.2] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': True
                }
                continue
            
            # Split data
            train_features = feature_df[:split_point]
            val_features = feature_df[split_point:]
            
            X_train = train_features.drop('target', axis=1)
            y_train = train_features['target']
            X_val = val_features.drop('target', axis=1)
            y_val = val_features['target']
            
            # Scale features using RobustScaler (better for outliers)
            scaler = RobustScaler()
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
                print(f"  Optimizing hyperparameters for {item_id}...")
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train, item_id)
                self.best_params = best_params
            elif hasattr(self, 'best_params'):
                best_params = self.best_params
            else:
                best_params = {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'max_features': self.max_features,
                    'bootstrap': self.bootstrap
                }
            
            # Train Random Forest model
            model = RandomForestRegressor(
                **best_params,
                random_state=42,
                n_jobs=1  # Use single job to avoid nested parallelization
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Store model and scaler
            self.models[item_id] = model
            self.scalers[item_id] = scaler
            
            # Validation predictions
            val_pred = model.predict(X_val_scaled)
            val_pred = np.maximum(val_pred, 0)  # Ensure non-negative
            
            # Calculate accuracy metrics
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            r2 = r2_score(y_val, val_pred)
            
            # Calculate MAPE
            mape_values = []
            for actual, pred in zip(y_val, val_pred):
                if actual != 0:
                    mape_values.append(abs((actual - pred) / actual))
            mape = np.mean(mape_values) * 100 if mape_values else 0
            
            results['model_performance'][item_id] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            # Feature importance
            importance = model.feature_importances_
            results['feature_importance'][item_id] = dict(zip(X_train_scaled.columns, importance))
            
            # Generate forecasts using ensemble approach
            X_last = X_train_scaled.iloc[-1:].copy()
            forecasts, uncertainties = self.create_ensemble_forecast(
                model, X_last, forecast_periods, demand_series
            )
            
            results['item_forecasts'][item_id] = {
                'historical_demand': demand_series.tolist(),
                'monthly_forecasts': forecasts,
                'forecast_uncertainty': uncertainties,
                'validation_predictions': val_pred.tolist(),
                'validation_actual': y_val.tolist(),
                'item_name': row['item_name'],
                'category': row['category'],
                'model_fitted': True,
                'insufficient_data': False
            }
            
            results['ensemble_uncertainty'][item_id] = {
                'mean_uncertainty': np.mean(uncertainties),
                'max_uncertainty': np.max(uncertainties),
                'uncertainty_trend': np.polyfit(range(len(uncertainties)), uncertainties, 1)[0] if len(uncertainties) > 1 else 0
            }
        
        print(f"Completed Random Forest forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of Random Forest forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for item_id, forecast_data in results['item_forecasts'].items():
            # Get additional metrics if available
            performance = results['model_performance'].get(item_id, {})
            uncertainty = results['ensemble_uncertainty'].get(item_id, {})
            
            summary_data.append({
                'Item_ID': item_id,
                'Item_Name': forecast_data['item_name'][:50] + '...' if len(forecast_data['item_name']) > 50 else forecast_data['item_name'],
                'Category': forecast_data['category'],
                'Model_Fitted': forecast_data.get('model_fitted', False),
                'Historical_Total': sum(forecast_data['historical_demand']),
                'Avg_Monthly_Historical': round(np.mean(forecast_data['historical_demand']), 2),
                'Monthly_Forecast_Avg': round(np.mean(forecast_data['monthly_forecasts']), 2),
                'Annual_Forecast': round(sum(forecast_data['monthly_forecasts']), 2),
                'Forecast_Uncertainty_Avg': round(np.mean(forecast_data['forecast_uncertainty']), 3),
                'MAE': round(performance.get('MAE', 0), 3),
                'RMSE': round(performance.get('RMSE', 0), 3),
                'R2_Score': round(performance.get('R2', 0), 3),
                'MAPE': round(performance.get('MAPE', 0), 2)
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create comprehensive visualization of Random Forest forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to analyze
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Model Performance Distribution
        fitted_items = [item_id for item_id, data in results['item_forecasts'].items() 
                       if data.get('model_fitted', False)]
        
        if results['model_performance']:
            r2_values = [results['model_performance'][item_id]['R2'] 
                        for item_id in fitted_items if item_id in results['model_performance']]
            axes[0, 0].hist(r2_values, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('R² Score')
            axes[0, 0].set_ylabel('Number of Models')
            axes[0, 0].set_title('Random Forest Model Performance (R²)')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
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
        
        # Plot 3: Forecast Uncertainty Distribution
        all_uncertainties = []
        for data in results['item_forecasts'].values():
            if data.get('model_fitted', False):
                all_uncertainties.extend(data['forecast_uncertainty'])
        
        if all_uncertainties:
            axes[1, 0].hist(all_uncertainties, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Forecast Uncertainty (Std Dev)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Forecast Uncertainties')
        
        # Plot 4: Top Item Time Series with Uncertainty Bands
        if top_items:
            top_item_id = top_items[0][0]
            top_item_data = results['item_forecasts'][top_item_id]
            
            historical = top_item_data['historical_demand']
            forecasts = top_item_data['monthly_forecasts']
            uncertainties = top_item_data['forecast_uncertainty']
            
            # Plot historical data
            hist_months = list(range(1, len(historical) + 1))
            axes[1, 1].plot(hist_months, historical, 'b-o', alpha=0.7, label='Historical', markersize=4)
            
            # Plot validation if available
            if 'validation_actual' in top_item_data and top_item_data.get('model_fitted', False):
                val_actual = top_item_data['validation_actual']
                val_pred = top_item_data['validation_predictions']
                split_point = len(historical) - len(val_actual)
                val_months = list(range(split_point + 1, len(historical) + 1))
                
                axes[1, 1].plot(val_months, val_actual, 'g-s', alpha=0.7, label='Validation Actual', markersize=4)
                axes[1, 1].plot(val_months, val_pred, 'orange', linestyle='--', marker='s', 
                               alpha=0.7, label='Validation Predicted', markersize=4)
            
            # Plot forecasts with uncertainty bands
            forecast_months = list(range(len(historical) + 1, len(historical) + 13))
            axes[1, 1].plot(forecast_months, forecasts, 'r--^', alpha=0.7, label='RF Forecast', markersize=4)
            
            # Add uncertainty bands
            lower_bound = [max(0, f - 1.96 * u) for f, u in zip(forecasts, uncertainties)]
            upper_bound = [f + 1.96 * u for f, u in zip(forecasts, uncertainties)]
            axes[1, 1].fill_between(forecast_months, lower_bound, upper_bound, 
                                   alpha=0.2, color='red', label='95% Confidence')
            
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].set_title(f'Random Forest Forecast: {top_item_id}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Model Performance Scatter
        if results['model_performance']:
            mae_values = []
            r2_values = []
            item_totals_list = []
            
            for item_id in fitted_items:
                if item_id in results['model_performance']:
                    mae_values.append(results['model_performance'][item_id]['MAE'])
                    r2_values.append(results['model_performance'][item_id]['R2'])
                    item_totals_list.append(sum(results['item_forecasts'][item_id]['historical_demand']))
            
            scatter = axes[2, 0].scatter(mae_values, r2_values, c=item_totals_list, 
                                       alpha=0.6, cmap='viridis')
            axes[2, 0].set_xlabel('MAE')
            axes[2, 0].set_ylabel('R² Score')
            axes[2, 0].set_title('Model Performance: MAE vs R²')
            plt.colorbar(scatter, ax=axes[2, 0], label='Total Historical Demand')
        
        # Plot 6: Historical vs Forecast Comparison
        if top_items:
            top_item_ids = [item[0] for item in top_items]
            historical_avgs = [np.mean(results['item_forecasts'][item_id]['historical_demand']) 
                              for item_id in top_item_ids]
            forecast_avgs = [np.mean(results['item_forecasts'][item_id]['monthly_forecasts']) 
                            for item_id in top_item_ids]
            
            x_pos = np.arange(len(top_item_ids))
            width = 0.35
            
            axes[2, 1].bar(x_pos - width/2, historical_avgs, width, label='Historical Avg', alpha=0.7)
            axes[2, 1].bar(x_pos + width/2, forecast_avgs, width, label='RF Forecast', alpha=0.7)
            axes[2, 1].set_xlabel('Items')
            axes[2, 1].set_ylabel('Average Monthly Demand')
            axes[2, 1].set_title('Top Items: Historical vs RF Forecast')
            axes[2, 1].set_xticks(x_pos)
            axes[2, 1].set_xticklabels([f"Item_{i+1}" for i in range(len(top_item_ids))], rotation=45)
            axes[2, 1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate Random Forest forecasting
    """
    print("Random Forest Forecasting for Spare Parts Demand")
    print("=" * 50)
    
    # Initialize Random Forest model
    rf_forecaster = RandomForestForecasting(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        optimize_hyperparams=True,
        forecast_strategy='recursive'
    )
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = rf_forecaster.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit models and generate forecasts
    print("\nRunning Random Forest forecasting...")
    results = rf_forecaster.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = rf_forecaster.create_forecast_summary(results)
    
    print(f"\nRandom Forest Forecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Model performance summary
    fitted_models = summary_df[summary_df['Model_Fitted'] == True]
    if len(fitted_models) > 0:
        print(f"\nRandom Forest Performance Summary:")
        print(f"  Items with fitted models: {len(fitted_models)}")
        print(f"  Average MAE: {fitted_models['MAE'].mean():.3f}")
        print(f"  Average RMSE: {fitted_models['RMSE'].mean():.3f}")
        print(f"  Average R² Score: {fitted_models['R2_Score'].mean():.3f}")
        print(f"  Average MAPE: {fitted_models['MAPE'].mean():.2f}%")
        print(f"  Average Forecast Uncertainty: {fitted_models['Forecast_Uncertainty_Avg'].mean():.3f}")
    
    # Feature importance summary
    if results['feature_importance']:
        print(f"\nTop 5 Most Important Features (Average):")
        all_features = {}
        for importance_dict in results['feature_importance'].values():
            for feature, importance in importance_dict.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        avg_importance = {feature: np.mean(values) for feature, values in all_features.items()}
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # Create visualizations
    print("\nGenerating Random Forest forecast visualizations...")
    rf_forecaster.plot_forecast_analysis(results, top_n=5)
    
    # Save results to Excel
    output_file = 'RandomForest_Forecast_Results.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Forecast_Summary', index=False)
        
        # Detailed forecasts with uncertainty
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
                    'Forecast_Uncertainty': round(item_data['forecast_uncertainty'][month], 3),
                    'Lower_95CI': round(max(0, item_data['monthly_forecasts'][month] - 1.96 * item_data['forecast_uncertainty'][month]), 2),
                    'Upper_95CI': round(item_data['monthly_forecasts'][month] + 1.96 * item_data['forecast_uncertainty'][month], 2),
                    'Model_Fitted': item_data.get('model_fitted', False)
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
                        'Importance': round(importance, 6)
                    })
            
            pd.DataFrame(importance_data).to_excel(writer, sheet_name='Feature_Importance', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print("Random Forest forecasting completed successfully!")

if __name__ == "__main__":
    main()
