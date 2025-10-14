import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, MultiHeadAttention,
    Input, Concatenate, RepeatVector, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class LSTMForecasting:
    """
    LSTM-based forecasting for spare parts demand
    
    Uses Long Short-Term Memory neural networks to capture temporal
    dependencies and complex patterns in spare parts demand data.
    Includes multiple architectures and advanced deep learning techniques.
    """
    
    def __init__(self, 
                 sequence_length: int = 12,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 architecture: str = 'stacked',  # 'vanilla', 'stacked', 'bidirectional', 'attention'
                 use_attention: bool = True,
                 ensemble_size: int = 5):
        """
        Initialize LSTM forecasting model
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            batch_size: Training batch size
            epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            architecture: LSTM architecture type
            use_attention: Whether to use attention mechanism
            ensemble_size: Number of models for ensemble
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.use_attention = use_attention
        self.ensemble_size = ensemble_size
        
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.training_history = {}
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
    
    def create_sequences(self, data: np.array, features: Optional[np.array] = None) -> Tuple[np.array, np.array]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Time series data
            features: Additional features (optional)
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            # Input sequence
            seq_x = data[i:(i + self.sequence_length)]
            
            # Add features if available
            if features is not None:
                feature_seq = features[i:(i + self.sequence_length)]
                # Combine demand data with features
                seq_x = np.column_stack([seq_x, feature_seq])
            
            X.append(seq_x)
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def create_lstm_features(self, demand_series: np.array, item_info: Dict) -> np.array:
        """
        Create additional features for LSTM model
        
        Args:
            demand_series: Historical demand values
            item_info: Item metadata
            
        Returns:
            Feature array
        """
        features_list = []
        dates = [self.date_mapping[col] for col in self.time_columns[:len(demand_series)]]
        
        for i in range(len(demand_series)):
            features = []
            current_date = dates[i]
            
            # Time-based features
            features.extend([
                current_date.month / 12.0,  # Normalized month
                (current_date.month - 1) // 3 / 4.0,  # Normalized quarter
                np.sin(2 * np.pi * current_date.month / 12),  # Month sine
                np.cos(2 * np.pi * current_date.month / 12),  # Month cosine
            ])
            
            # Trend features
            features.extend([
                i / len(demand_series),  # Normalized time index
                (i / len(demand_series)) ** 2,  # Quadratic trend
            ])
            
            # Lag features (normalized)
            max_val = np.max(demand_series) if np.max(demand_series) > 0 else 1
            for lag in [1, 3, 6, 12]:
                if i >= lag:
                    features.append(demand_series[i - lag] / max_val)
                else:
                    features.append(0)
            
            # Rolling statistics (normalized)
            for window in [3, 6]:
                if i >= window - 1:
                    window_data = demand_series[max(0, i - window + 1):i + 1]
                    features.extend([
                        np.mean(window_data) / max_val,
                        np.std(window_data) / max_val,
                    ])
                else:
                    features.extend([0, 0])
            
            # Intermittency features
            if i > 0:
                recent_data = demand_series[:i + 1]
                features.extend([
                    np.sum(recent_data > 0) / len(recent_data),  # Demand frequency
                    (i - np.where(recent_data > 0)[0][-1] if np.sum(recent_data > 0) > 0 else i) / len(recent_data)  # Periods since last demand
                ])
            else:
                features.extend([1 if demand_series[0] > 0 else 0, 0])
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def build_vanilla_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build vanilla LSTM model
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=False, input_shape=input_shape),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_stacked_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build stacked LSTM model
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled stacked LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for units in self.lstm_units[1:-1]:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # Final LSTM layer
        model.add(LSTM(self.lstm_units[-1], return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_bidirectional_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build bidirectional LSTM model
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled bidirectional LSTM model
        """
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units[0], return_sequences=True), input_shape=input_shape),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            Bidirectional(LSTM(self.lstm_units[1], return_sequences=False)),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01)),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_attention_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build LSTM model with attention mechanism
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled LSTM model with attention
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm1 = LSTM(self.lstm_units[0], return_sequences=True)(inputs)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(self.lstm_units[1], return_sequences=True)(lstm1)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm2, lstm2)
        attention = Dropout(self.dropout_rate)(attention)
        
        # Global average pooling to reduce sequence dimension
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01))(pooled)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense2 = Dense(32, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build LSTM model based on specified architecture
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled LSTM model
        """
        if self.architecture == 'vanilla':
            return self.build_vanilla_lstm(input_shape)
        elif self.architecture == 'stacked':
            return self.build_stacked_lstm(input_shape)
        elif self.architecture == 'bidirectional':
            return self.build_bidirectional_lstm(input_shape)
        elif self.architecture == 'attention':
            return self.build_attention_lstm(input_shape)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def create_callbacks(self, item_id: str) -> List:
        """
        Create training callbacks
        
        Args:
            item_id: Item identifier
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        return callbacks
    
    def train_ensemble(self, X_train: np.array, y_train: np.array, 
                      X_val: np.array, y_val: np.array, item_id: str) -> List[Model]:
        """
        Train ensemble of LSTM models
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            item_id: Item identifier
            
        Returns:
            List of trained models
        """
        models = []
        histories = []
        
        for i in range(self.ensemble_size):
            print(f"    Training ensemble model {i+1}/{self.ensemble_size}...")
            
            # Build model
            model = self.build_model(X_train.shape[1:])
            
            # Create callbacks
            callbacks = self.create_callbacks(item_id)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            models.append(model)
            histories.append(history)
        
        # Store best model and history
        val_losses = [min(h.history['val_loss']) for h in histories]
        best_idx = np.argmin(val_losses)
        
        self.models[item_id] = models
        self.training_history[item_id] = histories[best_idx]
        
        return models
    
    def predict_ensemble(self, models: List[Model], X: np.array) -> Tuple[np.array, np.array]:
        """
        Make ensemble predictions
        
        Args:
            models: List of trained models
            X: Input sequences
            
        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        predictions = []
        
        for model in models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def generate_forecasts(self, models: List[Model], last_sequence: np.array, 
                          scaler: MinMaxScaler, forecast_periods: int = 12) -> Tuple[List[float], List[float]]:
        """
        Generate multi-step forecasts using ensemble
        
        Args:
            models: List of trained models
            last_sequence: Last input sequence
            scaler: Fitted scaler
            forecast_periods: Number of periods to forecast
            
        Returns:
            Tuple of (forecasts, uncertainties)
        """
        forecasts = []
        uncertainties = []
        
        # Use the last sequence as starting point
        current_seq = last_sequence.copy()
        
        for step in range(forecast_periods):
            # Make ensemble prediction
            mean_pred, std_pred = self.predict_ensemble(models, current_seq.reshape(1, *current_seq.shape))
            
            # Denormalize prediction
            pred_denorm = scaler.inverse_transform([[mean_pred[0]]])[0][0]
            pred_denorm = max(0, pred_denorm)  # Ensure non-negative
            
            # Calculate uncertainty (denormalized)
            uncertainty = std_pred[0] * (scaler.data_max_[0] - scaler.data_min_[0])
            
            forecasts.append(pred_denorm)
            uncertainties.append(uncertainty)
            
            # Update sequence for next prediction
            # Remove first element and append normalized prediction
            pred_norm = scaler.transform([[pred_denorm]])[0][0]
            
            if len(current_seq.shape) == 2:  # With features
                # Update only the demand column (first column)
                new_step = current_seq[-1].copy()
                new_step[0] = pred_norm
                current_seq = np.vstack([current_seq[1:], new_step.reshape(1, -1)])
            else:  # Only demand data
                current_seq = np.append(current_seq[1:], pred_norm)
        
        return forecasts, uncertainties
    
    def fit_and_forecast(self, df: pd.DataFrame, forecast_periods: int = 12) -> Dict:
        """
        Fit LSTM models and generate forecasts for all items
        
        Args:
            df: DataFrame with item data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecasting results
        """
        results = {
            'item_forecasts': {},
            'model_performance': {},
            'training_history': {},
            'ensemble_uncertainty': {}
        }
        
        print(f"Starting LSTM forecasting with {self.architecture} architecture...")
        
        for idx, row in df.iterrows():
            item_id = row['item_id']
            print(f"Processing item {idx + 1}/{len(df)}: {item_id}")
            
            # Extract demand series
            demand_series = np.array([row[col] for col in self.time_columns])
            
            # Skip items with no demand or insufficient data
            min_length = self.sequence_length + 12  # Need enough data for sequences + validation
            if np.sum(demand_series) == 0 or len(demand_series) < min_length:
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
            
            try:
                # Create features
                item_info = {
                    'category': row['category'],
                    'item_name': row['item_name']
                }
                
                features = self.create_lstm_features(demand_series, item_info)
                
                # Scale demand data
                demand_scaler = MinMaxScaler(feature_range=(0, 1))
                demand_scaled = demand_scaler.fit_transform(demand_series.reshape(-1, 1)).flatten()
                
                # Scale features
                feature_scaler = MinMaxScaler(feature_range=(0, 1))
                features_scaled = feature_scaler.fit_transform(features)
                
                # Create sequences
                X, y = self.create_sequences(demand_scaled, features_scaled)
                
                if len(X) < 12:  # Need minimum sequences
                    raise ValueError("Insufficient sequences for training")
                
                # Train-validation split (80-20)
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Train ensemble
                models = self.train_ensemble(X_train, y_train, X_val, y_val, item_id)
                
                # Store scalers
                self.scalers[item_id] = demand_scaler
                self.feature_scalers[item_id] = feature_scaler
                
                # Validation predictions
                val_pred_mean, val_pred_std = self.predict_ensemble(models, X_val)
                val_pred_denorm = demand_scaler.inverse_transform(val_pred_mean.reshape(-1, 1)).flatten()
                val_actual_denorm = demand_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mae = mean_absolute_error(val_actual_denorm, val_pred_denorm)
                rmse = np.sqrt(mean_squared_error(val_actual_denorm, val_pred_denorm))
                r2 = r2_score(val_actual_denorm, val_pred_denorm)
                
                # Calculate MAPE
                mape_values = []
                for actual, pred in zip(val_actual_denorm, val_pred_denorm):
                    if actual != 0:
                        mape_values.append(abs((actual - pred) / actual))
                mape = np.mean(mape_values) * 100 if mape_values else 0
                
                results['model_performance'][item_id] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape,
                    'val_loss': min(self.training_history[item_id].history['val_loss'])
                }
                
                # Generate forecasts
                last_sequence = X[-1]  # Use last sequence for forecasting
                forecasts, uncertainties = self.generate_forecasts(
                    models, last_sequence, demand_scaler, forecast_periods
                )
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': forecasts,
                    'forecast_uncertainty': uncertainties,
                    'validation_predictions': val_pred_denorm.tolist(),
                    'validation_actual': val_actual_denorm.tolist(),
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
                
            except Exception as e:
                print(f"  LSTM modeling failed for {item_id}: {e}")
                
                # Fallback to simple average
                avg_demand = np.mean(demand_series[demand_series > 0]) if np.sum(demand_series > 0) > 0 else 0
                
                results['item_forecasts'][item_id] = {
                    'historical_demand': demand_series.tolist(),
                    'monthly_forecasts': [avg_demand] * forecast_periods,
                    'forecast_uncertainty': [avg_demand * 0.2] * forecast_periods,
                    'item_name': row['item_name'],
                    'category': row['category'],
                    'model_fitted': False,
                    'insufficient_data': False
                }
        
        print(f"Completed LSTM forecasting for {len(results['item_forecasts'])} items")
        return results
    
    def create_forecast_summary(self, results: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of LSTM forecasting results
        
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
                'MAPE': round(performance.get('MAPE', 0), 2),
                'Val_Loss': round(performance.get('val_loss', 0), 6)
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_forecast_analysis(self, results: Dict, top_n: int = 5):
        """
        Create comprehensive visualization of LSTM forecasting results
        
        Args:
            results: Results dictionary from fit_and_forecast
            top_n: Number of top items to analyze
        """
        # Get top items by historical demand
        item_totals = {item_id: sum(data['historical_demand']) 
                      for item_id, data in results['item_forecasts'].items()}
        top_items = sorted(item_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Model Performance Distribution (R² Score)
        fitted_items = [item_id for item_id, data in results['item_forecasts'].items() 
                       if data.get('model_fitted', False)]
        
        if results['model_performance']:
            r2_values = [results['model_performance'][item_id]['R2'] 
                        for item_id in fitted_items if item_id in results['model_performance']]
            axes[0, 0].hist(r2_values, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 0].set_xlabel('R² Score')
            axes[0, 0].set_ylabel('Number of Models')
            axes[0, 0].set_title(f'LSTM Model Performance (R²)\nArchitecture: {self.architecture.title()}')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training Loss Distribution
        if results['model_performance']:
            val_losses = [results['model_performance'][item_id]['val_loss'] 
                         for item_id in fitted_items if item_id in results['model_performance']]
            if val_losses:
                axes[0, 1].hist(val_losses, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
                axes[0, 1].set_xlabel('Validation Loss')
                axes[0, 1].set_ylabel('Number of Models')
                axes[0, 1].set_title('LSTM Training Validation Loss')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Ensemble Uncertainty Distribution
        all_uncertainties = []
        for data in results['item_forecasts'].values():
            if data.get('model_fitted', False):
                all_uncertainties.extend(data['forecast_uncertainty'])
        
        if all_uncertainties:
            axes[1, 0].hist(all_uncertainties, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
            axes[1, 0].set_xlabel('Forecast Uncertainty')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Ensemble Uncertainties')
            axes[1, 0].grid(True, alpha=0.3)
        
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
            axes[1, 1].plot(forecast_months, forecasts, 'r--^', alpha=0.7, label='LSTM Forecast', markersize=4)
            
            # Add uncertainty bands
            lower_bound = [max(0, f - 1.96 * u) for f, u in zip(forecasts, uncertainties)]
            upper_bound = [f + 1.96 * u for f, u in zip(forecasts, uncertainties)]
            axes[1, 1].fill_between(forecast_months, lower_bound, upper_bound, 
                                   alpha=0.2, color='red', label='95% Confidence')
            
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Demand')
            axes[1, 1].set_title(f'LSTM Forecast: {top_item_id}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Model Performance Scatter (MAE vs R²)
        if results['model_performance']:
            mae_values = []
            r2_values = []
            item_totals_list = []
            
            for item_id in fitted_items:
                if item_id in results['model_performance']:
                    mae_values.append(results['model_performance'][item_id]['MAE'])
                    r2_values.append(results['model_performance'][item_id]['R2'])
                    item_totals_list.append(sum(results['item_forecasts'][item_id]['historical_demand']))
            
            if mae_values and r2_values:
                scatter = axes[2, 0].scatter(mae_values, r2_values, c=item_totals_list, 
                                           alpha=0.6, cmap='viridis', s=60)
                axes[2, 0].set_xlabel('MAE')
                axes[2, 0].set_ylabel('R² Score')
                axes[2, 0].set_title('LSTM Performance: MAE vs R²')
                axes[2, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[2, 0], label='Historical Total Demand')
        
        # Plot 6: Historical vs Forecast Comparison
        if top_items:
            top_item_ids = [item[0] for item in top_items]
            historical_avgs = [np.mean(results['item_forecasts'][item_id]['historical_demand']) 
                              for item_id in top_item_ids]
            forecast_avgs = [np.mean(results['item_forecasts'][item_id]['monthly_forecasts']) 
                            for item_id in top_item_ids]
            
            x_pos = np.arange(len(top_item_ids))
            width = 0.35
            
            axes[2, 1].bar(x_pos - width/2, historical_avgs, width, label='Historical Avg', alpha=0.7, color='steelblue')
            axes[2, 1].bar(x_pos + width/2, forecast_avgs, width, label='LSTM Forecast', alpha=0.7, color='coral')
            axes[2, 1].set_xlabel('Items')
            axes[2, 1].set_ylabel('Average Monthly Demand')
            axes[2, 1].set_title('Top Items: Historical vs LSTM Forecast')
            axes[2, 1].set_xticks(x_pos)
            axes[2, 1].set_xticklabels([f"Item_{i+1}" for i in range(len(top_item_ids))], rotation=45)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, results: Dict, item_id: str):
        """
        Plot training history for a specific item
        
        Args:
            results: Results dictionary
            item_id: Item to analyze
        """
        if item_id not in self.training_history:
            print(f"No training history available for {item_id}")
            return
        
        history = self.training_history[item_id]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss', alpha=0.7)
        ax1.plot(history.history['val_loss'], label='Validation Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training History: {item_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE', alpha=0.7)
            ax2.plot(history.history['val_mae'], label='Validation MAE', alpha=0.7)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title(f'MAE History: {item_id}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate LSTM forecasting
    """
    print("LSTM Forecasting for Spare Parts Demand")
    print("=" * 45)
    
    # Initialize LSTM model
    lstm_forecaster = LSTMForecasting(
        sequence_length=12,
        lstm_units=[128, 64],
        dropout_rate=0.2,
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        architecture='stacked',  # 'vanilla', 'stacked', 'bidirectional', 'attention'
        use_attention=True,
        ensemble_size=3  # Reduced for faster execution
    )
    
    # Load data
    file_path = 'Sample_FiveYears_Sales_SpareParts.xlsx'
    df = lstm_forecaster.load_data(file_path)
    
    if df.empty:
        print("Failed to load data. Please check the file path and format.")
        return
    
    # Fit models and generate forecasts
    print(f"\nRunning LSTM forecasting...")
    results = lstm_forecaster.fit_and_forecast(df, forecast_periods=12)
    
    # Create summary
    summary_df = lstm_forecaster.create_forecast_summary(results)
    
    print(f"\nLSTM Forecast Summary (Top 10 by Historical Demand):")
    print(summary_df.nlargest(10, 'Historical_Total').to_string(index=False))
    
    # Model performance summary
    fitted_models = summary_df[summary_df['Model_Fitted'] == True]
    if len(fitted_models) > 0:
        print(f"\nLSTM Performance Summary:")
        print(f"  Items with fitted models: {len(fitted_models)}")
        print(f"  Average MAE: {fitted_models['MAE'].mean():.3f}")
        print(f"  Average RMSE: {fitted_models['RMSE'].mean():.3f}")
        print(f"  Average R² Score: {fitted_models['R2_Score'].mean():.3f}")
        print(f"  Average MAPE: {fitted_models['MAPE'].mean():.2f}%")
        print(f"  Average Validation Loss: {fitted_models['Val_Loss'].mean():.6f}")
        print(f"  Average Forecast Uncertainty: {fitted_models['Forecast_Uncertainty_Avg'].mean():.3f}")
    
    # Create visualizations
    print("\nGenerating LSTM forecast visualizations...")
    lstm_forecaster.plot_forecast_analysis(results, top_n=5)
    
    # Show training history for top item
    top_item = summary_df.nlargest(1, 'Historical_Total')['Item_ID'].iloc[0]
    if summary_df[summary_df['Item_ID'] == top_item]['Model_Fitted'].iloc[0]:
        print(f"\nTraining history for top item: {top_item}")
        lstm_forecaster.plot_training_history(results, top_item)
    
    # Save results to Excel
    output_file = 'LSTM_Forecast_Results.xlsx'
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
        
        # Model performance details
        if results['model_performance']:
            performance_data = []
            for item_id, metrics in results['model_performance'].items():
                performance_data.append({
                    'Item_ID': item_id,
                    'MAE': round(metrics['MAE'], 4),
                    'RMSE': round(metrics['RMSE'], 4),
                    'R2_Score': round(metrics['R2'], 4),
                    'MAPE': round(metrics['MAPE'], 2),
                    'Validation_Loss': round(metrics['val_loss'], 6)
                })
            
            pd.DataFrame(performance_data).to_excel(writer, sheet_name='Model_Performance', index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"LSTM forecasting completed successfully!")
    print(f"Architecture used: {lstm_forecaster.architecture}")
    print(f"Ensemble size: {lstm_forecaster.ensemble_size}")

if __name__ == "__main__":
    main()
