"""
Data preprocessing utilities for allocation data
"""

import numpy as np
import pandas as pd
import logging
import io
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from fastapi import UploadFile

# from models.data_models import AllocationRecord  # Avoid circular imports for now


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing utilities for ML models and optimization"""
    
    def __init__(self):
        """Initialize data preprocessor"""
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self._preprocessing_stats = {}
    
    def records_to_dataframe(self, records: List[Any]) -> pd.DataFrame:
        """
        Convert AllocationRecord objects to pandas DataFrame
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            pandas DataFrame
        """
        if not records:
            return pd.DataFrame()
        
        # Convert records to dictionaries
        data = []
        for record in records:
            record_dict = record.dict()
            # Convert enums to string values
            record_dict['customer_tier'] = record.customer_tier.value
            record_dict['sla_level'] = record.sla_level.value
            data.append(record_dict)
        
        df = pd.DataFrame(data)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def create_feature_matrix(
        self, 
        records: List[Any],
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for ML models
        
        Args:
            records: List of AllocationRecord objects
            features: Specific features to include (None for all numeric features)
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        df = self.records_to_dataframe(records)
        
        if df.empty:
            return np.array([]), []
        
        # Default numeric features for ML models
        default_features = [
            'dc_priority',
            'current_inventory',
            'forecasted_demand',
            'historical_demand',
            'revenue_per_unit',
            'cost_per_unit',
            'margin',
            'risk_score',
            'allocated_quantity',
            'fulfillment_rate',
            'lead_time_days',
            'min_order_quantity',
            'safety_stock'
        ]
        
        feature_names = features or default_features
        
        # Filter to only include available features
        available_features = [f for f in feature_names if f in df.columns]
        
        if not available_features:
            logger.warning("No valid features found in data")
            return np.array([]), []
        
        # Create feature matrix
        feature_matrix = df[available_features].values
        
        # Handle any remaining NaN values
        if np.isnan(feature_matrix).any():
            logger.warning("Found NaN values in feature matrix, imputing...")
            imputer = SimpleImputer(strategy='median')
            feature_matrix = imputer.fit_transform(feature_matrix)
        
        logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
        
        return feature_matrix, available_features
    
    def create_categorical_features(self, records: List[Any]) -> pd.DataFrame:
        """
        Create one-hot encoded categorical features
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            DataFrame with one-hot encoded categorical features
        """
        df = self.records_to_dataframe(records)
        
        if df.empty:
            return pd.DataFrame()
        
        categorical_features = [
            'dc_region',
            'sku_category', 
            'customer_tier',
            'customer_region',
            'sla_level'
        ]
        
        categorical_df = pd.DataFrame()
        
        for feature in categorical_features:
            if feature in df.columns:
                # One-hot encode
                encoded = pd.get_dummies(df[feature], prefix=feature)
                categorical_df = pd.concat([categorical_df, encoded], axis=1)
        
        return categorical_df
    
    def scale_features(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        method: str = 'standard',
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Scale numerical features
        
        Args:
            feature_matrix: Input feature matrix
            feature_names: List of feature names
            method: Scaling method ('standard' or 'minmax')
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled feature matrix
        """
        if feature_matrix.size == 0:
            return feature_matrix
        
        scaler_key = f"{method}_{hash(tuple(feature_names))}"
        
        if fit_scaler or scaler_key not in self.scalers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            scaled_matrix = scaler.fit_transform(feature_matrix)
            self.scalers[scaler_key] = scaler
            
            logger.info(f"Fitted and applied {method} scaler")
        else:
            scaler = self.scalers[scaler_key]
            scaled_matrix = scaler.transform(feature_matrix)
            
            logger.info(f"Applied existing {method} scaler")
        
        return scaled_matrix
    
    def prepare_time_series_data(
        self,
        records: List[Any],
        target_column: str = 'allocated_quantity',
        window_size: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for forecasting models
        
        Args:
            records: List of AllocationRecord objects
            target_column: Target variable for forecasting
            window_size: Number of previous time steps to use as features
            
        Returns:
            Tuple of (X, y) arrays for time series modeling
        """
        df = self.records_to_dataframe(records)
        
        if df.empty or len(df) < window_size + 1:
            return np.array([]), np.array([])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Group by DC and SKU for time series
        time_series_data = []
        
        for (dc_id, sku_id), group in df.groupby(['dc_id', 'sku_id']):
            if len(group) < window_size + 1:
                continue
            
            values = group[target_column].values
            
            # Create sliding windows
            for i in range(len(values) - window_size):
                X_window = values[i:i + window_size]
                y_target = values[i + window_size]
                time_series_data.append((X_window, y_target))
        
        if not time_series_data:
            return np.array([]), np.array([])
        
        X = np.array([item[0] for item in time_series_data])
        y = np.array([item[1] for item in time_series_data])
        
        logger.info(f"Created time series data with {len(X)} samples")
        
        return X, y
    
    def create_demand_forecast_features(
        self, 
        records: List[Any]
    ) -> pd.DataFrame:
        """
        Create features specifically for demand forecasting
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            DataFrame with forecasting features
        """
        df = self.records_to_dataframe(records)
        
        if df.empty:
            return pd.DataFrame()
        
        # Create time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Create lag features
        df = df.sort_values(['dc_id', 'sku_id', 'date'])
        
        for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
            df[f'demand_lag_{lag}'] = df.groupby(['dc_id', 'sku_id'])['forecasted_demand'].shift(lag)
            df[f'allocated_lag_{lag}'] = df.groupby(['dc_id', 'sku_id'])['allocated_quantity'].shift(lag)
        
        # Create rolling statistics
        for window in [7, 30]:  # 1 week, 1 month windows
            df[f'demand_rolling_mean_{window}'] = (
                df.groupby(['dc_id', 'sku_id'])['forecasted_demand']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            
            df[f'demand_rolling_std_{window}'] = (
                df.groupby(['dc_id', 'sku_id'])['forecasted_demand']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
        
        # Create ratio features
        df['demand_to_inventory_ratio'] = df['forecasted_demand'] / (df['current_inventory'] + 1e-8)
        df['historical_vs_forecast_ratio'] = df['historical_demand'] / (df['forecasted_demand'] + 1e-8)
        df['fulfillment_gap'] = df['forecasted_demand'] - df['allocated_quantity']
        
        return df
    
    def detect_outliers(
        self,
        records: List[Any],
        method: str = 'iqr',
        columns: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Detect outliers in the data
        
        Args:
            records: List of AllocationRecord objects
            method: Outlier detection method ('iqr' or 'zscore')
            columns: Columns to check for outliers
            
        Returns:
            Dictionary mapping column names to lists of outlier indices
        """
        df = self.records_to_dataframe(records)
        
        if df.empty:
            return {}
        
        numeric_columns = columns or [
            'current_inventory',
            'forecasted_demand', 
            'historical_demand',
            'revenue_per_unit',
            'cost_per_unit',
            'allocated_quantity'
        ]
        
        outliers = {}
        
        for column in numeric_columns:
            if column not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[
                    (df[column] < lower_bound) | 
                    (df[column] > upper_bound)
                ].index.tolist()
                
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outlier_indices = df[z_scores > 3].index.tolist()
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            if outlier_indices:
                outliers[column] = outlier_indices
        
        logger.info(f"Found outliers in {len(outliers)} columns using {method} method")
        
        return outliers
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics
        
        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            'scalers_fitted': len(self.scalers),
            'imputers_fitted': len(self.imputers),
            'scaler_types': list(set(key.split('_')[0] for key in self.scalers.keys())),
            'stats': self._preprocessing_stats
        }


# Global preprocessor instance
preprocessor = DataPreprocessor()


async def process_uploaded_file(file: UploadFile, use_cache: bool = True) -> pd.DataFrame:
    """
    Process uploaded allocation data file with caching support
    
    Args:
        file: Uploaded file (CSV, Excel, or JSON)
        use_cache: Whether to use caching for file processing
        
    Returns:
        pandas DataFrame with allocation data
        
    Raises:
        ValueError: If file format is unsupported or data is invalid
    """
    try:
        # Read file content
        content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        # Generate cache key based on file content and name
        if use_cache:
            content_hash = hashlib.sha256(content).hexdigest()[:16]
            cache_key = f"file_data:{file.filename}:{content_hash}"
            
            # Try to get from cache first
            try:
                from services.cache_service import get_cache_service
                cache_service = await get_cache_service()
                
                if hasattr(cache_service, 'get'):
                    cached_df_dict = await cache_service.get(cache_key)
                    if cached_df_dict:
                        logger.info(f"Cache hit for file processing: {file.filename}")
                        # Convert back to DataFrame
                        df = pd.DataFrame(cached_df_dict['data'])
                        if 'columns' in cached_df_dict:
                            df.columns = cached_df_dict['columns']
                        return df
            except Exception as e:
                logger.warning(f"Cache retrieval failed for file processing: {e}")
        
        # Parse based on file type
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        elif file_extension == 'json':
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Cache the processed DataFrame
        if use_cache:
            try:
                from services.cache_service import get_cache_service
                cache_service = await get_cache_service()
                
                if hasattr(cache_service, 'set'):
                    # Convert DataFrame to cacheable format
                    df_dict = {
                        'data': df.to_dict('records'),
                        'columns': df.columns.tolist()
                    }
                    # Cache for 30 minutes
                    await cache_service.set(cache_key, df_dict, ttl=timedelta(minutes=30))
                    logger.info(f"Cached file processing result for: {file.filename}")
            except Exception as e:
                logger.warning(f"Cache storage failed for file processing: {e}")
        
        logger.info(f"Successfully processed {file_extension} file with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        raise ValueError(f"Failed to process file: {str(e)}")


def validate_allocation_data(df: pd.DataFrame) -> List[str]:
    """
    Validate allocation data DataFrame
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df.empty:
        errors.append("Data is empty")
        return errors
    
    # Required columns
    required_columns = [
        'dc_id', 'sku_id', 'customer_id', 'current_inventory',
        'forecasted_demand', 'dc_priority', 'customer_tier', 'sla_level'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Data type validations
    if 'current_inventory' in df.columns:
        try:
            pd.to_numeric(df['current_inventory'], errors='raise')
            if (df['current_inventory'] < 0).any():
                errors.append("Current inventory cannot be negative")
        except ValueError:
            errors.append("Current inventory must be numeric")
    
    if 'forecasted_demand' in df.columns:
        try:
            pd.to_numeric(df['forecasted_demand'], errors='raise')
            if (df['forecasted_demand'] < 0).any():
                errors.append("Forecasted demand cannot be negative")
        except ValueError:
            errors.append("Forecasted demand must be numeric")
    
    if 'dc_priority' in df.columns:
        try:
            priorities = pd.to_numeric(df['dc_priority'], errors='raise')
            if not priorities.between(1, 5).all():
                errors.append("DC priority must be between 1 and 5")
        except ValueError:
            errors.append("DC priority must be numeric")
    
    # Check for duplicates
    if len(required_columns) <= len(df.columns):
        duplicate_keys = ['dc_id', 'sku_id', 'customer_id']
        if all(col in df.columns for col in duplicate_keys):
            duplicates = df.duplicated(subset=duplicate_keys)
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate records based on DC, SKU, and Customer")
    
    # Value range checks
    if 'customer_tier' in df.columns:
        valid_tiers = ['A', 'B', 'C', 'a', 'b', 'c']
        invalid_tiers = ~df['customer_tier'].isin(valid_tiers)
        if invalid_tiers.any():
            errors.append("Customer tier must be A, B, or C")
    
    if 'sla_level' in df.columns:
        valid_slas = ['Premium', 'Standard', 'Basic', 'premium', 'standard', 'basic']
        invalid_slas = ~df['sla_level'].isin(valid_slas)
        if invalid_slas.any():
            errors.append("SLA level must be Premium, Standard, or Basic")
    
    logger.info(f"Validation completed with {len(errors)} errors")
    return errors


def normalize_allocation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize allocation data for consistent processing
    
    Args:
        df: Input DataFrame
        
    Returns:
        Normalized DataFrame
    """
    df_normalized = df.copy()
    
    # Normalize string columns to lowercase
    if 'customer_tier' in df_normalized.columns:
        df_normalized['customer_tier'] = df_normalized['customer_tier'].str.upper()
    
    if 'sla_level' in df_normalized.columns:
        df_normalized['sla_level'] = df_normalized['sla_level'].str.title()
    
    # Ensure numeric columns are proper numeric types
    numeric_columns = ['current_inventory', 'forecasted_demand', 'dc_priority']
    for col in numeric_columns:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
    
    # Fill missing min_order_quantity with default value
    if 'min_order_quantity' not in df_normalized.columns:
        df_normalized['min_order_quantity'] = 1.0
    else:
        df_normalized['min_order_quantity'] = df_normalized['min_order_quantity'].fillna(1.0)
    
    # Add default sku_category if not present
    if 'sku_category' not in df_normalized.columns:
        df_normalized['sku_category'] = 'default'
    
    logger.info("Data normalization completed")
    return df_normalized