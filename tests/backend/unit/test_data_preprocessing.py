"""
Phase 1 Test: Data preprocessing utilities
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from backend.api.module4.models.data_models import AllocationRecord, CustomerTier, SLALevel
from backend.api.module4.utils.data_preprocessing import DataPreprocessor


class TestDataPreprocessing:
    """Test data preprocessing utilities"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create data preprocessor instance"""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_records(self):
        """Create sample allocation records"""
        records = []
        for i in range(5):
            record = AllocationRecord(
                dc_id=f'DC{i:03d}',
                dc_name=f'DC {i}',
                dc_location=f'Location {i}',
                dc_region='North America' if i < 3 else 'Europe',
                dc_priority=5 - i % 3,  # Varying priorities
                sku_id=f'SKU-{i:03d}',
                sku_name=f'Product {i}',
                sku_category='Networking' if i % 2 == 0 else 'Storage',
                customer_id=f'CUST-{i:03d}',
                customer_name=f'Customer {i}',
                customer_tier=CustomerTier.STRATEGIC if i % 3 == 0 else CustomerTier.PREMIUM,
                customer_region='North America',
                current_inventory=100 + i * 50,
                forecasted_demand=80 + i * 20,
                historical_demand=75 + i * 15,
                revenue_per_unit=100.0 + i * 10,
                cost_per_unit=75.0 + i * 7.5,
                margin=25.0 - i * 0.5,
                sla_level=SLALevel.GOLD if i % 2 == 0 else SLALevel.SILVER,
                risk_score=0.1 + i * 0.05,
                substitution_sku_id=None,
                date=date.today() - timedelta(days=i),
                allocated_quantity=70 + i * 15,
                fulfillment_rate=85.0 + i * 2,
                lead_time_days=3 + i,
                min_order_quantity=5 + i,
                safety_stock=20 + i * 5
            )
            records.append(record)
        return records
    
    def test_records_to_dataframe_empty(self, preprocessor):
        """Test converting empty records to DataFrame"""
        df = preprocessor.records_to_dataframe([])
        
        assert df.empty
        assert isinstance(df, pd.DataFrame)
    
    def test_records_to_dataframe_success(self, preprocessor, sample_records):
        """Test successful conversion of records to DataFrame"""
        df = preprocessor.records_to_dataframe(sample_records)
        
        assert len(df) == 5
        assert 'dc_id' in df.columns
        assert 'customer_tier' in df.columns
        assert 'sla_level' in df.columns
        
        # Check that enums are converted to string values
        assert df['customer_tier'].iloc[0] == 'Strategic'
        assert df['sla_level'].iloc[0] == 'Gold'
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_create_feature_matrix_empty(self, preprocessor):
        """Test creating feature matrix with empty records"""
        features, feature_names = preprocessor.create_feature_matrix([])
        
        assert features.size == 0
        assert feature_names == []
    
    def test_create_feature_matrix_success(self, preprocessor, sample_records):
        """Test successful creation of feature matrix"""
        features, feature_names = preprocessor.create_feature_matrix(sample_records)
        
        assert features.shape[0] == 5  # 5 records
        assert features.shape[1] == len(feature_names)
        assert len(feature_names) > 0
        
        # Check that all features are numeric
        assert np.issubdtype(features.dtype, np.number)
        
        # Check specific features are included
        expected_features = [
            'dc_priority', 'current_inventory', 'forecasted_demand',
            'revenue_per_unit', 'allocated_quantity'
        ]
        for feature in expected_features:
            assert feature in feature_names
    
    def test_create_feature_matrix_custom_features(self, preprocessor, sample_records):
        """Test creating feature matrix with custom feature selection"""
        custom_features = ['dc_priority', 'current_inventory', 'risk_score']
        features, feature_names = preprocessor.create_feature_matrix(
            sample_records, features=custom_features
        )
        
        assert features.shape[1] == 3
        assert feature_names == custom_features
    
    def test_create_categorical_features_empty(self, preprocessor):
        """Test creating categorical features with empty records"""
        cat_df = preprocessor.create_categorical_features([])
        
        assert cat_df.empty
    
    def test_create_categorical_features_success(self, preprocessor, sample_records):
        """Test successful creation of categorical features"""
        cat_df = preprocessor.create_categorical_features(sample_records)
        
        assert len(cat_df) == 5
        
        # Check for one-hot encoded columns
        assert any(col.startswith('dc_region_') for col in cat_df.columns)
        assert any(col.startswith('customer_tier_') for col in cat_df.columns)
        assert any(col.startswith('sla_level_') for col in cat_df.columns)
        
        # Check binary encoding
        for col in cat_df.columns:
            assert cat_df[col].dtype == 'bool' or set(cat_df[col].unique()).issubset({0, 1})
    
    def test_scale_features_empty(self, preprocessor):
        """Test scaling empty feature matrix"""
        empty_matrix = np.array([])
        scaled = preprocessor.scale_features(empty_matrix, [], method='standard')
        
        assert scaled.size == 0
    
    def test_scale_features_standard(self, preprocessor, sample_records):
        """Test standard scaling of features"""
        features, feature_names = preprocessor.create_feature_matrix(sample_records)
        
        scaled_features = preprocessor.scale_features(
            features, feature_names, method='standard', fit_scaler=True
        )
        
        assert scaled_features.shape == features.shape
        
        # Check that scaled features have approximately zero mean and unit variance
        means = np.mean(scaled_features, axis=0)
        stds = np.std(scaled_features, axis=0, ddof=1)
        
        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-10)
    
    def test_scale_features_minmax(self, preprocessor, sample_records):
        """Test min-max scaling of features"""
        features, feature_names = preprocessor.create_feature_matrix(sample_records)
        
        scaled_features = preprocessor.scale_features(
            features, feature_names, method='minmax', fit_scaler=True
        )
        
        assert scaled_features.shape == features.shape
        
        # Check that scaled features are in [0, 1] range
        assert np.all(scaled_features >= 0)
        assert np.all(scaled_features <= 1)
        
        # Check that min and max values are 0 and 1
        mins = np.min(scaled_features, axis=0)
        maxs = np.max(scaled_features, axis=0)
        
        assert np.allclose(mins, 0, atol=1e-10)
        assert np.allclose(maxs, 1, atol=1e-10)
    
    def test_scale_features_invalid_method(self, preprocessor, sample_records):
        """Test scaling with invalid method"""
        features, feature_names = preprocessor.create_feature_matrix(sample_records)
        
        with pytest.raises(ValueError, match="Unknown scaling method"):
            preprocessor.scale_features(features, feature_names, method='invalid')
    
    def test_prepare_time_series_data_empty(self, preprocessor):
        """Test preparing time series data with empty records"""
        X, y = preprocessor.prepare_time_series_data([])
        
        assert X.size == 0
        assert y.size == 0
    
    def test_prepare_time_series_data_insufficient(self, preprocessor, sample_records):
        """Test preparing time series data with insufficient data"""
        # Use only 3 records with window_size=7 (insufficient)
        small_sample = sample_records[:3]
        X, y = preprocessor.prepare_time_series_data(small_sample, window_size=7)
        
        assert X.size == 0
        assert y.size == 0
    
    def test_prepare_time_series_data_success(self, preprocessor):
        """Test successful preparation of time series data"""
        # Create more records with same DC and SKU for time series
        records = []
        for i in range(10):
            record = AllocationRecord(
                dc_id='DC001',  # Same DC
                dc_name='Test DC',
                dc_location='Test Location',
                dc_region='North America',
                dc_priority=5,
                sku_id='SKU-001',  # Same SKU
                sku_name='Test Product',
                sku_category='Networking',
                customer_id=f'CUST-{i:03d}',
                customer_name=f'Customer {i}',
                customer_tier=CustomerTier.STRATEGIC,
                customer_region='North America',
                current_inventory=100,
                forecasted_demand=80 + i,
                historical_demand=75,
                revenue_per_unit=100.0,
                cost_per_unit=75.0,
                margin=25.0,
                sla_level=SLALevel.GOLD,
                risk_score=0.1,
                substitution_sku_id=None,
                date=date.today() - timedelta(days=10-i),  # Sequential dates
                allocated_quantity=70 + i,  # Target variable
                fulfillment_rate=85.0,
                lead_time_days=3,
                min_order_quantity=5,
                safety_stock=20
            )
            records.append(record)
        
        X, y = preprocessor.prepare_time_series_data(
            records, target_column='allocated_quantity', window_size=3
        )
        
        assert X.shape[0] > 0  # Should have some samples
        assert X.shape[1] == 3  # Window size
        assert len(y) == len(X)  # Same number of samples
        
        # Check that X and y have correct relationship
        # First X sample should be first 3 allocated_quantity values
        # First y should be the 4th allocated_quantity value
        assert X[0, 0] == 70  # First allocated_quantity
        assert y[0] == 73     # Fourth allocated_quantity
    
    def test_create_demand_forecast_features_empty(self, preprocessor):
        """Test creating demand forecast features with empty records"""
        df = preprocessor.create_demand_forecast_features([])
        
        assert df.empty
    
    def test_create_demand_forecast_features_success(self, preprocessor, sample_records):
        """Test successful creation of demand forecast features"""
        df = preprocessor.create_demand_forecast_features(sample_records)
        
        assert len(df) == 5
        
        # Check time-based features
        time_features = ['year', 'month', 'quarter', 'day_of_year', 'week_of_year']
        for feature in time_features:
            assert feature in df.columns
        
        # Check lag features
        lag_features = ['demand_lag_1', 'demand_lag_7', 'demand_lag_30']
        for feature in lag_features:
            assert feature in df.columns
        
        # Check rolling features
        rolling_features = ['demand_rolling_mean_7', 'demand_rolling_std_7']
        for feature in rolling_features:
            assert feature in df.columns
        
        # Check ratio features
        ratio_features = ['demand_to_inventory_ratio', 'historical_vs_forecast_ratio', 'fulfillment_gap']
        for feature in ratio_features:
            assert feature in df.columns
    
    def test_detect_outliers_empty(self, preprocessor):
        """Test outlier detection with empty records"""
        outliers = preprocessor.detect_outliers([])
        
        assert outliers == {}
    
    def test_detect_outliers_iqr(self, preprocessor):
        """Test outlier detection using IQR method"""
        # Create records with one clear outlier
        records = []
        for i in range(10):
            inventory = 100 if i < 9 else 1000  # Last one is outlier
            record = AllocationRecord(
                dc_id=f'DC{i:03d}',
                dc_name=f'DC {i}',
                dc_location=f'Location {i}',
                dc_region='North America',
                dc_priority=5,
                sku_id=f'SKU-{i:03d}',
                sku_name=f'Product {i}',
                sku_category='Networking',
                customer_id=f'CUST-{i:03d}',
                customer_name=f'Customer {i}',
                customer_tier=CustomerTier.STRATEGIC,
                customer_region='North America',
                current_inventory=inventory,
                forecasted_demand=80,
                historical_demand=75,
                revenue_per_unit=100.0,
                cost_per_unit=75.0,
                margin=25.0,
                sla_level=SLALevel.GOLD,
                risk_score=0.1,
                substitution_sku_id=None,
                date=date.today() - timedelta(days=i),
                allocated_quantity=70,
                fulfillment_rate=85.0,
                lead_time_days=3,
                min_order_quantity=5,
                safety_stock=20
            )
            records.append(record)
        
        outliers = preprocessor.detect_outliers(records, method='iqr')
        
        assert 'current_inventory' in outliers
        assert 9 in outliers['current_inventory']  # Index of outlier record
    
    def test_detect_outliers_zscore(self, preprocessor):
        """Test outlier detection using Z-score method"""
        # Create records with extreme values
        records = []
        for i in range(20):
            demand = 100 if i < 19 else 1000  # Last one is extreme outlier
            record = AllocationRecord(
                dc_id=f'DC{i:03d}',
                dc_name=f'DC {i}',
                dc_location=f'Location {i}',
                dc_region='North America',
                dc_priority=5,
                sku_id=f'SKU-{i:03d}',
                sku_name=f'Product {i}',
                sku_category='Networking',
                customer_id=f'CUST-{i:03d}',
                customer_name=f'Customer {i}',
                customer_tier=CustomerTier.STRATEGIC,
                customer_region='North America',
                current_inventory=100,
                forecasted_demand=demand,
                historical_demand=75,
                revenue_per_unit=100.0,
                cost_per_unit=75.0,
                margin=25.0,
                sla_level=SLALevel.GOLD,
                risk_score=0.1,
                substitution_sku_id=None,
                date=date.today() - timedelta(days=i),
                allocated_quantity=70,
                fulfillment_rate=85.0,
                lead_time_days=3,
                min_order_quantity=5,
                safety_stock=20
            )
            records.append(record)
        
        outliers = preprocessor.detect_outliers(records, method='zscore')
        
        assert 'forecasted_demand' in outliers
        assert 19 in outliers['forecasted_demand']  # Index of outlier record
    
    def test_detect_outliers_invalid_method(self, preprocessor, sample_records):
        """Test outlier detection with invalid method"""
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            preprocessor.detect_outliers(sample_records, method='invalid')
    
    def test_get_preprocessing_stats(self, preprocessor, sample_records):
        """Test getting preprocessing statistics"""
        # Perform some operations to generate stats
        features, feature_names = preprocessor.create_feature_matrix(sample_records)
        preprocessor.scale_features(features, feature_names, method='standard')
        
        stats = preprocessor.get_preprocessing_stats()
        
        assert 'scalers_fitted' in stats
        assert 'imputers_fitted' in stats
        assert 'scaler_types' in stats
        assert stats['scalers_fitted'] >= 1
        assert 'standard' in stats['scaler_types']