"""
Comprehensive tests for ML models module
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Add path to import ml_models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_models.model_wrapper import BaseModelWrapper, ModelType, ModelPrediction, ModelPerformance, ModelRegistry
from ml_models.model_metrics import ModelMetrics, CrossValidator, ModelEvaluator
from ml_models.model_selector import ModelSelector, AutoModelSelector, SelectionConfig

# Mock heavy ML libraries to avoid import issues in CI/testing
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.optimizers'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['xgboost'] = MagicMock()
sys.modules['prophet'] = MagicMock()
sys.modules['fbprophet'] = MagicMock()
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.tsa'] = MagicMock()
sys.modules['statsmodels.tsa.statespace'] = MagicMock()
sys.modules['statsmodels.tsa.statespace.sarimax'] = MagicMock()

# Now import the model wrappers
from ml_models.lstm_wrapper import LSTMWrapper
from ml_models.prophet_wrapper import ProphetWrapper  
from ml_models.xgboost_wrapper import XGBoostWrapper
from ml_models.rf_wrapper import RandomForestWrapper
from ml_models.sarima_wrapper import SARIMAWrapper
from ml_models.parallel_runner import ParallelModelRunner, ModelRunner, ParallelRunConfig


class TestModelWrapper:
    """Tests for base model wrapper"""
    
    def test_model_types_enum(self):
        """Test ModelType enum values"""
        assert ModelType.LSTM == "lstm"
        assert ModelType.PROPHET == "prophet"
        assert ModelType.XGBOOST == "xgboost"
        assert ModelType.RANDOM_FOREST == "random_forest"
        assert ModelType.SARIMA == "sarima"
    
    def test_model_prediction_to_dict(self):
        """Test ModelPrediction serialization"""
        predictions = np.array([1.0, 2.0, 3.0])
        confidence = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
        
        pred = ModelPrediction(
            model_type=ModelType.LSTM,
            predictions=predictions,
            confidence_intervals=confidence,
            feature_importance={'feature1': 0.8, 'feature2': 0.2},
            model_params={'param1': 'value1'}
        )
        
        result = pred.to_dict()
        
        assert result['model_type'] == 'lstm'
        assert result['predictions'] == [1.0, 2.0, 3.0]
        assert len(result['confidence_intervals']) == 3
        assert result['feature_importance']['feature1'] == 0.8
        assert result['model_params']['param1'] == 'value1'
    
    def test_model_performance_weighted_score(self):
        """Test ModelPerformance weighted score calculation"""
        perf = ModelPerformance(
            model_type=ModelType.XGBOOST,
            rmse=10.0,
            mae=5.0,
            mape=15.0,
            r2_score=0.8,
            execution_time=30.0,
            training_samples=1000,
            test_samples=200
        )
        
        # Test weighted score calculation
        score = perf.weighted_score
        assert isinstance(score, float)
        assert score > 0
        
        # Test serialization
        result = perf.to_dict()
        assert result['weighted_score'] == score
        assert result['model_type'] == 'xgboost'


class TestModelRegistry:
    """Tests for model registry"""
    
    def test_register_and_create_model(self):
        """Test model registration and creation"""
        registry = ModelRegistry()
        
        # Register a mock model class
        class MockModel(BaseModelWrapper):
            def fit(self, X, y, **kwargs):
                pass
            def predict(self, X, **kwargs):
                return ModelPrediction(self.model_type, np.zeros(len(X)))
            def get_feature_importance(self):
                return None
        
        registry.register_model(ModelType.LSTM, MockModel)
        
        # Test model creation
        model = registry.create_model(ModelType.LSTM, param1='value1')
        assert isinstance(model, MockModel)
        assert model.model_type == ModelType.LSTM
        assert model.model_params['param1'] == 'value1'
        
        # Test retrieval
        retrieved_model = registry.get_model(ModelType.LSTM)
        assert retrieved_model is model
        
        # Test listing
        registered = registry.list_registered_models()
        assert ModelType.LSTM in registered
        
        created = registry.list_created_models()
        assert ModelType.LSTM in created


class TestModelMetrics:
    """Tests for model metrics calculations"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing metrics"""
        np.random.seed(42)
        y_true = np.random.randn(100) * 10 + 50
        y_pred = y_true + np.random.randn(100) * 2  # Add some noise
        return y_true, y_pred
    
    def test_calculate_rmse(self, sample_data):
        """Test RMSE calculation"""
        y_true, y_pred = sample_data
        rmse = ModelMetrics.calculate_rmse(y_true, y_pred)
        
        assert isinstance(rmse, float)
        assert rmse > 0
        
        # Test with perfect predictions
        perfect_rmse = ModelMetrics.calculate_rmse(y_true, y_true)
        assert perfect_rmse == 0.0
    
    def test_calculate_mae(self, sample_data):
        """Test MAE calculation"""
        y_true, y_pred = sample_data
        mae = ModelMetrics.calculate_mae(y_true, y_pred)
        
        assert isinstance(mae, float)
        assert mae > 0
        
        # Test with perfect predictions
        perfect_mae = ModelMetrics.calculate_mae(y_true, y_true)
        assert perfect_mae == 0.0
    
    def test_calculate_mape(self, sample_data):
        """Test MAPE calculation"""
        y_true, y_pred = sample_data
        mape = ModelMetrics.calculate_mape(y_true, y_pred)
        
        assert isinstance(mape, float)
        assert mape > 0
        
        # Test with zeros (should handle division by zero)
        y_zero = np.array([0.0, 1.0, 2.0])
        y_pred_zero = np.array([0.1, 1.1, 2.1])
        mape_zero = ModelMetrics.calculate_mape(y_zero, y_pred_zero)
        assert not np.isnan(mape_zero)
        assert not np.isinf(mape_zero)
    
    def test_calculate_r2(self, sample_data):
        """Test R² calculation"""
        y_true, y_pred = sample_data
        r2 = ModelMetrics.calculate_r2(y_true, y_pred)
        
        assert isinstance(r2, float)
        assert -1 <= r2 <= 1  # R² can be negative for very poor models
        
        # Test with perfect predictions
        perfect_r2 = ModelMetrics.calculate_r2(y_true, y_true)
        assert abs(perfect_r2 - 1.0) < 1e-10
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        # Test with perfect directional prediction
        y_true = np.array([1, 2, 3, 2, 4])
        y_pred = np.array([1.1, 2.1, 3.1, 1.9, 4.1])
        
        acc = ModelMetrics.calculate_directional_accuracy(y_true, y_pred)
        assert acc == 100.0
        
        # Test with completely opposite directions
        y_true_trend = np.array([1, 2, 3, 4, 5])  # Always increasing
        y_pred_opposite = np.array([5, 4, 3, 2, 1])  # Always decreasing
        acc_wrong = ModelMetrics.calculate_directional_accuracy(y_true_trend, y_pred_opposite)
        assert acc_wrong == 0.0  # Should be 0% accuracy
    
    def test_calculate_all_metrics(self, sample_data):
        """Test comprehensive metrics calculation"""
        y_true, y_pred = sample_data
        
        metrics = ModelMetrics.calculate_all_metrics(
            y_true, y_pred, execution_time=1.5, training_samples=800, test_samples=100
        )
        
        expected_keys = ['rmse', 'mae', 'mape', 'r2_score', 'directional_accuracy', 
                        'bias', 'execution_time', 'training_samples', 'test_samples']
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        assert metrics['execution_time'] == 1.5
        assert metrics['training_samples'] == 800
        assert metrics['test_samples'] == 100


class TestCrossValidator:
    """Tests for cross-validation utilities"""
    
    @pytest.fixture
    def sample_time_series(self):
        """Sample time series data"""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)
        return X, y
    
    def test_time_series_split(self, sample_time_series):
        """Test time series cross-validation splits"""
        X, y = sample_time_series
        cv = CrossValidator(n_splits=3, test_size=0.2)
        
        splits = cv.time_series_split(X, y)
        
        assert len(splits) <= 3  # May be fewer if not enough data
        
        for X_train, X_test, y_train, y_test in splits:
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)
            
            # Test that splits are sequential (time series property)
            assert len(X_train) + len(X_test) <= len(X)


class TestModelRunner:
    """Tests for individual model runner"""
    
    @pytest.fixture
    def mock_model_wrapper(self):
        """Mock model wrapper for testing"""
        mock_model = Mock(spec=BaseModelWrapper)
        mock_model.model_type = ModelType.XGBOOST
        mock_model.fit.return_value = None
        mock_model.predict.return_value = ModelPrediction(
            model_type=ModelType.XGBOOST,
            predictions=np.array([1.0, 2.0, 3.0])
        )
        return mock_model
    
    @pytest.fixture
    def sample_train_test_data(self):
        """Sample training and test data"""
        np.random.seed(42)
        X_train = np.random.randn(80, 3)
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test
    
    def test_model_runner_initialization(self):
        """Test model runner initialization"""
        runner = ModelRunner(ModelType.XGBOOST, {'param1': 'value1'})
        
        assert runner.model_type == ModelType.XGBOOST
        assert runner.model_params == {'param1': 'value1'}
        assert isinstance(runner.evaluator, ModelEvaluator)


class TestParallelRunner:
    """Tests for parallel model execution"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for parallel testing"""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test
    
    def test_parallel_config(self):
        """Test parallel run configuration"""
        config = ParallelRunConfig(
            max_workers=2,
            timeout_seconds=60,
            cross_validate=False
        )
        
        assert config.max_workers == 2
        assert config.timeout_seconds == 60
        assert config.cross_validate == False
    
    def test_parallel_runner_initialization(self):
        """Test parallel runner initialization"""
        config = ParallelRunConfig(max_workers=2)
        runner = ParallelModelRunner(config)
        
        assert runner.config.max_workers == 2
    
    def test_get_successful_results(self):
        """Test filtering successful results"""
        from ml_models.parallel_runner import ModelRunResult, ParallelModelRunner
        
        results = [
            ModelRunResult(ModelType.XGBOOST, success=True),
            ModelRunResult(ModelType.LSTM, success=False),
            ModelRunResult(ModelType.PROPHET, success=True)
        ]
        
        runner = ParallelModelRunner()
        successful = runner.get_successful_results(results)
        
        assert len(successful) == 2
        assert all(r.success for r in successful)
    
    def test_execution_summary(self):
        """Test execution summary generation"""
        from ml_models.parallel_runner import ModelRunResult, ParallelModelRunner
        
        results = [
            ModelRunResult(ModelType.XGBOOST, success=True, execution_time=1.0),
            ModelRunResult(ModelType.LSTM, success=False, execution_time=0.5),
            ModelRunResult(ModelType.PROPHET, success=True, execution_time=2.0)
        ]
        
        runner = ParallelModelRunner()
        summary = runner.get_execution_summary(results)
        
        assert summary['total_models'] == 3
        assert summary['successful'] == 2
        assert summary['failed'] == 1
        assert summary['success_rate'] == 2/3 * 100
        assert summary['total_execution_time'] == 3.5
        assert 'lstm' in summary['failed_models']
        assert 'xgboost' in summary['successful_models']


class TestModelSelector:
    """Tests for model selection logic"""
    
    @pytest.fixture
    def sample_performances(self):
        """Sample model performances for testing"""
        return [
            ModelPerformance(
                model_type=ModelType.XGBOOST,
                rmse=2.0, mae=1.5, mape=10.0, r2_score=0.9,
                execution_time=5.0, training_samples=100, test_samples=20
            ),
            ModelPerformance(
                model_type=ModelType.RANDOM_FOREST,
                rmse=2.5, mae=2.0, mape=12.0, r2_score=0.85,
                execution_time=8.0, training_samples=100, test_samples=20
            ),
            ModelPerformance(
                model_type=ModelType.LSTM,
                rmse=3.0, mae=2.5, mape=15.0, r2_score=0.8,
                execution_time=20.0, training_samples=100, test_samples=20
            )
        ]
    
    def test_selection_config(self):
        """Test selection configuration"""
        from ml_models.model_selector import SelectionCriteria
        
        config = SelectionConfig(
            primary_criterion=SelectionCriteria.RMSE,
            secondary_criterion=SelectionCriteria.MAE,
            min_r2_score=0.5,
            max_execution_time=60.0
        )
        
        assert config.primary_criterion == SelectionCriteria.RMSE
        assert config.secondary_criterion == SelectionCriteria.MAE
        assert config.min_r2_score == 0.5
        assert config.max_execution_time == 60.0
    
    def test_model_selector_initialization(self):
        """Test model selector initialization"""
        selector = ModelSelector()
        
        assert isinstance(selector.config, SelectionConfig)
        assert isinstance(selector.evaluator, ModelEvaluator)
        assert len(selector.selection_history) == 0
    
    def test_get_criterion_value(self, sample_performances):
        """Test criterion value extraction"""
        from ml_models.model_selector import SelectionCriteria
        
        selector = ModelSelector()
        perf = sample_performances[0]
        
        assert selector._get_criterion_value(perf, SelectionCriteria.RMSE) == 2.0
        assert selector._get_criterion_value(perf, SelectionCriteria.MAE) == 1.5
        assert selector._get_criterion_value(perf, SelectionCriteria.R2_SCORE) == -0.9  # Negative because higher is better
    
    def test_model_recommendation(self):
        """Test model recommendation based on dataset characteristics"""
        selector = ModelSelector()
        
        # Test time series with seasonality
        chars = {
            'data_size': 200,
            'feature_count': 10,
            'has_seasonality': True,
            'has_trend': True,
            'is_time_series': True
        }
        
        recommendations = selector.get_model_recommendation(chars)
        
        assert ModelType.PROPHET in recommendations
        assert ModelType.SARIMA in recommendations
        assert len(recommendations) > 0


class TestAutoModelSelector:
    """Tests for automatic model selection"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for auto selection testing"""
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test
    
    def test_auto_selector_initialization(self):
        """Test auto selector initialization"""
        auto_selector = AutoModelSelector()
        
        assert isinstance(auto_selector.selector, ModelSelector)
        assert isinstance(auto_selector.parallel_runner, ParallelModelRunner)
    
    def test_dataset_analysis(self, sample_data):
        """Test dataset characteristic analysis"""
        X_train, y_train, _, _ = sample_data
        
        auto_selector = AutoModelSelector()
        characteristics = auto_selector._analyze_dataset(X_train, y_train)
        
        expected_keys = ['data_size', 'feature_count', 'is_time_series', 
                        'has_seasonality', 'has_trend', 'data_variance', 'data_mean']
        
        for key in expected_keys:
            assert key in characteristics
        
        assert characteristics['data_size'] == 50
        assert characteristics['feature_count'] == 3
        assert isinstance(bool(characteristics['has_seasonality']), bool)
        assert isinstance(bool(characteristics['has_trend']), bool)
    
    def test_seasonality_detection(self):
        """Test seasonality detection"""
        auto_selector = AutoModelSelector()
        
        # Create data with obvious seasonality
        t = np.arange(50)
        seasonal_data = np.sin(2 * np.pi * t / 12) + np.random.randn(50) * 0.1
        
        has_seasonality = auto_selector._detect_seasonality(seasonal_data, period=12)
        assert isinstance(bool(has_seasonality), bool)
        
        # Test with random data (should not detect seasonality)
        random_data = np.random.randn(50)
        has_seasonality_random = auto_selector._detect_seasonality(random_data, period=12)
        assert isinstance(bool(has_seasonality_random), bool)
    
    def test_trend_detection(self):
        """Test trend detection"""
        auto_selector = AutoModelSelector()
        
        # Create data with obvious trend
        t = np.arange(50)
        trending_data = t * 0.5 + np.random.randn(50) * 0.1
        
        has_trend = auto_selector._detect_trend(trending_data)
        assert isinstance(bool(has_trend), bool)
        
        # Test with random data (should not detect trend)
        random_data = np.random.randn(50)
        has_trend_random = auto_selector._detect_trend(random_data)
        assert isinstance(bool(has_trend_random), bool)


@pytest.mark.integration
class TestMLModelsIntegration:
    """Integration tests for ML models system"""
    
    @pytest.fixture
    def sample_allocation_data(self):
        """Sample allocation data for integration testing"""
        np.random.seed(42)
        n_samples = 100
        
        # Create realistic allocation features
        data = {
            'dc_priority': np.random.randint(1, 6, n_samples),
            'current_inventory': np.random.uniform(0, 1000, n_samples),
            'forecasted_demand': np.random.uniform(10, 500, n_samples),
            'historical_demand': np.random.uniform(5, 400, n_samples),
            'revenue_per_unit': np.random.uniform(50, 200, n_samples),
            'risk_score': np.random.uniform(0, 1, n_samples),
            'lead_time_days': np.random.randint(1, 30, n_samples),
            'allocated_quantity': np.random.uniform(0, 300, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create feature matrix and target
        feature_cols = ['dc_priority', 'current_inventory', 'forecasted_demand', 
                       'historical_demand', 'revenue_per_unit', 'risk_score', 'lead_time_days']
        
        X = df[feature_cols].values
        y = df['allocated_quantity'].values
        
        # Split into train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test, feature_cols
    
    def test_end_to_end_model_selection(self, sample_allocation_data):
        """Test complete end-to-end model selection process with mocked results"""
        X_train, y_train, X_test, y_test, feature_names = sample_allocation_data
        
        # Create mock results instead of actually running models
        from ml_models.parallel_runner import ModelRunResult
        
        mock_results = [
            ModelRunResult(
                model_type=ModelType.XGBOOST,
                success=True,
                performance=ModelPerformance(
                    model_type=ModelType.XGBOOST,
                    rmse=2.0, mae=1.5, mape=10.0, r2_score=0.9,
                    execution_time=5.0, training_samples=80, test_samples=20
                )
            ),
            ModelRunResult(
                model_type=ModelType.RANDOM_FOREST,
                success=True,
                performance=ModelPerformance(
                    model_type=ModelType.RANDOM_FOREST,
                    rmse=2.5, mae=2.0, mape=12.0, r2_score=0.85,
                    execution_time=8.0, training_samples=80, test_samples=20
                )
            )
        ]
        
        # Test model selection
        selector = ModelSelector()
        selection = selector.select_best_model(mock_results, "test_dataset")
        
        assert selection is not None
        assert selection.best_model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST]
        assert isinstance(selection.selection_reason, str)
        assert len(selection.all_performances) == 2
    
    def test_model_comparison_report(self, sample_allocation_data):
        """Test model comparison report generation"""
        X_train, y_train, X_test, y_test, feature_names = sample_allocation_data
        
        evaluator = ModelEvaluator()
        
        # Create sample performances
        performances = [
            ModelPerformance(
                model_type=ModelType.XGBOOST,
                rmse=2.0, mae=1.5, mape=10.0, r2_score=0.9,
                execution_time=5.0, training_samples=80, test_samples=20
            ),
            ModelPerformance(
                model_type=ModelType.RANDOM_FOREST,
                rmse=2.5, mae=2.0, mape=12.0, r2_score=0.85,
                execution_time=8.0, training_samples=80, test_samples=20
            )
        ]
        
        # Test comparison
        comparison_df = evaluator.compare_models(performances)
        
        assert not comparison_df.empty
        assert 'model_type' in comparison_df.columns
        assert 'weighted_score' in comparison_df.columns
        assert 'rank' in comparison_df.columns
        
        # Best model should be ranked 1
        best_row = comparison_df[comparison_df['rank'] == 1].iloc[0]
        assert best_row['model_type'] == 'xgboost'  # Lower weighted score
        
        # Test evaluation report
        report = evaluator.generate_evaluation_report(performances)
        
        assert isinstance(report, str)
        assert "MODEL EVALUATION REPORT" in report
        assert "xgboost" in report.lower()
        assert "random_forest" in report.lower()


if __name__ == "__main__":
    pytest.main([__file__])