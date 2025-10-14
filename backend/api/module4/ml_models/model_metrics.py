"""
Model performance metrics and evaluation utilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from .model_wrapper import ModelType, ModelPrediction, ModelPerformance, BaseModelWrapper

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Utility class for calculating model performance metrics"""
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value as percentage
        """
        # Avoid division by zero
        y_true_nonzero = np.where(y_true == 0, 1e-8, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        return float(mape)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared score
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² value
        """
        return float(r2_score(y_true, y_pred))
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (trend prediction accuracy)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy as percentage
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate direction changes
        true_direction = np.diff(y_true) >= 0
        pred_direction = np.diff(y_pred) >= 0
        
        # Calculate accuracy
        correct_directions = np.sum(true_direction == pred_direction)
        total_directions = len(true_direction)
        
        return float(correct_directions / total_directions * 100)
    
    @staticmethod
    def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate prediction bias (mean error)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Bias value
        """
        return float(np.mean(y_pred - y_true))
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            execution_time: float = 0.0,
                            training_samples: int = 0,
                            test_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            execution_time: Model execution time
            training_samples: Number of training samples
            test_samples: Number of test samples
            
        Returns:
            Dictionary of all calculated metrics
        """
        if test_samples is None:
            test_samples = len(y_true)
        
        metrics = {
            'rmse': ModelMetrics.calculate_rmse(y_true, y_pred),
            'mae': ModelMetrics.calculate_mae(y_true, y_pred),
            'mape': ModelMetrics.calculate_mape(y_true, y_pred),
            'r2_score': ModelMetrics.calculate_r2(y_true, y_pred),
            'directional_accuracy': ModelMetrics.calculate_directional_accuracy(y_true, y_pred),
            'bias': ModelMetrics.calculate_bias(y_true, y_pred),
            'execution_time': execution_time,
            'training_samples': training_samples,
            'test_samples': test_samples
        }
        
        return metrics


class CrossValidator:
    """Cross-validation utility for time series data"""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        """
        Initialize cross-validator
        
        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set as fraction
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create time series splits (expanding window)
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        n_samples = len(X)
        min_train_size = int(n_samples * 0.3)  # Minimum 30% for training
        test_size = int(n_samples * self.test_size)
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split point
            train_end = min_train_size + i * (n_samples - min_train_size - test_size) // (self.n_splits - 1) if self.n_splits > 1 else n_samples - test_size
            train_end = min(train_end, n_samples - test_size)
            
            if train_end < min_train_size:
                continue
            
            # Create splits
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[train_end:train_end + test_size]
            y_test = y[train_end:train_end + test_size]
            
            if len(X_test) > 0 and len(y_test) > 0:
                splits.append((X_train, X_test, y_train, y_test))
        
        return splits
    
    def cross_validate_model(self, model: BaseModelWrapper, X: np.ndarray, y: np.ndarray) -> List[Dict[str, float]]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model wrapper to validate
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of metrics for each fold
        """
        splits = self.time_series_split(X, y)
        cv_results = []
        
        for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
            try:
                logger.info(f"Cross-validation fold {i+1}/{len(splits)}")
                
                # Clone model parameters for clean fit
                model_params = model.model_params.copy()
                
                # Create fresh model instance
                model_class = type(model)
                fold_model = model_class(**model_params)
                
                # Time the fitting and prediction
                start_time = time.time()
                
                # Fit and predict
                fold_model.fit(X_train, y_train)
                predictions = fold_model.predict(X_test)
                
                execution_time = time.time() - start_time
                
                # Calculate metrics
                metrics = ModelMetrics.calculate_all_metrics(
                    y_test, 
                    predictions.predictions,
                    execution_time,
                    len(y_train),
                    len(y_test)
                )
                
                cv_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Cross-validation fold {i+1} failed: {e}")
                # Add placeholder metrics for failed fold
                cv_results.append({
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'mape': float('inf'),
                    'r2_score': -float('inf'),
                    'directional_accuracy': 0.0,
                    'bias': 0.0,
                    'execution_time': 0.0,
                    'training_samples': len(y_train) if 'y_train' in locals() else 0,
                    'test_samples': len(y_test) if 'y_test' in locals() else 0
                })
        
        return cv_results


class ModelEvaluator:
    """Comprehensive model evaluation utility"""
    
    def __init__(self):
        """Initialize model evaluator"""
        self.metrics = ModelMetrics()
        self.cross_validator = CrossValidator()
    
    def evaluate_model(self, model: BaseModelWrapper, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, 
                      cross_validate: bool = True) -> ModelPerformance:
        """
        Comprehensive model evaluation
        
        Args:
            model: Model wrapper to evaluate
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            cross_validate: Whether to perform cross-validation
            
        Returns:
            ModelPerformance object
        """
        logger.info(f"Evaluating {model.model_type.value} model")
        
        # Time the training and prediction
        start_time = time.time()
        
        # Fit and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        execution_time = time.time() - start_time
        
        # Calculate main metrics
        rmse = self.metrics.calculate_rmse(y_test, predictions.predictions)
        mae = self.metrics.calculate_mae(y_test, predictions.predictions)
        mape = self.metrics.calculate_mape(y_test, predictions.predictions)
        r2 = self.metrics.calculate_r2(y_test, predictions.predictions)
        
        # Cross-validation scores
        cv_scores = None
        if cross_validate and len(X_train) > 20:  # Only CV with sufficient data
            try:
                cv_results = self.cross_validator.cross_validate_model(
                    model, X_train, y_train
                )
                cv_scores = [result['rmse'] for result in cv_results if result['rmse'] != float('inf')]
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model.model_type.value}: {e}")
        
        return ModelPerformance(
            model_type=model.model_type,
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2_score=r2,
            execution_time=execution_time,
            training_samples=len(y_train),
            test_samples=len(y_test),
            cross_val_scores=cv_scores
        )
    
    def compare_models(self, model_performances: List[ModelPerformance]) -> pd.DataFrame:
        """
        Compare multiple model performances
        
        Args:
            model_performances: List of ModelPerformance objects
            
        Returns:
            DataFrame with comparison results
        """
        if not model_performances:
            return pd.DataFrame()
        
        # Create comparison data
        comparison_data = []
        for perf in model_performances:
            data = perf.to_dict()
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by weighted score (lower is better)
        if 'weighted_score' in df.columns:
            df = df.sort_values('weighted_score')
        
        # Add ranking
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_best_model(self, model_performances: List[ModelPerformance]) -> Optional[ModelPerformance]:
        """
        Get the best performing model
        
        Args:
            model_performances: List of ModelPerformance objects
            
        Returns:
            Best ModelPerformance or None
        """
        if not model_performances:
            return None
        
        # Filter out failed models
        valid_performances = [
            perf for perf in model_performances 
            if not (np.isinf(perf.rmse) or np.isnan(perf.rmse))
        ]
        
        if not valid_performances:
            return None
        
        # Find best model by weighted score
        best_model = min(valid_performances, key=lambda x: x.weighted_score)
        return best_model
    
    def generate_evaluation_report(self, model_performances: List[ModelPerformance]) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            model_performances: List of ModelPerformance objects
            
        Returns:
            Formatted evaluation report
        """
        if not model_performances:
            return "No model performances to report."
        
        # Get comparison dataframe
        comparison_df = self.compare_models(model_performances)
        
        # Start building report
        report = ["=" * 60]
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append(f"Models Evaluated: {len(model_performances)}")
        report.append(f"Best Model: {comparison_df.iloc[0]['model_type'] if not comparison_df.empty else 'None'}")
        report.append("")
        
        # Detailed comparison table
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 40)
        
        if not comparison_df.empty:
            # Select key columns for display
            display_columns = ['rank', 'model_type', 'rmse', 'mae', 'mape', 'r2_score', 'weighted_score', 'execution_time']
            display_df = comparison_df[display_columns].round(4)
            report.append(display_df.to_string(index=False))
        
        report.append("")
        
        # Individual model details
        report.append("DETAILED MODEL METRICS")
        report.append("-" * 40)
        
        for i, perf in enumerate(sorted(model_performances, key=lambda x: x.weighted_score)):
            report.append(f"\n{i+1}. {perf.model_type.value.upper()}")
            report.append(f"   RMSE: {perf.rmse:.4f}")
            report.append(f"   MAE: {perf.mae:.4f}")
            report.append(f"   MAPE: {perf.mape:.2f}%")
            report.append(f"   R²: {perf.r2_score:.4f}")
            report.append(f"   Weighted Score: {perf.weighted_score:.4f}")
            report.append(f"   Execution Time: {perf.execution_time:.2f}s")
            report.append(f"   Training Samples: {perf.training_samples}")
            report.append(f"   Test Samples: {perf.test_samples}")
            
            if perf.cross_val_scores:
                cv_mean = np.mean(perf.cross_val_scores)
                cv_std = np.std(perf.cross_val_scores)
                report.append(f"   CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class ModelBenchmark:
    """Benchmark utility for model performance tracking"""
    
    def __init__(self):
        """Initialize benchmark tracker"""
        self.benchmarks: Dict[str, List[ModelPerformance]] = {}
    
    def add_benchmark(self, dataset_name: str, performance: ModelPerformance) -> None:
        """
        Add benchmark result
        
        Args:
            dataset_name: Name of the dataset
            performance: Model performance result
        """
        if dataset_name not in self.benchmarks:
            self.benchmarks[dataset_name] = []
        
        self.benchmarks[dataset_name].append(performance)
    
    def get_benchmarks(self, dataset_name: str) -> List[ModelPerformance]:
        """
        Get benchmark results for dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of model performances
        """
        return self.benchmarks.get(dataset_name, [])
    
    def get_best_benchmark(self, dataset_name: str) -> Optional[ModelPerformance]:
        """
        Get best benchmark for dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Best model performance or None
        """
        benchmarks = self.get_benchmarks(dataset_name)
        if not benchmarks:
            return None
        
        evaluator = ModelEvaluator()
        return evaluator.get_best_model(benchmarks)
    
    def compare_to_benchmark(self, dataset_name: str, performance: ModelPerformance) -> Dict[str, Any]:
        """
        Compare performance to existing benchmarks
        
        Args:
            dataset_name: Name of the dataset
            performance: Model performance to compare
            
        Returns:
            Comparison results
        """
        best_benchmark = self.get_best_benchmark(dataset_name)
        
        if best_benchmark is None:
            return {
                'is_new_best': True,
                'improvement': None,
                'benchmark_model': None
            }
        
        improvement = (best_benchmark.weighted_score - performance.weighted_score) / best_benchmark.weighted_score * 100
        
        return {
            'is_new_best': performance.weighted_score < best_benchmark.weighted_score,
            'improvement': improvement,
            'benchmark_model': best_benchmark.model_type.value,
            'benchmark_score': best_benchmark.weighted_score,
            'new_score': performance.weighted_score
        }