"""
Model selection logic for choosing the best performing ML model
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from .model_wrapper import BaseModelWrapper, ModelType, ModelPerformance
from .parallel_runner import ParallelModelRunner, ModelRunResult, ParallelRunConfig
from .model_metrics import ModelEvaluator, ModelBenchmark

logger = logging.getLogger(__name__)


class SelectionCriteria(str, Enum):
    """Model selection criteria"""
    WEIGHTED_SCORE = "weighted_score"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R2_SCORE = "r2_score"
    EXECUTION_TIME = "execution_time"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class SelectionConfig:
    """Configuration for model selection"""
    primary_criterion: SelectionCriteria = SelectionCriteria.WEIGHTED_SCORE
    secondary_criterion: SelectionCriteria = SelectionCriteria.RMSE
    min_r2_score: float = 0.0  # Minimum acceptable R² score
    max_execution_time: float = 300.0  # Maximum acceptable execution time (seconds)
    require_cross_validation: bool = True
    cross_validation_weight: float = 0.3  # Weight for CV scores in selection
    performance_threshold: float = 0.1  # Minimum improvement required to switch models


@dataclass
class ModelSelection:
    """Container for model selection results"""
    best_model_type: ModelType
    best_performance: ModelPerformance
    selection_reason: str
    all_performances: List[ModelPerformance]
    selection_metadata: Dict[str, Any]


class ModelSelector:
    """Intelligent model selector for choosing the best ML model"""
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize model selector
        
        Args:
            config: Selection configuration
        """
        self.config = config or SelectionConfig()
        self.evaluator = ModelEvaluator()
        self.benchmark = ModelBenchmark()
        self.selection_history: List[ModelSelection] = []
    
    def select_best_model(self, results: List[ModelRunResult],
                         dataset_name: Optional[str] = None) -> Optional[ModelSelection]:
        """
        Select the best model from run results
        
        Args:
            results: List of model run results
            dataset_name: Name of dataset for benchmarking
            
        Returns:
            ModelSelection object or None if no valid models
        """
        logger.info(f"Selecting best model from {len(results)} results")
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.performance]
        
        if not successful_results:
            logger.warning("No successful model results to select from")
            return None
        
        # Apply filters
        filtered_results = self._apply_filters(successful_results)
        
        if not filtered_results:
            logger.warning("No models passed filtering criteria")
            # Fallback to all successful results with relaxed criteria
            filtered_results = successful_results
        
        # Calculate selection scores
        scored_results = self._calculate_selection_scores(filtered_results)
        
        # Select best model
        best_result = self._select_by_criteria(scored_results)
        
        if not best_result:
            logger.error("Failed to select best model")
            return None
        
        # Create selection object
        selection = ModelSelection(
            best_model_type=best_result.model_type,
            best_performance=best_result.performance,
            selection_reason=self._generate_selection_reason(best_result, scored_results),
            all_performances=[r.performance for r in successful_results],
            selection_metadata={
                'selection_time': datetime.now().isoformat(),
                'total_models_evaluated': len(results),
                'successful_models': len(successful_results),
                'filtered_models': len(filtered_results),
                'selection_criteria': {
                    'primary': self.config.primary_criterion.value,
                    'secondary': self.config.secondary_criterion.value
                },
                'dataset_name': dataset_name
            }
        )
        
        # Update benchmarks if dataset name provided
        if dataset_name:
            self.benchmark.add_benchmark(dataset_name, best_result.performance)
        
        # Store selection history
        self.selection_history.append(selection)
        
        logger.info(f"Selected {best_result.model_type.value} as best model")
        return selection
    
    def _apply_filters(self, results: List[ModelRunResult]) -> List[ModelRunResult]:
        """Apply filtering criteria to results"""
        filtered = []
        
        for result in results:
            if not result.performance:
                continue
            
            perf = result.performance
            
            # Check R² score
            if perf.r2_score < self.config.min_r2_score:
                logger.debug(f"{result.model_type.value} filtered out: R² {perf.r2_score:.4f} < {self.config.min_r2_score}")
                continue
            
            # Check execution time
            if perf.execution_time > self.config.max_execution_time:
                logger.debug(f"{result.model_type.value} filtered out: time {perf.execution_time:.2f}s > {self.config.max_execution_time}s")
                continue
            
            # Check cross-validation requirement
            if self.config.require_cross_validation and not perf.cross_val_scores:
                logger.debug(f"{result.model_type.value} filtered out: no cross-validation scores")
                continue
            
            filtered.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered)}")
        return filtered
    
    def _calculate_selection_scores(self, results: List[ModelRunResult]) -> List[Tuple[ModelRunResult, float]]:
        """Calculate selection scores for each result"""
        scored_results = []
        
        for result in results:
            score = self._calculate_model_score(result.performance)
            scored_results.append((result, score))
        
        # Sort by score (lower is better for most metrics)
        scored_results.sort(key=lambda x: x[1])
        
        return scored_results
    
    def _calculate_model_score(self, performance: ModelPerformance) -> float:
        """
        Calculate overall model score based on multiple criteria
        
        Args:
            performance: Model performance object
            
        Returns:
            Overall score (lower is better)
        """
        # Base score from primary criterion
        primary_score = self._get_criterion_value(performance, self.config.primary_criterion)
        
        # Secondary criterion adjustment (10% weight)
        secondary_score = self._get_criterion_value(performance, self.config.secondary_criterion)
        combined_score = 0.9 * primary_score + 0.1 * secondary_score
        
        # Cross-validation adjustment
        if self.config.cross_validation_weight > 0 and performance.cross_val_scores:
            cv_mean = np.mean(performance.cross_val_scores)
            cv_std = np.std(performance.cross_val_scores)
            
            # Penalize high variance in CV scores
            cv_penalty = cv_std / cv_mean if cv_mean > 0 else 0
            combined_score += self.config.cross_validation_weight * cv_penalty
        
        return combined_score
    
    def _get_criterion_value(self, performance: ModelPerformance, criterion: SelectionCriteria) -> float:
        """Get value for specific selection criterion"""
        if criterion == SelectionCriteria.WEIGHTED_SCORE:
            return performance.weighted_score
        elif criterion == SelectionCriteria.RMSE:
            return performance.rmse
        elif criterion == SelectionCriteria.MAE:
            return performance.mae
        elif criterion == SelectionCriteria.MAPE:
            return performance.mape
        elif criterion == SelectionCriteria.R2_SCORE:
            return -performance.r2_score  # Negative because higher R² is better
        elif criterion == SelectionCriteria.EXECUTION_TIME:
            return performance.execution_time
        elif criterion == SelectionCriteria.CROSS_VALIDATION:
            if performance.cross_val_scores:
                return np.mean(performance.cross_val_scores)
            else:
                return float('inf')  # Penalize missing CV scores
        else:
            return performance.weighted_score
    
    def _select_by_criteria(self, scored_results: List[Tuple[ModelRunResult, float]]) -> Optional[ModelRunResult]:
        """Select best model from scored results"""
        if not scored_results:
            return None
        
        # Best result is first (lowest score)
        return scored_results[0][0]
    
    def _generate_selection_reason(self, best_result: ModelRunResult,
                                  scored_results: List[Tuple[ModelRunResult, float]]) -> str:
        """Generate explanation for model selection"""
        perf = best_result.performance
        
        reasons = [
            f"Selected {best_result.model_type.value} based on {self.config.primary_criterion.value}"
        ]
        
        # Add performance details
        reasons.append(f"RMSE: {perf.rmse:.4f}")
        reasons.append(f"MAE: {perf.mae:.4f}")
        reasons.append(f"R²: {perf.r2_score:.4f}")
        reasons.append(f"Weighted Score: {perf.weighted_score:.4f}")
        
        # Add comparison with second best
        if len(scored_results) > 1:
            second_best = scored_results[1][0]
            improvement = ((scored_results[1][1] - scored_results[0][1]) / scored_results[1][1]) * 100
            reasons.append(f"Outperformed {second_best.model_type.value} by {improvement:.1f}%")
        
        # Add cross-validation info
        if perf.cross_val_scores:
            cv_mean = np.mean(perf.cross_val_scores)
            cv_std = np.std(perf.cross_val_scores)
            reasons.append(f"CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return ". ".join(reasons)
    
    def should_retrain(self, current_performance: ModelPerformance,
                      new_data_size: int, last_training_size: int) -> bool:
        """
        Determine if model should be retrained based on new data
        
        Args:
            current_performance: Current model performance
            new_data_size: Size of new dataset
            last_training_size: Size of dataset used for last training
            
        Returns:
            True if retraining is recommended
        """
        # Retrain if data size increased significantly
        data_growth = (new_data_size - last_training_size) / last_training_size
        if data_growth > 0.2:  # 20% increase
            logger.info(f"Retraining recommended: data grew by {data_growth*100:.1f}%")
            return True
        
        # Retrain if current performance is poor
        if current_performance.weighted_score > 0.5:  # Threshold for poor performance
            logger.info("Retraining recommended: poor current performance")
            return True
        
        # Retrain if cross-validation shows high variance
        if current_performance.cross_val_scores:
            cv_std = np.std(current_performance.cross_val_scores)
            cv_mean = np.mean(current_performance.cross_val_scores)
            if cv_std / cv_mean > 0.3:  # High coefficient of variation
                logger.info("Retraining recommended: high model variance")
                return True
        
        return False
    
    def get_model_recommendation(self, dataset_characteristics: Dict[str, Any]) -> List[ModelType]:
        """
        Recommend models based on dataset characteristics
        
        Args:
            dataset_characteristics: Dictionary with dataset info
            
        Returns:
            List of recommended model types
        """
        recommendations = []
        
        data_size = dataset_characteristics.get('data_size', 0)
        has_seasonality = dataset_characteristics.get('has_seasonality', False)
        has_trend = dataset_characteristics.get('has_trend', False)
        feature_count = dataset_characteristics.get('feature_count', 0)
        is_time_series = dataset_characteristics.get('is_time_series', True)
        
        # Time series specific models
        if is_time_series:
            if has_seasonality:
                recommendations.extend([ModelType.PROPHET, ModelType.SARIMA])
            
            if data_size > 100:
                recommendations.append(ModelType.LSTM)
        
        # General ML models
        if feature_count > 5:
            recommendations.extend([ModelType.XGBOOST, ModelType.RANDOM_FOREST])
        
        # Default to all models if no specific recommendations
        if not recommendations:
            recommendations = [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.LSTM]
        
        logger.info(f"Recommended models: {[m.value for m in recommendations]}")
        return recommendations
    
    def compare_with_benchmark(self, performance: ModelPerformance,
                              dataset_name: str) -> Dict[str, Any]:
        """
        Compare performance with benchmark
        
        Args:
            performance: Model performance to compare
            dataset_name: Name of dataset
            
        Returns:
            Comparison results
        """
        return self.benchmark.compare_to_benchmark(dataset_name, performance)
    
    def get_selection_history(self, limit: int = 10) -> List[ModelSelection]:
        """Get recent selection history"""
        return self.selection_history[-limit:]
    
    def export_selection_report(self, selection: ModelSelection) -> str:
        """
        Export detailed selection report
        
        Args:
            selection: Model selection result
            
        Returns:
            Formatted report string
        """
        report = ["=" * 60]
        report.append("MODEL SELECTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Selection summary
        report.append(f"Selected Model: {selection.best_model_type.value.upper()}")
        report.append(f"Selection Time: {selection.selection_metadata.get('selection_time', 'Unknown')}")
        report.append(f"Dataset: {selection.selection_metadata.get('dataset_name', 'Unknown')}")
        report.append("")
        
        # Performance details
        perf = selection.best_performance
        report.append("PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"RMSE: {perf.rmse:.4f}")
        report.append(f"MAE: {perf.mae:.4f}")
        report.append(f"MAPE: {perf.mape:.2f}%")
        report.append(f"R² Score: {perf.r2_score:.4f}")
        report.append(f"Weighted Score: {perf.weighted_score:.4f}")
        report.append(f"Execution Time: {perf.execution_time:.2f}s")
        
        if perf.cross_val_scores:
            cv_mean = np.mean(perf.cross_val_scores)
            cv_std = np.std(perf.cross_val_scores)
            report.append(f"Cross-Validation RMSE: {cv_mean:.4f} ± {cv_std:.4f}")
        
        report.append("")
        
        # Selection reasoning
        report.append("SELECTION REASONING")
        report.append("-" * 30)
        report.append(selection.selection_reason)
        report.append("")
        
        # Comparison with other models
        report.append("MODEL COMPARISON")
        report.append("-" * 30)
        
        for i, other_perf in enumerate(selection.all_performances):
            if other_perf.model_type != selection.best_model_type:
                report.append(f"{other_perf.model_type.value}: RMSE={other_perf.rmse:.4f}, Score={other_perf.weighted_score:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class AutoModelSelector:
    """Automated model selection with intelligent defaults"""
    
    def __init__(self):
        """Initialize automatic model selector"""
        self.selector = ModelSelector()
        self.parallel_runner = ParallelModelRunner()
    
    def auto_select_best_model(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              dataset_name: Optional[str] = None) -> Optional[ModelSelection]:
        """
        Automatically run all models and select the best one
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            dataset_name: Name of dataset for benchmarking
            
        Returns:
            ModelSelection object or None
        """
        logger.info("Starting automatic model selection")
        
        # Analyze dataset characteristics
        dataset_chars = self._analyze_dataset(X_train, y_train)
        
        # Get model recommendations
        recommended_models = self.selector.get_model_recommendation(dataset_chars)
        
        # Configure parallel runner
        config = ParallelRunConfig(
            max_workers=min(len(recommended_models), 4),
            cross_validate=True,
            timeout_seconds=600  # 10 minutes
        )
        
        runner = ParallelModelRunner(config)
        
        # Run all recommended models
        results = runner.run_all_models(
            X_train, y_train, X_test, y_test, recommended_models
        )
        
        # Select best model
        selection = self.selector.select_best_model(results, dataset_name)
        
        if selection:
            # Log selection report
            report = self.selector.export_selection_report(selection)
            logger.info(f"Model selection completed:\n{report}")
        
        return selection
    
    def _analyze_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze dataset characteristics for model recommendation"""
        characteristics = {
            'data_size': len(X),
            'feature_count': X.shape[1],
            'is_time_series': True,  # Assume time series for demand forecasting
            'has_seasonality': self._detect_seasonality(y),
            'has_trend': self._detect_trend(y),
            'data_variance': float(np.var(y)),
            'data_mean': float(np.mean(y))
        }
        
        logger.info(f"Dataset characteristics: {characteristics}")
        return characteristics
    
    def _detect_seasonality(self, y: np.ndarray, period: int = 12) -> bool:
        """Simple seasonality detection"""
        if len(y) < period * 2:
            return False
        
        try:
            # Calculate autocorrelation at seasonal lag
            n = len(y)
            y_centered = y - np.mean(y)
            
            autocorr = np.correlate(y_centered, y_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            if len(autocorr) > period:
                seasonal_autocorr = autocorr[period]
                return abs(seasonal_autocorr) > 0.3  # Threshold for seasonality
            
        except Exception:
            pass
        
        return False
    
    def _detect_trend(self, y: np.ndarray) -> bool:
        """Simple trend detection using linear regression slope"""
        if len(y) < 10:
            return False
        
        try:
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalized slope threshold
            slope_threshold = np.std(y) * 0.01  # 1% of std dev per time step
            return abs(slope) > slope_threshold
            
        except Exception:
            return False