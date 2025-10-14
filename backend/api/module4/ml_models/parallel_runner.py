"""
Parallel model runner for training and evaluating multiple ML models simultaneously
"""

import logging
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import time
import multiprocessing as mp

from .model_wrapper import BaseModelWrapper, ModelType, ModelPrediction, ModelPerformance
from .model_metrics import ModelEvaluator
from .lstm_wrapper import LSTMWrapper
from .prophet_wrapper import ProphetWrapper
from .xgboost_wrapper import XGBoostWrapper
from .rf_wrapper import RandomForestWrapper
from .sarima_wrapper import SARIMAWrapper

logger = logging.getLogger(__name__)


@dataclass
class ModelRunResult:
    """Container for model run results"""
    model_type: ModelType
    success: bool
    performance: Optional[ModelPerformance] = None
    predictions: Optional[ModelPrediction] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ParallelRunConfig:
    """Configuration for parallel model execution"""
    max_workers: Optional[int] = None
    use_multiprocessing: bool = False
    timeout_seconds: int = 300  # 5 minutes default timeout
    cross_validate: bool = True
    model_params: Optional[Dict[ModelType, Dict[str, Any]]] = None


class ModelRunner:
    """Individual model runner for parallel execution"""
    
    def __init__(self, model_type: ModelType, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize model runner
        
        Args:
            model_type: Type of model to run
            model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.evaluator = ModelEvaluator()
    
    def _create_model(self) -> BaseModelWrapper:
        """Create model instance based on type"""
        model_classes = {
            ModelType.LSTM: LSTMWrapper,
            ModelType.PROPHET: ProphetWrapper,
            ModelType.XGBOOST: XGBoostWrapper,
            ModelType.RANDOM_FOREST: RandomForestWrapper,
            ModelType.SARIMA: SARIMAWrapper
        }
        
        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model_class = model_classes[self.model_type]
        return model_class(**self.model_params)
    
    def run_model(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  cross_validate: bool = True) -> ModelRunResult:
        """
        Run a single model with training and evaluation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            cross_validate: Whether to perform cross-validation
            
        Returns:
            ModelRunResult object
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting {self.model_type.value} model execution")
            
            # Create model
            model = self._create_model()
            
            # Evaluate model
            performance = self.evaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test, cross_validate
            )
            
            # Get predictions for return
            predictions = model.predict(X_test)
            
            execution_time = time.time() - start_time
            
            logger.info(f"{self.model_type.value} completed successfully in {execution_time:.2f}s")
            
            return ModelRunResult(
                model_type=self.model_type,
                success=True,
                performance=performance,
                predictions=predictions,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{self.model_type.value} failed: {str(e)}"
            logger.error(error_msg)
            
            return ModelRunResult(
                model_type=self.model_type,
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )


def run_single_model(args: Tuple) -> ModelRunResult:
    """
    Wrapper function for running a single model (used for multiprocessing)
    
    Args:
        args: Tuple containing (model_type, model_params, X_train, y_train, X_test, y_test, cross_validate)
        
    Returns:
        ModelRunResult object
    """
    model_type, model_params, X_train, y_train, X_test, y_test, cross_validate = args
    
    runner = ModelRunner(model_type, model_params)
    return runner.run_model(X_train, y_train, X_test, y_test, cross_validate)


class ParallelModelRunner:
    """Parallel execution manager for multiple ML models"""
    
    def __init__(self, config: Optional[ParallelRunConfig] = None):
        """
        Initialize parallel model runner
        
        Args:
            config: Configuration for parallel execution
        """
        self.config = config or ParallelRunConfig()
        
        # Set default max workers
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 8)  # Cap at 8 workers
        
        logger.info(f"Initialized parallel runner with {self.config.max_workers} workers")
        logger.info(f"Using multiprocessing: {self.config.use_multiprocessing}")
    
    def run_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      model_types: Optional[List[ModelType]] = None) -> List[ModelRunResult]:
        """
        Run all specified models in parallel
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_types: List of model types to run (default: all)
            
        Returns:
            List of ModelRunResult objects
        """
        if model_types is None:
            model_types = [ModelType.LSTM, ModelType.PROPHET, ModelType.XGBOOST, 
                          ModelType.RANDOM_FOREST, ModelType.SARIMA]
        
        logger.info(f"Running {len(model_types)} models in parallel")
        
        if self.config.use_multiprocessing:
            return self._run_with_multiprocessing(
                X_train, y_train, X_test, y_test, model_types
            )
        else:
            return self._run_with_threading(
                X_train, y_train, X_test, y_test, model_types
            )
    
    def _run_with_threading(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           model_types: List[ModelType]) -> List[ModelRunResult]:
        """Run models using ThreadPoolExecutor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all models
            future_to_model = {}
            
            for model_type in model_types:
                model_params = (self.config.model_params or {}).get(model_type, {})
                runner = ModelRunner(model_type, model_params)
                
                future = executor.submit(
                    runner.run_model,
                    X_train, y_train, X_test, y_test,
                    self.config.cross_validate
                )
                future_to_model[future] = model_type
            
            # Collect results as they complete
            for future in as_completed(future_to_model, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    model_type = future_to_model[future]
                    logger.error(f"Model {model_type.value} execution failed: {e}")
                    
                    results.append(ModelRunResult(
                        model_type=model_type,
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _run_with_multiprocessing(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 model_types: List[ModelType]) -> List[ModelRunResult]:
        """Run models using ProcessPoolExecutor"""
        results = []
        
        # Prepare arguments for each model
        model_args = []
        for model_type in model_types:
            model_params = (self.config.model_params or {}).get(model_type, {})
            args = (model_type, model_params, X_train, y_train, X_test, y_test, self.config.cross_validate)
            model_args.append(args)
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all models
            future_to_model = {}
            
            for i, args in enumerate(model_args):
                future = executor.submit(run_single_model, args)
                future_to_model[future] = model_types[i]
            
            # Collect results as they complete
            for future in as_completed(future_to_model, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    model_type = future_to_model[future]
                    logger.error(f"Model {model_type.value} execution failed: {e}")
                    
                    results.append(ModelRunResult(
                        model_type=model_type,
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    async def run_all_models_async(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  model_types: Optional[List[ModelType]] = None) -> List[ModelRunResult]:
        """
        Run all models asynchronously
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_types: List of model types to run (default: all)
            
        Returns:
            List of ModelRunResult objects
        """
        if model_types is None:
            model_types = [ModelType.LSTM, ModelType.PROPHET, ModelType.XGBOOST, 
                          ModelType.RANDOM_FOREST, ModelType.SARIMA]
        
        logger.info(f"Running {len(model_types)} models asynchronously")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            None, 
            self.run_all_models, 
            X_train, y_train, X_test, y_test, model_types
        )
    
    def get_successful_results(self, results: List[ModelRunResult]) -> List[ModelRunResult]:
        """Filter to successful results only"""
        return [result for result in results if result.success]
    
    def get_failed_results(self, results: List[ModelRunResult]) -> List[ModelRunResult]:
        """Filter to failed results only"""
        return [result for result in results if not result.success]
    
    def get_execution_summary(self, results: List[ModelRunResult]) -> Dict[str, Any]:
        """
        Get summary of parallel execution
        
        Args:
            results: List of ModelRunResult objects
            
        Returns:
            Summary dictionary
        """
        successful = self.get_successful_results(results)
        failed = self.get_failed_results(results)
        
        total_time = sum(result.execution_time for result in results)
        avg_time = total_time / len(results) if results else 0
        
        return {
            'total_models': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'failed_models': [result.model_type.value for result in failed],
            'successful_models': [result.model_type.value for result in successful]
        }


class ModelBatchRunner:
    """Batch runner for multiple datasets or parameter combinations"""
    
    def __init__(self, parallel_config: Optional[ParallelRunConfig] = None):
        """
        Initialize batch runner
        
        Args:
            parallel_config: Configuration for parallel execution
        """
        self.runner = ParallelModelRunner(parallel_config)
    
    def run_parameter_sweep(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           parameter_grid: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, List[ModelRunResult]]:
        """
        Run parameter sweep for multiple models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            parameter_grid: Grid of parameters for each model type
            
        Returns:
            Dictionary mapping model types to lists of results
        """
        all_results = {}
        
        for model_type, param_list in parameter_grid.items():
            logger.info(f"Running parameter sweep for {model_type.value} with {len(param_list)} configurations")
            
            model_results = []
            
            for i, params in enumerate(param_list):
                logger.info(f"Running {model_type.value} configuration {i+1}/{len(param_list)}")
                
                # Update runner config with current parameters
                config = ParallelRunConfig(
                    max_workers=1,  # Run one at a time for parameter sweep
                    model_params={model_type: params}
                )
                
                temp_runner = ParallelModelRunner(config)
                
                results = temp_runner.run_all_models(
                    X_train, y_train, X_test, y_test, [model_type]
                )
                
                if results:
                    model_results.append(results[0])
            
            all_results[model_type] = model_results
        
        return all_results
    
    def find_best_parameters(self, sweep_results: Dict[ModelType, List[ModelRunResult]]) -> Dict[ModelType, Dict[str, Any]]:
        """
        Find best parameters for each model type
        
        Args:
            sweep_results: Results from parameter sweep
            
        Returns:
            Dictionary mapping model types to best parameters
        """
        best_params = {}
        
        for model_type, results in sweep_results.items():
            successful_results = [r for r in results if r.success and r.performance]
            
            if not successful_results:
                logger.warning(f"No successful results for {model_type.value}")
                continue
            
            # Find best result by weighted score
            best_result = min(successful_results, key=lambda r: r.performance.weighted_score)
            
            # Extract parameters (this would need to be stored in the result)
            # For now, we'll return the model type
            best_params[model_type] = {
                'weighted_score': best_result.performance.weighted_score,
                'rmse': best_result.performance.rmse,
                'mae': best_result.performance.mae
            }
            
            logger.info(f"Best {model_type.value} score: {best_result.performance.weighted_score:.4f}")
        
        return best_params