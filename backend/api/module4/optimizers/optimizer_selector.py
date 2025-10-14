"""
Optimizer selection and management interface
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .base_optimizer import (
    BaseOptimizer, OptimizerType, AllocationResult, AllocationConstraints, 
    OptimizerFactory, optimizer_registry
)

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """High-level optimization strategies that users can select from"""
    PRIORITY_BASED = "priority_based"
    FAIR_SHARE = "fair_share"
    HYBRID = "hybrid"
    AUTO_SELECT = "auto_select"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer selection and execution"""
    strategy: OptimizationStrategy
    constraints: Optional[AllocationConstraints] = None
    strategy_params: Optional[Dict[str, Any]] = None
    
    # Auto-selection parameters
    auto_select_criteria: Optional[Dict[str, float]] = None
    
    # Hybrid strategy parameters
    priority_weight: float = 0.6  # Weight for priority in hybrid mode
    fairness_weight: float = 0.4  # Weight for fairness in hybrid mode
    
    # Performance preferences
    prefer_efficiency: bool = True  # Prefer allocation efficiency over fairness
    prefer_speed: bool = False  # Prefer faster execution over optimal results


@dataclass
class OptimizationComparison:
    """Container for comparing multiple optimization results"""
    results: Dict[OptimizationStrategy, AllocationResult]
    best_strategy: OptimizationStrategy
    comparison_metrics: Dict[str, Any]
    recommendation: str


class OptimizerSelector:
    """
    Intelligent optimizer selection and management system
    
    Provides user-friendly interface for selecting and running optimization strategies
    """
    
    def __init__(self):
        """Initialize optimizer selector"""
        self.available_strategies = {
            OptimizationStrategy.PRIORITY_BASED: OptimizerType.PRIORITY,
            OptimizationStrategy.FAIR_SHARE: OptimizerType.FAIR_SHARE
        }
        self.execution_history: List[AllocationResult] = []
    
    def select_and_run_optimizer(self, 
                                allocation_data: pd.DataFrame,
                                config: OptimizerConfig) -> AllocationResult:
        """
        Select and run optimizer based on user configuration
        
        Args:
            allocation_data: DataFrame with allocation data
            config: Optimizer configuration
            
        Returns:
            AllocationResult from selected optimizer
        """
        logger.info(f"Running optimization with strategy: {config.strategy.value}")
        
        if config.strategy == OptimizationStrategy.AUTO_SELECT:
            return self._auto_select_and_run(allocation_data, config)
        
        elif config.strategy == OptimizationStrategy.HYBRID:
            return self._run_hybrid_optimization(allocation_data, config)
        
        else:
            # Direct strategy selection
            optimizer_type = self.available_strategies.get(config.strategy)
            if not optimizer_type:
                raise ValueError(f"Unsupported optimization strategy: {config.strategy}")
            
            optimizer = OptimizerFactory.create_optimizer(optimizer_type, config.constraints)
            
            # Pass strategy-specific parameters
            strategy_params = config.strategy_params or {}
            result = optimizer.optimize(allocation_data, **strategy_params)
            
            # Store in history
            self.execution_history.append(result)
            
            return result
    
    def compare_strategies(self, 
                          allocation_data: pd.DataFrame,
                          strategies: List[OptimizationStrategy],
                          constraints: Optional[AllocationConstraints] = None) -> OptimizationComparison:
        """
        Compare multiple optimization strategies on the same data
        
        Args:
            allocation_data: DataFrame with allocation data
            strategies: List of strategies to compare
            constraints: Allocation constraints to use for all strategies
            
        Returns:
            OptimizationComparison with results from all strategies
        """
        logger.info(f"Comparing {len(strategies)} optimization strategies")
        
        results = {}
        
        for strategy in strategies:
            try:
                config = OptimizerConfig(
                    strategy=strategy,
                    constraints=constraints,
                    prefer_speed=True  # Use faster settings for comparison
                )
                
                result = self.select_and_run_optimizer(allocation_data, config)
                results[strategy] = result
                
                logger.info(f"{strategy.value}: {result.allocation_efficiency:.1f}% efficiency")
                
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                continue
        
        if not results:
            raise ValueError("All optimization strategies failed")
        
        # Analyze and compare results
        comparison_metrics = self._analyze_strategy_comparison(results)
        best_strategy = self._select_best_strategy(results, comparison_metrics)
        recommendation = self._generate_strategy_recommendation(results, comparison_metrics, best_strategy)
        
        return OptimizationComparison(
            results=results,
            best_strategy=best_strategy,
            comparison_metrics=comparison_metrics,
            recommendation=recommendation
        )
    
    def get_strategy_recommendation(self, 
                                  allocation_data: pd.DataFrame,
                                  user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze data and provide strategy recommendation
        
        Args:
            allocation_data: DataFrame with allocation data
            user_preferences: User preferences for optimization
            
        Returns:
            Dictionary with strategy recommendation and reasoning
        """
        logger.info("Analyzing data for strategy recommendation")
        
        # Analyze data characteristics
        data_analysis = self._analyze_allocation_data(allocation_data)
        
        # Default preferences
        preferences = user_preferences or {}
        prefer_efficiency = preferences.get('prefer_efficiency', True)
        prefer_fairness = preferences.get('prefer_fairness', False)
        prefer_speed = preferences.get('prefer_speed', False)
        max_execution_time = preferences.get('max_execution_time_seconds', 60)
        
        recommendations = []
        
        # Analyze priority distribution
        priority_variance = data_analysis.get('priority_variance', 0)
        demand_variance = data_analysis.get('demand_variance', 0)
        inventory_utilization = data_analysis.get('inventory_utilization', 0)
        
        # Priority-based recommendation logic
        if priority_variance > 0.5:  # High priority variance
            recommendations.append({
                'strategy': OptimizationStrategy.PRIORITY_BASED,
                'confidence': 0.8,
                'reason': 'High variance in DC priorities suggests priority-based allocation would be effective'
            })
        
        # Fair share recommendation logic
        if priority_variance < 0.3 and demand_variance > 0.2:  # Low priority variance, high demand variance
            recommendations.append({
                'strategy': OptimizationStrategy.FAIR_SHARE,
                'confidence': 0.7,
                'reason': 'Similar DC priorities with varied demand suggests fair share allocation'
            })
        
        # Hybrid recommendation logic
        if 0.3 <= priority_variance <= 0.7:  # Moderate priority variance
            recommendations.append({
                'strategy': OptimizationStrategy.HYBRID,
                'confidence': 0.6,
                'reason': 'Moderate priority variance suggests hybrid approach balancing priority and fairness'
            })
        
        # Apply user preferences
        if prefer_efficiency:
            for rec in recommendations:
                if rec['strategy'] == OptimizationStrategy.PRIORITY_BASED:
                    rec['confidence'] += 0.1
        
        if prefer_fairness:
            for rec in recommendations:
                if rec['strategy'] == OptimizationStrategy.FAIR_SHARE:
                    rec['confidence'] += 0.1
        
        if prefer_speed:
            # Priority-based is generally faster
            for rec in recommendations:
                if rec['strategy'] == OptimizationStrategy.PRIORITY_BASED:
                    rec['confidence'] += 0.05
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Default to priority-based if no clear recommendation
        if not recommendations:
            recommendations.append({
                'strategy': OptimizationStrategy.PRIORITY_BASED,
                'confidence': 0.5,
                'reason': 'Default recommendation - priority-based allocation is generally robust'
            })
        
        return {
            'recommended_strategy': recommendations[0]['strategy'],
            'confidence': recommendations[0]['confidence'],
            'reason': recommendations[0]['reason'],
            'all_recommendations': recommendations,
            'data_analysis': data_analysis
        }
    
    def _auto_select_and_run(self, allocation_data: pd.DataFrame, config: OptimizerConfig) -> AllocationResult:
        """Automatically select and run the best optimizer for the data"""
        logger.info("Auto-selecting optimization strategy")
        
        # Get recommendation
        recommendation = self.get_strategy_recommendation(
            allocation_data,
            config.auto_select_criteria or {}
        )
        
        recommended_strategy = recommendation['recommended_strategy']
        
        # Create new config with recommended strategy
        auto_config = OptimizerConfig(
            strategy=recommended_strategy,
            constraints=config.constraints,
            strategy_params=config.strategy_params
        )
        
        logger.info(f"Auto-selected strategy: {recommended_strategy.value} (confidence: {recommendation['confidence']:.2f})")
        
        return self.select_and_run_optimizer(allocation_data, auto_config)
    
    def _run_hybrid_optimization(self, allocation_data: pd.DataFrame, config: OptimizerConfig) -> AllocationResult:
        """Run hybrid optimization combining priority and fair share approaches"""
        logger.info("Running hybrid optimization")
        
        # Run both optimizers
        priority_optimizer = OptimizerFactory.create_optimizer(OptimizerType.PRIORITY, config.constraints)
        fairshare_optimizer = OptimizerFactory.create_optimizer(OptimizerType.FAIR_SHARE, config.constraints)
        
        priority_params = config.strategy_params or {}
        fairshare_params = config.strategy_params or {}
        
        priority_result = priority_optimizer.optimize(allocation_data, **priority_params)
        fairshare_result = fairshare_optimizer.optimize(allocation_data, **fairshare_params)
        
        # Combine results based on weights
        hybrid_allocations = self._combine_allocation_results(
            priority_result.allocations,
            fairshare_result.allocations,
            config.priority_weight,
            config.fairness_weight
        )
        
        # Create hybrid result
        total_allocated = hybrid_allocations['allocated_quantity'].sum()
        total_demand = hybrid_allocations['forecasted_demand'].sum()
        
        hybrid_result = AllocationResult(
            optimizer_type=OptimizerType.PRIORITY,  # Use priority as base type
            allocations=hybrid_allocations,
            total_allocated=total_allocated,
            total_demand=total_demand,
            allocation_efficiency=(total_allocated / total_demand * 100) if total_demand > 0 else 0,
            unallocated_demand=total_demand - total_allocated,
            substitutions_made=priority_result.substitutions_made + fairshare_result.substitutions_made,
            optimization_time=priority_result.optimization_time + fairshare_result.optimization_time,
            constraints_violated=list(set(priority_result.constraints_violated + fairshare_result.constraints_violated)),
            allocation_summary={}
        )
        
        # Generate hybrid summary
        hybrid_result.allocation_summary = {
            'hybrid_approach': True,
            'priority_weight': config.priority_weight,
            'fairness_weight': config.fairness_weight,
            'priority_efficiency': priority_result.allocation_efficiency,
            'fairshare_efficiency': fairshare_result.allocation_efficiency,
            'hybrid_efficiency': hybrid_result.allocation_efficiency
        }
        
        self.execution_history.append(hybrid_result)
        
        logger.info(f"Hybrid optimization completed: {hybrid_result.allocation_efficiency:.1f}% efficiency")
        
        return hybrid_result
    
    def _combine_allocation_results(self, 
                                  priority_df: pd.DataFrame,
                                  fairshare_df: pd.DataFrame,
                                  priority_weight: float,
                                  fairness_weight: float) -> pd.DataFrame:
        """Combine allocation results from two strategies"""
        
        # Ensure dataframes have the same index
        combined_df = priority_df.copy()
        
        # Weighted combination of allocations
        if 'allocated_quantity' in priority_df.columns and 'allocated_quantity' in fairshare_df.columns:
            combined_allocations = (
                priority_df['allocated_quantity'] * priority_weight +
                fairshare_df['allocated_quantity'] * fairness_weight
            )
            
            # Ensure combined allocation doesn't exceed inventory or demand
            if 'current_inventory' in combined_df.columns:
                combined_allocations = np.minimum(combined_allocations, combined_df['current_inventory'])
            
            if 'forecasted_demand' in combined_df.columns:
                combined_allocations = np.minimum(combined_allocations, combined_df['forecasted_demand'])
            
            combined_df['allocated_quantity'] = combined_allocations
            combined_df['allocation_method'] = 'hybrid'
        
        return combined_df
    
    def _analyze_allocation_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze allocation data characteristics for strategy recommendation"""
        
        analysis = {
            'total_records': len(df),
            'unique_dcs': df['dc_id'].nunique() if 'dc_id' in df.columns else 0,
            'unique_skus': df['sku_id'].nunique() if 'sku_id' in df.columns else 0,
            'priority_variance': 0,
            'demand_variance': 0,
            'inventory_utilization': 0,
            'demand_concentration': 0
        }
        
        # Priority variance analysis
        if 'dc_priority' in df.columns:
            priorities = df['dc_priority'].values
            if len(set(priorities)) > 1:
                analysis['priority_variance'] = np.var(priorities) / np.mean(priorities) if np.mean(priorities) > 0 else 0
        
        # Demand variance analysis
        if 'forecasted_demand' in df.columns:
            demands = df['forecasted_demand'].values
            demands = demands[demands > 0]  # Remove zero demands
            if len(demands) > 1:
                analysis['demand_variance'] = np.var(demands) / np.mean(demands) if np.mean(demands) > 0 else 0
        
        # Inventory utilization
        if 'current_inventory' in df.columns and 'forecasted_demand' in df.columns:
            total_inventory = df['current_inventory'].sum()
            total_demand = df['forecasted_demand'].sum()
            if total_inventory > 0:
                analysis['inventory_utilization'] = min(1.0, total_demand / total_inventory)
        
        # Demand concentration (Gini coefficient)
        if 'forecasted_demand' in df.columns:
            demands = df['forecasted_demand'].values
            demands = demands[demands > 0]
            if len(demands) > 1:
                # Calculate Gini coefficient
                sorted_demands = np.sort(demands)
                n = len(sorted_demands)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_demands)) / (n * np.sum(sorted_demands)) - (n + 1) / n
                analysis['demand_concentration'] = gini
        
        return analysis
    
    def _analyze_strategy_comparison(self, results: Dict[OptimizationStrategy, AllocationResult]) -> Dict[str, Any]:
        """Analyze comparison between multiple strategy results"""
        
        metrics = {
            'efficiency_comparison': {},
            'execution_time_comparison': {},
            'fairness_comparison': {},
            'overall_scores': {}
        }
        
        # Efficiency comparison
        for strategy, result in results.items():
            metrics['efficiency_comparison'][strategy.value] = result.allocation_efficiency
            metrics['execution_time_comparison'][strategy.value] = result.optimization_time
        
        # Calculate overall scores (weighted combination of metrics)
        for strategy, result in results.items():
            # Normalize metrics (0-1 scale)
            efficiency_score = result.allocation_efficiency / 100.0
            time_score = max(0, 1 - (result.optimization_time / 60.0))  # Penalty for long execution
            constraint_score = max(0, 1 - (len(result.constraints_violated) / 10.0))  # Penalty for violations
            
            # Weighted overall score
            overall_score = (
                0.5 * efficiency_score +
                0.2 * time_score +
                0.3 * constraint_score
            )
            
            metrics['overall_scores'][strategy.value] = overall_score
        
        return metrics
    
    def _select_best_strategy(self, 
                            results: Dict[OptimizationStrategy, AllocationResult],
                            metrics: Dict[str, Any]) -> OptimizationStrategy:
        """Select the best strategy based on comparison metrics"""
        
        overall_scores = metrics['overall_scores']
        
        # Find strategy with highest overall score
        best_strategy_value = max(overall_scores, key=overall_scores.get)
        
        # Convert back to enum
        for strategy in results.keys():
            if strategy.value == best_strategy_value:
                return strategy
        
        # Fallback to first strategy
        return list(results.keys())[0]
    
    def _generate_strategy_recommendation(self, 
                                        results: Dict[OptimizationStrategy, AllocationResult],
                                        metrics: Dict[str, Any],
                                        best_strategy: OptimizationStrategy) -> str:
        """Generate human-readable recommendation"""
        
        best_result = results[best_strategy]
        best_score = metrics['overall_scores'][best_strategy.value]
        
        recommendation = [
            f"Recommended Strategy: {best_strategy.value.replace('_', ' ').title()}",
            f"Overall Score: {best_score:.2f}/1.0",
            f"Allocation Efficiency: {best_result.allocation_efficiency:.1f}%",
            f"Execution Time: {best_result.optimization_time:.1f}s"
        ]
        
        # Add specific insights
        if best_strategy == OptimizationStrategy.PRIORITY_BASED:
            recommendation.append("Priority-based allocation maximizes efficiency by respecting DC hierarchy.")
        elif best_strategy == OptimizationStrategy.FAIR_SHARE:
            recommendation.append("Fair share allocation ensures equitable distribution across all DCs.")
        
        return "\n".join(recommendation)
    
    def get_execution_history(self, limit: int = 10) -> List[AllocationResult]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear execution history"""
        self.execution_history.clear()
        logger.info("Execution history cleared")


# Global optimizer selector instance
optimizer_selector = OptimizerSelector()