"""
Comprehensive tests for allocation optimizers
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, patch
from typing import Dict, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizers.base_optimizer import (
    BaseOptimizer, OptimizerType, AllocationResult, AllocationConstraints,
    OptimizerFactory, OptimizerRegistry
)
from optimizers.priority_optimizer import PriorityOptimizer
from optimizers.fair_share_optimizer import FairShareOptimizer
from optimizers.optimizer_selector import (
    OptimizerSelector, OptimizerConfig, OptimizationStrategy
)


class TestAllocationConstraints:
    """Tests for allocation constraints"""
    
    def test_default_constraints(self):
        """Test default constraint values"""
        constraints = AllocationConstraints()
        
        assert constraints.min_allocation == 0.0
        assert constraints.max_allocation is None
        assert constraints.min_order_quantity == 1.0
        assert constraints.safety_stock_buffer == 0.1
        assert constraints.allow_substitution == True
        assert constraints.max_substitution_ratio == 0.3
        assert constraints.respect_customer_tier == True
        assert constraints.respect_sla_levels == True
    
    def test_custom_constraints(self):
        """Test custom constraint values"""
        constraints = AllocationConstraints(
            min_allocation=5.0,
            max_allocation=1000.0,
            allow_substitution=False,
            safety_stock_buffer=0.15
        )
        
        assert constraints.min_allocation == 5.0
        assert constraints.max_allocation == 1000.0
        assert constraints.allow_substitution == False
        assert constraints.safety_stock_buffer == 0.15


class TestAllocationResult:
    """Tests for allocation result container"""
    
    @pytest.fixture
    def sample_allocation_df(self):
        """Sample allocation dataframe"""
        return pd.DataFrame({
            'dc_id': ['DC1', 'DC2', 'DC3'],
            'sku_id': ['SKU1', 'SKU1', 'SKU2'],
            'allocated_quantity': [100.0, 150.0, 200.0],
            'forecasted_demand': [120.0, 180.0, 180.0]
        })
    
    def test_allocation_result_creation(self, sample_allocation_df):
        """Test allocation result creation"""
        result = AllocationResult(
            optimizer_type=OptimizerType.PRIORITY,
            allocations=sample_allocation_df,
            total_allocated=450.0,
            total_demand=480.0,
            allocation_efficiency=93.75,
            unallocated_demand=30.0,
            substitutions_made=[],
            optimization_time=1.5,
            constraints_violated=[],
            allocation_summary={}
        )
        
        assert result.optimizer_type == OptimizerType.PRIORITY
        assert result.total_allocated == 450.0
        assert result.allocation_efficiency == 93.75
        assert len(result.allocations) == 3
    
    def test_allocation_result_serialization(self, sample_allocation_df):
        """Test allocation result to_dict method"""
        result = AllocationResult(
            optimizer_type=OptimizerType.FAIR_SHARE,
            allocations=sample_allocation_df,
            total_allocated=450.0,
            total_demand=480.0,
            allocation_efficiency=93.75,
            unallocated_demand=30.0,
            substitutions_made=[{'test': 'substitution'}],
            optimization_time=2.0,
            constraints_violated=['test_violation'],
            allocation_summary={'test': 'summary'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['optimizer_type'] == 'fair_share'
        assert result_dict['total_allocated'] == 450.0
        assert result_dict['allocation_efficiency'] == 93.75
        assert isinstance(result_dict['allocations'], list)
        assert len(result_dict['allocations']) == 3
        assert result_dict['substitutions_made'] == [{'test': 'substitution'}]


class TestBaseOptimizer:
    """Tests for base optimizer functionality"""
    
    @pytest.fixture
    def sample_allocation_data(self):
        """Sample allocation data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'dc_id': ['DC1', 'DC2', 'DC3', 'DC4', 'DC5'] * 4,
            'sku_id': ['SKU1'] * 10 + ['SKU2'] * 10,
            'customer_id': ['CUST1', 'CUST2'] * 10,
            'current_inventory': np.random.uniform(50, 500, 20),
            'forecasted_demand': np.random.uniform(10, 200, 20),
            'dc_priority': np.random.randint(1, 6, 20),
            'customer_tier': ['PREMIUM', 'STANDARD'] * 10,
            'sla_level': ['HIGH', 'MEDIUM', 'LOW'] * 6 + ['HIGH', 'MEDIUM'],
            'sku_category': ['Electronics'] * 20,
            'min_order_quantity': [5.0] * 20
        })
    
    def test_input_validation_valid_data(self, sample_allocation_data):
        """Test input validation with valid data"""
        constraints = AllocationConstraints()
        
        # Create a mock optimizer to test base functionality
        class MockOptimizer(BaseOptimizer):
            def optimize(self, allocation_data, **kwargs):
                return AllocationResult(
                    optimizer_type=self.optimizer_type,
                    allocations=allocation_data,
                    total_allocated=0, total_demand=0, allocation_efficiency=0,
                    unallocated_demand=0, substitutions_made=[], optimization_time=0,
                    constraints_violated=[], allocation_summary={}
                )
            def get_feature_importance(self):
                return None
        
        optimizer = MockOptimizer(OptimizerType.PRIORITY, constraints)
        
        # Should not raise any exception
        optimizer.validate_input_data(sample_allocation_data)
    
    def test_input_validation_missing_columns(self):
        """Test input validation with missing required columns"""
        constraints = AllocationConstraints()
        
        class MockOptimizer(BaseOptimizer):
            def optimize(self, allocation_data, **kwargs):
                pass
            def get_feature_importance(self):
                return None
        
        optimizer = MockOptimizer(OptimizerType.PRIORITY, constraints)
        
        # Missing required columns
        invalid_df = pd.DataFrame({
            'dc_id': ['DC1', 'DC2'],
            'sku_id': ['SKU1', 'SKU2']
            # Missing other required columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            optimizer.validate_input_data(invalid_df)
    
    def test_input_validation_empty_data(self):
        """Test input validation with empty data"""
        constraints = AllocationConstraints()
        
        class MockOptimizer(BaseOptimizer):
            def optimize(self, allocation_data, **kwargs):
                pass
            def get_feature_importance(self):
                return None
        
        optimizer = MockOptimizer(OptimizerType.PRIORITY, constraints)
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            optimizer.validate_input_data(empty_df)
    
    def test_apply_constraints(self, sample_allocation_data):
        """Test applying business constraints"""
        constraints = AllocationConstraints(
            min_allocation=10.0,
            max_allocation=100.0,
            safety_stock_buffer=0.2
        )
        
        class MockOptimizer(BaseOptimizer):
            def optimize(self, allocation_data, **kwargs):
                pass
            def get_feature_importance(self):
                return None
        
        optimizer = MockOptimizer(OptimizerType.PRIORITY, constraints)
        
        # Create test allocation data
        test_df = sample_allocation_data.copy()
        test_df['allocated_quantity'] = [5.0, 150.0, 50.0] + [25.0] * 17  # Mix of values
        
        constrained_df = optimizer.apply_constraints(test_df)
        
        # Check minimum allocation constraint
        small_allocations = constrained_df[constrained_df['allocated_quantity'] > 0]['allocated_quantity']
        assert all(small_allocations >= 10.0)
        
        # Check maximum allocation constraint
        assert all(constrained_df['allocated_quantity'] <= 100.0)
    
    def test_allocation_efficiency_calculation(self):
        """Test allocation efficiency calculation"""
        constraints = AllocationConstraints()
        
        class MockOptimizer(BaseOptimizer):
            def optimize(self, allocation_data, **kwargs):
                pass
            def get_feature_importance(self):
                return None
        
        optimizer = MockOptimizer(OptimizerType.PRIORITY, constraints)
        
        # Test perfect efficiency
        assert optimizer.calculate_allocation_efficiency(100.0, 100.0) == 100.0
        
        # Test partial efficiency
        assert optimizer.calculate_allocation_efficiency(80.0, 100.0) == 80.0
        
        # Test over-allocation (should cap at 100%)
        assert optimizer.calculate_allocation_efficiency(120.0, 100.0) == 100.0
        
        # Test zero demand
        assert optimizer.calculate_allocation_efficiency(0.0, 0.0) == 100.0
        assert optimizer.calculate_allocation_efficiency(50.0, 0.0) == 0.0


class TestOptimizerFactory:
    """Tests for optimizer factory"""
    
    def test_create_priority_optimizer(self):
        """Test creating priority optimizer"""
        constraints = AllocationConstraints(min_allocation=5.0)
        optimizer = OptimizerFactory.create_optimizer(OptimizerType.PRIORITY, constraints)
        
        assert isinstance(optimizer, PriorityOptimizer)
        assert optimizer.optimizer_type == OptimizerType.PRIORITY
        assert optimizer.constraints.min_allocation == 5.0
    
    def test_create_fair_share_optimizer(self):
        """Test creating fair share optimizer"""
        constraints = AllocationConstraints(allow_substitution=False)
        optimizer = OptimizerFactory.create_optimizer(OptimizerType.FAIR_SHARE, constraints)
        
        assert isinstance(optimizer, FairShareOptimizer)
        assert optimizer.optimizer_type == OptimizerType.FAIR_SHARE
        assert optimizer.constraints.allow_substitution == False
    
    def test_create_invalid_optimizer(self):
        """Test creating invalid optimizer type"""
        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            OptimizerFactory.create_optimizer("invalid_type")
    
    def test_get_available_optimizers(self):
        """Test getting available optimizers"""
        available = OptimizerFactory.get_available_optimizers()
        
        assert OptimizerType.PRIORITY in available
        assert OptimizerType.FAIR_SHARE in available
        assert len(available) == 2


class TestOptimizerRegistry:
    """Tests for optimizer registry"""
    
    def test_register_and_get_optimizer(self):
        """Test registering and retrieving optimizer"""
        registry = OptimizerRegistry()
        optimizer = PriorityOptimizer()
        
        registry.register_optimizer(optimizer)
        retrieved = registry.get_optimizer(OptimizerType.PRIORITY)
        
        assert retrieved is optimizer
        assert retrieved.optimizer_type == OptimizerType.PRIORITY
    
    def test_create_and_register(self):
        """Test creating and registering optimizer"""
        registry = OptimizerRegistry()
        constraints = AllocationConstraints(min_allocation=10.0)
        
        optimizer = registry.create_and_register(OptimizerType.FAIR_SHARE, constraints)
        
        assert isinstance(optimizer, FairShareOptimizer)
        assert optimizer.constraints.min_allocation == 10.0
        
        # Should be retrievable
        retrieved = registry.get_optimizer(OptimizerType.FAIR_SHARE)
        assert retrieved is optimizer
    
    def test_list_registered_optimizers(self):
        """Test listing registered optimizers"""
        registry = OptimizerRegistry()
        
        # Initially empty
        assert registry.list_registered_optimizers() == []
        
        # Register optimizers
        registry.create_and_register(OptimizerType.PRIORITY)
        registry.create_and_register(OptimizerType.FAIR_SHARE)
        
        registered = registry.list_registered_optimizers()
        assert OptimizerType.PRIORITY in registered
        assert OptimizerType.FAIR_SHARE in registered
        assert len(registered) == 2


class TestPriorityOptimizer:
    """Tests for priority-based optimizer"""
    
    @pytest.fixture
    def sample_priority_data(self):
        """Sample data for priority optimizer testing"""
        return pd.DataFrame({
            'dc_id': ['DC1', 'DC2', 'DC3', 'DC4'],
            'sku_id': ['SKU1', 'SKU1', 'SKU1', 'SKU2'],
            'customer_id': ['CUST1', 'CUST1', 'CUST2', 'CUST1'],
            'current_inventory': [100.0, 200.0, 150.0, 300.0],
            'forecasted_demand': [80.0, 120.0, 100.0, 200.0],
            'dc_priority': [1, 3, 2, 1],  # DC1 and DC4 highest priority
            'customer_tier': ['PREMIUM', 'STANDARD', 'PREMIUM', 'PREMIUM'],
            'sla_level': ['HIGH', 'MEDIUM', 'HIGH', 'HIGH'],
            'sku_category': ['Electronics', 'Electronics', 'Electronics', 'Home'],
            'min_order_quantity': [5.0, 5.0, 5.0, 10.0]
        })
    
    def test_priority_optimizer_initialization(self):
        """Test priority optimizer initialization"""
        constraints = AllocationConstraints(min_allocation=5.0)
        optimizer = PriorityOptimizer(constraints)
        
        assert optimizer.optimizer_type == OptimizerType.PRIORITY
        assert optimizer.constraints.min_allocation == 5.0
        assert len(optimizer.priority_weights) == 5  # Default priority levels
        assert optimizer.priority_weights[1] == 1.0  # Highest priority
        assert optimizer.priority_weights[5] == 0.2  # Lowest priority
    
    def test_priority_optimization(self, sample_priority_data):
        """Test basic priority optimization"""
        optimizer = PriorityOptimizer()
        
        result = optimizer.optimize(sample_priority_data)
        
        assert isinstance(result, AllocationResult)
        assert result.optimizer_type == OptimizerType.PRIORITY
        assert result.total_allocated > 0
        assert result.allocation_efficiency > 0
        assert len(result.allocations) == len(sample_priority_data)
    
    def test_priority_allocation_order(self, sample_priority_data):
        """Test that higher priority DCs get preference"""
        optimizer = PriorityOptimizer()
        
        result = optimizer.optimize(sample_priority_data)
        allocations = result.allocations
        
        # Group by SKU and check priority allocation
        for sku_id in allocations['sku_id'].unique():
            sku_data = allocations[allocations['sku_id'] == sku_id].copy()
            sku_data = sku_data.sort_values('dc_priority')
            
            # Higher priority (lower number) should generally get more relative to demand
            if len(sku_data) > 1:
                priority_1_records = sku_data[sku_data['dc_priority'] == 1]
                other_records = sku_data[sku_data['dc_priority'] > 1]
                
                if not priority_1_records.empty and not other_records.empty:
                    # Check fulfillment ratios
                    priority_1_fulfillment = (
                        priority_1_records['allocated_quantity'].sum() /
                        priority_1_records['forecasted_demand'].sum()
                    )
                    
                    other_fulfillment = (
                        other_records['allocated_quantity'].sum() /
                        other_records['forecasted_demand'].sum()
                    )
                    
                    # Priority 1 should have equal or better fulfillment
                    assert priority_1_fulfillment >= other_fulfillment * 0.95  # Allow small tolerance
    
    def test_priority_weights_customization(self, sample_priority_data):
        """Test custom priority weights"""
        optimizer = PriorityOptimizer()
        
        # Set custom weights
        custom_weights = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.1}
        optimizer.set_priority_weights(custom_weights)
        
        weights = optimizer.get_priority_weights()
        assert weights[1] == 2.0
        assert weights[2] == 1.5
        assert weights[3] == 1.0


class TestFairShareOptimizer:
    """Tests for fair share optimizer"""
    
    @pytest.fixture
    def sample_fairshare_data(self):
        """Sample data for fair share optimizer testing"""
        return pd.DataFrame({
            'dc_id': ['DC1', 'DC2', 'DC3', 'DC4'],
            'sku_id': ['SKU1', 'SKU1', 'SKU1', 'SKU1'],
            'customer_id': ['CUST1', 'CUST1', 'CUST1', 'CUST1'],
            'current_inventory': [100.0, 100.0, 100.0, 100.0],  # Equal inventory
            'forecasted_demand': [50.0, 100.0, 150.0, 200.0],   # Different demands
            'dc_priority': [1, 2, 3, 4],  # Different priorities (should be ignored)
            'customer_tier': ['PREMIUM', 'PREMIUM', 'PREMIUM', 'PREMIUM'],
            'sla_level': ['HIGH', 'HIGH', 'HIGH', 'HIGH'],
            'sku_category': ['Electronics', 'Electronics', 'Electronics', 'Electronics'],
            'min_order_quantity': [5.0, 5.0, 5.0, 5.0]
        })
    
    def test_fairshare_optimizer_initialization(self):
        """Test fair share optimizer initialization"""
        constraints = AllocationConstraints(allow_substitution=False)
        optimizer = FairShareOptimizer(constraints)
        
        assert optimizer.optimizer_type == OptimizerType.FAIR_SHARE
        assert optimizer.constraints.allow_substitution == False
        assert optimizer.fairness_threshold == 0.05  # Default fairness threshold
    
    def test_fairshare_optimization(self, sample_fairshare_data):
        """Test basic fair share optimization"""
        optimizer = FairShareOptimizer()
        
        result = optimizer.optimize(sample_fairshare_data)
        
        assert isinstance(result, AllocationResult)
        assert result.optimizer_type == OptimizerType.FAIR_SHARE
        assert result.total_allocated > 0
        assert result.allocation_efficiency > 0
        assert len(result.allocations) == len(sample_fairshare_data)
    
    def test_fairshare_proportional_allocation(self, sample_fairshare_data):
        """Test that allocation is proportional to demand"""
        optimizer = FairShareOptimizer()
        
        result = optimizer.optimize(sample_fairshare_data)
        allocations = result.allocations
        
        # Calculate fulfillment ratios
        allocations['fulfillment_ratio'] = np.where(
            allocations['forecasted_demand'] > 0,
            allocations['allocated_quantity'] / allocations['forecasted_demand'],
            0.0
        )
        
        # Fair share should result in similar fulfillment ratios
        fulfillment_ratios = allocations[allocations['fulfillment_ratio'] > 0]['fulfillment_ratio'].values
        
        if len(fulfillment_ratios) > 1:
            # Check coefficient of variation (lower is more fair)
            cv = np.std(fulfillment_ratios) / np.mean(fulfillment_ratios)
            assert cv < 0.5  # Reasonably fair distribution
    
    def test_fairness_metrics_calculation(self, sample_fairshare_data):
        """Test fairness metrics calculation"""
        optimizer = FairShareOptimizer()
        
        result = optimizer.optimize(sample_fairshare_data)
        
        # Check that fairness metrics are included in summary
        assert 'overall_fairness_score' in result.allocation_summary
        assert 'fairness_by_sku' in result.allocation_summary
        assert 'gini_coefficient' in result.allocation_summary
        
        fairness_score = result.allocation_summary['overall_fairness_score']
        assert 0.0 <= fairness_score <= 1.0
    
    def test_fairness_report_generation(self, sample_fairshare_data):
        """Test fairness report generation"""
        optimizer = FairShareOptimizer()
        
        result = optimizer.optimize(sample_fairshare_data)
        report = optimizer.get_fairness_report(result)
        
        assert isinstance(report, str)
        assert "FAIR SHARE ALLOCATION REPORT" in report
        assert "FAIRNESS OVERVIEW" in report
        assert "Overall Fairness Score" in report


class TestOptimizerSelector:
    """Tests for optimizer selector"""
    
    @pytest.fixture
    def sample_selector_data(self):
        """Sample data for optimizer selector testing"""
        return pd.DataFrame({
            'dc_id': ['DC1', 'DC2', 'DC3'] * 3,
            'sku_id': ['SKU1', 'SKU2', 'SKU3'] * 3,
            'customer_id': ['CUST1'] * 9,
            'current_inventory': [100.0] * 9,
            'forecasted_demand': [80.0, 120.0, 90.0] * 3,
            'dc_priority': [1, 2, 3] * 3,
            'customer_tier': ['PREMIUM'] * 9,
            'sla_level': ['HIGH'] * 9,
            'sku_category': ['Electronics'] * 9,
            'min_order_quantity': [5.0] * 9
        })
    
    def test_selector_initialization(self):
        """Test optimizer selector initialization"""
        selector = OptimizerSelector()
        
        assert len(selector.available_strategies) == 2
        assert OptimizationStrategy.PRIORITY_BASED in selector.available_strategies
        assert OptimizationStrategy.FAIR_SHARE in selector.available_strategies
        assert len(selector.execution_history) == 0
    
    def test_direct_strategy_selection(self, sample_selector_data):
        """Test direct strategy selection and execution"""
        selector = OptimizerSelector()
        
        config = OptimizerConfig(
            strategy=OptimizationStrategy.PRIORITY_BASED,
            constraints=AllocationConstraints(min_allocation=5.0)
        )
        
        result = selector.select_and_run_optimizer(sample_selector_data, config)
        
        assert isinstance(result, AllocationResult)
        assert result.optimizer_type == OptimizerType.PRIORITY
        assert len(selector.execution_history) == 1
    
    def test_strategy_comparison(self, sample_selector_data):
        """Test strategy comparison functionality"""
        selector = OptimizerSelector()
        
        strategies = [OptimizationStrategy.PRIORITY_BASED, OptimizationStrategy.FAIR_SHARE]
        constraints = AllocationConstraints()
        
        comparison = selector.compare_strategies(sample_selector_data, strategies, constraints)
        
        assert isinstance(comparison.results, dict)
        assert len(comparison.results) == 2
        assert OptimizationStrategy.PRIORITY_BASED in comparison.results
        assert OptimizationStrategy.FAIR_SHARE in comparison.results
        assert comparison.best_strategy in strategies
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.comparison_metrics) > 0
    
    def test_strategy_recommendation(self, sample_selector_data):
        """Test strategy recommendation based on data analysis"""
        selector = OptimizerSelector()
        
        recommendation = selector.get_strategy_recommendation(sample_selector_data)
        
        assert 'recommended_strategy' in recommendation
        assert 'confidence' in recommendation
        assert 'reason' in recommendation
        assert 'data_analysis' in recommendation
        
        assert isinstance(recommendation['recommended_strategy'], OptimizationStrategy)
        assert 0.0 <= recommendation['confidence'] <= 1.0
        assert isinstance(recommendation['reason'], str)
        
        # Data analysis should contain key metrics
        analysis = recommendation['data_analysis']
        assert 'total_records' in analysis
        assert 'priority_variance' in analysis
        assert 'demand_variance' in analysis
    
    def test_auto_select_strategy(self, sample_selector_data):
        """Test auto-select strategy functionality"""
        selector = OptimizerSelector()
        
        config = OptimizerConfig(
            strategy=OptimizationStrategy.AUTO_SELECT,
            constraints=AllocationConstraints()
        )
        
        result = selector.select_and_run_optimizer(sample_selector_data, config)
        
        assert isinstance(result, AllocationResult)
        assert result.optimizer_type in [OptimizerType.PRIORITY, OptimizerType.FAIR_SHARE]
        assert len(selector.execution_history) == 1


@pytest.mark.integration
class TestOptimizerIntegration:
    """Integration tests for complete optimizer workflow"""
    
    @pytest.fixture
    def comprehensive_allocation_data(self):
        """Comprehensive allocation data for integration testing"""
        np.random.seed(42)
        
        n_records = 50
        dcs = ['DC1', 'DC2', 'DC3', 'DC4', 'DC5']
        skus = ['SKU1', 'SKU2', 'SKU3', 'SKU4']
        customers = ['CUST1', 'CUST2', 'CUST3']
        
        data = []
        for i in range(n_records):
            record = {
                'dc_id': np.random.choice(dcs),
                'sku_id': np.random.choice(skus),
                'customer_id': np.random.choice(customers),
                'current_inventory': np.random.uniform(50, 500),
                'forecasted_demand': np.random.uniform(10, 200),
                'dc_priority': np.random.randint(1, 6),
                'customer_tier': np.random.choice(['PREMIUM', 'STANDARD', 'BASIC']),
                'sla_level': np.random.choice(['HIGH', 'MEDIUM', 'LOW']),
                'sku_category': np.random.choice(['Electronics', 'Home', 'Apparel']),
                'min_order_quantity': np.random.uniform(1, 10),
                'revenue_per_unit': np.random.uniform(10, 100),
                'cost_per_unit': np.random.uniform(5, 50),
                'lead_time_days': np.random.randint(1, 30)
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def test_end_to_end_priority_optimization(self, comprehensive_allocation_data):
        """Test complete priority optimization workflow"""
        constraints = AllocationConstraints(
            min_allocation=5.0,
            allow_substitution=True,
            max_substitution_ratio=0.2
        )
        
        optimizer = PriorityOptimizer(constraints)
        result = optimizer.optimize(comprehensive_allocation_data)
        
        # Verify result completeness
        assert isinstance(result, AllocationResult)
        assert result.total_allocated > 0
        assert result.total_demand > 0
        assert 0 <= result.allocation_efficiency <= 100
        assert result.optimization_time > 0
        assert len(result.allocations) == len(comprehensive_allocation_data)
        
        # Verify business rules
        allocated_records = result.allocations[result.allocations['allocated_quantity'] > 0]
        if not allocated_records.empty:
            # Check minimum allocation constraint
            assert all(allocated_records['allocated_quantity'] >= 5.0)
            
            # Check inventory constraints
            over_allocated = allocated_records[
                allocated_records['allocated_quantity'] > allocated_records['current_inventory']
            ]
            assert len(over_allocated) == 0  # No over-allocation
        
        # Verify allocation summary
        assert len(result.allocation_summary) > 0
        assert 'total_records' in result.allocation_summary
    
    def test_end_to_end_fairshare_optimization(self, comprehensive_allocation_data):
        """Test complete fair share optimization workflow"""
        constraints = AllocationConstraints(
            allow_substitution=False,
            respect_customer_tier=True
        )
        
        optimizer = FairShareOptimizer(constraints)
        result = optimizer.optimize(comprehensive_allocation_data)
        
        # Verify result completeness
        assert isinstance(result, AllocationResult)
        assert result.total_allocated > 0
        assert result.allocation_efficiency > 0
        assert result.optimization_time > 0
        
        # Verify fairness metrics
        assert 'overall_fairness_score' in result.allocation_summary
        assert 'gini_coefficient' in result.allocation_summary
        
        fairness_score = result.allocation_summary['overall_fairness_score']
        assert 0.0 <= fairness_score <= 1.0
    
    def test_strategy_comparison_integration(self, comprehensive_allocation_data):
        """Test complete strategy comparison workflow"""
        selector = OptimizerSelector()
        
        strategies = [
            OptimizationStrategy.PRIORITY_BASED,
            OptimizationStrategy.FAIR_SHARE
        ]
        
        constraints = AllocationConstraints(min_allocation=1.0)
        comparison = selector.compare_strategies(comprehensive_allocation_data, strategies, constraints)
        
        # Verify comparison results
        assert len(comparison.results) == 2
        assert comparison.best_strategy in strategies
        assert isinstance(comparison.recommendation, str)
        
        # Verify both strategies executed successfully
        for strategy, result in comparison.results.items():
            assert result.allocation_efficiency > 0
            assert result.total_allocated > 0
            assert result.optimization_time > 0
        
        # Verify comparison metrics
        metrics = comparison.comparison_metrics
        assert 'efficiency_comparison' in metrics
        assert 'execution_time_comparison' in metrics
        assert 'overall_scores' in metrics
        
        # Verify all strategies have metrics
        for strategy in strategies:
            assert strategy.value in metrics['efficiency_comparison']
            assert strategy.value in metrics['execution_time_comparison']
            assert strategy.value in metrics['overall_scores']
    
    def test_performance_benchmarking(self, comprehensive_allocation_data):
        """Test optimizer performance with larger datasets"""
        # Create larger dataset
        large_data = pd.concat([comprehensive_allocation_data] * 5, ignore_index=True)
        
        constraints = AllocationConstraints()
        
        # Test priority optimizer performance
        priority_optimizer = PriorityOptimizer(constraints)
        start_time = time.time()
        priority_result = priority_optimizer.optimize(large_data)
        priority_time = time.time() - start_time
        
        # Test fair share optimizer performance
        fairshare_optimizer = FairShareOptimizer(constraints)
        start_time = time.time()
        fairshare_result = fairshare_optimizer.optimize(large_data)
        fairshare_time = time.time() - start_time
        
        # Verify reasonable performance (should complete within 30 seconds)
        assert priority_time < 30.0
        assert fairshare_time < 30.0
        
        # Verify results quality
        assert priority_result.allocation_efficiency > 50.0  # At least 50% efficiency
        assert fairshare_result.allocation_efficiency > 50.0
        
        print(f"Performance benchmark - Priority: {priority_time:.2f}s, Fair Share: {fairshare_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__])