"""
Base optimizer interface and common allocation logic
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    """Optimizer types enumeration"""
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"


@dataclass
class AllocationConstraints:
    """Container for allocation constraints and business rules"""
    min_allocation: float = 0.0
    max_allocation: Optional[float] = None
    min_order_quantity: float = 1.0
    safety_stock_buffer: float = 0.1  # 10% buffer
    allow_substitution: bool = True
    max_substitution_ratio: float = 0.3  # Max 30% substitution
    respect_customer_tier: bool = True
    respect_sla_levels: bool = True


@dataclass
class AllocationResult:
    """Container for allocation optimization results"""
    optimizer_type: OptimizerType
    allocations: pd.DataFrame  # Final allocation per DC/SKU/Customer
    total_allocated: float
    total_demand: float
    allocation_efficiency: float  # % of demand satisfied
    unallocated_demand: float
    substitutions_made: List[Dict[str, Any]]
    optimization_time: float
    constraints_violated: List[str]
    allocation_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization"""
        return {
            'optimizer_type': self.optimizer_type.value,
            'allocations': self.allocations.to_dict('records'),
            'total_allocated': float(self.total_allocated),
            'total_demand': float(self.total_demand),
            'allocation_efficiency': float(self.allocation_efficiency),
            'unallocated_demand': float(self.unallocated_demand),
            'substitutions_made': self.substitutions_made,
            'optimization_time': float(self.optimization_time),
            'constraints_violated': self.constraints_violated,
            'allocation_summary': self.allocation_summary
        }


class BaseOptimizer(ABC):
    """Abstract base class for all allocation optimizers"""
    
    def __init__(self, optimizer_type: OptimizerType, constraints: Optional[AllocationConstraints] = None):
        """
        Initialize optimizer
        
        Args:
            optimizer_type: Type of optimizer
            constraints: Allocation constraints and business rules
        """
        self.optimizer_type = optimizer_type
        self.constraints = constraints or AllocationConstraints()
        self.allocation_history: List[AllocationResult] = []
        
    @abstractmethod
    def optimize(self, allocation_data: pd.DataFrame, **kwargs) -> AllocationResult:
        """
        Perform allocation optimization
        
        Args:
            allocation_data: DataFrame with allocation records
            **kwargs: Additional optimization parameters
            
        Returns:
            AllocationResult object
        """
        pass
    
    def validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input allocation data
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If data is invalid
        """
        # Check for empty data first
        if df.empty:
            raise ValueError("Input data cannot be empty")
        
        required_columns = [
            'dc_id', 'sku_id', 'current_inventory', 'forecasted_demand',
            'dc_priority', 'customer_tier', 'sla_level'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for negative values
        numeric_columns = ['current_inventory', 'forecasted_demand']
        for col in numeric_columns:
            if col in df.columns:
                negative_values = df[df[col] < 0]
                if not negative_values.empty:
                    logger.warning(f"Found {len(negative_values)} negative values in {col}")
        
        logger.info(f"Validated input data: {len(df)} records, {len(df.columns)} columns")
    
    def apply_constraints(self, allocations: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business constraints to allocations
        
        Args:
            allocations: DataFrame with proposed allocations
            
        Returns:
            DataFrame with constrained allocations
        """
        constrained = allocations.copy()
        
        # Apply minimum allocation constraint
        if self.constraints.min_allocation > 0:
            constrained['allocated_quantity'] = np.maximum(
                constrained['allocated_quantity'], 
                self.constraints.min_allocation
            )
        
        # Apply maximum allocation constraint
        if self.constraints.max_allocation is not None:
            constrained['allocated_quantity'] = np.minimum(
                constrained['allocated_quantity'],
                self.constraints.max_allocation
            )
        
        # Respect minimum order quantities
        if 'min_order_quantity' in constrained.columns:
            mask = constrained['allocated_quantity'] > 0
            constrained.loc[mask, 'allocated_quantity'] = np.maximum(
                constrained.loc[mask, 'allocated_quantity'],
                constrained.loc[mask, 'min_order_quantity']
            )
        
        # Apply safety stock buffer
        if 'current_inventory' in constrained.columns:
            max_available = constrained['current_inventory'] * (1 - self.constraints.safety_stock_buffer)
            constrained['allocated_quantity'] = np.minimum(
                constrained['allocated_quantity'],
                max_available
            )
        
        return constrained
    
    def calculate_allocation_efficiency(self, allocated: float, demand: float) -> float:
        """
        Calculate allocation efficiency percentage
        
        Args:
            allocated: Total allocated quantity
            demand: Total demand quantity
            
        Returns:
            Efficiency percentage (0-100)
        """
        if demand <= 0:
            return 100.0 if allocated == 0 else 0.0
        
        return min(100.0, (allocated / demand) * 100.0)
    
    def generate_allocation_summary(self, result: AllocationResult) -> Dict[str, Any]:
        """
        Generate summary statistics for allocation result
        
        Args:
            result: Allocation result
            
        Returns:
            Summary dictionary
        """
        allocations = result.allocations
        
        summary = {
            'total_records': len(allocations),
            'unique_dcs': allocations['dc_id'].nunique() if 'dc_id' in allocations else 0,
            'unique_skus': allocations['sku_id'].nunique() if 'sku_id' in allocations else 0,
            'unique_customers': allocations['customer_id'].nunique() if 'customer_id' in allocations else 0,
            'avg_allocation_per_record': float(result.total_allocated / len(allocations)) if len(allocations) > 0 else 0.0,
            'allocation_distribution': {},
            'top_allocated_dcs': [],
            'constraint_violations': len(result.constraints_violated)
        }
        
        # Allocation distribution by DC priority
        if 'dc_priority' in allocations.columns and 'allocated_quantity' in allocations.columns:
            priority_summary = allocations.groupby('dc_priority')['allocated_quantity'].sum().to_dict()
            summary['allocation_distribution']['by_priority'] = {
                str(k): float(v) for k, v in priority_summary.items()
            }
        
        # Top allocated DCs
        if 'dc_id' in allocations.columns and 'allocated_quantity' in allocations.columns:
            top_dcs = (allocations.groupby('dc_id')['allocated_quantity']
                      .sum()
                      .sort_values(ascending=False)
                      .head(5)
                      .to_dict())
            summary['top_allocated_dcs'] = [
                {'dc_id': str(dc), 'total_allocated': float(qty)} 
                for dc, qty in top_dcs.items()
            ]
        
        return summary
    
    def validate_allocation_result(self, result: AllocationResult) -> List[str]:
        """
        Validate allocation result for business rule violations
        
        Args:
            result: Allocation result to validate
            
        Returns:
            List of validation errors/warnings
        """
        violations = []
        allocations = result.allocations
        
        # Check for negative allocations
        if 'allocated_quantity' in allocations.columns:
            negative_allocs = allocations[allocations['allocated_quantity'] < 0]
            if not negative_allocs.empty:
                violations.append(f"Found {len(negative_allocs)} negative allocations")
        
        # Check inventory constraints
        if 'current_inventory' in allocations.columns and 'allocated_quantity' in allocations.columns:
            over_allocated = allocations[
                allocations['allocated_quantity'] > allocations['current_inventory']
            ]
            if not over_allocated.empty:
                violations.append(f"Found {len(over_allocated)} over-allocated records (allocated > inventory)")
        
        # Check demand fulfillment
        efficiency_threshold = 50.0  # Warn if less than 50% efficiency
        if result.allocation_efficiency < efficiency_threshold:
            violations.append(
                f"Low allocation efficiency: {result.allocation_efficiency:.1f}% "
                f"(threshold: {efficiency_threshold}%)"
            )
        
        # Check for extremely unbalanced allocations
        if 'allocated_quantity' in allocations.columns and len(allocations) > 1:
            alloc_std = allocations['allocated_quantity'].std()
            alloc_mean = allocations['allocated_quantity'].mean()
            
            if alloc_mean > 0 and (alloc_std / alloc_mean) > 2.0:  # High coefficient of variation
                violations.append("Highly unbalanced allocation distribution detected")
        
        return violations
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """
        Get optimizer information and configuration
        
        Returns:
            Dictionary with optimizer information
        """
        return {
            'optimizer_type': self.optimizer_type.value,
            'constraints': {
                'min_allocation': self.constraints.min_allocation,
                'max_allocation': self.constraints.max_allocation,
                'min_order_quantity': self.constraints.min_order_quantity,
                'safety_stock_buffer': self.constraints.safety_stock_buffer,
                'allow_substitution': self.constraints.allow_substitution,
                'max_substitution_ratio': self.constraints.max_substitution_ratio,
                'respect_customer_tier': self.constraints.respect_customer_tier,
                'respect_sla_levels': self.constraints.respect_sla_levels
            },
            'allocation_history_count': len(self.allocation_history)
        }
    
    def add_to_history(self, result: AllocationResult) -> None:
        """
        Add allocation result to history
        
        Args:
            result: Allocation result to add
        """
        self.allocation_history.append(result)
        
        # Keep only last 100 results to prevent memory issues
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]
    
    def get_allocation_history(self, limit: int = 10) -> List[AllocationResult]:
        """
        Get recent allocation history
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent allocation results
        """
        return self.allocation_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear allocation history"""
        self.allocation_history.clear()
        logger.info("Allocation history cleared")


class OptimizerFactory:
    """Factory for creating optimizer instances"""
    
    @staticmethod
    def create_optimizer(optimizer_type: OptimizerType, 
                        constraints: Optional[AllocationConstraints] = None) -> BaseOptimizer:
        """
        Create optimizer instance based on type
        
        Args:
            optimizer_type: Type of optimizer to create
            constraints: Allocation constraints
            
        Returns:
            Optimizer instance
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        if optimizer_type == OptimizerType.PRIORITY:
            from .priority_optimizer import PriorityOptimizer
            return PriorityOptimizer(constraints)
        
        elif optimizer_type == OptimizerType.FAIR_SHARE:
            from .fair_share_optimizer import FairShareOptimizer
            return FairShareOptimizer(constraints)
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_available_optimizers() -> List[OptimizerType]:
        """
        Get list of available optimizer types
        
        Returns:
            List of available optimizer types
        """
        return [OptimizerType.PRIORITY, OptimizerType.FAIR_SHARE]


class OptimizerRegistry:
    """Registry for managing optimizer instances"""
    
    def __init__(self):
        """Initialize optimizer registry"""
        self._optimizers: Dict[OptimizerType, BaseOptimizer] = {}
    
    def register_optimizer(self, optimizer: BaseOptimizer) -> None:
        """
        Register an optimizer instance
        
        Args:
            optimizer: Optimizer instance to register
        """
        self._optimizers[optimizer.optimizer_type] = optimizer
        logger.info(f"Registered optimizer: {optimizer.optimizer_type.value}")
    
    def get_optimizer(self, optimizer_type: OptimizerType) -> Optional[BaseOptimizer]:
        """
        Get optimizer instance by type
        
        Args:
            optimizer_type: Type of optimizer
            
        Returns:
            Optimizer instance or None if not found
        """
        return self._optimizers.get(optimizer_type)
    
    def create_and_register(self, optimizer_type: OptimizerType, 
                          constraints: Optional[AllocationConstraints] = None) -> BaseOptimizer:
        """
        Create and register optimizer instance
        
        Args:
            optimizer_type: Type of optimizer to create
            constraints: Allocation constraints
            
        Returns:
            Created optimizer instance
        """
        optimizer = OptimizerFactory.create_optimizer(optimizer_type, constraints)
        self.register_optimizer(optimizer)
        return optimizer
    
    def list_registered_optimizers(self) -> List[OptimizerType]:
        """
        List all registered optimizer types
        
        Returns:
            List of registered optimizer types
        """
        return list(self._optimizers.keys())
    
    def clear_registry(self) -> None:
        """Clear all registered optimizers"""
        self._optimizers.clear()
        logger.info("Optimizer registry cleared")


# Global optimizer registry instance
optimizer_registry = OptimizerRegistry()