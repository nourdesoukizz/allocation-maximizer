"""
Priority-based allocation optimizer - allocates based on DC priority rankings
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import time

from .base_optimizer import BaseOptimizer, OptimizerType, AllocationResult, AllocationConstraints

logger = logging.getLogger(__name__)


class PriorityOptimizer(BaseOptimizer):
    """
    Priority-based allocation optimizer
    
    Allocates inventory based on DC priority rankings:
    - Higher priority DCs get allocated first
    - Within same priority, allocates based on demand ratios
    - Respects customer tiers and SLA levels
    - Handles remaining inventory redistribution
    """
    
    def __init__(self, constraints: Optional[AllocationConstraints] = None):
        """
        Initialize priority optimizer
        
        Args:
            constraints: Allocation constraints and business rules
        """
        super().__init__(OptimizerType.PRIORITY, constraints)
        self.priority_weights = {
            1: 1.0,   # Highest priority
            2: 0.8,
            3: 0.6, 
            4: 0.4,
            5: 0.2    # Lowest priority
        }
    
    def optimize(self, allocation_data: pd.DataFrame, **kwargs) -> AllocationResult:
        """
        Perform priority-based allocation optimization
        
        Args:
            allocation_data: DataFrame with allocation records
            **kwargs: Additional optimization parameters
                - respect_customer_tier: bool (default True)
                - allow_overflow: bool (default False) 
                - max_iterations: int (default 5)
            
        Returns:
            AllocationResult object
        """
        start_time = time.time()
        
        # Validate input data
        self.validate_input_data(allocation_data)
        
        logger.info(f"Starting priority-based optimization for {len(allocation_data)} records")
        
        # Initialize working dataframe
        df = allocation_data.copy()
        df['allocated_quantity'] = 0.0
        df['allocation_round'] = 0
        df['priority_weight'] = df['dc_priority'].map(self.priority_weights).fillna(0.1)
        
        # Extract parameters
        respect_customer_tier = kwargs.get('respect_customer_tier', self.constraints.respect_customer_tier)
        allow_overflow = kwargs.get('allow_overflow', False)
        max_iterations = kwargs.get('max_iterations', 5)
        
        substitutions_made = []
        constraints_violated = []
        
        try:
            # Phase 1: Priority-based allocation
            df = self._allocate_by_priority(df, respect_customer_tier)
            
            # Phase 2: Handle remaining inventory
            df = self._redistribute_remaining_inventory(df, max_iterations)
            
            # Phase 3: Apply business constraints
            df = self.apply_constraints(df)
            
            # Phase 4: Handle substitutions if enabled
            if self.constraints.allow_substitution:
                df, substitutions = self._handle_substitutions(df)
                substitutions_made.extend(substitutions)
            
            # Calculate results
            total_allocated = df['allocated_quantity'].sum()
            total_demand = df['forecasted_demand'].sum()
            allocation_efficiency = self.calculate_allocation_efficiency(total_allocated, total_demand)
            unallocated_demand = total_demand - total_allocated
            
            # Create result
            result = AllocationResult(
                optimizer_type=self.optimizer_type,
                allocations=df,
                total_allocated=total_allocated,
                total_demand=total_demand,
                allocation_efficiency=allocation_efficiency,
                unallocated_demand=unallocated_demand,
                substitutions_made=substitutions_made,
                optimization_time=time.time() - start_time,
                constraints_violated=constraints_violated,
                allocation_summary={}
            )
            
            # Validate result and generate summary
            result.constraints_violated = self.validate_allocation_result(result)
            result.allocation_summary = self.generate_allocation_summary(result)
            
            # Add to history
            self.add_to_history(result)
            
            logger.info(
                f"Priority optimization completed: {allocation_efficiency:.1f}% efficiency, "
                f"{total_allocated:.0f} allocated, {len(substitutions_made)} substitutions"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Priority optimization failed: {e}")
            raise ValueError(f"Priority optimization failed: {e}")
    
    def _allocate_by_priority(self, df: pd.DataFrame, respect_customer_tier: bool) -> pd.DataFrame:
        """
        Allocate inventory based on DC priority levels
        
        Args:
            df: Working dataframe
            respect_customer_tier: Whether to consider customer tier in allocation
            
        Returns:
            Updated dataframe with allocations
        """
        logger.info("Phase 1: Priority-based allocation")
        
        # Sort by priority (lower number = higher priority), then by customer tier if enabled
        sort_columns = ['dc_priority']
        if respect_customer_tier and 'customer_tier' in df.columns:
            sort_columns.append('customer_tier')
        sort_columns.extend(['forecasted_demand'])  # Secondary sort by demand
        
        df_sorted = df.sort_values(sort_columns, ascending=[True, True, False])
        
        # Group by SKU to allocate each SKU separately
        for sku_id in df_sorted['sku_id'].unique():
            sku_data = df_sorted[df_sorted['sku_id'] == sku_id].copy()
            
            # Calculate total available inventory for this SKU
            total_inventory = sku_data['current_inventory'].sum()
            total_demand = sku_data['forecasted_demand'].sum()
            
            logger.debug(f"SKU {sku_id}: {total_inventory:.0f} inventory, {total_demand:.0f} demand")
            
            remaining_inventory = total_inventory
            
            # Allocate by priority groups
            for priority in sorted(sku_data['dc_priority'].unique()):
                priority_records = sku_data[sku_data['dc_priority'] == priority]
                
                if remaining_inventory <= 0:
                    break
                
                # Calculate demand for this priority group
                priority_demand = priority_records['forecasted_demand'].sum()
                
                if priority_demand <= 0:
                    continue
                
                # Allocate proportionally within priority group
                allocation_ratio = min(1.0, remaining_inventory / priority_demand)
                
                for idx in priority_records.index:
                    demand = df.loc[idx, 'forecasted_demand']
                    inventory_available = df.loc[idx, 'current_inventory']
                    
                    # Calculate allocation based on demand ratio and available inventory
                    proposed_allocation = demand * allocation_ratio
                    max_allocation = min(inventory_available, proposed_allocation)
                    
                    # Apply minimum order quantity if specified
                    if 'min_order_quantity' in df.columns:
                        min_qty = df.loc[idx, 'min_order_quantity']
                        if max_allocation > 0 and max_allocation < min_qty:
                            # Only allocate if we can meet minimum, otherwise skip
                            if inventory_available >= min_qty and remaining_inventory >= min_qty:
                                max_allocation = min_qty
                            else:
                                max_allocation = 0
                    
                    allocation = min(max_allocation, remaining_inventory)
                    
                    if allocation > 0:
                        df.loc[idx, 'allocated_quantity'] = allocation
                        df.loc[idx, 'allocation_round'] = 1
                        remaining_inventory -= allocation
                        
                        logger.debug(
                            f"Allocated {allocation:.0f} to DC {df.loc[idx, 'dc_id']} "
                            f"(priority {priority}) for SKU {sku_id}"
                        )
                    
                    if remaining_inventory <= 0:
                        break
        
        allocated_count = len(df[df['allocated_quantity'] > 0])
        total_allocated = df['allocated_quantity'].sum()
        logger.info(f"Priority allocation: {allocated_count} records allocated, {total_allocated:.0f} total")
        
        return df
    
    def _redistribute_remaining_inventory(self, df: pd.DataFrame, max_iterations: int) -> pd.DataFrame:
        """
        Redistribute any remaining unallocated inventory
        
        Args:
            df: Working dataframe 
            max_iterations: Maximum redistribution iterations
            
        Returns:
            Updated dataframe with redistributed allocations
        """
        logger.info("Phase 2: Redistributing remaining inventory")
        
        for iteration in range(max_iterations):
            initial_allocated = df['allocated_quantity'].sum()
            
            # Group by SKU
            for sku_id in df['sku_id'].unique():
                sku_data = df[df['sku_id'] == sku_id].copy()
                
                # Calculate remaining inventory for this SKU
                total_inventory = sku_data['current_inventory'].sum()
                allocated_inventory = sku_data['allocated_quantity'].sum()
                remaining_inventory = total_inventory - allocated_inventory
                
                if remaining_inventory <= 1.0:  # Small threshold to avoid floating point issues
                    continue
                
                # Find records that could take more inventory (have unmet demand)
                unmet_demand_mask = (sku_data['forecasted_demand'] > sku_data['allocated_quantity']) & \
                                   (sku_data['current_inventory'] > sku_data['allocated_quantity'])
                
                redistribution_candidates = sku_data[unmet_demand_mask]
                
                if redistribution_candidates.empty:
                    continue
                
                # Sort by priority and unmet demand ratio
                redistribution_candidates['unmet_demand'] = (
                    redistribution_candidates['forecasted_demand'] - 
                    redistribution_candidates['allocated_quantity']
                )
                
                redistribution_candidates = redistribution_candidates.sort_values([
                    'dc_priority', 'unmet_demand'
                ], ascending=[True, False])
                
                # Redistribute proportionally to unmet demand
                total_unmet_demand = redistribution_candidates['unmet_demand'].sum()
                
                if total_unmet_demand <= 0:
                    continue
                
                for idx in redistribution_candidates.index:
                    if remaining_inventory <= 0:
                        break
                    
                    unmet_demand = redistribution_candidates.loc[idx, 'unmet_demand']
                    current_allocation = df.loc[idx, 'allocated_quantity']
                    available_capacity = df.loc[idx, 'current_inventory'] - current_allocation
                    
                    # Calculate additional allocation
                    demand_ratio = unmet_demand / total_unmet_demand
                    additional_allocation = min(
                        remaining_inventory * demand_ratio,
                        available_capacity,
                        unmet_demand
                    )
                    
                    if additional_allocation > 0.1:  # Threshold to avoid tiny allocations
                        df.loc[idx, 'allocated_quantity'] += additional_allocation
                        df.loc[idx, 'allocation_round'] = iteration + 2
                        remaining_inventory -= additional_allocation
                        
                        logger.debug(
                            f"Redistributed {additional_allocation:.0f} to DC {df.loc[idx, 'dc_id']} "
                            f"for SKU {sku_id} (iteration {iteration + 1})"
                        )
            
            final_allocated = df['allocated_quantity'].sum()
            improvement = final_allocated - initial_allocated
            
            logger.debug(f"Redistribution iteration {iteration + 1}: +{improvement:.0f} allocated")
            
            # Stop if minimal improvement
            if improvement < 1.0:
                break
        
        return df
    
    def _handle_substitutions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Handle SKU substitutions for unmet demand
        
        Args:
            df: Working dataframe
            
        Returns:
            Tuple of (updated dataframe, list of substitutions made)
        """
        logger.info("Phase 3: Handling substitutions")
        substitutions_made = []
        
        if not self.constraints.allow_substitution:
            return df, substitutions_made
        
        # Simple substitution logic - can be enhanced based on business rules
        # For now, substitute within same category if available
        
        # Group by customer and find unmet demand
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id].copy()
            
            # Find records with unmet demand
            unmet_demand_records = customer_data[
                customer_data['forecasted_demand'] > customer_data['allocated_quantity']
            ]
            
            for idx in unmet_demand_records.index:
                unmet_demand = (df.loc[idx, 'forecasted_demand'] - 
                               df.loc[idx, 'allocated_quantity'])
                
                if unmet_demand <= 0:
                    continue
                
                # Look for substitute SKUs (simplified logic)
                original_sku = df.loc[idx, 'sku_id']
                original_category = df.loc[idx, 'sku_category'] if 'sku_category' in df.columns else 'default'
                
                # Find potential substitute SKUs with excess inventory
                potential_substitutes = df[
                    (df['customer_id'] == customer_id) &
                    (df['sku_id'] != original_sku) &
                    (df['allocated_quantity'] < df['current_inventory']) &
                    (df.get('sku_category', 'default') == original_category)
                ]
                
                for sub_idx in potential_substitutes.index:
                    available_for_substitution = (
                        df.loc[sub_idx, 'current_inventory'] - 
                        df.loc[sub_idx, 'allocated_quantity']
                    )
                    
                    max_substitution = min(
                        unmet_demand * self.constraints.max_substitution_ratio,
                        available_for_substitution
                    )
                    
                    if max_substitution > 1.0:  # Minimum viable substitution
                        # Record the substitution
                        substitution = {
                            'customer_id': customer_id,
                            'original_sku': original_sku,
                            'substitute_sku': df.loc[sub_idx, 'sku_id'],
                            'quantity': max_substitution,
                            'dc_id': df.loc[sub_idx, 'dc_id']
                        }
                        substitutions_made.append(substitution)
                        
                        # Update allocations
                        df.loc[sub_idx, 'allocated_quantity'] += max_substitution
                        unmet_demand -= max_substitution
                        
                        logger.debug(
                            f"Substituted {max_substitution:.0f} of {original_sku} "
                            f"with {substitution['substitute_sku']} for customer {customer_id}"
                        )
                        
                        if unmet_demand <= 0:
                            break
        
        logger.info(f"Substitutions completed: {len(substitutions_made)} substitutions made")
        return df, substitutions_made
    
    def get_priority_distribution(self, result: AllocationResult) -> Dict[int, Dict[str, float]]:
        """
        Get allocation distribution by DC priority
        
        Args:
            result: Allocation result
            
        Returns:
            Dictionary with priority-based allocation statistics
        """
        df = result.allocations
        
        if 'dc_priority' not in df.columns or 'allocated_quantity' not in df.columns:
            return {}
        
        distribution = {}
        
        for priority in sorted(df['dc_priority'].unique()):
            priority_data = df[df['dc_priority'] == priority]
            
            distribution[priority] = {
                'total_allocated': float(priority_data['allocated_quantity'].sum()),
                'total_demand': float(priority_data['forecasted_demand'].sum()),
                'record_count': len(priority_data),
                'avg_allocation': float(priority_data['allocated_quantity'].mean()),
                'fulfillment_rate': float(
                    (priority_data['allocated_quantity'].sum() / 
                     priority_data['forecasted_demand'].sum()) * 100
                    if priority_data['forecasted_demand'].sum() > 0 else 0
                )
            }
        
        return distribution
    
    def set_priority_weights(self, weights: Dict[int, float]) -> None:
        """
        Set custom priority weights
        
        Args:
            weights: Dictionary mapping priority levels to weights
        """
        self.priority_weights.update(weights)
        logger.info(f"Updated priority weights: {self.priority_weights}")
    
    def get_priority_weights(self) -> Dict[int, float]:
        """
        Get current priority weights
        
        Returns:
            Dictionary of priority weights
        """
        return self.priority_weights.copy()