"""
Fair share allocation optimizer - distributes proportionally based on demand ratios
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import time

from .base_optimizer import BaseOptimizer, OptimizerType, AllocationResult, AllocationConstraints

logger = logging.getLogger(__name__)


class FairShareOptimizer(BaseOptimizer):
    """
    Fair share allocation optimizer
    
    Distributes inventory proportionally based on demand ratios:
    - All DCs get proportional allocation based on their demand
    - No preferential treatment by priority
    - Respects customer tiers and SLA levels if configured
    - Redistributes remaining inventory fairly
    - Balances allocations across all participants
    """
    
    def __init__(self, constraints: Optional[AllocationConstraints] = None):
        """
        Initialize fair share optimizer
        
        Args:
            constraints: Allocation constraints and business rules
        """
        super().__init__(OptimizerType.FAIR_SHARE, constraints)
        self.fairness_threshold = 0.05  # 5% threshold for fairness
        
    def optimize(self, allocation_data: pd.DataFrame, **kwargs) -> AllocationResult:
        """
        Perform fair share allocation optimization
        
        Args:
            allocation_data: DataFrame with allocation records
            **kwargs: Additional optimization parameters
                - fairness_weight: float (0-1, default 0.7) - Balance between demand satisfaction and fairness
                - min_allocation_threshold: float (default 1.0) - Minimum viable allocation
                - max_rebalancing_iterations: int (default 10) - Maximum rebalancing rounds
            
        Returns:
            AllocationResult object
        """
        start_time = time.time()
        
        # Validate input data
        self.validate_input_data(allocation_data)
        
        logger.info(f"Starting fair share optimization for {len(allocation_data)} records")
        
        # Initialize working dataframe
        df = allocation_data.copy()
        df['allocated_quantity'] = 0.0
        df['allocation_round'] = 0
        df['fairness_ratio'] = 0.0  # Track fairness for each allocation
        
        # Extract parameters
        fairness_weight = kwargs.get('fairness_weight', 0.7)
        min_allocation_threshold = kwargs.get('min_allocation_threshold', 1.0)
        max_rebalancing_iterations = kwargs.get('max_rebalancing_iterations', 10)
        
        substitutions_made = []
        constraints_violated = []
        
        try:
            # Phase 1: Initial proportional allocation
            df = self._allocate_proportionally(df, fairness_weight)
            
            # Phase 2: Rebalance for fairness
            df = self._rebalance_for_fairness(df, max_rebalancing_iterations)
            
            # Phase 3: Handle remaining inventory fairly
            df = self._redistribute_remaining_fairly(df)
            
            # Phase 4: Apply business constraints
            df = self.apply_constraints(df)
            
            # Phase 5: Handle substitutions if enabled
            if self.constraints.allow_substitution:
                df, substitutions = self._handle_fair_substitutions(df)
                substitutions_made.extend(substitutions)
            
            # Phase 6: Final fairness adjustment
            df = self._final_fairness_adjustment(df, min_allocation_threshold)
            
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
            
            # Add fairness metrics to summary
            result.allocation_summary.update(self._calculate_fairness_metrics(result))
            
            # Add to history
            self.add_to_history(result)
            
            logger.info(
                f"Fair share optimization completed: {allocation_efficiency:.1f}% efficiency, "
                f"{total_allocated:.0f} allocated, {len(substitutions_made)} substitutions"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fair share optimization failed: {e}")
            raise ValueError(f"Fair share optimization failed: {e}")
    
    def _allocate_proportionally(self, df: pd.DataFrame, fairness_weight: float) -> pd.DataFrame:
        """
        Initial proportional allocation based on demand ratios
        
        Args:
            df: Working dataframe
            fairness_weight: Weight for fairness vs efficiency (0-1)
            
        Returns:
            Updated dataframe with proportional allocations
        """
        logger.info("Phase 1: Initial proportional allocation")
        
        # Group by SKU to allocate each SKU separately
        for sku_id in df['sku_id'].unique():
            sku_data = df[df['sku_id'] == sku_id].copy()
            
            # Calculate total available inventory and demand for this SKU
            total_inventory = sku_data['current_inventory'].sum()
            total_demand = sku_data['forecasted_demand'].sum()
            
            logger.debug(f"SKU {sku_id}: {total_inventory:.0f} inventory, {total_demand:.0f} demand")
            
            if total_demand <= 0:
                continue
            
            # Calculate initial allocation ratios
            if total_demand <= total_inventory:
                # Sufficient inventory - allocate full demand
                for idx in sku_data.index:
                    demand = df.loc[idx, 'forecasted_demand']
                    inventory_available = df.loc[idx, 'current_inventory']
                    allocation = min(demand, inventory_available)
                    
                    df.loc[idx, 'allocated_quantity'] = allocation
                    df.loc[idx, 'allocation_round'] = 1
                    df.loc[idx, 'fairness_ratio'] = 1.0 if demand > 0 else 0.0
                    
            else:
                # Insufficient inventory - allocate proportionally
                allocation_ratio = total_inventory / total_demand
                
                # Apply fairness weighting
                fair_allocation_ratio = allocation_ratio * fairness_weight
                remaining_ratio = allocation_ratio - fair_allocation_ratio
                
                for idx in sku_data.index:
                    demand = df.loc[idx, 'forecasted_demand']
                    inventory_available = df.loc[idx, 'current_inventory']
                    demand_ratio = demand / total_demand if total_demand > 0 else 0
                    
                    # Calculate fair share allocation
                    fair_share = demand * fair_allocation_ratio
                    
                    # Add proportional share of remaining
                    proportional_share = (total_inventory * remaining_ratio) * demand_ratio
                    
                    total_proposed = fair_share + proportional_share
                    
                    # Constrain by available inventory
                    allocation = min(total_proposed, inventory_available, demand)
                    
                    if allocation > 0:
                        df.loc[idx, 'allocated_quantity'] = allocation
                        df.loc[idx, 'allocation_round'] = 1
                        df.loc[idx, 'fairness_ratio'] = allocation / demand if demand > 0 else 0.0
                        
                        logger.debug(
                            f"Allocated {allocation:.0f} to DC {df.loc[idx, 'dc_id']} "
                            f"for SKU {sku_id} (ratio: {allocation/demand:.2f})"
                        )
        
        allocated_count = len(df[df['allocated_quantity'] > 0])
        total_allocated = df['allocated_quantity'].sum()
        logger.info(f"Proportional allocation: {allocated_count} records allocated, {total_allocated:.0f} total")
        
        return df
    
    def _rebalance_for_fairness(self, df: pd.DataFrame, max_iterations: int) -> pd.DataFrame:
        """
        Rebalance allocations to improve fairness across DCs
        
        Args:
            df: Working dataframe
            max_iterations: Maximum rebalancing iterations
            
        Returns:
            Updated dataframe with rebalanced allocations
        """
        logger.info("Phase 2: Rebalancing for fairness")
        
        for iteration in range(max_iterations):
            initial_fairness = self._calculate_fairness_score(df)
            rebalanced = False
            
            # Group by SKU
            for sku_id in df['sku_id'].unique():
                sku_data = df[df['sku_id'] == sku_id].copy()
                
                if len(sku_data) < 2:  # Need at least 2 records to rebalance
                    continue
                
                # Calculate fairness ratios for this SKU
                sku_data['fairness_ratio'] = np.where(
                    sku_data['forecasted_demand'] > 0,
                    sku_data['allocated_quantity'] / sku_data['forecasted_demand'],
                    0.0
                )
                
                # Find most and least fairly allocated records
                max_ratio_idx = sku_data['fairness_ratio'].idxmax()
                min_ratio_idx = sku_data['fairness_ratio'].idxmin()
                
                max_ratio = sku_data.loc[max_ratio_idx, 'fairness_ratio']
                min_ratio = sku_data.loc[min_ratio_idx, 'fairness_ratio']
                
                # Check if rebalancing is needed
                fairness_gap = max_ratio - min_ratio
                if fairness_gap > self.fairness_threshold:
                    
                    # Calculate rebalancing amount
                    over_allocated = df.loc[max_ratio_idx, 'allocated_quantity']
                    under_allocated_demand = df.loc[min_ratio_idx, 'forecasted_demand']
                    under_allocated_current = df.loc[min_ratio_idx, 'allocated_quantity']
                    under_allocated_capacity = (
                        df.loc[min_ratio_idx, 'current_inventory'] - under_allocated_current
                    )
                    
                    # Calculate optimal transfer amount
                    target_ratio = (max_ratio + min_ratio) / 2
                    
                    # Amount to transfer from over-allocated
                    from_amount = max(0, over_allocated - (under_allocated_demand * target_ratio))
                    
                    # Amount under-allocated can receive
                    to_amount = min(
                        under_allocated_capacity,
                        (under_allocated_demand * target_ratio) - under_allocated_current,
                        from_amount
                    )
                    
                    if to_amount > 1.0:  # Minimum transfer threshold
                        # Perform the transfer
                        df.loc[max_ratio_idx, 'allocated_quantity'] -= to_amount
                        df.loc[min_ratio_idx, 'allocated_quantity'] += to_amount
                        df.loc[max_ratio_idx, 'allocation_round'] = iteration + 2
                        df.loc[min_ratio_idx, 'allocation_round'] = iteration + 2
                        
                        rebalanced = True
                        
                        logger.debug(
                            f"Rebalanced {to_amount:.0f} from DC {df.loc[max_ratio_idx, 'dc_id']} "
                            f"to DC {df.loc[min_ratio_idx, 'dc_id']} for SKU {sku_id}"
                        )
            
            final_fairness = self._calculate_fairness_score(df)
            improvement = final_fairness - initial_fairness
            
            logger.debug(f"Rebalancing iteration {iteration + 1}: fairness improved by {improvement:.3f}")
            
            # Stop if no rebalancing occurred or minimal improvement
            if not rebalanced or improvement < 0.001:
                break
        
        return df
    
    def _redistribute_remaining_fairly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Redistribute any remaining inventory maintaining fairness
        
        Args:
            df: Working dataframe
            
        Returns:
            Updated dataframe with redistributed allocations
        """
        logger.info("Phase 3: Redistributing remaining inventory fairly")
        
        # Group by SKU
        for sku_id in df['sku_id'].unique():
            sku_data = df[df['sku_id'] == sku_id].copy()
            
            # Calculate remaining inventory
            total_inventory = sku_data['current_inventory'].sum()
            allocated_inventory = sku_data['allocated_quantity'].sum()
            remaining_inventory = total_inventory - allocated_inventory
            
            if remaining_inventory <= 1.0:
                continue
            
            # Find records with unmet demand and capacity
            sku_data['unmet_demand'] = np.maximum(0, 
                sku_data['forecasted_demand'] - sku_data['allocated_quantity']
            )
            sku_data['available_capacity'] = np.maximum(0,
                sku_data['current_inventory'] - sku_data['allocated_quantity']
            )
            
            # Filter to viable candidates
            candidates = sku_data[
                (sku_data['unmet_demand'] > 0) & 
                (sku_data['available_capacity'] > 0)
            ]
            
            if candidates.empty:
                continue
            
            # Distribute remaining inventory fairly based on unmet demand ratios
            total_unmet_demand = candidates['unmet_demand'].sum()
            
            if total_unmet_demand <= 0:
                continue
            
            for idx in candidates.index:
                unmet_demand = candidates.loc[idx, 'unmet_demand']
                available_capacity = candidates.loc[idx, 'available_capacity']
                
                # Fair share of remaining inventory
                demand_ratio = unmet_demand / total_unmet_demand
                fair_share = remaining_inventory * demand_ratio
                
                # Constrain by capacity and demand
                additional_allocation = min(fair_share, available_capacity, unmet_demand)
                
                if additional_allocation > 0.1:
                    df.loc[idx, 'allocated_quantity'] += additional_allocation
                    remaining_inventory -= additional_allocation
                    
                    logger.debug(
                        f"Redistributed {additional_allocation:.0f} to DC {df.loc[idx, 'dc_id']} "
                        f"for SKU {sku_id}"
                    )
                
                if remaining_inventory <= 0:
                    break
        
        return df
    
    def _handle_fair_substitutions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Handle substitutions while maintaining fairness
        
        Args:
            df: Working dataframe
            
        Returns:
            Tuple of (updated dataframe, list of substitutions made)
        """
        logger.info("Phase 4: Handling fair substitutions")
        substitutions_made = []
        
        if not self.constraints.allow_substitution:
            return df, substitutions_made
        
        # Group by customer and maintain fairness across substitutions
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id].copy()
            
            # Calculate overall fulfillment ratio for this customer
            total_demand = customer_data['forecasted_demand'].sum()
            total_allocated = customer_data['allocated_quantity'].sum()
            
            if total_demand <= 0:
                continue
            
            current_fulfillment_ratio = total_allocated / total_demand
            
            # Find records with unmet demand
            unmet_records = customer_data[
                customer_data['forecasted_demand'] > customer_data['allocated_quantity']
            ]
            
            # Find potential substitute sources
            potential_sources = customer_data[
                customer_data['allocated_quantity'] < customer_data['current_inventory']
            ]
            
            for unmet_idx in unmet_records.index:
                unmet_demand = (df.loc[unmet_idx, 'forecasted_demand'] - 
                               df.loc[unmet_idx, 'allocated_quantity'])
                
                if unmet_demand <= 0:
                    continue
                
                original_sku = df.loc[unmet_idx, 'sku_id']
                
                # Look for substitution sources maintaining fairness
                for source_idx in potential_sources.index:
                    if source_idx == unmet_idx:
                        continue
                    
                    source_sku = df.loc[source_idx, 'sku_id']
                    if source_sku == original_sku:  # Skip same SKU
                        continue
                    
                    # Check substitution viability
                    available_for_substitution = (
                        df.loc[source_idx, 'current_inventory'] - 
                        df.loc[source_idx, 'allocated_quantity']
                    )
                    
                    # Limit substitution to maintain fairness
                    max_substitution = min(
                        unmet_demand * self.constraints.max_substitution_ratio,
                        available_for_substitution * 0.5,  # Conservative substitution
                        unmet_demand
                    )
                    
                    if max_substitution > 1.0:
                        # Record substitution
                        substitution = {
                            'customer_id': customer_id,
                            'original_sku': original_sku,
                            'substitute_sku': source_sku,
                            'quantity': max_substitution,
                            'dc_id': df.loc[source_idx, 'dc_id'],
                            'fairness_maintained': True
                        }
                        substitutions_made.append(substitution)
                        
                        # Update allocation
                        df.loc[source_idx, 'allocated_quantity'] += max_substitution
                        unmet_demand -= max_substitution
                        
                        logger.debug(
                            f"Fair substitution: {max_substitution:.0f} of {original_sku} "
                            f"with {source_sku} for customer {customer_id}"
                        )
                        
                        if unmet_demand <= 0:
                            break
        
        logger.info(f"Fair substitutions completed: {len(substitutions_made)} substitutions made")
        return df, substitutions_made
    
    def _final_fairness_adjustment(self, df: pd.DataFrame, min_threshold: float) -> pd.DataFrame:
        """
        Final adjustment to maximize fairness
        
        Args:
            df: Working dataframe
            min_threshold: Minimum allocation threshold
            
        Returns:
            Updated dataframe with final fairness adjustments
        """
        logger.info("Phase 5: Final fairness adjustment")
        
        # Remove allocations below minimum threshold and redistribute
        small_allocations = df[
            (df['allocated_quantity'] > 0) & 
            (df['allocated_quantity'] < min_threshold)
        ]
        
        if not small_allocations.empty:
            logger.debug(f"Removing {len(small_allocations)} small allocations below threshold")
            
            # Collect small allocation quantities for redistribution
            for idx in small_allocations.index:
                small_qty = df.loc[idx, 'allocated_quantity']
                sku_id = df.loc[idx, 'sku_id']
                
                # Remove small allocation
                df.loc[idx, 'allocated_quantity'] = 0.0
                
                # Find other records for this SKU that can absorb the quantity
                sku_records = df[
                    (df['sku_id'] == sku_id) & 
                    (df['allocated_quantity'] > 0) &
                    (df['allocated_quantity'] + small_qty <= df['current_inventory'])
                ].index
                
                if len(sku_records) > 0:
                    # Distribute the small quantity proportionally
                    redistribution_per_record = small_qty / len(sku_records)
                    for redistribute_idx in sku_records:
                        df.loc[redistribute_idx, 'allocated_quantity'] += redistribution_per_record
        
        # Calculate final fairness scores
        for sku_id in df['sku_id'].unique():
            sku_mask = df['sku_id'] == sku_id
            sku_data = df[sku_mask]
            
            # Update fairness ratios
            df.loc[sku_mask, 'fairness_ratio'] = np.where(
                sku_data['forecasted_demand'] > 0,
                sku_data['allocated_quantity'] / sku_data['forecasted_demand'],
                0.0
            )
        
        return df
    
    def _calculate_fairness_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall fairness score (higher is better)
        
        Args:
            df: Allocation dataframe
            
        Returns:
            Fairness score between 0 and 1
        """
        if df.empty:
            return 0.0
        
        fairness_scores = []
        
        # Calculate fairness per SKU
        for sku_id in df['sku_id'].unique():
            sku_data = df[df['sku_id'] == sku_id]
            
            if len(sku_data) < 2:
                fairness_scores.append(1.0)  # Perfect fairness for single record
                continue
            
            # Calculate fulfillment ratios
            fulfillment_ratios = []
            for _, row in sku_data.iterrows():
                if row['forecasted_demand'] > 0:
                    ratio = row['allocated_quantity'] / row['forecasted_demand']
                    fulfillment_ratios.append(ratio)
            
            if not fulfillment_ratios:
                fairness_scores.append(1.0)
                continue
            
            # Calculate coefficient of variation (lower is more fair)
            if len(fulfillment_ratios) > 1:
                mean_ratio = np.mean(fulfillment_ratios)
                if mean_ratio > 0:
                    std_ratio = np.std(fulfillment_ratios)
                    cv = std_ratio / mean_ratio
                    fairness_score = max(0.0, 1.0 - cv)  # Convert to fairness score
                else:
                    fairness_score = 1.0
            else:
                fairness_score = 1.0
            
            fairness_scores.append(fairness_score)
        
        return np.mean(fairness_scores) if fairness_scores else 0.0
    
    def _calculate_fairness_metrics(self, result: AllocationResult) -> Dict[str, Any]:
        """
        Calculate detailed fairness metrics
        
        Args:
            result: Allocation result
            
        Returns:
            Dictionary with fairness metrics
        """
        df = result.allocations
        
        fairness_metrics = {
            'overall_fairness_score': self._calculate_fairness_score(df),
            'fairness_by_sku': {},
            'fairness_distribution': {},
            'gini_coefficient': 0.0
        }
        
        # Per-SKU fairness
        for sku_id in df['sku_id'].unique():
            sku_data = df[df['sku_id'] == sku_id]
            
            fulfillment_ratios = []
            for _, row in sku_data.iterrows():
                if row['forecasted_demand'] > 0:
                    ratio = row['allocated_quantity'] / row['forecasted_demand']
                    fulfillment_ratios.append(min(1.0, ratio))  # Cap at 100%
            
            if fulfillment_ratios:
                fairness_metrics['fairness_by_sku'][str(sku_id)] = {
                    'mean_fulfillment_ratio': float(np.mean(fulfillment_ratios)),
                    'std_fulfillment_ratio': float(np.std(fulfillment_ratios)),
                    'min_fulfillment_ratio': float(np.min(fulfillment_ratios)),
                    'max_fulfillment_ratio': float(np.max(fulfillment_ratios)),
                    'coefficient_of_variation': float(np.std(fulfillment_ratios) / np.mean(fulfillment_ratios)) 
                                                if np.mean(fulfillment_ratios) > 0 else 0.0
                }
        
        # Calculate Gini coefficient for allocation distribution
        allocations = df['allocated_quantity'].values
        allocations = allocations[allocations > 0]  # Remove zero allocations
        
        if len(allocations) > 1:
            # Sort allocations
            sorted_allocs = np.sort(allocations)
            n = len(sorted_allocs)
            
            # Calculate Gini coefficient
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_allocs)) / (n * np.sum(sorted_allocs)) - (n + 1) / n
            fairness_metrics['gini_coefficient'] = float(gini)
        
        return fairness_metrics
    
    def get_fairness_report(self, result: AllocationResult) -> str:
        """
        Generate detailed fairness report
        
        Args:
            result: Allocation result
            
        Returns:
            Formatted fairness report
        """
        metrics = self._calculate_fairness_metrics(result)
        
        report = ["=" * 60]
        report.append("FAIR SHARE ALLOCATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("FAIRNESS OVERVIEW")
        report.append("-" * 30)
        report.append(f"Overall Fairness Score: {metrics['overall_fairness_score']:.3f}")
        report.append(f"Gini Coefficient: {metrics['gini_coefficient']:.3f}")
        report.append(f"Total Allocated: {result.total_allocated:.0f}")
        report.append(f"Allocation Efficiency: {result.allocation_efficiency:.1f}%")
        report.append("")
        
        # Per-SKU fairness
        if metrics['fairness_by_sku']:
            report.append("FAIRNESS BY SKU")
            report.append("-" * 30)
            
            for sku_id, sku_metrics in metrics['fairness_by_sku'].items():
                report.append(f"\nSKU {sku_id}:")
                report.append(f"  Mean Fulfillment: {sku_metrics['mean_fulfillment_ratio']:.1%}")
                report.append(f"  Std Dev: {sku_metrics['std_fulfillment_ratio']:.3f}")
                report.append(f"  Range: {sku_metrics['min_fulfillment_ratio']:.1%} - {sku_metrics['max_fulfillment_ratio']:.1%}")
                report.append(f"  CV: {sku_metrics['coefficient_of_variation']:.3f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)