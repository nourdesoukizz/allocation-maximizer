"""
Data validation and cleaning utilities
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, date
from collections import defaultdict

from ..models.data_models import AllocationRecord, ValidationReport, CustomerTier, SLALevel


logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and cleaning utilities"""
    
    def __init__(self):
        """Initialize data validator"""
        self.validation_rules = {}
        self.cleaning_stats = defaultdict(int)
    
    def validate_allocation_records(
        self, 
        records: List[AllocationRecord]
    ) -> ValidationReport:
        """
        Comprehensive validation of allocation records
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            ValidationReport with validation results
        """
        if not records:
            return ValidationReport(
                total_records=0,
                valid_records=0,
                invalid_records=0,
                errors=["No records provided for validation"]
            )
        
        errors = []
        warnings = []
        duplicates_found = 0
        missing_values = defaultdict(int)
        
        # Check for duplicates
        seen_combinations = set()
        for idx, record in enumerate(records):
            key = (record.dc_id, record.customer_id, record.sku_id, record.date)
            if key in seen_combinations:
                duplicates_found += 1
                errors.append(f"Duplicate record at index {idx}: {key}")
            else:
                seen_combinations.add(key)
        
        # Business logic validations
        for idx, record in enumerate(records):
            record_errors = self._validate_single_record(record, idx)
            errors.extend(record_errors)
        
        # Data consistency checks
        consistency_errors = self._check_data_consistency(records)
        errors.extend(consistency_errors)
        
        # Statistical validations
        stat_warnings = self._statistical_validations(records)
        warnings.extend(stat_warnings)
        
        valid_records = len(records) if not errors else len(records) - len([e for e in errors if "index" in e])
        invalid_records = len(records) - valid_records
        
        return ValidationReport(
            total_records=len(records),
            valid_records=valid_records,
            invalid_records=invalid_records,
            errors=errors,
            warnings=warnings,
            duplicates_found=duplicates_found,
            missing_values=dict(missing_values)
        )
    
    def _validate_single_record(self, record: AllocationRecord, index: int) -> List[str]:
        """
        Validate a single allocation record
        
        Args:
            record: AllocationRecord to validate
            index: Record index for error reporting
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Business rule validations
        if record.allocated_quantity > record.current_inventory:
            errors.append(
                f"Record {index}: Allocated quantity ({record.allocated_quantity}) "
                f"exceeds current inventory ({record.current_inventory})"
            )
        
        if record.allocated_quantity > record.forecasted_demand:
            # This might be intentional for strategic customers, so just warn
            pass
        
        if record.cost_per_unit >= record.revenue_per_unit:
            errors.append(
                f"Record {index}: Cost per unit ({record.cost_per_unit}) "
                f"should be less than revenue per unit ({record.revenue_per_unit})"
            )
        
        if record.min_order_quantity > record.allocated_quantity > 0:
            errors.append(
                f"Record {index}: Allocated quantity ({record.allocated_quantity}) "
                f"is below minimum order quantity ({record.min_order_quantity})"
            )
        
        if record.safety_stock > record.current_inventory:
            errors.append(
                f"Record {index}: Safety stock ({record.safety_stock}) "
                f"exceeds current inventory ({record.current_inventory})"
            )
        
        # Logical consistency checks
        if record.historical_demand == 0 and record.forecasted_demand > 1000:
            errors.append(
                f"Record {index}: High forecasted demand ({record.forecasted_demand}) "
                f"with zero historical demand seems inconsistent"
            )
        
        # Date validation
        if record.date > date.today():
            errors.append(f"Record {index}: Future date ({record.date}) not allowed")
        
        # Risk score validation
        if record.risk_score > 0.8 and record.dc_priority == 5:
            errors.append(
                f"Record {index}: High risk score ({record.risk_score}) "
                f"inconsistent with highest priority ({record.dc_priority})"
            )
        
        return errors
    
    def _check_data_consistency(self, records: List[AllocationRecord]) -> List[str]:
        """
        Check for data consistency across records
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            List of consistency errors
        """
        errors = []
        
        # Group records for consistency checks
        dc_info = defaultdict(list)
        customer_info = defaultdict(list)
        sku_info = defaultdict(list)
        
        for record in records:
            dc_info[record.dc_id].append(record)
            customer_info[record.customer_id].append(record)
            sku_info[record.sku_id].append(record)
        
        # Check DC consistency
        for dc_id, dc_records in dc_info.items():
            names = set(r.dc_name for r in dc_records)
            locations = set(r.dc_location for r in dc_records)
            regions = set(r.dc_region for r in dc_records)
            priorities = set(r.dc_priority for r in dc_records)
            
            if len(names) > 1:
                errors.append(f"DC {dc_id} has inconsistent names: {names}")
            if len(locations) > 1:
                errors.append(f"DC {dc_id} has inconsistent locations: {locations}")
            if len(regions) > 1:
                errors.append(f"DC {dc_id} has inconsistent regions: {regions}")
            if len(priorities) > 1:
                errors.append(f"DC {dc_id} has inconsistent priorities: {priorities}")
        
        # Check customer consistency
        for customer_id, customer_records in customer_info.items():
            names = set(r.customer_name for r in customer_records)
            tiers = set(r.customer_tier for r in customer_records)
            regions = set(r.customer_region for r in customer_records)
            
            if len(names) > 1:
                errors.append(f"Customer {customer_id} has inconsistent names: {names}")
            if len(tiers) > 1:
                errors.append(f"Customer {customer_id} has inconsistent tiers: {tiers}")
            if len(regions) > 1:
                errors.append(f"Customer {customer_id} has inconsistent regions: {regions}")
        
        # Check SKU consistency
        for sku_id, sku_records in sku_info.items():
            names = set(r.sku_name for r in sku_records)
            categories = set(r.sku_category for r in sku_records)
            
            if len(names) > 1:
                errors.append(f"SKU {sku_id} has inconsistent names: {names}")
            if len(categories) > 1:
                errors.append(f"SKU {sku_id} has inconsistent categories: {categories}")
        
        return errors
    
    def _statistical_validations(self, records: List[AllocationRecord]) -> List[str]:
        """
        Statistical validations and anomaly detection
        
        Args:
            records: List of AllocationRecord objects
            
        Returns:
            List of statistical warnings
        """
        warnings = []
        
        if len(records) < 10:
            warnings.append("Very small dataset - statistical validations may not be reliable")
            return warnings
        
        # Convert to DataFrame for statistical analysis
        data = []
        for record in records:
            data.append({
                'current_inventory': record.current_inventory,
                'forecasted_demand': record.forecasted_demand,
                'historical_demand': record.historical_demand,
                'revenue_per_unit': record.revenue_per_unit,
                'cost_per_unit': record.cost_per_unit,
                'margin': record.margin,
                'risk_score': record.risk_score,
                'allocated_quantity': record.allocated_quantity,
                'fulfillment_rate': record.fulfillment_rate,
                'dc_priority': record.dc_priority
            })
        
        df = pd.DataFrame(data)
        
        # Check for statistical anomalies
        for column in df.select_dtypes(include=[np.number]).columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            if len(outliers) > len(df) * 0.1:  # More than 10% outliers
                warnings.append(
                    f"High number of outliers in {column}: {len(outliers)} "
                    f"({len(outliers)/len(df)*100:.1f}%)"
                )
        
        # Check for suspicious patterns
        zero_inventory_count = (df['current_inventory'] == 0).sum()
        if zero_inventory_count > len(df) * 0.3:
            warnings.append(f"High percentage of zero inventory records: {zero_inventory_count/len(df)*100:.1f}%")
        
        # Check fulfillment rates
        low_fulfillment_count = (df['fulfillment_rate'] < 50).sum()
        if low_fulfillment_count > len(df) * 0.4:
            warnings.append(f"High percentage of low fulfillment rates: {low_fulfillment_count/len(df)*100:.1f}%")
        
        # Check risk distribution
        high_risk_count = (df['risk_score'] > 0.7).sum()
        if high_risk_count > len(df) * 0.2:
            warnings.append(f"High percentage of high-risk records: {high_risk_count/len(df)*100:.1f}%")
        
        return warnings
    
    def clean_data(
        self, 
        records: List[AllocationRecord],
        fix_inconsistencies: bool = True,
        remove_duplicates: bool = True
    ) -> Tuple[List[AllocationRecord], Dict[str, int]]:
        """
        Clean and fix data issues
        
        Args:
            records: List of AllocationRecord objects
            fix_inconsistencies: Whether to fix data inconsistencies
            remove_duplicates: Whether to remove duplicate records
            
        Returns:
            Tuple of (cleaned_records, cleaning_stats)
        """
        if not records:
            return records, {}
        
        self.cleaning_stats.clear()
        cleaned_records = records.copy()
        
        if remove_duplicates:
            cleaned_records = self._remove_duplicates(cleaned_records)
        
        if fix_inconsistencies:
            cleaned_records = self._fix_inconsistencies(cleaned_records)
        
        # Fix obvious data errors
        cleaned_records = self._fix_data_errors(cleaned_records)
        
        return cleaned_records, dict(self.cleaning_stats)
    
    def _remove_duplicates(self, records: List[AllocationRecord]) -> List[AllocationRecord]:
        """Remove duplicate records"""
        seen_keys = set()
        unique_records = []
        
        for record in records:
            key = (record.dc_id, record.customer_id, record.sku_id, record.date)
            if key not in seen_keys:
                unique_records.append(record)
                seen_keys.add(key)
            else:
                self.cleaning_stats['duplicates_removed'] += 1
        
        return unique_records
    
    def _fix_inconsistencies(self, records: List[AllocationRecord]) -> List[AllocationRecord]:
        """Fix data inconsistencies"""
        # Group records to find most common values for each entity
        dc_info = defaultdict(lambda: defaultdict(list))
        customer_info = defaultdict(lambda: defaultdict(list))
        sku_info = defaultdict(lambda: defaultdict(list))
        
        # Collect all values for each entity
        for record in records:
            dc_info[record.dc_id]['name'].append(record.dc_name)
            dc_info[record.dc_id]['location'].append(record.dc_location)
            dc_info[record.dc_id]['region'].append(record.dc_region)
            dc_info[record.dc_id]['priority'].append(record.dc_priority)
            
            customer_info[record.customer_id]['name'].append(record.customer_name)
            customer_info[record.customer_id]['tier'].append(record.customer_tier)
            customer_info[record.customer_id]['region'].append(record.customer_region)
            
            sku_info[record.sku_id]['name'].append(record.sku_name)
            sku_info[record.sku_id]['category'].append(record.sku_category)
        
        # Find most common values
        def most_common(lst):
            return max(set(lst), key=lst.count)
        
        dc_canonical = {}
        for dc_id, values in dc_info.items():
            dc_canonical[dc_id] = {
                'name': most_common(values['name']),
                'location': most_common(values['location']),
                'region': most_common(values['region']),
                'priority': most_common(values['priority'])
            }
        
        customer_canonical = {}
        for customer_id, values in customer_info.items():
            customer_canonical[customer_id] = {
                'name': most_common(values['name']),
                'tier': most_common(values['tier']),
                'region': most_common(values['region'])
            }
        
        sku_canonical = {}
        for sku_id, values in sku_info.items():
            sku_canonical[sku_id] = {
                'name': most_common(values['name']),
                'category': most_common(values['category'])
            }
        
        # Apply fixes
        fixed_records = []
        for record in records:
            record_dict = record.dict()
            
            # Fix DC info
            if record.dc_id in dc_canonical:
                canonical = dc_canonical[record.dc_id]
                if record_dict['dc_name'] != canonical['name']:
                    record_dict['dc_name'] = canonical['name']
                    self.cleaning_stats['dc_name_fixed'] += 1
                if record_dict['dc_location'] != canonical['location']:
                    record_dict['dc_location'] = canonical['location']
                    self.cleaning_stats['dc_location_fixed'] += 1
                if record_dict['dc_region'] != canonical['region']:
                    record_dict['dc_region'] = canonical['region']
                    self.cleaning_stats['dc_region_fixed'] += 1
                if record_dict['dc_priority'] != canonical['priority']:
                    record_dict['dc_priority'] = canonical['priority']
                    self.cleaning_stats['dc_priority_fixed'] += 1
            
            # Fix customer info
            if record.customer_id in customer_canonical:
                canonical = customer_canonical[record.customer_id]
                if record_dict['customer_name'] != canonical['name']:
                    record_dict['customer_name'] = canonical['name']
                    self.cleaning_stats['customer_name_fixed'] += 1
                if record_dict['customer_tier'] != canonical['tier']:
                    record_dict['customer_tier'] = canonical['tier']
                    self.cleaning_stats['customer_tier_fixed'] += 1
                if record_dict['customer_region'] != canonical['region']:
                    record_dict['customer_region'] = canonical['region']
                    self.cleaning_stats['customer_region_fixed'] += 1
            
            # Fix SKU info
            if record.sku_id in sku_canonical:
                canonical = sku_canonical[record.sku_id]
                if record_dict['sku_name'] != canonical['name']:
                    record_dict['sku_name'] = canonical['name']
                    self.cleaning_stats['sku_name_fixed'] += 1
                if record_dict['sku_category'] != canonical['category']:
                    record_dict['sku_category'] = canonical['category']
                    self.cleaning_stats['sku_category_fixed'] += 1
            
            fixed_records.append(AllocationRecord(**record_dict))
        
        return fixed_records
    
    def _fix_data_errors(self, records: List[AllocationRecord]) -> List[AllocationRecord]:
        """Fix obvious data errors"""
        fixed_records = []
        
        for record in records:
            record_dict = record.dict()
            
            # Fix negative values that should be positive
            for field in ['current_inventory', 'forecasted_demand', 'historical_demand', 
                         'allocated_quantity', 'min_order_quantity', 'safety_stock']:
                if record_dict[field] < 0:
                    record_dict[field] = 0
                    self.cleaning_stats[f'{field}_negative_fixed'] += 1
            
            # Fix fulfillment rate calculation if inconsistent
            if record.forecasted_demand > 0:
                calculated_rate = (record.allocated_quantity / record.forecasted_demand) * 100
                if abs(calculated_rate - record.fulfillment_rate) > 1.0:
                    record_dict['fulfillment_rate'] = round(calculated_rate, 1)
                    self.cleaning_stats['fulfillment_rate_recalculated'] += 1
            
            # Fix margin calculation if inconsistent
            if record.revenue_per_unit > 0:
                calculated_margin = ((record.revenue_per_unit - record.cost_per_unit) / 
                                   record.revenue_per_unit) * 100
                if abs(calculated_margin - record.margin) > 1.0:
                    record_dict['margin'] = round(calculated_margin, 1)
                    self.cleaning_stats['margin_recalculated'] += 1
            
            fixed_records.append(AllocationRecord(**record_dict))
        
        return fixed_records


# Global validator instance
validator = DataValidator()