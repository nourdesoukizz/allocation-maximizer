"""
CSV Data Service for loading and managing allocation data
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from pydantic import ValidationError

from ..models.data_models import (
    AllocationRecord, 
    AllocationData, 
    DataSummary, 
    ValidationReport
)


logger = logging.getLogger(__name__)


class CSVDataService:
    """Service for loading and managing CSV allocation data"""
    
    def __init__(self, csv_path: str = "data/allocation_data.csv"):
        """
        Initialize CSV data service
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self._data: Optional[AllocationData] = None
        self._last_loaded: Optional[datetime] = None
        self._validation_report: Optional[ValidationReport] = None
    
    def load_data(self, force_reload: bool = False) -> AllocationData:
        """
        Load data from CSV file
        
        Args:
            force_reload: Force reload even if data is already loaded
            
        Returns:
            AllocationData object with all records
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV data is invalid
        """
        if not force_reload and self._data is not None:
            logger.info("Using cached data")
            return self._data
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        logger.info(f"Loading data from {self.csv_path}")
        
        try:
            # Load CSV using pandas for better performance and handling
            df = pd.read_csv(self.csv_path)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Convert DataFrame to list of dictionaries
            records_data = df.to_dict('records')
            
            # Validate and create AllocationRecord objects
            valid_records = []
            validation_errors = []
            
            for idx, record_data in enumerate(records_data):
                try:
                    # Handle NaN values by converting to None
                    cleaned_data = {}
                    for key, value in record_data.items():
                        if pd.isna(value):
                            if key == 'substitution_sku_id':
                                cleaned_data[key] = None
                            else:
                                raise ValueError(f"Missing required field: {key}")
                        else:
                            cleaned_data[key] = value
                    
                    record = AllocationRecord(**cleaned_data)
                    valid_records.append(record)
                    
                except ValidationError as e:
                    validation_errors.append(f"Row {idx + 2}: {e}")
                    logger.warning(f"Validation error in row {idx + 2}: {e}")
                except Exception as e:
                    validation_errors.append(f"Row {idx + 2}: {e}")
                    logger.error(f"Error processing row {idx + 2}: {e}")
            
            if not valid_records:
                raise ValueError("No valid records found in CSV file")
            
            self._data = AllocationData(
                records=valid_records,
                total_records=len(valid_records)
            )
            
            self._last_loaded = datetime.now()
            
            # Create validation report
            self._validation_report = ValidationReport(
                total_records=len(records_data),
                valid_records=len(valid_records),
                invalid_records=len(records_data) - len(valid_records),
                errors=validation_errors
            )
            
            logger.info(f"Successfully loaded {len(valid_records)} valid records")
            
            if validation_errors:
                logger.warning(f"Found {len(validation_errors)} validation errors")
            
            return self._data
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise ValueError(f"Failed to load CSV data: {e}")
    
    def get_data_summary(self) -> DataSummary:
        """
        Generate summary statistics for the loaded data
        
        Returns:
            DataSummary object with statistics
        """
        if self._data is None:
            self.load_data()
        
        records = self._data.records
        
        # Calculate unique counts
        unique_dcs = len(set(r.dc_id for r in records))
        unique_customers = len(set(r.customer_id for r in records))
        unique_skus = len(set(r.sku_id for r in records))
        unique_regions = len(set(r.dc_region for r in records))
        
        # Calculate totals
        total_inventory = sum(r.current_inventory for r in records)
        total_demand = sum(r.forecasted_demand for r in records)
        total_allocated = sum(r.allocated_quantity for r in records)
        
        # Calculate averages
        overall_fulfillment_rate = (total_allocated / total_demand * 100) if total_demand > 0 else 0
        avg_risk_score = sum(r.risk_score for r in records) / len(records)
        avg_margin = sum(r.margin for r in records) / len(records)
        
        # Group by categories
        records_by_tier = {}
        records_by_sla = {}
        records_by_priority = {}
        
        for record in records:
            tier = record.customer_tier.value
            records_by_tier[tier] = records_by_tier.get(tier, 0) + 1
            
            sla = record.sla_level.value
            records_by_sla[sla] = records_by_sla.get(sla, 0) + 1
            
            priority = record.dc_priority
            records_by_priority[priority] = records_by_priority.get(priority, 0) + 1
        
        return DataSummary(
            total_records=len(records),
            unique_dcs=unique_dcs,
            unique_customers=unique_customers,
            unique_skus=unique_skus,
            unique_regions=unique_regions,
            total_inventory=total_inventory,
            total_demand=total_demand,
            total_allocated=total_allocated,
            overall_fulfillment_rate=overall_fulfillment_rate,
            avg_risk_score=avg_risk_score,
            avg_margin=avg_margin,
            records_by_tier=records_by_tier,
            records_by_sla=records_by_sla,
            records_by_priority=records_by_priority
        )
    
    def get_validation_report(self) -> Optional[ValidationReport]:
        """
        Get the validation report from the last data load
        
        Returns:
            ValidationReport or None if no data has been loaded
        """
        return self._validation_report
    
    def filter_records(
        self,
        dc_ids: Optional[List[str]] = None,
        customer_ids: Optional[List[str]] = None,
        sku_ids: Optional[List[str]] = None,
        customer_tiers: Optional[List[str]] = None,
        sla_levels: Optional[List[str]] = None,
        min_priority: Optional[int] = None,
        max_risk_score: Optional[float] = None
    ) -> List[AllocationRecord]:
        """
        Filter allocation records based on criteria
        
        Args:
            dc_ids: List of DC IDs to include
            customer_ids: List of customer IDs to include
            sku_ids: List of SKU IDs to include
            customer_tiers: List of customer tiers to include
            sla_levels: List of SLA levels to include
            min_priority: Minimum DC priority level
            max_risk_score: Maximum risk score
            
        Returns:
            List of filtered AllocationRecord objects
        """
        if self._data is None:
            self.load_data()
        
        filtered_records = self._data.records
        
        if dc_ids:
            filtered_records = [r for r in filtered_records if r.dc_id in dc_ids]
        
        if customer_ids:
            filtered_records = [r for r in filtered_records if r.customer_id in customer_ids]
        
        if sku_ids:
            filtered_records = [r for r in filtered_records if r.sku_id in sku_ids]
        
        if customer_tiers:
            filtered_records = [r for r in filtered_records if r.customer_tier.value in customer_tiers]
        
        if sla_levels:
            filtered_records = [r for r in filtered_records if r.sla_level.value in sla_levels]
        
        if min_priority is not None:
            filtered_records = [r for r in filtered_records if r.dc_priority >= min_priority]
        
        if max_risk_score is not None:
            filtered_records = [r for r in filtered_records if r.risk_score <= max_risk_score]
        
        return filtered_records
    
    def get_records_by_dc(self, dc_id: str) -> List[AllocationRecord]:
        """Get all records for a specific DC"""
        return self.filter_records(dc_ids=[dc_id])
    
    def get_records_by_customer(self, customer_id: str) -> List[AllocationRecord]:
        """Get all records for a specific customer"""
        return self.filter_records(customer_ids=[customer_id])
    
    def get_records_by_sku(self, sku_id: str) -> List[AllocationRecord]:
        """Get all records for a specific SKU"""
        return self.filter_records(sku_ids=[sku_id])
    
    def get_unique_values(self, field: str) -> List[str]:
        """
        Get unique values for a specific field
        
        Args:
            field: Field name to get unique values for
            
        Returns:
            List of unique values
        """
        if self._data is None:
            self.load_data()
        
        if field == 'dc_id':
            return list(set(r.dc_id for r in self._data.records))
        elif field == 'customer_id':
            return list(set(r.customer_id for r in self._data.records))
        elif field == 'sku_id':
            return list(set(r.sku_id for r in self._data.records))
        elif field == 'customer_tier':
            return list(set(r.customer_tier.value for r in self._data.records))
        elif field == 'sla_level':
            return list(set(r.sla_level.value for r in self._data.records))
        elif field == 'dc_region':
            return list(set(r.dc_region for r in self._data.records))
        elif field == 'sku_category':
            return list(set(r.sku_category for r in self._data.records))
        else:
            raise ValueError(f"Unsupported field: {field}")
    
    def is_data_stale(self, max_age_minutes: int = 15) -> bool:
        """
        Check if loaded data is stale
        
        Args:
            max_age_minutes: Maximum age in minutes before data is considered stale
            
        Returns:
            True if data is stale or not loaded
        """
        if self._last_loaded is None:
            return True
        
        age = datetime.now() - self._last_loaded
        return age.total_seconds() > (max_age_minutes * 60)
    
    def refresh_if_stale(self, max_age_minutes: int = 15) -> AllocationData:
        """
        Refresh data if it's stale
        
        Args:
            max_age_minutes: Maximum age in minutes before refresh
            
        Returns:
            Current AllocationData (refreshed if necessary)
        """
        if self.is_data_stale(max_age_minutes):
            logger.info("Data is stale, refreshing...")
            return self.load_data(force_reload=True)
        
        return self._data or self.load_data()


# Global instance for easy access
csv_service = CSVDataService()