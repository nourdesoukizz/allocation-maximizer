"""
Pydantic models for CSV data validation and type safety
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


class CustomerTier(str, Enum):
    """Customer tier enumeration"""
    STRATEGIC = "Strategic"
    PREMIUM = "Premium"
    STANDARD = "Standard"
    BASIC = "Basic"


class SLALevel(str, Enum):
    """SLA level enumeration"""
    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"


class AllocationRecord(BaseModel):
    """Model for a single allocation record from CSV"""
    
    # DC Information
    dc_id: str = Field(..., description="Distribution Center ID")
    dc_name: str = Field(..., description="DC Name")
    dc_location: str = Field(..., description="DC Location")
    dc_region: str = Field(..., description="DC Region")
    dc_priority: int = Field(..., ge=1, le=5, description="Priority level 1-5")
    
    # SKU Information
    sku_id: str = Field(..., description="SKU identifier")
    sku_name: str = Field(..., description="Product name")
    sku_category: str = Field(..., description="Product category")
    
    # Customer Information
    customer_id: str = Field(..., description="Customer identifier")
    customer_name: str = Field(..., description="Customer name")
    customer_tier: CustomerTier = Field(..., description="Customer tier")
    customer_region: str = Field(..., description="Customer region")
    
    # Inventory & Demand
    current_inventory: int = Field(..., ge=0, description="Current stock level")
    forecasted_demand: int = Field(..., ge=0, description="Predicted demand")
    historical_demand: int = Field(..., ge=0, description="Past period demand")
    
    # Financial Information
    revenue_per_unit: float = Field(..., gt=0, description="Revenue per unit")
    cost_per_unit: float = Field(..., gt=0, description="Cost per unit")
    margin: float = Field(..., ge=0, le=100, description="Profit margin %")
    
    # Service & Risk
    sla_level: SLALevel = Field(..., description="Service level agreement")
    risk_score: float = Field(..., ge=0, le=1, description="Risk assessment 0-1")
    substitution_sku_id: Optional[str] = Field(None, description="Alternative SKU")
    
    # Allocation Results
    date: date = Field(..., description="Record date")
    allocated_quantity: int = Field(..., ge=0, description="Allocated amount")
    fulfillment_rate: float = Field(..., ge=0, le=100, description="Fulfillment %")
    lead_time_days: int = Field(..., ge=0, description="Delivery lead time")
    min_order_quantity: int = Field(..., ge=1, description="Minimum order qty")
    safety_stock: int = Field(..., ge=0, description="Safety stock level")
    
    @validator('margin')
    def validate_margin(cls, v, values):
        """Validate margin calculation"""
        if 'revenue_per_unit' in values and 'cost_per_unit' in values:
            revenue = values['revenue_per_unit']
            cost = values['cost_per_unit']
            if revenue > 0:
                calculated_margin = ((revenue - cost) / revenue) * 100
                # Allow 1% tolerance for rounding
                if abs(calculated_margin - v) > 1.0:
                    raise ValueError(f"Margin {v}% doesn't match calculated {calculated_margin:.1f}%")
        return v
    
    @validator('fulfillment_rate')
    def validate_fulfillment_rate(cls, v, values):
        """Validate fulfillment rate calculation"""
        if 'allocated_quantity' in values and 'forecasted_demand' in values:
            allocated = values['allocated_quantity']
            demand = values['forecasted_demand']
            if demand > 0:
                calculated_rate = (allocated / demand) * 100
                # Allow 1% tolerance for rounding
                if abs(calculated_rate - v) > 1.0:
                    raise ValueError(f"Fulfillment rate {v}% doesn't match calculated {calculated_rate:.1f}%")
        return v

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True


class AllocationData(BaseModel):
    """Container for all allocation records"""
    
    records: list[AllocationRecord] = Field(..., description="List of allocation records")
    total_records: int = Field(..., description="Total number of records")
    
    @validator('total_records')
    def validate_total_records(cls, v, values):
        """Ensure total_records matches actual record count"""
        if 'records' in values and len(values['records']) != v:
            raise ValueError(f"Total records {v} doesn't match actual count {len(values['records'])}")
        return v


class DataSummary(BaseModel):
    """Summary statistics for the allocation data"""
    
    total_records: int
    unique_dcs: int
    unique_customers: int
    unique_skus: int
    unique_regions: int
    
    total_inventory: int
    total_demand: int
    total_allocated: int
    overall_fulfillment_rate: float
    
    avg_risk_score: float
    avg_margin: float
    
    records_by_tier: dict[str, int]
    records_by_sla: dict[str, int]
    records_by_priority: dict[int, int]


class ValidationReport(BaseModel):
    """Report of data validation results"""
    
    total_records: int
    valid_records: int
    invalid_records: int
    
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    duplicates_found: int = 0
    missing_values: dict[str, int] = Field(default_factory=dict)
    
    @property
    def validation_success_rate(self) -> float:
        """Calculate validation success rate"""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self.invalid_records == 0