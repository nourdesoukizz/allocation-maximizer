"""
Pydantic schemas for Module 4 API request/response models
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class OptimizationStrategy(str, Enum):
    """Available optimization strategies"""
    PRIORITY_BASED = "priority_based"
    FAIR_SHARE = "fair_share"
    HYBRID = "hybrid"
    AUTO_SELECT = "auto_select"


class AllocationRecord(BaseModel):
    """Individual allocation record"""
    dc_id: str = Field(..., description="Distribution Center ID")
    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    customer_id: str = Field(..., description="Customer ID")
    current_inventory: float = Field(..., ge=0, description="Current inventory quantity")
    forecasted_demand: float = Field(..., ge=0, description="Forecasted demand quantity")
    dc_priority: int = Field(..., ge=1, le=5, description="DC priority (1=highest, 5=lowest)")
    customer_tier: str = Field(..., description="Customer tier (A, B, C)")
    sla_level: str = Field(..., description="Service Level Agreement level")
    min_order_quantity: Optional[float] = Field(1.0, ge=0, description="Minimum order quantity")
    sku_category: Optional[str] = Field(None, description="SKU category for substitution")


class AllocationConstraints(BaseModel):
    """Business constraints for allocation optimization"""
    min_allocation: float = Field(0.0, ge=0, description="Minimum allocation per record")
    max_allocation: Optional[float] = Field(None, ge=0, description="Maximum allocation per record")
    min_order_quantity: float = Field(1.0, ge=0, description="Global minimum order quantity")
    safety_stock_buffer: float = Field(0.1, ge=0, le=1, description="Safety stock buffer (0-1)")
    allow_substitution: bool = Field(True, description="Allow SKU substitution")
    max_substitution_ratio: float = Field(0.3, ge=0, le=1, description="Max substitution ratio")
    respect_customer_tier: bool = Field(True, description="Respect customer tier in allocation")
    respect_sla_levels: bool = Field(True, description="Respect SLA levels in allocation")


class OptimizationRequest(BaseModel):
    """Request model for optimization"""
    strategy: OptimizationStrategy = Field(..., description="Optimization strategy to use")
    allocation_data: List[AllocationRecord] = Field(..., min_items=1, description="Allocation records")
    constraints: Optional[AllocationConstraints] = Field(None, description="Business constraints")
    strategy_params: Optional[Dict[str, Any]] = Field(None, description="Strategy-specific parameters")
    
    # Strategy-specific parameters
    priority_weight: Optional[float] = Field(0.6, ge=0, le=1, description="Priority weight for hybrid strategy")
    fairness_weight: Optional[float] = Field(0.4, ge=0, le=1, description="Fairness weight for hybrid strategy")
    
    # User preferences
    prefer_efficiency: Optional[bool] = Field(True, description="Prefer efficiency over fairness")
    prefer_speed: Optional[bool] = Field(False, description="Prefer speed over optimal results")
    max_execution_time: Optional[int] = Field(60, ge=1, le=300, description="Max execution time in seconds")

    @validator('priority_weight', 'fairness_weight')
    def validate_weights(cls, v, values):
        """Validate that priority and fairness weights sum to 1.0"""
        if 'priority_weight' in values and 'fairness_weight' in values:
            total = values['priority_weight'] + v
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError('Priority weight and fairness weight must sum to 1.0')
        return v


class AllocationResult(BaseModel):
    """Individual allocation result"""
    dc_id: str
    sku_id: str
    customer_id: str
    allocated_quantity: float
    forecasted_demand: float
    current_inventory: float
    allocation_efficiency: float
    allocation_round: Optional[int] = None


class SubstitutionRecord(BaseModel):
    """SKU substitution record"""
    customer_id: str
    original_sku: str
    substitute_sku: str
    quantity: float
    dc_id: str


class OptimizationResponse(BaseModel):
    """Response model for optimization results"""
    success: bool = Field(..., description="Whether optimization was successful")
    strategy_used: OptimizationStrategy = Field(..., description="Strategy that was used")
    total_allocated: float = Field(..., description="Total quantity allocated")
    total_demand: float = Field(..., description="Total demand quantity")
    allocation_efficiency: float = Field(..., description="Overall allocation efficiency (0-100)")
    unallocated_demand: float = Field(..., description="Unallocated demand quantity")
    optimization_time: float = Field(..., description="Optimization execution time in seconds")
    
    allocations: List[AllocationResult] = Field(..., description="Individual allocation results")
    substitutions_made: List[SubstitutionRecord] = Field(default_factory=list, description="SKU substitutions made")
    constraints_violated: List[str] = Field(default_factory=list, description="Business constraint violations")
    
    allocation_summary: Dict[str, Any] = Field(default_factory=dict, description="Allocation summary statistics")
    recommendation: Optional[str] = Field(None, description="Strategy recommendation and insights")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class StrategyComparison(BaseModel):
    """Strategy comparison request"""
    allocation_data: List[AllocationRecord] = Field(..., min_items=1)
    strategies: List[OptimizationStrategy] = Field(..., min_items=2, max_items=4)
    constraints: Optional[AllocationConstraints] = Field(None)


class StrategyComparisonResponse(BaseModel):
    """Strategy comparison results"""
    success: bool = Field(..., description="Whether comparison was successful")
    best_strategy: OptimizationStrategy = Field(..., description="Recommended best strategy")
    recommendation: str = Field(..., description="Human-readable recommendation")
    
    results: Dict[str, OptimizationResponse] = Field(..., description="Results for each strategy")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison metrics")
    
    execution_time: float = Field(..., description="Total comparison time")
    timestamp: datetime = Field(default_factory=datetime.now)


class StrategyRecommendationRequest(BaseModel):
    """Request for strategy recommendation"""
    allocation_data: List[AllocationRecord] = Field(..., min_items=1)
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for optimization")


class StrategyRecommendationResponse(BaseModel):
    """Strategy recommendation response"""
    recommended_strategy: OptimizationStrategy = Field(..., description="Recommended strategy")
    confidence: float = Field(..., ge=0, le=1, description="Recommendation confidence (0-1)")
    reason: str = Field(..., description="Reason for recommendation")
    all_recommendations: List[Dict[str, Any]] = Field(..., description="All strategy recommendations with scores")
    data_analysis: Dict[str, Any] = Field(..., description="Analysis of input data characteristics")
    
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.now)


class FileUploadResponse(BaseModel):
    """File upload response"""
    success: bool
    filename: str
    file_size: int
    records_count: int
    message: str
    validation_errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Health check models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(..., description="Check timestamp")
    service: str = Field(..., description="Service name")
    module: str = Field(..., description="Module identifier")
    version: str = Field(..., description="API version")


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response"""
    environment: str = Field(..., description="Environment (development, production, etc.)")
    components: Dict[str, Any] = Field(..., description="Component health status")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    error: Optional[str] = Field(None, description="Error message if unhealthy")