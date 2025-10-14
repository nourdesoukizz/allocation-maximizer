"""
Optimization endpoints for Module 4 API
"""

import logging
import pandas as pd
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from schemas import (
    OptimizationRequest, OptimizationResponse, 
    StrategyComparison, StrategyComparisonResponse,
    StrategyRecommendationRequest, StrategyRecommendationResponse,
    ErrorResponse, FileUploadResponse, AllocationRecord, AllocationResult, SubstitutionRecord
)
from optimizers.optimizer_selector import optimizer_selector, OptimizerConfig, OptimizationStrategy as OptimizerStrategy
from optimizers.base_optimizer import AllocationConstraints as BaseConstraints, AllocationResult as BaseResult
from utils.data_preprocessing import process_uploaded_file, validate_allocation_data
from services.cache_service import get_cache_service, CacheKeys
from middleware.security import get_rate_limiter, security_middleware
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for request tracking (fallback if cache is not available)
optimization_results: Dict[str, Dict[str, Any]] = {}

# Get rate limiter instance
limiter = get_rate_limiter()

def _generate_request_hash(request: OptimizationRequest) -> str:
    """Generate a hash for the optimization request to use as cache key"""
    # Create a consistent hash based on request parameters
    request_data = {
        'strategy': request.strategy.value,
        'allocation_data': [
            {
                'dc_id': record.dc_id,
                'sku_id': record.sku_id,
                'customer_id': record.customer_id,
                'current_inventory': record.current_inventory,
                'forecasted_demand': record.forecasted_demand,
                'dc_priority': record.dc_priority,
                'customer_tier': record.customer_tier,
                'sla_level': record.sla_level,
                'min_order_quantity': record.min_order_quantity,
                'sku_category': record.sku_category
            }
            for record in request.allocation_data
        ],
        'constraints': request.constraints.dict() if request.constraints else None,
        'strategy_params': request.strategy_params or {},
        'priority_weight': request.priority_weight or 0.6,
        'fairness_weight': request.fairness_weight or 0.4,
        'prefer_efficiency': request.prefer_efficiency or True,
        'prefer_speed': request.prefer_speed or False
    }
    
    # Create hash from JSON representation
    request_json = json.dumps(request_data, sort_keys=True, default=str)
    return hashlib.sha256(request_json.encode()).hexdigest()[:16]

@router.post("/optimize", response_model=OptimizationResponse)
@limiter.limit("100/minute")
async def optimize_allocation(optimization_request: OptimizationRequest, request: Request) -> OptimizationResponse:
    """
    Perform allocation optimization with specified strategy
    
    Args:
        request: Optimization request with strategy and data
        
    Returns:
        Optimization results
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Starting optimization request {request_id} with strategy: {optimization_request.strategy}")
    
    try:
        # Generate cache key for this request
        request_hash = _generate_request_hash(optimization_request)
        cache_key = CacheKeys.optimization_result(request_hash)
        
        # Get cache service
        cache_service = await get_cache_service()
        
        # Check cache first
        if hasattr(cache_service, 'get'):
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for optimization request {request_id} (hash: {request_hash})")
                cached_result["request_id"] = request_id  # Update request ID for tracking
                return OptimizationResponse(**cached_result)
        
        # Validate input data using security middleware
        request_data = optimization_request.dict()
        security_middleware.validate_input_data(request_data)
        
        # Convert request data to DataFrame
        allocation_df = _convert_request_to_dataframe(optimization_request.allocation_data)
        
        # Validate allocation data format
        validation_errors = validate_allocation_data(allocation_df)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid allocation data",
                    "validation_errors": validation_errors
                }
            )
        
        # Create optimizer configuration
        config = OptimizerConfig(
            strategy=OptimizerStrategy(optimization_request.strategy.value),
            constraints=_convert_constraints(optimization_request.constraints) if optimization_request.constraints else None,
            strategy_params=optimization_request.strategy_params or {},
            priority_weight=optimization_request.priority_weight or 0.6,
            fairness_weight=optimization_request.fairness_weight or 0.4,
            prefer_efficiency=optimization_request.prefer_efficiency or True,
            prefer_speed=optimization_request.prefer_speed or False
        )
        
        # Run optimization
        result = optimizer_selector.select_and_run_optimizer(allocation_df, config)
        
        # Convert result to response format
        response = _convert_result_to_response(result, optimization_request.strategy, request_id)
        
        # Cache the result (TTL: 1 hour for optimization results)
        if hasattr(cache_service, 'set'):
            cache_data = response.dict()
            await cache_service.set(cache_key, cache_data, ttl=timedelta(hours=1))
            logger.info(f"Cached optimization result for request {request_id} (hash: {request_hash})")
        
        # Store result for tracking (fallback)
        optimization_results[request_id] = {
            "response": response.dict(),
            "timestamp": datetime.now(),
            "strategy": optimization_request.strategy.value
        }
        
        logger.info(f"Optimization request {request_id} completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization request {request_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Optimization failed",
                "message": str(e),
                "request_id": request_id
            }
        )

@router.post("/compare-strategies", response_model=StrategyComparisonResponse)
@limiter.limit("50/minute")
async def compare_strategies(comparison_request: StrategyComparison, request: Request) -> StrategyComparisonResponse:
    """
    Compare multiple optimization strategies on the same data
    
    Args:
        request: Strategy comparison request
        
    Returns:
        Comparison results with recommendations
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Starting strategy comparison request {request_id}")
    
    try:
        # Convert request data to DataFrame
        allocation_df = _convert_request_to_dataframe(optimization_request.allocation_data)
        
        # Validate input data
        validation_errors = validate_allocation_data(allocation_df)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid allocation data",
                    "validation_errors": validation_errors
                }
            )
        
        # Convert strategies
        strategies = [OptimizerStrategy(s.value) for s in optimization_request.strategies]
        constraints = _convert_constraints(optimization_request.constraints) if optimization_request.constraints else None
        
        # Run comparison
        comparison_result = optimizer_selector.compare_strategies(
            allocation_df, 
            strategies, 
            constraints
        )
        
        # Convert results to response format
        strategy_results = {}
        for strategy, result in comparison_result.results.items():
            strategy_results[strategy.value] = _convert_result_to_response(
                result, 
                strategy.value, 
                f"{request_id}_{strategy.value}"
            )
        
        response = StrategyComparisonResponse(
            success=True,
            best_strategy=comparison_result.best_strategy.value,
            recommendation=comparison_result.recommendation,
            results=strategy_results,
            comparison_metrics=comparison_result.comparison_metrics,
            execution_time=sum(r.optimization_time for r in comparison_result.results.values())
        )
        
        logger.info(f"Strategy comparison request {request_id} completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy comparison request {request_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Strategy comparison failed",
                "message": str(e),
                "request_id": request_id
            }
        )

@router.post("/recommend-strategy", response_model=StrategyRecommendationResponse)
async def recommend_strategy(request: StrategyRecommendationRequest) -> StrategyRecommendationResponse:
    """
    Get strategy recommendation based on data analysis
    
    Args:
        request: Strategy recommendation request
        
    Returns:
        Strategy recommendation with reasoning
    """
    try:
        # Convert request data to DataFrame
        allocation_df = _convert_request_to_dataframe(optimization_request.allocation_data)
        
        # Get recommendation
        recommendation = optimizer_selector.get_strategy_recommendation(
            allocation_df,
            optimization_request.user_preferences
        )
        
        response = StrategyRecommendationResponse(
            recommended_strategy=recommendation['recommended_strategy'].value,
            confidence=recommendation['confidence'],
            reason=recommendation['reason'],
            all_recommendations=recommendation['all_recommendations'],
            data_analysis=recommendation['data_analysis']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Strategy recommendation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Strategy recommendation failed",
                "message": str(e)
            }
        )

@router.post("/upload-file", response_model=FileUploadResponse)
@limiter.limit("20/minute")
async def upload_allocation_file(
    request: Request,
    file: UploadFile = File(...),
    validate_only: bool = Form(False)
) -> FileUploadResponse:
    """
    Upload and process allocation data file
    
    Args:
        file: Uploaded CSV or Excel file
        validate_only: Only validate data without storing
        
    Returns:
        Upload and validation results
    """
    try:
        settings = get_settings()
        
        # Check file size
        if file.size > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Check file format
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.supported_file_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {settings.supported_file_formats}"
            )
        
        # Process file
        allocation_df = await process_uploaded_file(file)
        
        # Validate data
        validation_errors = validate_allocation_data(allocation_df)
        
        response = FileUploadResponse(
            success=len(validation_errors) == 0,
            filename=file.filename,
            file_size=file.size,
            records_count=len(allocation_df),
            message="File processed successfully" if len(validation_errors) == 0 else "File processed with validation errors",
            validation_errors=validation_errors if validation_errors else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "File upload failed",
                "message": str(e)
            }
        )

@router.get("/results/{request_id}")
async def get_optimization_result(request_id: str) -> Dict[str, Any]:
    """
    Get stored optimization result by request ID
    
    Args:
        request_id: Optimization request ID
        
    Returns:
        Stored optimization result
    """
    if request_id not in optimization_results:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization result not found for request ID: {request_id}"
        )
    
    return optimization_results[request_id]

@router.get("/data/csv-preview")
async def get_csv_data_preview() -> Dict[str, Any]:
    """
    Load and preview CSV allocation data for the frontend
    
    Returns:
        Parsed CSV data with customers, DCs, and products
    """
    try:
        # Load CSV data
        csv_path = "../../../data/allocation_data.csv"
        df = pd.read_csv(csv_path)
        
        # Extract unique customers
        customers = df[['customer_id', 'customer_name', 'customer_tier', 'customer_region']].drop_duplicates()
        customers_list = [
            {
                'id': row['customer_id'],
                'name': row['customer_name'],
                'tier': row['customer_tier'],
                'region': row['customer_region']
            }
            for _, row in customers.iterrows()
        ]
        
        # Extract unique distribution centers
        dcs = df[['dc_id', 'dc_name', 'dc_location', 'dc_region', 'dc_priority']].drop_duplicates()
        dcs_list = [
            {
                'id': row['dc_id'],
                'name': row['dc_name'],
                'location': row['dc_location'],
                'region': row['dc_region'],
                'priority': row['dc_priority']
            }
            for _, row in dcs.iterrows()
        ]
        
        # Extract unique products
        products = df[['sku_id', 'sku_name', 'sku_category']].drop_duplicates()
        products_list = [
            {
                'id': row['sku_id'],
                'name': row['sku_name'],
                'category': row['sku_category']
            }
            for _, row in products.iterrows()
        ]
        
        return {
            'customers': customers_list,
            'distributionCenters': dcs_list,
            'products': products_list,
            'totalRecords': len(df),
            'lastUpdated': df['date'].iloc[0] if 'date' in df.columns else None
        }
        
    except Exception as e:
        logger.error(f"Failed to load CSV data: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to load CSV data",
                "message": str(e)
            }
        )

@router.get("/available-strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """
    Get list of available optimization strategies
    
    Returns:
        Available strategies and their descriptions
    """
    return {
        "strategies": {
            "priority_based": {
                "name": "Priority Based",
                "description": "Allocates based on DC priority rankings - higher priority DCs get allocated first",
                "best_for": "Maximizing efficiency with clear DC hierarchy",
                "parameters": ["respect_customer_tier", "allow_overflow", "max_iterations"]
            },
            "fair_share": {
                "name": "Fair Share",
                "description": "Distributes proportionally based on demand ratios across all DCs",
                "best_for": "Ensuring equitable distribution and fairness",
                "parameters": ["fairness_weight", "rebalancing_iterations", "min_fairness_threshold"]
            },
            "hybrid": {
                "name": "Hybrid",
                "description": "Combines priority and fair share approaches with configurable weights",
                "best_for": "Balancing efficiency and fairness",
                "parameters": ["priority_weight", "fairness_weight"]
            },
            "auto_select": {
                "name": "Auto Select",
                "description": "Automatically selects the best strategy based on data analysis",
                "best_for": "When unsure which strategy to use",
                "parameters": ["prefer_efficiency", "prefer_fairness", "prefer_speed"]
            }
        }
    }

# Helper functions
def _convert_request_to_dataframe(allocation_data: List[AllocationRecord]) -> pd.DataFrame:
    """Convert request allocation data to pandas DataFrame"""
    data = []
    for record in allocation_data:
        data.append({
            'dc_id': record.dc_id,
            'sku_id': record.sku_id,
            'customer_id': record.customer_id,
            'current_inventory': record.current_inventory,
            'forecasted_demand': record.forecasted_demand,
            'dc_priority': record.dc_priority,
            'customer_tier': record.customer_tier,
            'sla_level': record.sla_level,
            'min_order_quantity': record.min_order_quantity or 1.0,
            'sku_category': record.sku_category or 'default'
        })
    
    return pd.DataFrame(data)

def _convert_constraints(constraints) -> BaseConstraints:
    """Convert API constraints to optimizer constraints"""
    return BaseConstraints(
        min_allocation=constraints.min_allocation,
        max_allocation=constraints.max_allocation,
        min_order_quantity=constraints.min_order_quantity,
        safety_stock_buffer=constraints.safety_stock_buffer,
        allow_substitution=constraints.allow_substitution,
        max_substitution_ratio=constraints.max_substitution_ratio,
        respect_customer_tier=constraints.respect_customer_tier,
        respect_sla_levels=constraints.respect_sla_levels
    )

def _convert_result_to_response(result: BaseResult, strategy: str, request_id: str) -> OptimizationResponse:
    """Convert optimizer result to API response"""
    
    # Convert allocations DataFrame to list of AllocationResult
    allocations = []
    for _, row in result.allocations.iterrows():
        allocation = AllocationResult(
            dc_id=str(row['dc_id']),
            sku_id=str(row['sku_id']),
            customer_id=str(row['customer_id']),
            allocated_quantity=float(row['allocated_quantity']),
            forecasted_demand=float(row['forecasted_demand']),
            current_inventory=float(row['current_inventory']),
            allocation_efficiency=float(row['allocated_quantity'] / row['forecasted_demand'] * 100) if row['forecasted_demand'] > 0 else 0.0,
            allocation_round=int(row.get('allocation_round', 1)) if 'allocation_round' in row else None
        )
        allocations.append(allocation)
    
    # Convert substitutions
    substitutions = []
    for sub in result.substitutions_made:
        substitution = SubstitutionRecord(
            customer_id=str(sub['customer_id']),
            original_sku=str(sub['original_sku']),
            substitute_sku=str(sub['substitute_sku']),
            quantity=float(sub['quantity']),
            dc_id=str(sub['dc_id'])
        )
        substitutions.append(substitution)
    
    return OptimizationResponse(
        success=True,
        strategy_used=strategy,
        total_allocated=result.total_allocated,
        total_demand=result.total_demand,
        allocation_efficiency=result.allocation_efficiency,
        unallocated_demand=result.unallocated_demand,
        optimization_time=result.optimization_time,
        allocations=allocations,
        substitutions_made=substitutions,
        constraints_violated=result.constraints_violated,
        allocation_summary=result.allocation_summary,
        request_id=request_id
    )