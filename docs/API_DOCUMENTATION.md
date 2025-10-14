# Allocation Maximizer API Documentation

## Overview
The Allocation Maximizer API provides advanced allocation optimization capabilities with multiple strategies including priority-based, fair share, hybrid, and automatic selection approaches.

**Base URL:** `http://localhost:8000`  
**API Version:** 1.0.0  
**Documentation:** `http://localhost:8000/docs` (Interactive Swagger UI)

## Authentication
Optional API key authentication via Authorization header:
```bash
Authorization: Bearer your_api_key_here
```

## Rate Limits
- Optimization endpoints: 100 requests/minute
- Strategy comparison: 50 requests/minute  
- File uploads: 20 requests/minute

## Core Endpoints

### 1. Health Check Endpoints

#### Basic Health Check
```http
GET /health/
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-14T10:30:00Z",
  "service": "Allocation Maximizer API",
  "module": "Module 4",
  "version": "1.0.0"
}
```

#### Detailed Health Check
```http
GET /health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-14T10:30:00Z",
  "service": "Allocation Maximizer API",
  "module": "Module 4",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "optimizers": {
      "status": "healthy",
      "available_optimizers": 4
    },
    "configuration": {
      "status": "healthy",
      "environment": "development"
    },
    "cache": {
      "status": "healthy",
      "type": "redis"
    }
  },
  "performance": {
    "response_time_ms": 45.2,
    "uptime_seconds": 1234.5
  }
}
```

### 2. Optimization Endpoints

#### Optimize Allocation
```http
POST /optimization/optimize
```

**Request Body:**
```json
{
  "strategy": "priority_based",
  "allocation_data": [
    {
      "dc_id": "DC001",
      "sku_id": "SKU001",
      "customer_id": "CUST001",
      "current_inventory": 1000,
      "forecasted_demand": 800,
      "dc_priority": 1,
      "customer_tier": "premium",
      "sla_level": "standard",
      "min_order_quantity": 50,
      "sku_category": "electronics"
    }
  ],
  "constraints": {
    "min_allocation": 0,
    "max_allocation": null,
    "min_order_quantity": 1,
    "safety_stock_buffer": 0.1,
    "allow_substitution": true,
    "max_substitution_ratio": 0.2,
    "respect_customer_tier": true,
    "respect_sla_levels": true
  },
  "strategy_params": {
    "respect_customer_tier": true,
    "allow_overflow": false
  },
  "priority_weight": 0.6,
  "fairness_weight": 0.4,
  "prefer_efficiency": true,
  "prefer_speed": false
}
```

**Response:**
```json
{
  "success": true,
  "strategy_used": "priority_based",
  "total_allocated": 800,
  "total_demand": 800,
  "allocation_efficiency": 100.0,
  "unallocated_demand": 0,
  "optimization_time": 0.025,
  "allocations": [
    {
      "dc_id": "DC001",
      "sku_id": "SKU001",
      "customer_id": "CUST001",
      "allocated_quantity": 800,
      "forecasted_demand": 800,
      "current_inventory": 1000,
      "allocation_efficiency": 100.0,
      "allocation_round": 1
    }
  ],
  "substitutions_made": [],
  "constraints_violated": [],
  "allocation_summary": {
    "total_customers": 1,
    "total_dcs": 1,
    "total_skus": 1,
    "allocation_rounds": 1
  },
  "request_id": "req_12345"
}
```

#### Compare Strategies
```http
POST /optimization/compare-strategies
```

**Request Body:**
```json
{
  "strategies": ["priority_based", "fair_share", "hybrid"],
  "allocation_data": [
    {
      "dc_id": "DC001",
      "sku_id": "SKU001",
      "customer_id": "CUST001",
      "current_inventory": 1000,
      "forecasted_demand": 800,
      "dc_priority": 1,
      "customer_tier": "premium",
      "sla_level": "standard"
    }
  ],
  "constraints": {
    "min_allocation": 0,
    "safety_stock_buffer": 0.1
  }
}
```

**Response:**
```json
{
  "success": true,
  "best_strategy": "priority_based",
  "recommendation": "Priority-based strategy recommended for highest efficiency",
  "results": {
    "priority_based": {
      "allocation_efficiency": 100.0,
      "total_allocated": 800,
      "optimization_time": 0.025
    },
    "fair_share": {
      "allocation_efficiency": 95.0,
      "total_allocated": 760,
      "optimization_time": 0.030
    },
    "hybrid": {
      "allocation_efficiency": 98.0,
      "total_allocated": 784,
      "optimization_time": 0.035
    }
  },
  "comparison_metrics": {
    "efficiency_range": [95.0, 100.0],
    "time_range": [0.025, 0.035],
    "recommended_for_efficiency": "priority_based",
    "recommended_for_fairness": "fair_share"
  },
  "execution_time": 0.090
}
```

#### Get Strategy Recommendation
```http
POST /optimization/recommend-strategy
```

**Request Body:**
```json
{
  "allocation_data": [
    {
      "dc_id": "DC001",
      "sku_id": "SKU001",
      "customer_id": "CUST001",
      "current_inventory": 1000,
      "forecasted_demand": 800,
      "dc_priority": 1,
      "customer_tier": "premium"
    }
  ],
  "user_preferences": {
    "prefer_efficiency": true,
    "prefer_fairness": false,
    "prefer_speed": true
  }
}
```

**Response:**
```json
{
  "recommended_strategy": "priority_based",
  "confidence": 0.85,
  "reason": "High inventory levels and clear priority hierarchy favor priority-based allocation",
  "all_recommendations": [
    {
      "strategy": "priority_based",
      "score": 0.85,
      "reason": "Optimal for efficiency with clear hierarchy"
    },
    {
      "strategy": "hybrid",
      "score": 0.75,
      "reason": "Good balance of efficiency and fairness"
    }
  ],
  "data_analysis": {
    "total_inventory": 1000,
    "total_demand": 800,
    "inventory_ratio": 1.25,
    "priority_spread": 0.0,
    "customer_tiers": ["premium"]
  }
}
```

### 3. File Upload Endpoint

#### Upload Allocation File
```http
POST /optimization/upload-file
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: CSV/Excel file with allocation data
- `validate_only`: boolean (optional, default: false)

**Response:**
```json
{
  "success": true,
  "filename": "allocation_data.csv",
  "file_size": 2048,
  "records_count": 100,
  "message": "File processed successfully",
  "validation_errors": null
}
```

### 4. Utility Endpoints

#### Get Available Strategies
```http
GET /optimization/available-strategies
```

**Response:**
```json
{
  "strategies": {
    "priority_based": {
      "name": "Priority Based",
      "description": "Allocates based on DC priority rankings",
      "best_for": "Maximizing efficiency with clear DC hierarchy",
      "parameters": ["respect_customer_tier", "allow_overflow"]
    },
    "fair_share": {
      "name": "Fair Share",
      "description": "Distributes proportionally based on demand ratios",
      "best_for": "Ensuring equitable distribution and fairness",
      "parameters": ["fairness_weight", "rebalancing_iterations"]
    },
    "hybrid": {
      "name": "Hybrid",
      "description": "Combines priority and fair share approaches",
      "best_for": "Balancing efficiency and fairness",
      "parameters": ["priority_weight", "fairness_weight"]
    },
    "auto_select": {
      "name": "Auto Select",
      "description": "Automatically selects the best strategy",
      "best_for": "When unsure which strategy to use",
      "parameters": ["prefer_efficiency", "prefer_fairness"]
    }
  }
}
```

#### Get Optimization Result
```http
GET /optimization/results/{request_id}
```

**Response:**
```json
{
  "response": {
    "success": true,
    "strategy_used": "priority_based",
    "total_allocated": 800
  },
  "timestamp": "2024-10-14T10:30:00Z",
  "strategy": "priority_based"
}
```

### 5. Monitoring Endpoints

#### Performance Metrics
```http
GET /health/metrics
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-14T10:30:00Z",
  "performance": {
    "requests": {
      "total_requests": 1250,
      "total_errors": 5,
      "error_rate": 0.4,
      "average_response_time": 0.156
    },
    "cache": {
      "type": "redis",
      "hit_rate": 75.5,
      "entries": 234
    }
  },
  "security": {
    "api_keys_configured": 2,
    "security_headers_enabled": true,
    "input_validation_enabled": true,
    "rate_limiting_enabled": true
  }
}
```

#### Cache Statistics
```http
GET /health/cache-stats
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-14T10:30:00Z",
  "cache_type": "redis",
  "statistics": {
    "hit_rate": 75.5,
    "entries": 234,
    "memory_used": "2.5MB",
    "connected": true
  }
}
```

## Data Models

### AllocationRecord
```json
{
  "dc_id": "string",
  "sku_id": "string", 
  "customer_id": "string",
  "current_inventory": "number",
  "forecasted_demand": "number",
  "dc_priority": "integer",
  "customer_tier": "string (optional)",
  "sla_level": "string (optional)",
  "min_order_quantity": "number (optional)",
  "sku_category": "string (optional)"
}
```

### AllocationConstraints
```json
{
  "min_allocation": "number (optional)",
  "max_allocation": "number (optional)",
  "min_order_quantity": "number (optional)",
  "safety_stock_buffer": "number (optional)",
  "allow_substitution": "boolean (optional)",
  "max_substitution_ratio": "number (optional)",
  "respect_customer_tier": "boolean (optional)",
  "respect_sla_levels": "boolean (optional)"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "request_id": "req_12345",
  "validation_errors": ["List of validation errors"]
}
```

### Common HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Invalid API key
- `413 Payload Too Large`: File size exceeds limit
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Example Usage with cURL

### Basic Optimization
```bash
curl -X POST "http://localhost:8000/optimization/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "priority_based",
    "allocation_data": [
      {
        "dc_id": "DC001",
        "sku_id": "SKU001", 
        "customer_id": "CUST001",
        "current_inventory": 1000,
        "forecasted_demand": 800,
        "dc_priority": 1,
        "customer_tier": "premium"
      }
    ]
  }'
```

### File Upload
```bash
curl -X POST "http://localhost:8000/optimization/upload-file" \
  -F "file=@allocation_data.csv" \
  -F "validate_only=false"
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health/"
```

## Client Libraries

### Python Example
```python
import requests

# Basic optimization
response = requests.post(
    "http://localhost:8000/optimization/optimize",
    json={
        "strategy": "priority_based",
        "allocation_data": [
            {
                "dc_id": "DC001",
                "sku_id": "SKU001",
                "customer_id": "CUST001", 
                "current_inventory": 1000,
                "forecasted_demand": 800,
                "dc_priority": 1
            }
        ]
    }
)

result = response.json()
print(f"Allocation efficiency: {result['allocation_efficiency']}%")
```

### JavaScript Example
```javascript
// Basic optimization
const response = await fetch('http://localhost:8000/optimization/optimize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    strategy: 'priority_based',
    allocation_data: [
      {
        dc_id: 'DC001',
        sku_id: 'SKU001',
        customer_id: 'CUST001',
        current_inventory: 1000,
        forecasted_demand: 800,
        dc_priority: 1
      }
    ]
  })
});

const result = await response.json();
console.log(`Allocation efficiency: ${result.allocation_efficiency}%`);
```

## Support
For issues and questions:
- Check the troubleshooting guide
- Review server logs for detailed error information
- Verify input data format matches the schema
- Ensure rate limits are not exceeded