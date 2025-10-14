"""
Integration tests for Module 4 API endpoints
"""

import pytest
import requests
import json
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8004"
TIMEOUT = 30


class TestAPIIntegration:
    """Integration tests for the Allocation Maximizer API"""
    
    @classmethod
    def setup_class(cls):
        """Setup test class"""
        # Wait for server to be ready
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    pytest.skip("API server is not running")
                time.sleep(1)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health/", timeout=TIMEOUT)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Allocation Maximizer API"
        assert data["module"] == "Module 4"
        assert data["version"] == "1.0.0"
    
    def test_detailed_health_endpoint(self):
        """Test detailed health check endpoint"""
        response = requests.get(f"{BASE_URL}/health/detailed", timeout=TIMEOUT)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
        assert "performance" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Allocation Maximizer API"
        assert data["status"] == "running"
    
    def test_available_strategies_endpoint(self):
        """Test available strategies endpoint"""
        response = requests.get(f"{BASE_URL}/optimization/available-strategies", timeout=TIMEOUT)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "strategies" in data
        
        strategies = data["strategies"]
        assert "priority_based" in strategies
        assert "fair_share" in strategies
        assert "hybrid" in strategies
        assert "auto_select" in strategies
    
    def test_priority_based_optimization(self):
        """Test priority-based optimization"""
        request_data = {
            "strategy": "priority_based",
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001",
                    "customer_id": "CUST001", 
                    "current_inventory": 100,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A",
                    "sla_level": "Premium"
                },
                {
                    "dc_id": "DC002",
                    "sku_id": "SKU001",
                    "customer_id": "CUST002",
                    "current_inventory": 50,
                    "forecasted_demand": 60,
                    "dc_priority": 2, 
                    "customer_tier": "B",
                    "sla_level": "Standard"
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["strategy_used"] == "priority_based"
        assert data["total_allocated"] > 0
        assert data["total_demand"] > 0
        assert 0 <= data["allocation_efficiency"] <= 100
        assert len(data["allocations"]) == 2
        
        # Check allocation results
        allocations = data["allocations"]
        dc1_allocation = next(a for a in allocations if a["dc_id"] == "DC001")
        dc2_allocation = next(a for a in allocations if a["dc_id"] == "DC002")
        
        # DC001 has higher priority, should be allocated first
        assert dc1_allocation["allocated_quantity"] == 80.0  # Full demand
        assert dc2_allocation["allocated_quantity"] <= 50.0  # Limited by inventory
    
    def test_fair_share_optimization(self):
        """Test fair share optimization"""
        request_data = {
            "strategy": "fair_share",
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001", 
                    "customer_id": "CUST001",
                    "current_inventory": 60,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A",
                    "sla_level": "Premium"
                },
                {
                    "dc_id": "DC002", 
                    "sku_id": "SKU001",
                    "customer_id": "CUST002",
                    "current_inventory": 40,
                    "forecasted_demand": 60,
                    "dc_priority": 1,  # Same priority for fair share test
                    "customer_tier": "A",
                    "sla_level": "Premium"
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["strategy_used"] == "fair_share"
        assert data["total_allocated"] > 0
        assert len(data["allocations"]) == 2
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation endpoint"""
        request_data = {
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001",
                    "customer_id": "CUST001",
                    "current_inventory": 100,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A",
                    "sla_level": "Premium"
                },
                {
                    "dc_id": "DC002",
                    "sku_id": "SKU001", 
                    "customer_id": "CUST002",
                    "current_inventory": 50,
                    "forecasted_demand": 60,
                    "dc_priority": 3,  # Different priority for recommendation test
                    "customer_tier": "C",
                    "sla_level": "Basic"
                }
            ],
            "user_preferences": {
                "prefer_efficiency": True
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/recommend-strategy",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "recommended_strategy" in data
        assert "confidence" in data
        assert "reason" in data
        assert "data_analysis" in data
        assert 0 <= data["confidence"] <= 1
    
    def test_strategy_comparison(self):
        """Test strategy comparison endpoint"""
        request_data = {
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001",
                    "customer_id": "CUST001",
                    "current_inventory": 100,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A",
                    "sla_level": "Premium"
                },
                {
                    "dc_id": "DC002",
                    "sku_id": "SKU001",
                    "customer_id": "CUST002", 
                    "current_inventory": 50,
                    "forecasted_demand": 60,
                    "dc_priority": 2,
                    "customer_tier": "B",
                    "sla_level": "Standard"
                }
            ],
            "strategies": ["priority_based", "fair_share"]
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/compare-strategies",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "best_strategy" in data
        assert "results" in data
        assert "comparison_metrics" in data
        
        results = data["results"]
        assert "priority_based" in results
        assert "fair_share" in results
    
    def test_invalid_optimization_request(self):
        """Test invalid optimization request"""
        request_data = {
            "strategy": "priority_based",
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    # Missing required fields
                    "current_inventory": 100
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize", 
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_empty_allocation_data(self):
        """Test optimization with empty allocation data"""
        request_data = {
            "strategy": "priority_based",
            "allocation_data": []
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_unsupported_strategy(self):
        """Test optimization with unsupported strategy"""
        request_data = {
            "strategy": "invalid_strategy",
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001", 
                    "customer_id": "CUST001",
                    "current_inventory": 100,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A",
                    "sla_level": "Premium"
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error for invalid enum
    
    def test_optimization_with_constraints(self):
        """Test optimization with custom constraints"""
        request_data = {
            "strategy": "priority_based",
            "allocation_data": [
                {
                    "dc_id": "DC001",
                    "sku_id": "SKU001",
                    "customer_id": "CUST001",
                    "current_inventory": 100,
                    "forecasted_demand": 80,
                    "dc_priority": 1,
                    "customer_tier": "A", 
                    "sla_level": "Premium"
                }
            ],
            "constraints": {
                "min_allocation": 10,
                "safety_stock_buffer": 0.2,
                "allow_substitution": False,
                "respect_customer_tier": True
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/optimization/optimize",
            json=request_data,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total_allocated"] > 0


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])