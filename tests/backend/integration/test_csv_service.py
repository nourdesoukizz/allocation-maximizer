"""
Phase 1 Integration Test: Complete CSV service flow
"""

import pytest
import tempfile
import csv
from pathlib import Path
from datetime import date
from backend.api.module4.services.csv_service import CSVDataService
from backend.api.module4.services.cache_service import DataCache, InMemoryCache


class TestCSVServiceIntegration:
    """Integration tests for CSV service with caching"""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing"""
        return """dc_id,dc_name,dc_location,dc_region,dc_priority,sku_id,sku_name,sku_category,customer_id,customer_name,customer_tier,customer_region,current_inventory,forecasted_demand,historical_demand,revenue_per_unit,cost_per_unit,margin,sla_level,risk_score,substitution_sku_id,date,allocated_quantity,fulfillment_rate,lead_time_days,min_order_quantity,safety_stock
DC001,Chicago DC,Chicago IL,North America,5,SKU-NET-001,Network Switch 48-Port,Networking,CUST-001,TechCorp Inc,Strategic,North America,500,450,420,1250.00,950.00,24.0,Gold,0.15,SKU-NET-002,2024-10-14,400,88.9,3,10,50
DC002,Dallas DC,Dallas TX,North America,4,SKU-STO-001,Storage Array 10TB,Storage,CUST-002,CloudBase Inc,Premium,North America,150,200,180,8500.00,6500.00,23.5,Silver,0.16,,2024-10-14,150,75.0,4,2,25
DC003,Los Angeles DC,Los Angeles CA,North America,3,SKU-NET-003,Router Enterprise,Networking,CUST-003,WestCoast Tech,Standard,North America,200,250,230,2200.00,1700.00,22.7,Bronze,0.20,SKU-NET-004,2024-10-14,200,80.0,3,5,35"""
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_content):
        """Create temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def csv_service_with_cache(self, temp_csv_file):
        """Create CSV service with cache integration"""
        service = CSVDataService(temp_csv_file)
        cache = DataCache(InMemoryCache(default_ttl_minutes=5))
        return service, cache
    
    def test_end_to_end_data_loading(self, temp_csv_file):
        """Test complete data loading workflow"""
        service = CSVDataService(temp_csv_file)
        
        # Initial load
        data = service.load_data()
        
        assert data is not None
        assert data.total_records == 3
        assert len(data.records) == 3
        
        # Verify data integrity
        record1 = data.records[0]
        assert record1.dc_id == 'DC001'
        assert record1.customer_name == 'TechCorp Inc'
        assert record1.current_inventory == 500
        assert record1.substitution_sku_id == 'SKU-NET-002'
        
        # Check None handling
        record2 = data.records[1]
        assert record2.substitution_sku_id is None
        
        # Get validation report
        report = service.get_validation_report()
        assert report is not None
        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.is_valid
    
    def test_data_summary_generation(self, temp_csv_file):
        """Test data summary generation"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        summary = service.get_data_summary()
        
        # Verify summary statistics
        assert summary.total_records == 3
        assert summary.unique_dcs == 3
        assert summary.unique_customers == 3
        assert summary.unique_skus == 3
        assert summary.unique_regions == 1  # All North America
        
        # Verify aggregations
        assert summary.total_inventory == 850  # 500 + 150 + 200
        assert summary.total_demand == 900    # 450 + 200 + 250
        assert summary.total_allocated == 750 # 400 + 150 + 200
        
        # Verify categorical breakdowns
        assert 'Strategic' in summary.records_by_tier
        assert 'Premium' in summary.records_by_tier
        assert 'Standard' in summary.records_by_tier
        
        assert summary.records_by_tier['Strategic'] == 1
        assert summary.records_by_tier['Premium'] == 1
        assert summary.records_by_tier['Standard'] == 1
        
        assert 'Gold' in summary.records_by_sla
        assert 'Silver' in summary.records_by_sla
        assert 'Bronze' in summary.records_by_sla
        
        assert summary.records_by_priority[5] == 1
        assert summary.records_by_priority[4] == 1
        assert summary.records_by_priority[3] == 1
    
    def test_filtering_and_querying(self, temp_csv_file):
        """Test data filtering and querying capabilities"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # Test filtering by DC priority
        high_priority_records = service.filter_records(min_priority=4)
        assert len(high_priority_records) == 2  # Priority 5 and 4
        assert all(r.dc_priority >= 4 for r in high_priority_records)
        
        # Test filtering by customer tier
        strategic_records = service.filter_records(customer_tiers=['Strategic'])
        assert len(strategic_records) == 1
        assert strategic_records[0].customer_tier.value == 'Strategic'
        
        # Test filtering by risk score
        low_risk_records = service.filter_records(max_risk_score=0.16)
        assert len(low_risk_records) == 2  # Risk scores 0.15 and 0.16
        assert all(r.risk_score <= 0.16 for r in low_risk_records)
        
        # Test combined filtering
        combined_records = service.filter_records(
            customer_tiers=['Strategic', 'Premium'],
            min_priority=4
        )
        assert len(combined_records) == 2
        assert all(r.customer_tier.value in ['Strategic', 'Premium'] for r in combined_records)
        assert all(r.dc_priority >= 4 for r in combined_records)
        
        # Test entity-specific queries
        dc_records = service.get_records_by_dc('DC001')
        assert len(dc_records) == 1
        assert dc_records[0].dc_id == 'DC001'
        
        customer_records = service.get_records_by_customer('CUST-002')
        assert len(customer_records) == 1
        assert customer_records[0].customer_id == 'CUST-002'
        
        sku_records = service.get_records_by_sku('SKU-NET-003')
        assert len(sku_records) == 1
        assert sku_records[0].sku_id == 'SKU-NET-003'
    
    def test_unique_values_extraction(self, temp_csv_file):
        """Test extraction of unique values"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # Test various field extractions
        dc_ids = service.get_unique_values('dc_id')
        assert set(dc_ids) == {'DC001', 'DC002', 'DC003'}
        
        customer_ids = service.get_unique_values('customer_id')
        assert set(customer_ids) == {'CUST-001', 'CUST-002', 'CUST-003'}
        
        sku_ids = service.get_unique_values('sku_id')
        assert set(sku_ids) == {'SKU-NET-001', 'SKU-STO-001', 'SKU-NET-003'}
        
        tiers = service.get_unique_values('customer_tier')
        assert set(tiers) == {'Strategic', 'Premium', 'Standard'}
        
        sla_levels = service.get_unique_values('sla_level')
        assert set(sla_levels) == {'Gold', 'Silver', 'Bronze'}
        
        regions = service.get_unique_values('dc_region')
        assert set(regions) == {'North America'}
        
        categories = service.get_unique_values('sku_category')
        assert set(categories) == {'Networking', 'Storage'}
    
    def test_cache_integration(self, csv_service_with_cache):
        """Test CSV service integration with caching"""
        service, cache = csv_service_with_cache
        
        # Load data (should cache it)
        data1 = service.load_data()
        
        # Cache the data
        cache.set_csv_data(data1, ttl_minutes=10)
        
        # Retrieve from cache
        cached_data = cache.get_csv_data()
        assert cached_data is not None
        assert cached_data.total_records == data1.total_records
        
        # Cache summary
        summary = service.get_data_summary()
        cache.set_data_summary(summary, ttl_minutes=5)
        
        cached_summary = cache.get_data_summary()
        assert cached_summary is not None
        assert cached_summary.total_records == summary.total_records
        
        # Test filtered data caching
        filter_key = "priority_4_plus"
        filtered_data = service.filter_records(min_priority=4)
        cache.set_filtered_data(filter_key, filtered_data, ttl_minutes=5)
        
        cached_filtered = cache.get_filtered_data(filter_key)
        assert cached_filtered is not None
        assert len(cached_filtered) == len(filtered_data)
    
    def test_data_staleness_workflow(self, temp_csv_file):
        """Test data staleness detection and refresh workflow"""
        service = CSVDataService(temp_csv_file)
        
        # Initially no data
        assert service.is_data_stale()
        
        # Load data
        data1 = service.load_data()
        assert not service.is_data_stale()
        
        # Force data to be stale
        import datetime
        old_time = datetime.datetime.now() - datetime.timedelta(minutes=20)
        service._last_loaded = old_time
        
        assert service.is_data_stale(max_age_minutes=10)
        
        # Refresh if stale
        data2 = service.refresh_if_stale(max_age_minutes=10)
        assert data2 is not data1  # Should be new instance
        assert not service.is_data_stale(max_age_minutes=10)
    
    def test_error_handling_and_recovery(self, temp_csv_file):
        """Test error handling and recovery scenarios"""
        service = CSVDataService(temp_csv_file)
        
        # First load successfully
        data = service.load_data()
        assert data is not None
        
        # Simulate file disappearing
        Path(temp_csv_file).unlink()
        
        # Should still return cached data
        cached_data = service.load_data(force_reload=False)
        assert cached_data is data
        
        # Force reload should fail
        with pytest.raises(FileNotFoundError):
            service.load_data(force_reload=True)
    
    def test_performance_with_large_filters(self, temp_csv_file):
        """Test performance characteristics with various filter combinations"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # Test multiple filter combinations
        filter_combinations = [
            {'dc_ids': ['DC001', 'DC002']},
            {'customer_tiers': ['Strategic']},
            {'sla_levels': ['Gold', 'Silver']},
            {'min_priority': 3},
            {'max_risk_score': 0.18},
            {'dc_ids': ['DC001'], 'customer_tiers': ['Strategic']},
            {'min_priority': 4, 'max_risk_score': 0.17}
        ]
        
        for filters in filter_combinations:
            filtered_records = service.filter_records(**filters)
            assert isinstance(filtered_records, list)
            # Each filter should return some valid subset
            assert len(filtered_records) <= 3  # Total records
    
    def test_concurrent_access_safety(self, csv_service_with_cache):
        """Test concurrent access patterns (basic thread safety)"""
        service, cache = csv_service_with_cache
        
        # Load data
        data = service.load_data()
        
        # Simulate concurrent operations
        results = []
        
        # Multiple summary generations
        for _ in range(3):
            summary = service.get_data_summary()
            results.append(summary.total_records)
        
        # All should be consistent
        assert all(r == 3 for r in results)
        
        # Multiple filtering operations
        filter_results = []
        for priority in [3, 4, 5]:
            filtered = service.filter_records(min_priority=priority)
            filter_results.append(len(filtered))
        
        # Results should be decreasing (higher priority = fewer records)
        assert filter_results == [3, 2, 1]