"""
Phase 1 Performance Test: Large CSV handling
"""

import pytest
import tempfile
import csv
import time
from pathlib import Path
from datetime import date, timedelta
import random

from backend.api.module4.services.csv_service import CSVDataService
from backend.api.module4.services.cache_service import DataCache, InMemoryCache


class TestLargeCSVPerformance:
    """Performance tests for handling large CSV files"""
    
    @pytest.fixture(scope="class")
    def large_csv_file(self):
        """Create large CSV file for performance testing"""
        num_records = 10000  # 10K records for testing
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            fieldnames = [
                'dc_id', 'dc_name', 'dc_location', 'dc_region', 'dc_priority',
                'sku_id', 'sku_name', 'sku_category',
                'customer_id', 'customer_name', 'customer_tier', 'customer_region',
                'current_inventory', 'forecasted_demand', 'historical_demand',
                'revenue_per_unit', 'cost_per_unit', 'margin',
                'sla_level', 'risk_score', 'substitution_sku_id',
                'date', 'allocated_quantity', 'fulfillment_rate',
                'lead_time_days', 'min_order_quantity', 'safety_stock'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Generate test data
            dc_ids = [f'DC{i:03d}' for i in range(1, 51)]  # 50 DCs
            sku_ids = [f'SKU-{category}-{i:03d}' 
                      for category in ['NET', 'STO', 'SRV', 'PWR'] 
                      for i in range(1, 251)]  # 1000 SKUs
            customer_ids = [f'CUST-{i:04d}' for i in range(1, 1001)]  # 1000 customers
            
            regions = ['North America', 'Europe', 'Asia Pacific', 'South America']
            categories = ['Networking', 'Storage', 'Servers', 'Power']
            tiers = ['Strategic', 'Premium', 'Standard', 'Basic']
            sla_levels = ['Gold', 'Silver', 'Bronze']
            
            for i in range(num_records):
                dc_id = random.choice(dc_ids)
                dc_num = int(dc_id[2:])
                
                sku_id = random.choice(sku_ids)
                category = sku_id.split('-')[1]
                category_map = {'NET': 'Networking', 'STO': 'Storage', 'SRV': 'Servers', 'PWR': 'Power'}
                
                customer_id = random.choice(customer_ids)
                
                # Generate realistic data with some correlation
                priority = random.randint(1, 5)
                inventory = random.randint(10, 1000)
                demand = random.randint(5, min(inventory + 200, 1200))
                historical = demand + random.randint(-50, 50)
                allocated = min(demand, inventory) + random.randint(-20, 20)
                allocated = max(0, allocated)
                
                fulfillment_rate = (allocated / demand * 100) if demand > 0 else 0
                
                revenue = random.uniform(50, 10000)
                cost = revenue * random.uniform(0.6, 0.9)
                margin = ((revenue - cost) / revenue * 100)
                
                row = {
                    'dc_id': dc_id,
                    'dc_name': f'DC {dc_num}',
                    'dc_location': f'Location {dc_num}',
                    'dc_region': random.choice(regions),
                    'dc_priority': priority,
                    'sku_id': sku_id,
                    'sku_name': f'Product {sku_id}',
                    'sku_category': category_map[category],
                    'customer_id': customer_id,
                    'customer_name': f'Customer {customer_id[5:]}',
                    'customer_tier': random.choice(tiers),
                    'customer_region': random.choice(regions),
                    'current_inventory': inventory,
                    'forecasted_demand': demand,
                    'historical_demand': historical,
                    'revenue_per_unit': round(revenue, 2),
                    'cost_per_unit': round(cost, 2),
                    'margin': round(margin, 1),
                    'sla_level': random.choice(sla_levels),
                    'risk_score': round(random.uniform(0.05, 0.5), 3),
                    'substitution_sku_id': random.choice(sku_ids) if random.random() < 0.3 else '',
                    'date': (date.today() - timedelta(days=random.randint(0, 365))).isoformat(),
                    'allocated_quantity': allocated,
                    'fulfillment_rate': round(fulfillment_rate, 1),
                    'lead_time_days': random.randint(1, 14),
                    'min_order_quantity': random.randint(1, 20),
                    'safety_stock': random.randint(5, 100)
                }
                writer.writerow(row)
            
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.performance
    def test_large_csv_loading_performance(self, large_csv_file):
        """Test loading performance with large CSV file"""
        service = CSVDataService(large_csv_file)
        
        # Measure loading time
        start_time = time.time()
        data = service.load_data()
        load_time = time.time() - start_time
        
        assert data is not None
        assert data.total_records == 10000
        assert load_time < 30.0  # Should load within 30 seconds
        
        print(f"Loaded {data.total_records} records in {load_time:.2f} seconds")
        print(f"Loading rate: {data.total_records / load_time:.0f} records/second")
        
        # Verify data integrity
        assert len(data.records) == 10000
        assert all(record.dc_id.startswith('DC') for record in data.records[:100])
    
    @pytest.mark.performance
    def test_summary_generation_performance(self, large_csv_file):
        """Test data summary generation performance"""
        service = CSVDataService(large_csv_file)
        service.load_data()
        
        # Measure summary generation time
        start_time = time.time()
        summary = service.get_data_summary()
        summary_time = time.time() - start_time
        
        assert summary_time < 5.0  # Should generate summary within 5 seconds
        assert summary.total_records == 10000
        
        print(f"Generated summary in {summary_time:.2f} seconds")
        
        # Verify summary completeness
        assert summary.unique_dcs > 0
        assert summary.unique_customers > 0
        assert summary.unique_skus > 0
        assert summary.total_inventory > 0
        assert summary.total_demand > 0
    
    @pytest.mark.performance
    def test_filtering_performance(self, large_csv_file):
        """Test filtering performance with large dataset"""
        service = CSVDataService(large_csv_file)
        service.load_data()
        
        # Test various filter operations
        filter_tests = [
            ('high_priority', {'min_priority': 4}),
            ('strategic_customers', {'customer_tiers': ['Strategic']}),
            ('low_risk', {'max_risk_score': 0.2}),
            ('specific_dcs', {'dc_ids': ['DC001', 'DC002', 'DC003']}),
            ('combined_filter', {'min_priority': 3, 'customer_tiers': ['Strategic', 'Premium']})
        ]
        
        for test_name, filters in filter_tests:
            start_time = time.time()
            filtered_records = service.filter_records(**filters)
            filter_time = time.time() - start_time
            
            assert filter_time < 2.0  # Each filter should complete within 2 seconds
            assert isinstance(filtered_records, list)
            
            print(f"{test_name}: {len(filtered_records)} records in {filter_time:.3f} seconds")
    
    @pytest.mark.performance
    def test_unique_values_performance(self, large_csv_file):
        """Test unique values extraction performance"""
        service = CSVDataService(large_csv_file)
        service.load_data()
        
        fields_to_test = [
            'dc_id',
            'sku_id', 
            'customer_id',
            'customer_tier',
            'sla_level',
            'sku_category'
        ]
        
        for field in fields_to_test:
            start_time = time.time()
            unique_values = service.get_unique_values(field)
            extraction_time = time.time() - start_time
            
            assert extraction_time < 1.0  # Should extract within 1 second
            assert len(unique_values) > 0
            
            print(f"{field}: {len(unique_values)} unique values in {extraction_time:.3f} seconds")
    
    @pytest.mark.performance
    def test_cache_performance_impact(self, large_csv_file):
        """Test cache performance impact"""
        service = CSVDataService(large_csv_file)
        cache = DataCache(InMemoryCache(default_ttl_minutes=10))
        
        # First load without cache
        start_time = time.time()
        data = service.load_data()
        first_load_time = time.time() - start_time
        
        # Cache the data
        cache.set_csv_data(data)
        
        # Retrieve from cache
        start_time = time.time()
        cached_data = cache.get_csv_data()
        cache_retrieval_time = time.time() - start_time
        
        assert cache_retrieval_time < 0.1  # Cache should be very fast
        assert cached_data is not None
        assert cached_data.total_records == data.total_records
        
        print(f"First load: {first_load_time:.2f}s, Cache retrieval: {cache_retrieval_time:.4f}s")
        print(f"Cache speedup: {first_load_time / cache_retrieval_time:.0f}x")
    
    @pytest.mark.performance
    def test_memory_usage_estimation(self, large_csv_file):
        """Test memory usage with large dataset"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before loading
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        service = CSVDataService(large_csv_file)
        data = service.load_data()
        
        # Memory after loading
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        assert memory_used > 0
        assert memory_used < 500  # Should use less than 500MB for 10K records
        
        print(f"Memory usage: {memory_used:.1f} MB for {data.total_records} records")
        print(f"Memory per record: {memory_used * 1024 / data.total_records:.2f} KB")
        
        # Generate summary to test additional memory usage
        summary = service.get_data_summary()
        memory_after_summary = process.memory_info().rss / 1024 / 1024  # MB
        summary_memory = memory_after_summary - memory_after
        
        print(f"Additional memory for summary: {summary_memory:.1f} MB")
    
    @pytest.mark.performance
    def test_concurrent_operations_performance(self, large_csv_file):
        """Test performance under concurrent operations"""
        import concurrent.futures
        import threading
        
        service = CSVDataService(large_csv_file)
        service.load_data()
        
        def perform_operations(thread_id):
            """Perform various operations in parallel"""
            results = {}
            
            # Filtering operation
            start = time.time()
            filtered = service.filter_records(min_priority=random.randint(1, 5))
            results[f'filter_{thread_id}'] = (len(filtered), time.time() - start)
            
            # Summary operation
            start = time.time()
            summary = service.get_data_summary()
            results[f'summary_{thread_id}'] = (summary.total_records, time.time() - start)
            
            # Unique values operation
            start = time.time()
            unique_dcs = service.get_unique_values('dc_id')
            results[f'unique_{thread_id}'] = (len(unique_dcs), time.time() - start)
            
            return results
        
        # Run concurrent operations
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(perform_operations, i) for i in range(5)]
            all_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        assert total_time < 10.0  # All concurrent operations should complete within 10 seconds
        assert len(all_results) == 5
        
        # Verify all operations completed successfully
        for results in all_results:
            assert len(results) == 3  # Each thread should complete 3 operations
            for operation, (result_size, operation_time) in results.items():
                assert result_size > 0
                assert operation_time < 5.0  # Each operation should be reasonably fast
        
        print(f"5 concurrent threads completed in {total_time:.2f} seconds")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_test_repeated_operations(self, large_csv_file):
        """Stress test with repeated operations"""
        service = CSVDataService(large_csv_file)
        service.load_data()
        
        # Perform repeated operations
        num_iterations = 100
        
        start_time = time.time()
        for i in range(num_iterations):
            # Vary the operations to test different code paths
            if i % 3 == 0:
                service.filter_records(min_priority=random.randint(1, 5))
            elif i % 3 == 1:
                service.get_unique_values(random.choice(['dc_id', 'customer_tier', 'sla_level']))
            else:
                service.get_data_summary()
        
        total_time = time.time() - start_time
        avg_time_per_operation = total_time / num_iterations
        
        assert total_time < 60.0  # 100 operations should complete within 1 minute
        assert avg_time_per_operation < 0.6  # Average operation time should be reasonable
        
        print(f"Completed {num_iterations} operations in {total_time:.2f} seconds")
        print(f"Average time per operation: {avg_time_per_operation:.3f} seconds")