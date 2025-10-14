"""
Phase 1 Test: CSV loading functionality
"""

import pytest
import tempfile
import csv
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, mock_open

from backend.api.module4.services.csv_service import CSVDataService
from backend.api.module4.models.data_models import AllocationRecord, CustomerTier, SLALevel


class TestCSVLoader:
    """Test CSV data loading functionality"""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing"""
        return [
            {
                'dc_id': 'DC001',
                'dc_name': 'Chicago DC',
                'dc_location': 'Chicago IL',
                'dc_region': 'North America',
                'dc_priority': 5,
                'sku_id': 'SKU-NET-001',
                'sku_name': 'Network Switch 48-Port',
                'sku_category': 'Networking',
                'customer_id': 'CUST-001',
                'customer_name': 'TechCorp Inc',
                'customer_tier': 'Strategic',
                'customer_region': 'North America',
                'current_inventory': 500,
                'forecasted_demand': 450,
                'historical_demand': 420,
                'revenue_per_unit': 1250.00,
                'cost_per_unit': 950.00,
                'margin': 24.0,
                'sla_level': 'Gold',
                'risk_score': 0.15,
                'substitution_sku_id': 'SKU-NET-002',
                'date': '2024-10-14',
                'allocated_quantity': 400,
                'fulfillment_rate': 88.9,
                'lead_time_days': 3,
                'min_order_quantity': 10,
                'safety_stock': 50
            },
            {
                'dc_id': 'DC002',
                'dc_name': 'Dallas DC',
                'dc_location': 'Dallas TX',
                'dc_region': 'North America',
                'dc_priority': 4,
                'sku_id': 'SKU-STO-001',
                'sku_name': 'Storage Array 10TB',
                'sku_category': 'Storage',
                'customer_id': 'CUST-002',
                'customer_name': 'CloudBase Inc',
                'customer_tier': 'Premium',
                'customer_region': 'North America',
                'current_inventory': 150,
                'forecasted_demand': 200,
                'historical_demand': 180,
                'revenue_per_unit': 8500.00,
                'cost_per_unit': 6500.00,
                'margin': 23.5,
                'sla_level': 'Silver',
                'risk_score': 0.16,
                'substitution_sku_id': None,  # Test None value
                'date': '2024-10-14',
                'allocated_quantity': 150,
                'fulfillment_rate': 75.0,
                'lead_time_days': 4,
                'min_order_quantity': 2,
                'safety_stock': 25
            }
        ]
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            fieldnames = sample_csv_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_csv_data)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_csv_service_initialization(self):
        """Test CSV service initialization"""
        service = CSVDataService("test.csv")
        assert service.csv_path == Path("test.csv")
        assert service._data is None
        assert service._last_loaded is None
    
    def test_load_data_file_not_found(self):
        """Test loading data when CSV file doesn't exist"""
        service = CSVDataService("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            service.load_data()
    
    def test_load_data_success(self, temp_csv_file):
        """Test successful data loading"""
        service = CSVDataService(temp_csv_file)
        
        data = service.load_data()
        
        assert data is not None
        assert data.total_records == 2
        assert len(data.records) == 2
        assert service._last_loaded is not None
        
        # Check first record
        record = data.records[0]
        assert record.dc_id == 'DC001'
        assert record.customer_tier == CustomerTier.STRATEGIC
        assert record.sla_level == SLALevel.GOLD
        assert record.date == date(2024, 10, 14)
        assert record.substitution_sku_id == 'SKU-NET-002'
        
        # Check second record (with None substitution)
        record2 = data.records[1]
        assert record2.substitution_sku_id is None
    
    def test_load_data_caching(self, temp_csv_file):
        """Test data caching functionality"""
        service = CSVDataService(temp_csv_file)
        
        # First load
        data1 = service.load_data()
        first_load_time = service._last_loaded
        
        # Second load (should use cache)
        data2 = service.load_data()
        second_load_time = service._last_loaded
        
        assert data1 is data2  # Same object reference
        assert first_load_time == second_load_time
        
        # Force reload
        data3 = service.load_data(force_reload=True)
        third_load_time = service._last_loaded
        
        assert data3 is not data1  # Different object
        assert third_load_time > first_load_time
    
    def test_load_invalid_data(self):
        """Test loading invalid CSV data"""
        invalid_csv_content = """dc_id,invalid_field
DC001,invalid_value"""
        
        with patch('builtins.open', mock_open(read_data=invalid_csv_content)):
            service = CSVDataService("invalid.csv")
            
            with pytest.raises(ValueError):
                service.load_data()
    
    def test_get_data_summary(self, temp_csv_file):
        """Test data summary generation"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        summary = service.get_data_summary()
        
        assert summary.total_records == 2
        assert summary.unique_dcs == 2
        assert summary.unique_customers == 2
        assert summary.unique_skus == 2
        assert summary.unique_regions == 1  # Both North America
        
        assert summary.total_inventory == 650  # 500 + 150
        assert summary.total_demand == 650  # 450 + 200
        assert summary.total_allocated == 550  # 400 + 150
        
        assert 'Strategic' in summary.records_by_tier
        assert 'Premium' in summary.records_by_tier
        assert summary.records_by_tier['Strategic'] == 1
        assert summary.records_by_tier['Premium'] == 1
    
    def test_filter_records(self, temp_csv_file):
        """Test record filtering functionality"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # Filter by DC
        dc_records = service.filter_records(dc_ids=['DC001'])
        assert len(dc_records) == 1
        assert dc_records[0].dc_id == 'DC001'
        
        # Filter by customer tier
        strategic_records = service.filter_records(customer_tiers=['Strategic'])
        assert len(strategic_records) == 1
        assert strategic_records[0].customer_tier == CustomerTier.STRATEGIC
        
        # Filter by priority
        high_priority_records = service.filter_records(min_priority=5)
        assert len(high_priority_records) == 1
        assert high_priority_records[0].dc_priority == 5
        
        # Filter by risk score
        low_risk_records = service.filter_records(max_risk_score=0.15)
        assert len(low_risk_records) == 1
        assert low_risk_records[0].risk_score == 0.15
    
    def test_get_records_by_entity(self, temp_csv_file):
        """Test getting records by specific entity"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # By DC
        dc_records = service.get_records_by_dc('DC001')
        assert len(dc_records) == 1
        assert dc_records[0].dc_id == 'DC001'
        
        # By customer
        customer_records = service.get_records_by_customer('CUST-002')
        assert len(customer_records) == 1
        assert customer_records[0].customer_id == 'CUST-002'
        
        # By SKU
        sku_records = service.get_records_by_sku('SKU-NET-001')
        assert len(sku_records) == 1
        assert sku_records[0].sku_id == 'SKU-NET-001'
    
    def test_get_unique_values(self, temp_csv_file):
        """Test getting unique values for fields"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        # Test various fields
        dc_ids = service.get_unique_values('dc_id')
        assert set(dc_ids) == {'DC001', 'DC002'}
        
        customer_tiers = service.get_unique_values('customer_tier')
        assert set(customer_tiers) == {'Strategic', 'Premium'}
        
        categories = service.get_unique_values('sku_category')
        assert set(categories) == {'Networking', 'Storage'}
        
        # Test invalid field
        with pytest.raises(ValueError):
            service.get_unique_values('invalid_field')
    
    def test_data_staleness(self, temp_csv_file):
        """Test data staleness detection"""
        service = CSVDataService(temp_csv_file)
        
        # No data loaded yet
        assert service.is_data_stale()
        
        # Load data
        service.load_data()
        assert not service.is_data_stale()
        
        # Mock old timestamp
        old_time = datetime.now() - datetime.timedelta(minutes=20)
        service._last_loaded = old_time
        
        assert service.is_data_stale(max_age_minutes=15)
        assert not service.is_data_stale(max_age_minutes=30)
    
    def test_refresh_if_stale(self, temp_csv_file):
        """Test refreshing stale data"""
        service = CSVDataService(temp_csv_file)
        
        # No data loaded
        data1 = service.refresh_if_stale()
        assert data1 is not None
        
        # Data is fresh
        data2 = service.refresh_if_stale()
        assert data1 is data2
        
        # Make data stale
        old_time = datetime.now() - datetime.timedelta(minutes=20)
        service._last_loaded = old_time
        
        # Should refresh
        data3 = service.refresh_if_stale(max_age_minutes=15)
        assert data3 is not data1
    
    def test_validation_report(self, temp_csv_file):
        """Test validation report generation"""
        service = CSVDataService(temp_csv_file)
        service.load_data()
        
        report = service.get_validation_report()
        
        assert report is not None
        assert report.total_records == 2
        assert report.valid_records == 2
        assert report.invalid_records == 0
        assert report.is_valid