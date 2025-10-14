"""
Phase 1 Test: Data validation and cleaning
"""

import pytest
from datetime import date, timedelta
from backend.api.module4.models.data_models import AllocationRecord, CustomerTier, SLALevel
from backend.api.module4.utils.data_validation import DataValidator


class TestDataValidation:
    """Test data validation and cleaning"""
    
    @pytest.fixture
    def validator(self):
        """Create data validator instance"""
        return DataValidator()
    
    @pytest.fixture
    def valid_record(self):
        """Create a valid allocation record"""
        return AllocationRecord(
            dc_id='DC001',
            dc_name='Chicago DC',
            dc_location='Chicago IL',
            dc_region='North America',
            dc_priority=5,
            sku_id='SKU-001',
            sku_name='Test Product',
            sku_category='Networking',
            customer_id='CUST-001',
            customer_name='Test Customer',
            customer_tier=CustomerTier.STRATEGIC,
            customer_region='North America',
            current_inventory=100,
            forecasted_demand=80,
            historical_demand=75,
            revenue_per_unit=100.0,
            cost_per_unit=75.0,
            margin=25.0,
            sla_level=SLALevel.GOLD,
            risk_score=0.1,
            substitution_sku_id=None,
            date=date.today() - timedelta(days=1),
            allocated_quantity=80,
            fulfillment_rate=100.0,
            lead_time_days=3,
            min_order_quantity=5,
            safety_stock=20
        )
    
    def test_validate_empty_records(self, validator):
        """Test validation with empty record list"""
        report = validator.validate_allocation_records([])
        
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.invalid_records == 0
        assert len(report.errors) == 1
        assert "No records provided" in report.errors[0]
    
    def test_validate_single_valid_record(self, validator, valid_record):
        """Test validation with single valid record"""
        report = validator.validate_allocation_records([valid_record])
        
        assert report.total_records == 1
        assert report.valid_records == 1
        assert report.invalid_records == 0
        assert len(report.errors) == 0
        assert report.is_valid
        assert report.validation_success_rate == 100.0
    
    def test_validate_over_allocation(self, validator, valid_record):
        """Test validation when allocated > inventory"""
        invalid_record = valid_record.copy(update={
            'allocated_quantity': 150,  # More than current_inventory (100)
            'fulfillment_rate': 187.5  # Update to match
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert report.invalid_records > 0
        assert any('exceeds current inventory' in error for error in report.errors)
    
    def test_validate_cost_revenue_relationship(self, validator, valid_record):
        """Test validation of cost vs revenue relationship"""
        invalid_record = valid_record.copy(update={
            'cost_per_unit': 120.0,  # More than revenue_per_unit (100.0)
            'margin': -20.0  # Negative margin
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('Cost per unit' in error and 'should be less than revenue' in error 
                  for error in report.errors)
    
    def test_validate_min_order_quantity(self, validator, valid_record):
        """Test validation of minimum order quantity"""
        invalid_record = valid_record.copy(update={
            'allocated_quantity': 3,  # Less than min_order_quantity (5)
            'fulfillment_rate': 3.75  # Update to match
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('below minimum order quantity' in error for error in report.errors)
    
    def test_validate_future_date(self, validator, valid_record):
        """Test validation of future dates"""
        invalid_record = valid_record.copy(update={
            'date': date.today() + timedelta(days=1)  # Future date
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('Future date' in error and 'not allowed' in error for error in report.errors)
    
    def test_validate_high_risk_high_priority(self, validator, valid_record):
        """Test validation of high risk + high priority inconsistency"""
        invalid_record = valid_record.copy(update={
            'risk_score': 0.9,  # High risk
            'dc_priority': 5    # High priority (inconsistent)
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('High risk score' in error and 'inconsistent' in error for error in report.errors)
    
    def test_validate_duplicates(self, validator, valid_record):
        """Test duplicate detection"""
        duplicate_record = valid_record.copy()
        
        report = validator.validate_allocation_records([valid_record, duplicate_record])
        
        assert report.duplicates_found == 1
        assert any('Duplicate record' in error for error in report.errors)
    
    def test_validate_data_consistency_dc(self, validator):
        """Test DC data consistency validation"""
        record1 = AllocationRecord(
            dc_id='DC001',
            dc_name='Chicago DC',
            dc_location='Chicago IL',
            dc_region='North America',
            dc_priority=5,
            sku_id='SKU-001',
            sku_name='Test Product',
            sku_category='Networking',
            customer_id='CUST-001',
            customer_name='Test Customer',
            customer_tier=CustomerTier.STRATEGIC,
            customer_region='North America',
            current_inventory=100,
            forecasted_demand=80,
            historical_demand=75,
            revenue_per_unit=100.0,
            cost_per_unit=75.0,
            margin=25.0,
            sla_level=SLALevel.GOLD,
            risk_score=0.1,
            substitution_sku_id=None,
            date=date.today() - timedelta(days=1),
            allocated_quantity=80,
            fulfillment_rate=100.0,
            lead_time_days=3,
            min_order_quantity=5,
            safety_stock=20
        )
        
        # Same DC ID but different name (inconsistent)
        record2 = record1.copy(update={
            'dc_name': 'Different Name',
            'customer_id': 'CUST-002',
            'sku_id': 'SKU-002'
        })
        
        report = validator.validate_allocation_records([record1, record2])
        
        assert any('DC DC001 has inconsistent names' in error for error in report.errors)
    
    def test_statistical_validations_small_dataset(self, validator, valid_record):
        """Test statistical validation with small dataset"""
        records = [valid_record] * 5  # Very small dataset
        
        report = validator.validate_allocation_records(records)
        
        assert any('Very small dataset' in warning for warning in report.warnings)
    
    def test_clean_data_remove_duplicates(self, validator, valid_record):
        """Test data cleaning - duplicate removal"""
        duplicate_record = valid_record.copy()
        records = [valid_record, duplicate_record]
        
        cleaned_records, stats = validator.clean_data(records, remove_duplicates=True)
        
        assert len(cleaned_records) == 1
        assert stats['duplicates_removed'] == 1
    
    def test_clean_data_fix_inconsistencies(self, validator):
        """Test data cleaning - fix inconsistencies"""
        record1 = AllocationRecord(
            dc_id='DC001',
            dc_name='Chicago DC',
            dc_location='Chicago IL',
            dc_region='North America',
            dc_priority=5,
            sku_id='SKU-001',
            sku_name='Test Product',
            sku_category='Networking',
            customer_id='CUST-001',
            customer_name='Test Customer',
            customer_tier=CustomerTier.STRATEGIC,
            customer_region='North America',
            current_inventory=100,
            forecasted_demand=80,
            historical_demand=75,
            revenue_per_unit=100.0,
            cost_per_unit=75.0,
            margin=25.0,
            sla_level=SLALevel.GOLD,
            risk_score=0.1,
            substitution_sku_id=None,
            date=date.today() - timedelta(days=1),
            allocated_quantity=80,
            fulfillment_rate=100.0,
            lead_time_days=3,
            min_order_quantity=5,
            safety_stock=20
        )
        
        # Same DC ID but wrong name (should be fixed)
        record2 = record1.copy(update={
            'dc_name': 'Wrong Name',  # Will be fixed to 'Chicago DC'
            'customer_id': 'CUST-002',
            'sku_id': 'SKU-002'
        })
        
        # Third record with correct name (most common)
        record3 = record1.copy(update={
            'customer_id': 'CUST-003',
            'sku_id': 'SKU-003'
        })
        
        records = [record1, record2, record3]
        cleaned_records, stats = validator.clean_data(records, fix_inconsistencies=True)
        
        assert len(cleaned_records) == 3
        assert all(r.dc_name == 'Chicago DC' for r in cleaned_records)
        assert stats['dc_name_fixed'] == 1
    
    def test_clean_data_fix_negative_values(self, validator, valid_record):
        """Test data cleaning - fix negative values"""
        invalid_record = valid_record.copy(update={
            'current_inventory': -10,  # Should be fixed to 0
            'allocated_quantity': -5   # Should be fixed to 0
        })
        
        cleaned_records, stats = validator.clean_data([invalid_record])
        
        assert len(cleaned_records) == 1
        assert cleaned_records[0].current_inventory == 0
        assert cleaned_records[0].allocated_quantity == 0
        assert stats['current_inventory_negative_fixed'] == 1
        assert stats['allocated_quantity_negative_fixed'] == 1
    
    def test_clean_data_recalculate_fulfillment_rate(self, validator, valid_record):
        """Test data cleaning - recalculate fulfillment rate"""
        invalid_record = valid_record.copy(update={
            'forecasted_demand': 100,
            'allocated_quantity': 80,
            'fulfillment_rate': 50.0  # Wrong - should be 80.0
        })
        
        cleaned_records, stats = validator.clean_data([invalid_record])
        
        assert len(cleaned_records) == 1
        assert cleaned_records[0].fulfillment_rate == 80.0
        assert stats['fulfillment_rate_recalculated'] == 1
    
    def test_clean_data_recalculate_margin(self, validator, valid_record):
        """Test data cleaning - recalculate margin"""
        invalid_record = valid_record.copy(update={
            'revenue_per_unit': 100.0,
            'cost_per_unit': 80.0,
            'margin': 50.0  # Wrong - should be 20.0
        })
        
        cleaned_records, stats = validator.clean_data([invalid_record])
        
        assert len(cleaned_records) == 1
        assert cleaned_records[0].margin == 20.0
        assert stats['margin_recalculated'] == 1
    
    def test_validate_safety_stock_vs_inventory(self, validator, valid_record):
        """Test validation when safety stock > inventory"""
        invalid_record = valid_record.copy(update={
            'current_inventory': 10,
            'safety_stock': 20  # More than inventory
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('Safety stock' in error and 'exceeds current inventory' in error 
                  for error in report.errors)
    
    def test_validate_historical_vs_forecast_inconsistency(self, validator, valid_record):
        """Test validation of historical vs forecast inconsistency"""
        invalid_record = valid_record.copy(update={
            'historical_demand': 0,
            'forecasted_demand': 2000  # Very high forecast with no history
        })
        
        report = validator.validate_allocation_records([invalid_record])
        
        assert any('High forecasted demand' in error and 'zero historical demand' in error 
                  for error in report.errors)