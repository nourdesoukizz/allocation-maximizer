"""
Phase 0 Test: Verify backend directory structure
"""

import os
import pytest
from pathlib import Path


class TestDirectoryStructure:
    """Test backend directory structure is correctly set up"""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory"""
        return Path(__file__).parent.parent.parent.parent
    
    def test_backend_structure_exists(self, project_root):
        """Test that backend directory structure exists"""
        backend_path = project_root / "backend"
        api_path = backend_path / "api" / "module4"
        
        assert backend_path.exists(), "Backend directory should exist"
        assert api_path.exists(), "Backend API Module4 directory should exist"
        
        # Check main directories
        required_dirs = [
            "models",
            "services", 
            "routers",
            "ml_models",
            "utils"
        ]
        
        for dir_name in required_dirs:
            dir_path = api_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist in backend/api/module4"
    
    def test_ml_models_moved(self, project_root):
        """Test that ML model files were moved to ml_models directory"""
        ml_models_path = project_root / "backend" / "ml_models"
        
        assert ml_models_path.exists(), "ML models directory should exist"
        
        # Check for expected ML model files
        expected_files = [
            "LSTM.py",
            "prophet.py", 
            "random-forest.py",
            "sarima.py",
            "sba-forecasting.py",
            "xgboost.py"
        ]
        
        for file_name in expected_files:
            file_path = ml_models_path / file_name
            assert file_path.exists(), f"ML model file {file_name} should exist in ml_models directory"
    
    def test_data_directory_exists(self, project_root):
        """Test that data directory with CSV exists"""
        data_path = project_root / "data"
        csv_path = data_path / "allocation_data.csv"
        
        assert data_path.exists(), "Data directory should exist"
        assert csv_path.exists(), "allocation_data.csv should exist"
    
    def test_tests_structure_exists(self, project_root):
        """Test that tests directory structure exists"""
        tests_path = project_root / "tests"
        
        assert tests_path.exists(), "Tests directory should exist"
        
        # Check frontend test structure
        frontend_tests = tests_path / "frontend"
        assert frontend_tests.exists(), "Frontend tests directory should exist"
        assert (frontend_tests / "unit").exists(), "Frontend unit tests directory should exist"
        assert (frontend_tests / "integration").exists(), "Frontend integration tests directory should exist"
        assert (frontend_tests / "e2e").exists(), "Frontend e2e tests directory should exist"
        
        # Check backend test structure
        backend_tests = tests_path / "backend"
        assert backend_tests.exists(), "Backend tests directory should exist"
        assert (backend_tests / "unit").exists(), "Backend unit tests directory should exist"
        assert (backend_tests / "integration").exists(), "Backend integration tests directory should exist"
        assert (backend_tests / "performance").exists(), "Backend performance tests directory should exist"
    
    def test_test_config_files_exist(self, project_root):
        """Test that test configuration files exist"""
        tests_path = project_root / "tests"
        
        assert (tests_path / "pytest.ini").exists(), "pytest.ini should exist"
        assert (tests_path / "jest.config.js").exists(), "jest.config.js should exist"
        assert (tests_path / "frontend" / "setupTests.js").exists(), "setupTests.js should exist"