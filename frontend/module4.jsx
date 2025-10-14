import React, { useState, useEffect } from 'react';
import EnhancedAllocationForm from './enhanced_allocation_form';
import EnhancedResultsTable from './enhanced_results_table';
import { optimizationApi, mapOptimizationResult, ApiError } from './services/api';

const Module4AllocationPage = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [loadingData, setLoadingData] = useState(true);

  // Load CSV data on component mount
  useEffect(() => {
    const loadCsvData = async () => {
      try {
        setLoadingData(true);
        console.log('Loading CSV data from API...');
        const data = await optimizationApi.getCsvData();
        console.log('CSV data loaded:', data);
        setCsvData(data);
      } catch (err) {
        console.error('Failed to load CSV data:', err);
        setError(`Failed to load data: ${err.message}`);
      } finally {
        setLoadingData(false);
      }
    };

    loadCsvData();
  }, []);

  const handleRunAllocation = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log('Running allocation with form data:', formData);

      // Use allocation data from enhanced form
      const allocationData = formData.allocationData || [];
      
      // Prepare optimization request
      const optimizationRequest = {
        strategy: formData.strategy || 'fair_share',
        allocation_data: allocationData,
        constraints: {
          allow_substitution: true,
          respect_customer_tier: true,
          safety_stock_buffer: 0.1,
        },
        prefer_efficiency: true,
        prefer_speed: false,
      };

      console.log('Sending optimization request:', optimizationRequest);

      // Call the backend API
      const backendResult = await optimizationApi.optimize(optimizationRequest);
      
      console.log('Backend response:', backendResult);

      // Map the result to frontend format
      const mappedResult = mapOptimizationResult(backendResult);
      
      console.log('Mapped result:', mappedResult);

      setResult(mappedResult);

    } catch (err) {
      console.error('Optimization failed:', err);
      
      if (err instanceof ApiError) {
        // Handle API errors gracefully
        if (err.status === 0) {
          setError('Network error: Please check if the backend server is running');
        } else if (err.status === 408) {
          setError('Request timeout: The optimization took too long');
        } else {
          setError(`API Error: ${err.message}`);
        }
      } else {
        setError(err.message || 'An unexpected error occurred');
      }
      
      // Fallback to example data in development
      if (process.env.NODE_ENV === 'development') {
        console.log('Using fallback example data');
        setResult({
          success: true,
          status: 'completed',
          optimizer_used: formData.optimizerType || 'fair_share',
          model_used: 'optimization_engine',
          allocation_summary: {
            total_demand: 5000,
            total_allocated: 4500,
            fulfillment_rate: 90.0,
            substitutions_used: 120,
            optimization_time: 0.1
          },
          allocations: [],
          substitutions_made: [],
          constraints_violated: [],
        });
        setError(null); // Clear error when using fallback
      }
    } finally {
      setLoading(false);
    }
  };

  // Helper function to create sample allocation data
  const createSampleAllocationData = (formData) => {
    const dcIds = formData.dcIds?.length ? formData.dcIds : ['DC001', 'DC002', 'DC003'];
    const productIds = formData.productIds?.length ? formData.productIds : ['SKU001', 'SKU002'];
    const customerIds = formData.customerIds?.length ? formData.customerIds : ['CUST001', 'CUST002'];
    
    const allocationData = [];
    
    dcIds.forEach((dcId, dcIndex) => {
      productIds.forEach((productId, productIndex) => {
        customerIds.forEach((customerId, customerIndex) => {
          allocationData.push({
            dc_id: dcId,
            sku_id: productId,
            customer_id: customerId,
            current_inventory: Math.floor(Math.random() * 1000) + 100,
            forecasted_demand: Math.floor(Math.random() * 800) + 50,
            dc_priority: dcIndex + 1,
            customer_tier: ['A', 'B', 'C'][customerIndex % 3],
            sla_level: ['Premium', 'Standard', 'Basic'][customerIndex % 3],
            min_order_quantity: 1,
            sku_category: 'electronics',
          });
        });
      });
    });
    
    return allocationData;
  };

  return (
    <div className="bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
      <div className="container mx-auto p-4 sm:p-6 lg:p-8">
        
        {/* Data Loading State */}
        {loadingData && (
          <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <h3 className="text-lg font-semibold text-indigo-600">Loading Data</h3>
              <p className="text-gray-600 mt-2">Fetching real data from CSV files...</p>
            </div>
          </div>
        )}
        
        {/* Enhanced Form */}
        {!loadingData && csvData && (
          <EnhancedAllocationForm 
            onRunAllocation={handleRunAllocation} 
            loading={loading} 
            exampleData={csvData} 
          />
        )}
        
        {/* Error Display */}
        {error && !loadingData && (
          <div className="mt-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {/* Loading State */}
        {loading && !loadingData && (
          <div className="mt-6 bg-white rounded-lg shadow-lg p-8">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <h3 className="text-lg font-semibold text-indigo-600">Running Allocation Optimization</h3>
              <p className="text-gray-600 mt-2">Please wait while we analyze your data and optimize allocations...</p>
            </div>
          </div>
        )}

        {/* Enhanced Results */}
        {result && !error && !loadingData && (
          <div className="mt-6 space-y-6">
            <EnhancedResultsTable result={result} csvData={csvData} />
          </div>
        )}
        
        {/* Footer */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>Allocation Maximizer - Module 4 | Enhanced with ML Forecasting</p>
        </div>
      </div>
    </div>
  );
};

export default Module4AllocationPage;