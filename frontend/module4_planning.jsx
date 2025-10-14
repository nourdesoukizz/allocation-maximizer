import React, { useState, useEffect } from 'react';
import { FaFilter, FaUpload, FaSync, FaCloudUploadAlt, FaTable, FaChartBar, FaInfoCircle } from 'react-icons/fa';
import { motion } from 'framer-motion';
import PlanningInputForm from './planning_input_form';
import PlanningTable from './planning_table';
import examplePlanningData from './example_planning2.json';

const Module4Planning = () => {
  const [planData, setPlanData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState('table'); // 'table' or 'dashboard'
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishSuccess, setPublishSuccess] = useState(false);

  // Load the example planning data on component mount
  useEffect(() => {
    // Simulate API call to fetch current month's plan
    const fetchInitialPlan = async () => {
      setLoading(true);
      try {
        // In a real app, this would be an API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setPlanData(examplePlanningData);
      } catch (err) {
        setError('Failed to load the current month planning data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialPlan();
  }, []);

  const handleGeneratePlan = async (settings) => {
    setLoading(true);
    setError(null);
    setPublishSuccess(false);
    
    try {
      // In a real application, this would be an API call to a backend service
      // For now, we'll simulate an API call with a timeout and use example data
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Process the example data based on the settings
      const processedData = processDataBasedOnSettings(settings);
      
      setPlanData(processedData);
    } catch (err) {
      console.error('Error generating plan:', err);
      setError('An error occurred during plan generation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePublishToERP = async () => {
    setIsPublishing(true);
    setPublishSuccess(false);
    
    try {
      // Simulate API call to publish to ERP
      await new Promise(resolve => setTimeout(resolve, 2500));
      setPublishSuccess(true);
      
      // Reset publish success message after 5 seconds
      setTimeout(() => {
        setPublishSuccess(false);
      }, 5000);
    } catch (err) {
      setError('Failed to publish to ERP system. Please try again.');
    } finally {
      setIsPublishing(false);
    }
  };

  const handleUploadAllocations = () => {
    // This would open a file upload dialog in a real application
    alert('Upload functionality would be implemented here.');
  };

  // Function to process example data based on settings
  const processDataBasedOnSettings = (settings) => {
    // Clone the example data to avoid mutating the original
    const processedData = JSON.parse(JSON.stringify(examplePlanningData));
    
    // Apply allocation method (fair share or priority)
    if (settings.allocationMethod === 'fairShare') {
      processedData.summary.allocationMethod = 'Fair Share';
      // In a real app, this would adjust the allocation numbers
    } else if (settings.allocationMethod === 'priority') {
      processedData.summary.allocationMethod = 'Priority Based';
      // Adjust allocations to favor high priority customers
      processedData.allocations.forEach(allocation => {
        if (allocation.customerPriority === 'High') {
          allocation.allocatedQty = Math.min(
            allocation.forecastQty, 
            Math.floor(allocation.allocatedQty * 1.15)
          );
        }
      });
    }
    
    // Update other settings in the plan data
    processedData.summary.generatedDate = new Date().toISOString();
    processedData.summary.planMonth = settings.planMonth;
    processedData.summary.planYear = settings.planYear;
    
    return processedData;
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      <div className="container mx-auto p-4 sm:p-6 lg:p-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800">Monthly Allocation Planning</h1>
          <p className="mt-2 text-gray-600">
            Generate and manage monthly allocation plans for optimal distribution across customers.
          </p>
        </div>
        
        {/* Plan Settings Input Form */}
        <PlanningInputForm 
          onGeneratePlan={handleGeneratePlan} 
          loading={loading}
          initialData={planData?.summary || null}
        />
        
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {publishSuccess && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-50 border-l-4 border-green-400 p-4 mb-6"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-green-700">Plan successfully published to ERP system!</p>
              </div>
            </div>
          </motion.div>
        )}
        
        {planData && !loading && (
          <div className="mt-6">
            {/* View Toggle and Action Buttons */}
            <div className="flex flex-wrap justify-between items-center mb-6">
              <div className="flex space-x-2 mb-4 md:mb-0">
                <button
                  onClick={() => setView('table')}
                  className={`px-4 py-2 rounded-md flex items-center ${view === 'table' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700 border border-gray-300'}`}
                >
                  <FaTable className="mr-2" />
                  Table View
                </button>
                {/* Dashboard View to be implemented 
                <button
                  onClick={() => setView('dashboard')}
                  className={`px-4 py-2 rounded-md flex items-center ${view === 'dashboard' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-700 border border-gray-300'}`}
                >
                  <FaChartBar className="mr-2" />
                  Dashboard
                </button>
                */}
              </div>
              
              <div className="flex space-x-2">
                <button
                  onClick={handleUploadAllocations}
                  className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-md flex items-center hover:bg-gray-50"
                >
                  <FaUpload className="mr-2" />
                  Upload
                </button>
                <button
                  onClick={handlePublishToERP}
                  disabled={isPublishing || !planData}
                  className={`px-4 py-2 rounded-md flex items-center ${isPublishing ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'} text-white`}
                >
                  {isPublishing ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Publishing...
                    </>
                  ) : (
                    <>
                      <FaCloudUploadAlt className="mr-2" />
                      Publish to ERP
                    </>
                  )}
                </button>
              </div>
            </div>
            
            {/* Plan Summary */}
            <div className="bg-white p-4 rounded-lg shadow-sm mb-6">
              <div className="flex flex-wrap justify-between items-center">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800">
                    {planData.summary.planMonth} {planData.summary.planYear} Allocation Plan
                  </h3>
                  <p className="text-sm text-gray-600">
                    Generated on {new Date(planData.summary.generatedDate).toLocaleDateString()}
                  </p>
                </div>
                <div className="mt-2 md:mt-0">
                  <span className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                    Method: {planData.summary.allocationMethod}
                  </span>
                  <span className="ml-2 bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                    {planData.summary.totalAllocations} Total Allocations
                  </span>
                </div>
              </div>
            </div>
            
            {/* Planning Table or Dashboard based on view state */}
            {view === 'table' ? (
              <PlanningTable planData={planData} />
            ) : (
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex justify-center items-center h-64">
                  <div className="text-center">
                    <FaChartBar className="mx-auto text-4xl text-gray-300 mb-4" />
                    <p className="text-gray-500">Dashboard view will be implemented in a future update.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {loading && (
          <div className="flex justify-center items-center h-64">
            <div className="text-center">
              <svg className="animate-spin h-12 w-12 text-indigo-600 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-gray-600">Loading allocation plan data...</p>
            </div>
          </div>
        )}
        
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>Module 4 of the Supply Chain Execution Optimization Suite.</p>
        </div>
      </div>
    </div>
  );
};

export default Module4Planning;
