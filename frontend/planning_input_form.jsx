import React, { useState, useEffect } from 'react';
import { FaChevronDown, FaChevronUp, FaInfoCircle, FaCalendarAlt, FaUsers, FaWarehouse, FaBox } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';

const PlanningInputForm = ({ onGeneratePlan, loading, initialData }) => {
  const currentDate = new Date();
  const currentMonth = currentDate.toLocaleString('default', { month: 'long' });
  const currentYear = currentDate.getFullYear();
  
  const [isExpanded, setIsExpanded] = useState(true);
  const [formData, setFormData] = useState({
    planMonth: currentMonth,
    planYear: currentYear,
    allocationMethod: 'fairShare',
    includeAllCustomers: true,
    includeAllProducts: true,
    includeAllDCs: true,
    priority: 5, // Default middle value for priority slider
  });

  // Update form data when initialData changes
  useEffect(() => {
    if (initialData) {
      setFormData(prev => ({
        ...prev,
        planMonth: initialData.planMonth || prev.planMonth,
        planYear: initialData.planYear || prev.planYear,
        allocationMethod: initialData.allocationMethod === 'Fair Share' ? 'fairShare' : 'priority',
      }));
    }
  }, [initialData]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleSliderChange = (e) => {
    setFormData({
      ...formData,
      priority: parseInt(e.target.value),
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onGeneratePlan(formData);
  };

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  const years = Array.from({ length: 5 }, (_, i) => currentYear + i);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden mb-6 transition-all duration-200 hover:shadow-md">
      <div 
        className="flex justify-between items-center p-5 cursor-pointer hover:bg-gray-50 transition-colors duration-150"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          <FaCalendarAlt className="mr-3 text-indigo-600 text-xl" />
          Monthly Plan Settings
        </h2>
        <div className="flex items-center space-x-3">
          {initialData && (
            <span className="px-3 py-1 bg-indigo-50 text-indigo-700 text-sm font-medium rounded-full">
              Current: {initialData.planMonth} {initialData.planYear}
            </span>
          )}
          <div className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors">
            {isExpanded ? (
              <FaChevronUp className="text-gray-500" />
            ) : (
              <FaChevronDown className="text-gray-500" />
            )}
          </div>
        </div>
      </div>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <form onSubmit={handleSubmit} className="p-4 pt-0 border-t border-gray-100">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                {/* Plan Month & Year */}
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">Plan Month</label>
                  <div className="relative">
                    <select
                      name="planMonth"
                      value={formData.planMonth}
                      onChange={handleInputChange}
                      className="block w-full pl-3 pr-10 py-2.5 text-base border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent appearance-none transition-all duration-200 hover:border-gray-300"
                    >
                      {months.map(month => (
                        <option key={month} value={month}>{month}</option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                      <FaChevronDown className="h-4 w-4 text-gray-400" />
                    </div>
                  </div>
                </div>
                
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">Plan Year</label>
                  <div className="relative">
                    <select
                      name="planYear"
                      value={formData.planYear}
                      onChange={handleInputChange}
                      className="block w-full pl-3 pr-10 py-2.5 text-base border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent appearance-none transition-all duration-200 hover:border-gray-300"
                    >
                      {years.map(year => (
                        <option key={year} value={year}>{year}</option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                      <FaChevronDown className="h-4 w-4 text-gray-400" />
                    </div>
                  </div>
                </div>
                
                {/* Allocation Method */}
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">Allocation Method</label>
                  <div className="inline-flex rounded-md shadow-sm" role="group">
                    <button
                      type="button"
                      onClick={() => setFormData({...formData, allocationMethod: 'fairShare'})}
                      className={`px-4 py-2 text-sm font-medium rounded-l-lg border ${formData.allocationMethod === 'fairShare' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'}`}
                    >
                      Fair Share
                    </button>
                    <button
                      type="button"
                      onClick={() => setFormData({...formData, allocationMethod: 'priority'})}
                      className={`px-4 py-2 text-sm font-medium rounded-r-lg border-t border-b border-r ${formData.allocationMethod === 'priority' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'}`}
                    >
                      Priority Based
                    </button>
                  </div>
                </div>
              </div>
              
              {/* Scope Selection */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="includeAllCustomers"
                    name="includeAllCustomers"
                    checked={formData.includeAllCustomers}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                  />
                  <label htmlFor="includeAllCustomers" className="text-sm text-gray-700 flex items-center">
                    <FaUsers className="mr-1 text-indigo-500" />
                    Include All Customers
                  </label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="includeAllProducts"
                    name="includeAllProducts"
                    checked={formData.includeAllProducts}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                  />
                  <label htmlFor="includeAllProducts" className="text-sm text-gray-700 flex items-center">
                    <FaBox className="mr-1 text-indigo-500" />
                    Include All Products
                  </label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="includeAllDCs"
                    name="includeAllDCs"
                    checked={formData.includeAllDCs}
                    onChange={handleInputChange}
                    className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                  />
                  <label htmlFor="includeAllDCs" className="text-sm text-gray-700 flex items-center">
                    <FaWarehouse className="mr-1 text-indigo-500" />
                    Include All Distribution Centers
                  </label>
                </div>
              </div>
              
              {/* Info Box */}
              <div className="bg-blue-50 p-4 rounded-md mb-6 flex">
                <FaInfoCircle className="text-blue-500 mt-1 flex-shrink-0" />
                <div className="ml-3">
                  <p className="text-sm text-blue-700">
                    This allocation plan will be used as the basis for order fulfillment throughout the month. You can publish it to your ERP system once generated.
                  </p>
                </div>
              </div>
              
              {/* Submit Button */}
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-300 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Generating Plan...
                    </>
                  ) : (
                    'Reallocation Run'
                  )}
                </button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PlanningInputForm;
