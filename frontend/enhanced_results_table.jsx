import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaChartLine, FaEye, FaTimes, FaCube, FaBuilding, FaUser, FaArrowUp, FaArrowDown, FaMinus } from 'react-icons/fa';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Mock ML model forecasting data
const generateForecastData = (allocation) => {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  
  const models = {
    LSTM: [],
    Prophet: [],
    XGBoost: [],
    RandomForest: [],
    SARIMA: []
  };

  // Generate mock forecast data for each model
  months.forEach((month, index) => {
    const baseValue = allocation.allocated_quantity;
    const trend = index * 5; // Small upward trend
    const noise = () => (Math.random() - 0.5) * 20;

    models.LSTM.push({
      month,
      value: Math.max(0, baseValue + trend + noise() + Math.sin(index) * 10),
      confidence: 0.85 + Math.random() * 0.1
    });

    models.Prophet.push({
      month,
      value: Math.max(0, baseValue + trend * 0.8 + noise() + Math.cos(index) * 8),
      confidence: 0.82 + Math.random() * 0.12
    });

    models.XGBoost.push({
      month,
      value: Math.max(0, baseValue + trend * 1.2 + noise()),
      confidence: 0.88 + Math.random() * 0.08
    });

    models.RandomForest.push({
      month,
      value: Math.max(0, baseValue + trend * 0.9 + noise() + Math.sin(index * 0.5) * 6),
      confidence: 0.84 + Math.random() * 0.1
    });

    models.SARIMA.push({
      month,
      value: Math.max(0, baseValue + trend * 0.7 + noise() + Math.sin(index * 2) * 12),
      confidence: 0.79 + Math.random() * 0.14
    });
  });

  return models;
};

const ForecastModal = ({ allocation, isOpen, onClose, csvData }) => {
  const [selectedModel, setSelectedModel] = useState('LSTM');
  
  if (!isOpen || !allocation) return null;

  // Lookup functions for readable names
  const getCustomerName = (customerId) => {
    if (!csvData?.customers) return customerId;
    const customer = csvData.customers.find(c => c.id === customerId);
    return customer ? customer.name : customerId;
  };

  const getDCName = (dcId) => {
    if (!csvData?.distributionCenters) return dcId;
    const dc = csvData.distributionCenters.find(d => d.id === dcId);
    return dc ? dc.name : dcId;
  };

  const getProductName = (productId) => {
    if (!csvData?.products) return productId;
    const product = csvData.products.find(p => p.id === productId);
    return product ? product.name : productId;
  };

  const forecastData = generateForecastData(allocation);
  
  // Generate consistent historical data (should not change when switching models)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const baseValue = allocation.allocated_quantity;
  const historicalData = months.map((month, index) => {
    if (index < 6) {
      // Use allocation ID as seed for consistent historical data
      const seed = allocation.dc_id?.charCodeAt(0) || 1;
      const deterministicValue = Math.sin(index + seed) * 15 + (index * 3);
      return baseValue + deterministicValue;
    }
    return null;
  });

  const chartData = forecastData[selectedModel].map((item, index) => ({
    ...item,
    actual: historicalData[index], // Use consistent historical data
    upperBound: item.value + (item.value * 0.1),
    lowerBound: item.value - (item.value * 0.1)
  }));

  const modelMetrics = {
    LSTM: { accuracy: 92.3, mae: 15.2, rmse: 22.1, r2: 0.891 },
    Prophet: { accuracy: 88.7, mae: 18.4, rmse: 26.3, r2: 0.863 },
    XGBoost: { accuracy: 94.1, mae: 12.8, rmse: 19.5, r2: 0.912 },
    RandomForest: { accuracy: 89.5, mae: 17.1, rmse: 24.8, r2: 0.874 },
    SARIMA: { accuracy: 85.2, mae: 21.6, rmse: 30.2, r2: 0.842 }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white rounded-lg shadow-2xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">ML Forecasting Models</h2>
                <p className="text-gray-600 mt-1">
                  DC: {getDCName(allocation.dc_id)} | SKU: {getProductName(allocation.sku_id)} | Customer: {getCustomerName(allocation.customer_id)}
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <FaTimes className="text-gray-500" />
              </button>
            </div>
          </div>

          <div className="p-6">
            {/* Model Selection */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3">Select ML Model</h3>
              <div className="flex flex-wrap gap-2">
                {Object.keys(modelMetrics).map((model) => (
                  <button
                    key={model}
                    onClick={() => setSelectedModel(model)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      selectedModel === model
                        ? 'bg-indigo-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {model}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Chart */}
              <div className="lg:col-span-2">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-4">
                    {selectedModel} Forecast - Next 12 Months
                  </h4>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value, name) => [
                          Math.round(value), 
                          name === 'value' ? 'Forecast' : name === 'actual' ? 'Actual' : name
                        ]}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="actual" 
                        stroke="#10B981" 
                        strokeWidth={3}
                        name="Historical"
                        connectNulls={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#6366F1" 
                        strokeWidth={3}
                        name="Forecast"
                        strokeDasharray="5 5"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="upperBound" 
                        stroke="#EF4444" 
                        strokeWidth={1}
                        name="Upper Bound"
                        strokeDasharray="2 2"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="lowerBound" 
                        stroke="#EF4444" 
                        strokeWidth={1}
                        name="Lower Bound"
                        strokeDasharray="2 2"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Model Metrics */}
              <div className="space-y-4">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-800 mb-3">Model Performance</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Accuracy</span>
                      <span className="font-semibold text-green-600">
                        {modelMetrics[selectedModel].accuracy}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">MAE</span>
                      <span className="font-semibold">
                        {modelMetrics[selectedModel].mae}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">RMSE</span>
                      <span className="font-semibold">
                        {modelMetrics[selectedModel].rmse}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">R²</span>
                      <span className="font-semibold text-blue-600">
                        {modelMetrics[selectedModel].r2}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800 mb-2">Current Allocation</h4>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-blue-600">Allocated:</span>
                      <span className="ml-2 font-semibold">{allocation.allocated_quantity}</span>
                    </div>
                    <div>
                      <span className="text-blue-600">Demand:</span>
                      <span className="ml-2 font-semibold">{allocation.forecasted_demand}</span>
                    </div>
                    <div>
                      <span className="text-blue-600">Efficiency:</span>
                      <span className="ml-2 font-semibold">{allocation.allocation_efficiency?.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <h4 className="font-semibold text-yellow-800 mb-2">Model Insights</h4>
                  <ul className="text-sm text-yellow-700 space-y-1">
                    <li>• {selectedModel} shows {modelMetrics[selectedModel].accuracy > 90 ? 'high' : 'moderate'} accuracy</li>
                    <li>• Trend indicates {chartData[chartData.length - 1].value > chartData[0].value ? 'increasing' : 'stable'} demand</li>
                    <li>• Confidence interval: ±{Math.round(chartData[0].value * 0.1)} units</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Model Comparison */}
            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 mb-4">Model Comparison</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2">Model</th>
                      <th className="text-right py-2">Accuracy</th>
                      <th className="text-right py-2">MAE</th>
                      <th className="text-right py-2">RMSE</th>
                      <th className="text-right py-2">R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(modelMetrics).map(([model, metrics]) => (
                      <tr key={model} className={selectedModel === model ? 'bg-indigo-50' : ''}>
                        <td className="py-2 font-medium">{model}</td>
                        <td className="text-right py-2">{metrics.accuracy}%</td>
                        <td className="text-right py-2">{metrics.mae}</td>
                        <td className="text-right py-2">{metrics.rmse}</td>
                        <td className="text-right py-2">{metrics.r2}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

const EnhancedResultsTable = ({ result, csvData }) => {
  const [selectedAllocation, setSelectedAllocation] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  // Lookup functions to map IDs to readable names
  const getCustomerName = (customerId) => {
    if (!csvData?.customers) return customerId;
    const customer = csvData.customers.find(c => c.id === customerId);
    return customer ? customer.name : customerId;
  };

  const getDCName = (dcId) => {
    if (!csvData?.distributionCenters) return dcId;
    const dc = csvData.distributionCenters.find(d => d.id === dcId);
    return dc ? dc.name : dcId;
  };

  const getProductName = (productId) => {
    if (!csvData?.products) return productId;
    const product = csvData.products.find(p => p.id === productId);
    return product ? product.name : productId;
  };

  if (!result?.allocations) {
    return (
      <div className="text-center p-8 text-gray-500">
        No allocation results to display. Run an optimization to see results.
      </div>
    );
  }

  const handleRowClick = (allocation) => {
    setSelectedAllocation(allocation);
    setModalOpen(true);
  };

  const getEfficiencyColor = (efficiency) => {
    if (efficiency >= 90) return 'text-green-600 bg-green-100';
    if (efficiency >= 75) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getEfficiencyIcon = (efficiency) => {
    if (efficiency >= 90) return <FaArrowUp />;
    if (efficiency >= 75) return <FaMinus />;
    return <FaArrowDown />;
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Allocation Results</h2>
            <p className="text-gray-600 mt-1">
              Strategy: {result.strategy_used} | Efficiency: {result.allocation_efficiency?.toFixed(1)}%
            </p>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
              {result.allocations.length} Allocations
            </div>
            <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full">
              {result.total_allocated} Total Allocated
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="mb-4 bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500">
          <p className="text-blue-800 text-sm flex items-center gap-2">
            <FaChartLine />
            Click on any row to view ML forecasting models and detailed predictions
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-700">
                  <div className="flex items-center gap-2">
                    <FaBuilding className="text-gray-500" />
                    DC
                  </div>
                </th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">
                  <div className="flex items-center gap-2">
                    <FaCube className="text-gray-500" />
                    SKU
                  </div>
                </th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700">
                  <div className="flex items-center gap-2">
                    <FaUser className="text-gray-500" />
                    Customer
                  </div>
                </th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Allocated</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Demand</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Inventory</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700">Efficiency</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {result.allocations.map((allocation, index) => (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors"
                  onClick={() => handleRowClick(allocation)}
                >
                  <td className="py-3 px-4">
                    <span className="font-medium text-gray-900">{getDCName(allocation.dc_id)}</span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-gray-700">{getProductName(allocation.sku_id)}</span>
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-gray-700">{getCustomerName(allocation.customer_id)}</span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className="font-semibold text-indigo-600">
                      {allocation.allocated_quantity?.toLocaleString()}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className="text-gray-700">
                      {allocation.forecasted_demand?.toLocaleString()}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span className="text-gray-700">
                      {allocation.current_inventory?.toLocaleString()}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                      getEfficiencyColor(allocation.allocation_efficiency || 0)
                    }`}>
                      {getEfficiencyIcon(allocation.allocation_efficiency || 0)}
                      {allocation.allocation_efficiency?.toFixed(1)}%
                    </div>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRowClick(allocation);
                      }}
                      className="text-indigo-600 hover:text-indigo-800 p-2 rounded-lg hover:bg-indigo-50 transition-colors"
                      title="View ML Forecasts"
                    >
                      <FaEye />
                    </button>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {result.allocations.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            No allocations found for the selected criteria.
          </div>
        )}
      </div>

      <ForecastModal
        allocation={selectedAllocation}
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        csvData={csvData}
      />
    </div>
  );
};

export default EnhancedResultsTable;