import React from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaChartBar, FaTruck } from 'react-icons/fa';

const AllocationResults = ({ result }) => {
  if (!result) return null;

  const { status, optimizer_used, model_used, allocation_summary, model_performance } = result;
  const isSuccess = status === 'completed';

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <div className={`bg-white rounded-xl shadow-lg p-6 border-l-4 ${isSuccess ? 'border-green-500' : 'border-yellow-500'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {isSuccess ? (
              <FaCheckCircle className="text-green-500 text-3xl" />
            ) : (
              <FaExclamationTriangle className="text-yellow-500 text-3xl" />
            )}
            <div>
              <h2 className="text-xl font-semibold text-gray-800">Allocation {status}</h2>
              <p className="text-gray-600">Optimizer: {optimizer_used} | Model: {model_used}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-indigo-600">
              {allocation_summary?.fulfillment_rate?.toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Fulfillment Rate</p>
          </div>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Demand</p>
              <p className="text-xl font-semibold text-gray-800">
                {allocation_summary?.total_demand?.toLocaleString()}
              </p>
            </div>
            <FaChartBar className="text-indigo-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Allocated</p>
              <p className="text-xl font-semibold text-gray-800">
                {allocation_summary?.total_allocated?.toLocaleString()}
              </p>
            </div>
            <FaTruck className="text-green-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Substitutions</p>
              <p className="text-xl font-semibold text-gray-800">
                {allocation_summary?.substitutions_used || 0}
              </p>
            </div>
            <FaExclamationTriangle className="text-yellow-500 text-2xl" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Gap</p>
              <p className="text-xl font-semibold text-gray-800">
                {((allocation_summary?.total_demand || 0) - (allocation_summary?.total_allocated || 0)).toLocaleString()}
              </p>
            </div>
            <FaExclamationTriangle className="text-red-500 text-2xl" />
          </div>
        </div>
      </div>

      {/* Model Performance */}
      {model_performance && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Model Performance</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500">RMSE</p>
              <p className="text-lg font-medium text-gray-800">{model_performance.rmse?.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">MAE</p>
              <p className="text-lg font-medium text-gray-800">{model_performance.mae?.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">MAPE</p>
              <p className="text-lg font-medium text-gray-800">{model_performance.mape?.toFixed(2)}%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AllocationResults;