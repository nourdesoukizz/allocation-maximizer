import React, { useState, useMemo } from 'react';
import { FaSort, FaSortUp, FaSortDown, FaSearch, FaSpinner, FaChartLine } from 'react-icons/fa';
import ForecastModal from './forecast_modal';

const ResultTable = ({ allocationData, onSubmitOrder, isLoading }) => {
  const { allocationTable = [], status } = allocationData;
  const [sortField, setSortField] = useState('priority');
  const [sortDirection, setSortDirection] = useState('asc');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [showForecastModal, setShowForecastModal] = useState(false);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleRowClick = (customerData) => {
    if (customerData.demandForecast) {
      setSelectedCustomer(customerData);
      setShowForecastModal(true);
    }
  };

  const closeModal = () => {
    setSelectedCustomer(null);
    setShowForecastModal(false);
  };

  const filteredAndSortedAllocations = useMemo(() => {
    let result = [...allocationTable];

    if (searchTerm) {
      result = result.filter(item =>
        item.customer.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    result.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [allocationTable, searchTerm, sortField, sortDirection]);

  const getSortIndicator = (field) => {
    if (sortField === field) {
      return sortDirection === 'asc' ? <FaSortUp /> : <FaSortDown />;
    }
    return <FaSort />;
  };

  const getModelChipColor = (modelName) => {
    const colors = {
      'Deep Q-Networks': 'bg-purple-100 text-purple-800',
      'ARIMA': 'bg-blue-100 text-blue-800',
      'Monte Carlo Simulation': 'bg-green-100 text-green-800',
      'Markov Decision Processes (MDP)': 'bg-yellow-100 text-yellow-800'
    };
    return colors[modelName] || 'bg-gray-100 text-gray-800';
  };

  const parsePercentage = (percentageStr) => {
    const num = parseFloat(percentageStr);
    return isNaN(num) ? '100%' : Math.min(100, num) + '%';
  };

  return (
    <>
      <div className="mb-4">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <FaSearch className="text-gray-400" />
          </div>
          <input
            type="text"
            className="pl-10 w-full p-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
            placeholder="Search by customer name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {isLoading && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-4 rounded-lg shadow-lg flex items-center space-x-3">
            <FaSpinner className="animate-spin text-indigo-600 text-xl" />
            <span className="text-gray-700">Processing order...</span>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-200 rounded-lg overflow-hidden">
          <thead className="bg-gray-50">
            <tr>
              {[ 'Customer', 'Forecast', 'Allocated Qty', 'Consumed', 'Consumed %', 'Priority', 'Revenue', 'Selected Model' ].map(header => (
                <th
                  key={header}
                  className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSort(header.toLowerCase().replace(/ /g, ''))}
                >
                  <div className="flex items-center">
                    {header}
                    <span className="ml-2">{getSortIndicator(header.toLowerCase().replace(/ /g, ''))}</span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredAndSortedAllocations.map((item, index) => (
              <tr
                key={index}
                onClick={() => handleRowClick(item)}
                className={`hover:bg-gray-50 cursor-pointer ${
                  item.forecastHistory ? 'cursor-pointer' : ''
                } ${item.impacted ? 'bg-red-50' : ''}`}
              >
                <td className="px-4 py-3 text-sm">
                  <div className="font-medium text-gray-900">{item.customer}</div>
                  {item.currentOrder && <div className="text-xs text-blue-600">Current Order: {item.currentOrder}</div>}
                  {item.takenFrom && <div className="text-xs text-red-600">Reallocated: {item.takenFrom}</div>}
                </td>
                <td className="px-4 py-3 text-sm text-gray-800">{item.forecast}</td>
                <td className="px-4 py-3 text-sm text-gray-800">{item.allocatedQty}</td>
                <td className="px-4 py-3 text-sm text-gray-800">{item.consumedAllocation}</td>
                <td className="px-4 py-3 text-sm">
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div
                        className="bg-blue-600 h-2.5 rounded-full"
                        style={{ width: parsePercentage(item.consumedPercentage) }}
                      ></div>
                    </div>
                    <span className="ml-2 text-gray-700">{item.consumedPercentage}</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-center text-gray-800">{item.priority}</td>
                <td className="px-4 py-3 text-sm text-gray-800">${item.revenue}M</td>
                <td className="px-4 py-3 text-sm">
                  <div className="flex items-center">
                    {item.selectedModel ? (
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getModelChipColor(item.selectedModel)}`}>
                        {item.selectedModel}
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {filteredAndSortedAllocations.length === 0 && (
              <tr>
                <td colSpan="8" className="px-4 py-6 text-center text-gray-500">
                  No allocations found matching your criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {showForecastModal && selectedCustomer && (
        <ForecastModal
          isOpen={showForecastModal}
          onClose={closeModal}
          customerData={selectedCustomer}
        />
      )}
    </>
  );
};

export default ResultTable;
