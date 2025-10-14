import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaChevronDown, FaSearch, FaCheck, FaTimes, FaCalendarAlt, FaBoxOpen } from 'react-icons/fa';

const MultiSelectDropdown = ({ options, selectedIds, setSelectedIds, placeholder, labelField = 'name' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState(options);

  useEffect(() => {
    if (searchTerm) {
      setFilteredOptions(
        options.filter(option => 
          option[labelField].toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    } else {
      setFilteredOptions(options);
    }
  }, [searchTerm, options, labelField]);

  const toggleOption = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(selectedId => selectedId !== id));
    } else {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const selectAll = () => {
    const allIds = filteredOptions.map(option => option.id);
    setSelectedIds(allIds);
  };

  const clearAll = () => {
    setSelectedIds([]);
  };

  const selectedLabels = options
    .filter(option => selectedIds.includes(option.id))
    .map(option => option[labelField]);

  return (
    <div className="relative w-full">
      <div 
        className={`border ${isOpen ? 'border-indigo-500 ring-1 ring-indigo-500' : 'border-gray-300'} 
        rounded-lg p-2 flex justify-between items-center cursor-pointer bg-white`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex flex-wrap gap-1 flex-1">
          {selectedIds.length === 0 ? (
            <span className="text-gray-500">{placeholder}</span>
          ) : (
            <>
              {selectedLabels.slice(0, 2).map((label, index) => (
                <span key={index} className="bg-indigo-100 text-indigo-800 text-sm font-medium py-1 px-2 rounded">
                  {label}
                </span>
              ))}
              {selectedLabels.length > 2 && (
                <span className="bg-gray-200 text-gray-800 text-sm font-medium py-1 px-2 rounded">
                  +{selectedLabels.length - 2} more
                </span>
              )}
            </>
          )}
        </div>
        <FaChevronDown className={`transition-transform duration-300 ${isOpen ? 'transform rotate-180' : ''}`} />
      </div>

      {isOpen && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className="absolute mt-1 w-full bg-white border border-gray-300 rounded-lg shadow-lg z-10"
        >
          <div className="p-2 border-b">
            <div className="flex items-center rounded-md border border-gray-300 px-3 py-2 mb-2">
              <FaSearch className="text-gray-400 mr-2" />
              <input
                type="text"
                placeholder="Search..."
                className="w-full outline-none text-sm"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onClick={(e) => e.stopPropagation()}
              />
            </div>
            <div className="flex gap-2">
              <button 
                className="flex-1 text-xs bg-indigo-50 hover:bg-indigo-100 text-indigo-700 py-1 px-2 rounded"
                onClick={(e) => { e.stopPropagation(); selectAll(); }}
              >
                Select All
              </button>
              <button 
                className="flex-1 text-xs bg-gray-50 hover:bg-gray-100 text-gray-700 py-1 px-2 rounded"
                onClick={(e) => { e.stopPropagation(); clearAll(); }}
              >
                Clear All
              </button>
            </div>
          </div>
          <div className="max-h-60 overflow-y-auto">
            {filteredOptions.map((option) => (
              <div 
                key={option.id}
                className="flex items-center p-3 hover:bg-indigo-50 cursor-pointer"
                onClick={() => toggleOption(option.id)}
              >
                <div className={`w-5 h-5 border-2 rounded flex items-center justify-center mr-3 ${selectedIds.includes(option.id) ? 'bg-indigo-600 border-indigo-600' : 'border-gray-300'}`}>
                  {selectedIds.includes(option.id) && <FaCheck className="text-white text-xs" />}
                </div>
                <span>{option[labelField]}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

// Single select dropdown component for customer and product
const SingleSelectDropdown = ({ options, selectedId, setSelectedId, placeholder, labelField = 'name' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState(options);
  
  useEffect(() => {
    if (searchTerm) {
      setFilteredOptions(
        options.filter(option => 
          option[labelField].toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    } else {
      setFilteredOptions(options);
    }
  }, [searchTerm, options, labelField]);
  
  const selectedOption = options.find(option => option.id === selectedId);

  return (
    <div className="relative w-full">
      <div 
        className={`border ${isOpen ? 'border-indigo-500 ring-1 ring-indigo-500' : 'border-gray-300'} 
        rounded-lg p-3 flex justify-between items-center cursor-pointer bg-white`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className={selectedId ? 'text-gray-800' : 'text-gray-500'}>
          {selectedOption ? selectedOption[labelField] : placeholder}
        </span>
        <FaChevronDown className={`transition-transform duration-300 ${isOpen ? 'transform rotate-180' : ''}`} />
      </div>

      {isOpen && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className="absolute mt-1 w-full bg-white border border-gray-300 rounded-lg shadow-lg z-10"
        >
          <div className="p-2 border-b">
            <div className="flex items-center rounded-md border border-gray-300 px-3 py-2">
              <FaSearch className="text-gray-400 mr-2" />
              <input
                type="text"
                placeholder="Search..."
                className="w-full outline-none text-sm"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          </div>
          <div className="max-h-60 overflow-y-auto">
            {filteredOptions.map((option) => (
              <div 
                key={option.id}
                className="p-3 hover:bg-indigo-50 cursor-pointer"
                onClick={() => {
                  setSelectedId(option.id);
                  setIsOpen(false);
                }}
              >
                {option[labelField]}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

const AllocationInputForm = ({ onRunAllocation, loading, exampleData }) => {
  const [customerId, setCustomerId] = useState('');
  const [dcIds, setDcIds] = useState([]);
  const [productId, setProductId] = useState('');
  const [quantity, setQuantity] = useState(100);
  const [optimizerType, setOptimizerType] = useState('fair_share');

  const handleSubmit = (e) => {
    e.preventDefault();
    onRunAllocation({
      customerId,
      dcIds,
      productId,
      quantity,
      optimizerType
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-gray-700 text-sm font-medium mb-2">Customer</label>
          <SingleSelectDropdown 
            options={exampleData.customers}
            selectedId={customerId}
            setSelectedId={setCustomerId}
            placeholder="Select a customer"
          />
        </div>
        <div>
          <label className="block text-gray-700 text-sm font-medium mb-2">Distribution Centers <span className="text-xs text-gray-500">(Optional - Leave empty for all)</span></label>
          <MultiSelectDropdown 
            options={exampleData.distributionCenters.map(dc => ({
              ...dc,
              name: `${dc.name} (${dc.region})` // Add region to name
            }))}
            selectedIds={dcIds}
            setSelectedIds={setDcIds}
            placeholder="Select DCs or leave empty for all"
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
        <div>
          <label className="block text-gray-700 text-sm font-medium mb-2">Product</label>
          <SingleSelectDropdown 
            options={exampleData.products}
            selectedId={productId}
            setSelectedId={setProductId}
            placeholder="Select a product"
          />
        </div>
        <div className="w-full"> 
          <label className="block text-gray-700 text-sm font-medium mb-2">Quantity</label>
          <div className="flex items-center">
            <input 
              type="number" 
              min="1"
              value={quantity}
              onChange={(e) => setQuantity(parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>
        <div>
          <label className="block text-gray-700 text-sm font-medium mb-2">Optimizer Type</label>
          <select 
            value={optimizerType}
            onChange={(e) => setOptimizerType(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="fair_share">Fair Share - Equitable distribution across all DCs</option>
            <option value="priority_based">Priority Based - Allocate to higher priority DCs first</option>
            <option value="hybrid">Hybrid - Balance between priority and fairness</option>
            <option value="auto_select">Auto Select - Let AI choose the best strategy</option>
          </select>
        </div>
      </div>

      {/* <div className="mt-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">Optimization Type</label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className={`border rounded-lg p-4 cursor-pointer text-center ${optimizationType === 'reallocationPriority' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'}`} onClick={() => setOptimizationType('reallocationPriority')}>
                <h3 className="font-medium text-gray-800">Maximize Fulfillment</h3>
                <p className="text-sm text-gray-600 mt-1">Prioritize meeting the order quantity.</p>
            </div>
            <div className={`border rounded-lg p-4 cursor-pointer text-center ${optimizationType === 'marginPriority' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'}`} onClick={() => setOptimizationType('marginPriority')}>
                <h3 className="font-medium text-gray-800">Maximize Margin</h3>
                <p className="text-sm text-gray-600 mt-1">Prioritize the most profitable fulfillment option.</p>
            </div>
            <div className={`border rounded-lg p-4 cursor-pointer text-center ${optimizationType === 'balancedApproach' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'}`} onClick={() => setOptimizationType('balancedApproach')}>
                <h3 className="font-medium text-gray-800">Balanced</h3>
                <p className="text-sm text-gray-600 mt-1">Balance fulfillment rate and profitability.</p>
            </div>
        </div>
      </div> */}
      
      <div className="mt-8 flex justify-center">
        <button type="submit" className={`px-8 py-3 rounded-lg text-white font-semibold ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`} disabled={loading}>
          {loading ? 'Running Optimization...' : 'Run Allocation'}
        </button>
      </div>
    </form>
  );
};

export default AllocationInputForm;
