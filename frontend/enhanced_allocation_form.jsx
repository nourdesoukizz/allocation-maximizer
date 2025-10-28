import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaChevronDown, FaSearch, FaCheck, FaTimes, FaCalendarAlt, FaBoxOpen, FaBuilding, FaUsers, FaBalanceScale, FaTrophy, FaChartLine } from 'react-icons/fa';

const MultiSelectChecklist = ({ options, selectedIds, setSelectedIds, title, icon }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState(options);

  useEffect(() => {
    if (searchTerm) {
      setFilteredOptions(
        options.filter(option => 
          option.name.toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    } else {
      setFilteredOptions(options);
    }
  }, [searchTerm, options]);

  const toggleOption = (id) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(selectedId => selectedId !== id));
    } else {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const selectAll = () => {
    const allIds = filteredOptions.map(option => option.id);
    setSelectedIds([...new Set([...selectedIds, ...allIds])]);
  };

  const clearAll = () => {
    const currentSelected = filteredOptions.map(option => option.id);
    setSelectedIds(selectedIds.filter(id => !currentSelected.includes(id)));
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {icon}
          <h3 className="font-semibold text-gray-800">{title}</h3>
          <span className="text-sm text-gray-500">({selectedIds.length} selected)</span>
        </div>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="text-indigo-600 hover:text-indigo-800"
        >
          <FaChevronDown className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="space-y-3"
        >
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder={`Search ${title.toLowerCase()}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <button
              type="button"
              onClick={selectAll}
              className="px-3 py-2 text-xs font-medium text-indigo-600 hover:text-indigo-800"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={clearAll}
              className="px-3 py-2 text-xs font-medium text-gray-600 hover:text-gray-800"
            >
              Clear
            </button>
          </div>

          <div className="max-h-48 overflow-y-auto space-y-2">
            {filteredOptions.map((option) => (
              <label key={option.id} className="flex items-center gap-3 p-2 hover:bg-gray-50 rounded cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedIds.includes(option.id)}
                  onChange={() => toggleOption(option.id)}
                  className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                />
                <span className="text-sm text-gray-700">{option.name}</span>
                {option.region && <span className="text-xs text-gray-500">({option.region})</span>}
              </label>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

const StrategyButton = ({ strategy, selected, onSelect, icon, title, description }) => (
  <motion.button
    type="button"
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    onClick={() => onSelect(strategy)}
    className={`p-6 rounded-xl border-2 text-left transition-all ${
      selected 
        ? 'border-indigo-500 bg-indigo-50 shadow-lg' 
        : 'border-gray-200 bg-white hover:border-indigo-300 hover:shadow-md'
    }`}
  >
    <div className="flex items-center gap-3 mb-3">
      <div className={`p-3 rounded-lg ${selected ? 'bg-indigo-500 text-white' : 'bg-gray-100 text-gray-600'}`}>
        {icon}
      </div>
      <h3 className={`font-bold text-lg ${selected ? 'text-indigo-800' : 'text-gray-800'}`}>
        {title}
      </h3>
    </div>
    <p className="text-sm text-gray-600 leading-relaxed">{description}</p>
  </motion.button>
);

const EnhancedAllocationForm = ({ onRunAllocation, loading, exampleData }) => {
  // Date selection - Default to November 2024 (earliest future planning period)
  const getDefaultMonth = () => {
    const now = new Date();
    const currentYear = now.getFullYear();
    const currentMonth = now.getMonth() + 1;
    
    // If we're in 2024 and it's October or earlier, default to November 2024
    // If we're in 2024 and it's November or later, default to current month
    // If we're in 2025 or later, default to current month
    if (currentYear === 2024 && currentMonth <= 10) {
      return 11; // November
    }
    return currentMonth;
  };
  
  const getDefaultYear = () => {
    const now = new Date();
    const currentYear = now.getFullYear();
    const currentMonth = now.getMonth() + 1;
    
    // If we're in 2024 and it's October or earlier, default to 2024
    // Otherwise use current year
    if (currentYear === 2024 && currentMonth <= 10) {
      return 2024;
    }
    return currentYear;
  };

  const [selectedMonth, setSelectedMonth] = useState(6); // June
  const [selectedYear, setSelectedYear] = useState(2026); // Fixed year

  // Multi-select states
  const [selectedCustomers, setSelectedCustomers] = useState([]);
  const [selectedDCs, setSelectedDCs] = useState([]);
  const [selectedMaterials, setSelectedMaterials] = useState([]);

  // Strategy selection
  const [selectedStrategy, setSelectedStrategy] = useState('fair_share');

  // Generate month options
  const months = [
    { value: 1, label: 'January' }, { value: 2, label: 'February' }, { value: 3, label: 'March' },
    { value: 4, label: 'April' }, { value: 5, label: 'May' }, { value: 6, label: 'June' },
    { value: 7, label: 'July' }, { value: 8, label: 'August' }, { value: 9, label: 'September' },
    { value: 10, label: 'October' }, { value: 11, label: 'November' }, { value: 12, label: 'December' }
  ];

  // Generate year options (2024 onwards for future planning only)
  const currentYear = new Date().getFullYear();
  const startYear = Math.max(2024, currentYear); // Don't allow years before 2024
  const years = Array.from({ length: 5 }, (_, i) => startYear + i);

  // Initialize with all items selected by default
  useEffect(() => {
    if (exampleData?.customers) {
      setSelectedCustomers(exampleData.customers.map(c => c.id));
    }
    if (exampleData?.distributionCenters) {
      setSelectedDCs(exampleData.distributionCenters.map(dc => dc.id));
    }
    if (exampleData?.products) {
      setSelectedMaterials(exampleData.products.map(p => p.id));
    }
  }, [exampleData]);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Create allocation data based on selected items
    // Use only the CSV data combinations that exist and filter by user selections
    const allocationData = [];
    
    if (exampleData.csvData && exampleData.csvData.length > 0) {
      // Filter existing CSV data based on user selections
      const filteredRecords = exampleData.csvData.filter(record => 
        selectedCustomers.includes(record.customer_id) &&
        selectedDCs.includes(record.dc_id) &&
        selectedMaterials.includes(record.sku_id)
      );
      
      // Use the filtered CSV records directly
      filteredRecords.forEach(csvRecord => {
        allocationData.push({
          dc_id: csvRecord.dc_id,
          sku_id: csvRecord.sku_id,
          customer_id: csvRecord.customer_id,
          current_inventory: parseInt(csvRecord.current_inventory),
          forecasted_demand: parseInt(csvRecord.forecasted_demand),
          dc_priority: parseInt(csvRecord.dc_priority),
          customer_tier: csvRecord.customer_tier,
          sla_level: csvRecord.sla_level,
          min_order_quantity: parseInt(csvRecord.min_order_quantity),
          sku_category: csvRecord.sku_category
        });
      });
    } else {
      // Fallback: Create deterministic data for all combinations if no CSV data
      selectedCustomers.forEach(customerId => {
        selectedDCs.forEach(dcId => {
          selectedMaterials.forEach(materialId => {
            const customer = exampleData.customers.find(c => c.id === customerId);
            const dc = exampleData.distributionCenters.find(d => d.id === dcId);
            const material = exampleData.products.find(p => p.id === materialId);
            
            if (customer && dc && material) {
              // Create deterministic values based on IDs
              const baseInventory = 500 + (dc.priority || 1) * 100 + material.id.charCodeAt(material.id.length-1) * 10;
              const baseDemand = 300 + customer.id.charCodeAt(customer.id.length-1) * 50 + material.id.charCodeAt(material.id.length-2) * 20;
              
              allocationData.push({
                dc_id: dc.id,
                sku_id: material.id,
                customer_id: customer.id,
                current_inventory: parseInt(baseInventory),
                forecasted_demand: parseInt(baseDemand),
                dc_priority: dc.priority || 5,
                customer_tier: customer.tier === 'Strategic' ? 'Strategic' : customer.tier === 'Standard' ? 'Standard' : 'Premium',
                sla_level: customer.tier === 'Strategic' ? 'Gold' : 'Silver',
                min_order_quantity: 1,
                sku_category: material.category || 'general'
              });
            }
          });
        });
      });
    }

    onRunAllocation({
      selectedMonth,
      selectedYear,
      selectedCustomers,
      selectedDCs,
      selectedMaterials,
      strategy: selectedStrategy,
      allocationData
    });
  };

  if (!exampleData?.customers) {
    return <div className="text-center p-8">Loading form data...</div>;
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Allocation Maximizer</h2>
          <p className="text-gray-600">Configure your allocation parameters and strategy</p>
        </div>

        {/* Date Selection */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-4">
            <FaCalendarAlt className="text-indigo-600" />
            <h3 className="text-lg font-semibold text-gray-800">Time Period</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Month</label>
              <div className="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-700 font-medium">
                June
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Year</label>
              <div className="w-full p-3 border border-gray-300 rounded-lg bg-gray-50 text-gray-700 font-medium">
                2026
              </div>
            </div>
          </div>
        </div>

        {/* Selection Checklists */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <MultiSelectChecklist
            options={exampleData.customers}
            selectedIds={selectedCustomers}
            setSelectedIds={setSelectedCustomers}
            title="Customers"
            icon={<FaUsers className="text-blue-600" />}
          />
          
          <MultiSelectChecklist
            options={exampleData.distributionCenters}
            selectedIds={selectedDCs}
            setSelectedIds={setSelectedDCs}
            title="Distribution Centers"
            icon={<FaBuilding className="text-green-600" />}
          />
          
          <MultiSelectChecklist
            options={exampleData.products}
            selectedIds={selectedMaterials}
            setSelectedIds={setSelectedMaterials}
            title="Materials/Products"
            icon={<FaBoxOpen className="text-purple-600" />}
          />
        </div>

        {/* Strategy Selection */}
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="text-center mb-6">
            <h3 className="text-xl font-bold text-gray-800 mb-2">Choose Your Allocation Strategy</h3>
            <p className="text-gray-600">Select the optimization approach that best fits your business needs</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <StrategyButton
              strategy="fair_share"
              selected={selectedStrategy === 'fair_share'}
              onSelect={setSelectedStrategy}
              icon={<FaBalanceScale />}
              title="Fair Share"
              description="Distributes inventory proportionally based on demand ratios across all distribution centers. Ensures equitable allocation and balanced resource utilization."
            />
            
            <StrategyButton
              strategy="priority_based"
              selected={selectedStrategy === 'priority_based'}
              onSelect={setSelectedStrategy}
              icon={<FaTrophy />}
              title="Priority Based"
              description="Allocates inventory based on DC priority rankings, serving higher priority centers first. Maximizes efficiency with clear hierarchical allocation."
            />
          </div>
        </div>

        {/* Submit Button */}
        <div className="text-center">
          <motion.button
            type="submit"
            disabled={loading || selectedCustomers.length === 0 || selectedDCs.length === 0 || selectedMaterials.length === 0}
            whileHover={{ scale: loading ? 1 : 1.05 }}
            whileTap={{ scale: loading ? 1 : 0.95 }}
            className={`px-12 py-4 rounded-lg font-bold text-lg transition-all ${
              loading || selectedCustomers.length === 0 || selectedDCs.length === 0 || selectedMaterials.length === 0
                ? 'bg-gray-400 cursor-not-allowed text-gray-200'
                : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg'
            }`}
          >
            {loading ? (
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Running Optimization...
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <FaChartLine />
                Run Allocation Optimization
              </div>
            )}
          </motion.button>
        </div>

      </form>
    </div>
  );
};

export default EnhancedAllocationForm;