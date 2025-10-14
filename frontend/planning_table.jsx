import React, { useState, useEffect, useMemo } from 'react';
import { FaSort, FaSortUp, FaSortDown, FaFilter, FaDownload, FaSearch, FaTimes, FaEye, FaEyeSlash, FaColumns } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import resultTable from './result_table';
import DecisionFactorsTable from './decision_factors_table';
import fulfilled2 from './example.json';
import LatestAllocationUpdatesTable from './latest_allocation_updates_table';
import ForecastModal from './forecast_modal';

const ColumnManagerPopup = ({ columns, visibleColumns, onVisibilityChange, onClose }) => (
  <AnimatePresence>
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="absolute z-10 top-full right-0 mt-2 w-56 bg-white rounded-md shadow-lg border border-gray-200"
    >
      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
        <h4 className="font-semibold">Manage Columns</h4>
        <button onClick={onClose} className="p-1 rounded-full hover:bg-gray-200"><FaTimes className="text-gray-500" /></button>
      </div>
      <div className="p-2">
        {columns.map(col => (
          <label key={col.id} className="flex items-center space-x-3 px-2 py-1.5 rounded-md hover:bg-gray-100 cursor-pointer">
            <input
              type="checkbox"
              checked={!!visibleColumns[col.id]}
              onChange={() => onVisibilityChange(col.id)}
              disabled={col.alwaysVisible}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 disabled:bg-gray-200"
            />
            <span className="text-sm text-gray-700">{col.header}</span>
          </label>
        ))}
      </div>
    </motion.div>
  </AnimatePresence>
);

const allColumns = [
  { id: 'material', header: 'Material', defaultVisible: true, isSortable: true },
  { id: 'region', header: 'Region', defaultVisible: true, isSortable: true },
  { id: 'dc', header: 'Distribution Center', defaultVisible: true, isSortable: true },
  { id: 'customer', header: 'Customer', defaultVisible: true, isSortable: true },
  { id: 'forecastQty', header: 'Forecast Qty', defaultVisible: true, isSortable: true },
  { id: 'allocatedQty', header: 'Allocated Qty', defaultVisible: true, isSortable: true },
  { id: 'consumedQty', header: 'Consumed Qty', defaultVisible: true, isSortable: true },
  { id: 'consumedPercentage', header: 'Consumed %', defaultVisible: true, isSortable: true },
  { id: 'priority', header: 'Priority', defaultVisible: true, isSortable: true },
  { id: 'revenue', header: 'Revenue', defaultVisible: true, isSortable: true },
  { id: 'status', header: 'Status', defaultVisible: true, isSortable: true }
];

const PlanningTable = ({ planData }) => {
  const handleRowClick = (item) => {
    if (item.demandForecast && item.demandForecast.length > 0) {
      setSelectedCustomerData(item);
      setIsModalOpen(true);
    }
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedCustomerData(null);
  };

  const [allocations, setAllocations] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: 'material', direction: 'ascending' });
  const [filters, setFilters] = useState({
    customer: '',
    material: '',
    region: '',
    dc: '',
    priority: '',
  });
  const [activeFilters, setActiveFilters] = useState({});
  const [showFilters, setShowFilters] = useState(false);
  const [visibleColumns, setVisibleColumns] = useState(
    allColumns.reduce((acc, col) => ({ ...acc, [col.id]: col.defaultVisible }), {})
  );
  const [showColumnManager, setShowColumnManager] = useState(false);
  const [isMergedView, setIsMergedView] = useState(true); // Default to merged view
  const [materialFilter, setMaterialFilter] = useState('all');
  const [dcFilter, setDcFilter] = useState('all');
  const [customerFilter, setCustomerFilter] = useState('all');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedCustomerData, setSelectedCustomerData] = useState(null);

  useEffect(() => {
    if (planData && planData.allocations) {
      // Add consumed allocation calculations
      const enhancedAllocations = planData.allocations.map(item => {
        // Calculate consumed quantity (random value for example between 0 and allocated)
        const consumedQty = item.consumedAllocation > 0 ? item.consumedAllocation : Math.floor(Math.random() * (item.allocatedQty + 1));
        
        return {
          ...item,
          // Rename product to material
          material: item.product,
          // Convert priority from text to number (High=3, Medium=2, Low=1)
          priority: item.customerPriority === 'High' ? 3 : item.customerPriority === 'Medium' ? 2 : 1,
          // Add consumed quantities
          consumedQty,
          consumedPercentage: item.allocatedQty > 0 ? Math.round((consumedQty / item.allocatedQty) * 100) : 0
        };
      });

      setAllocations(enhancedAllocations);
    }
  }, [planData]);

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key === key) {
      return sortConfig.direction === 'ascending' ? <FaSortUp className="inline ml-1" /> : <FaSortDown className="inline ml-1" />;
    }
    return <FaSort className="inline ml-1 text-gray-300" />;
  };

  const formatNumber = (value) => {
    return new Intl.NumberFormat().format(value);
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  };

  const formatPercentage = (value) => {
    return `${value}%`;
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const applyFilters = () => {
    const newActiveFilters = {};
    Object.keys(filters).forEach(key => {
      if (filters[key]) {
        newActiveFilters[key] = filters[key];
      }
    });
    setActiveFilters(newActiveFilters);
  };

  const clearFilters = () => {
    setFilters({
      customer: '',
      material: '',
      region: '',
      dc: '',
      priority: '',
    });
    setActiveFilters({});
    setMaterialFilter('all');
    setDcFilter('all');
    setCustomerFilter('all');
  };

  const toggleColumnVisibility = (column) => {
    setVisibleColumns(prev => ({
      ...prev,
      [column]: !prev[column]
    }));
  };

  const handleVisibilityChange = (columnId) => {
    setVisibleColumns(prev => ({
      ...prev,
      [columnId]: !prev[columnId]
    }));
  };

  // Filter allocations based on active filters
  const filteredAllocations = useMemo(() => {
    return allocations.filter(item => {
      // Apply dropdown filters
      if (materialFilter !== 'all' && item.material !== materialFilter) return false;
      if (dcFilter !== 'all' && item.dc !== dcFilter) return false;
      if (customerFilter !== 'all' && item.customer !== customerFilter) return false;

      // Apply text filters
      for (const key in activeFilters) {
        const value = activeFilters[key];
        if (value && String(item[key]).toLowerCase().indexOf(value.toLowerCase()) === -1) {
          return false;
        }
      }
      return true;
    });
  }, [allocations, activeFilters, materialFilter, dcFilter, customerFilter]);

  // Sort and filter data
  const sortedAndFilteredData = useMemo(() => {
    // First apply filters
    let filteredData = [...filteredAllocations];
    
    // Then sort
    if (sortConfig.key) {
      filteredData.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    
    return filteredData;
  }, [filteredAllocations, sortConfig]);

  // Create merged data structure for cell merging
  const mergedData = useMemo(() => {
    if (!isMergedView) return [];
    // Group by material -> region -> dc -> customer
    const materialGroups = {};
    sortedAndFilteredData.forEach(item => {
      const material = item.material;
      const region = item.region;
      const dc = item.dc;
      const customer = item.customer;
      if (!materialGroups[material]) {
        materialGroups[material] = { materialId: material, regionGroups: {}, totalRowCount: 0 };
      }
      const matGroup = materialGroups[material];
      if (!matGroup.regionGroups[region]) {
        matGroup.regionGroups[region] = { regionId: region, dcGroups: {}, rowCount: 0 };
      }
      const regGroup = matGroup.regionGroups[region];
      if (!regGroup.dcGroups[dc]) {
        regGroup.dcGroups[dc] = { dcId: dc, customerGroups: {}, rowCount: 0 };
      }
      const dcGroup = regGroup.dcGroups[dc];
      if (!dcGroup.customerGroups[customer]) {
        dcGroup.customerGroups[customer] = { customerId: customer, allocations: [], rowCount: 0 };
      }
      const custGroup = dcGroup.customerGroups[customer];
      custGroup.allocations.push(item);
      custGroup.rowCount++;
      dcGroup.rowCount++;
      regGroup.rowCount++;
      matGroup.totalRowCount++;
    });
    // Convert to array and calculate row counts
    return Object.values(materialGroups).map(matGroup => ({
      materialId: matGroup.materialId,
      totalRowCount: matGroup.totalRowCount,
      regionGroups: Object.values(matGroup.regionGroups).map(regGroup => ({
        regionId: regGroup.regionId,
        rowCount: regGroup.rowCount,
        dcGroups: Object.values(regGroup.dcGroups).map(dcGroup => ({
          dcId: dcGroup.dcId,
          rowCount: dcGroup.rowCount,
          customerGroups: Object.values(dcGroup.customerGroups).map(custGroup => ({
            customerId: custGroup.customerId,
            rowCount: custGroup.rowCount,
            allocations: custGroup.allocations
          }))
        }))
      }))
    }));
  }, [isMergedView, sortedAndFilteredData]);
  
  // Create array of visible columns for table rendering
  const activeColumns = useMemo(() => {
    return allColumns.filter(col => visibleColumns[col.id]);
  }, [visibleColumns]);
  
  // Function to render cell content based on column type
  const renderCellContent = (item, col) => {
    switch (col.id) {
      case 'status':
        return getStatusBadge(item[col.id]);
      case 'priority':
        return item[col.id]; // Now a number instead of text
      case 'revenue':
        return formatCurrency(item[col.id]);
      case 'forecastQty':
      case 'allocatedQty':
      case 'consumedQty':
        return formatNumber(item[col.id]);
      case 'consumedPercentage':
        return formatPercentage(item[col.id]);
      default:
        return item[col.id];
    }
  };

  // Get unique materials for filter dropdown
  const uniqueMaterials = useMemo(() => {
    const materials = [...new Set(allocations.map(item => item.material))];
    return materials.sort();
  }, [allocations]);

  // Get unique DCs for filter dropdown
  const uniqueDCs = useMemo(() => {
    const dcs = [...new Set(allocations.map(item => item.dc))];
    return dcs.sort();
  }, [allocations]);

  // Get unique customers for filter dropdown
  const uniqueCustomers = useMemo(() => {
    const customers = [...new Set(allocations.map(item => item.customer))];
    return customers.sort();
  }, [allocations]);

  const getStatusBadge = (status) => {
    let bgColor = '';
    switch (status) {
      case 'Confirmed':
        bgColor = 'bg-green-100 text-green-800';
        break;
      case 'Pending':
        bgColor = 'bg-yellow-100 text-yellow-800';
        break;
      case 'At Risk':
        bgColor = 'bg-red-100 text-red-800';
        break;
      default:
        bgColor = 'bg-gray-100 text-gray-800';
    }
    return <span className={`px-2 py-1 text-xs font-medium rounded-full ${bgColor}`}>{status}</span>;
  };

  const getColumnLabel = (key) => {
    const column = allColumns.find(col => col.id === key);
    return column ? column.header : key.charAt(0).toUpperCase() + key.slice(1);
  };

  const exportToCSV = () => {
    // Create CSV content
    const headers = activeColumns.map(col => col.header);
    
    const csvContent = [
      headers.join(','),
      ...sortedAndFilteredData.map(row => 
        activeColumns.map(col => {
          let value = row[col.id];
          // Handle special formatting for CSV
          if (typeof value === 'string' && value.includes(',')) {
            return `"${value}"`;
          }
          return value;
        }).join(',')
      )
    ].join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `allocation_plan_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white shadow rounded-lg p-5">
      {/* Header with controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h2 className="text-xl font-semibold text-gray-800">Monthly Planning Table</h2>
        
        <div className="flex flex-wrap gap-2">
          {/* View toggle */}
          <div className="flex items-center">
            <span className="mr-2 text-sm text-gray-600">View:</span>
            <button
              onClick={() => setIsMergedView(true)}
              className={`px-3 py-1 text-sm rounded-l-md ${isMergedView ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
            >
              Merged
            </button>
            <button
              onClick={() => setIsMergedView(false)}
              className={`px-3 py-1 text-sm rounded-r-md ${!isMergedView ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
            >
              Normal
            </button>
          </div>
          
          {/* Filter button */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center px-3 py-1 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50"
          >
            <FaFilter className="mr-1" />
            Filters
          </button>
          
          {/* Export button */}
          <button
            onClick={exportToCSV}
            className="flex items-center px-3 py-1 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50"
          >
            <FaDownload className="mr-1" />
            Export
          </button>
          
          {/* Column manager */}
          <div className="relative">
            <button
              onClick={() => setShowColumnManager(!showColumnManager)}
              className="flex items-center px-3 py-1 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50"
            >
              <FaColumns className="mr-1" />
              Columns
            </button>
            {showColumnManager && (
              <ColumnManagerPopup
                columns={allColumns}
                visibleColumns={visibleColumns}
                onVisibilityChange={handleVisibilityChange}
                onClose={() => setShowColumnManager(false)}
              />
            )}
          </div>
        </div>
      </div>
      
      {/* Filters panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6 overflow-hidden"
          >
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {/* Material filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Material</label>
                  <select
                    value={materialFilter}
                    onChange={(e) => setMaterialFilter(e.target.value)}
                    className="w-full rounded-md border border-gray-300 py-2 px-3 text-sm"
                  >
                    <option value="all">All Materials</option>
                    {uniqueMaterials.map(material => (
                      <option key={material} value={material}>{material}</option>
                    ))}
                  </select>
                </div>
                
                {/* DC filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Distribution Center</label>
                  <select
                    value={dcFilter}
                    onChange={(e) => setDcFilter(e.target.value)}
                    className="w-full rounded-md border border-gray-300 py-2 px-3 text-sm"
                  >
                    <option value="all">All DCs</option>
                    {uniqueDCs.map(dc => (
                      <option key={dc} value={dc}>{dc}</option>
                    ))}
                  </select>
                </div>
                
                {/* Customer filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Customer</label>
                  <select
                    value={customerFilter}
                    onChange={(e) => setCustomerFilter(e.target.value)}
                    className="w-full rounded-md border border-gray-300 py-2 px-3 text-sm"
                  >
                    <option value="all">All Customers</option>
                    {uniqueCustomers.map(customer => (
                      <option key={customer} value={customer}>{customer}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {/* Text filters */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Region</label>
                  <div className="relative">
                    <input
                      type="text"
                      name="region"
                      value={filters.region}
                      onChange={handleFilterChange}
                      className="w-full rounded-md border border-gray-300 py-2 pl-9 pr-3 text-sm"
                      placeholder="Filter by region..."
                    />
                    <FaSearch className="absolute left-3 top-3 text-gray-400" />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                  <div className="relative">
                    <input
                      type="text"
                      name="priority"
                      value={filters.priority}
                      onChange={handleFilterChange}
                      className="w-full rounded-md border border-gray-300 py-2 pl-9 pr-3 text-sm"
                      placeholder="Filter by priority..."
                    />
                    <FaSearch className="absolute left-3 top-3 text-gray-400" />
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end space-x-2">
                <button
                  onClick={clearFilters}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Clear
                </button>
                <button
                  onClick={applyFilters}
                  className="px-4 py-2 bg-blue-600 border border-transparent rounded-md text-sm font-medium text-white hover:bg-blue-700"
                >
                  Apply Filters
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {activeColumns.map(col => (
                <th
                  key={col.id}
                  scope="col"
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  <div className="flex items-center">
                    <span>{col.header}</span>
                    {col.isSortable && (
                      <button
                        onClick={() => requestSort(col.id)}
                        className="ml-1 focus:outline-none"
                      >
                        {getSortIcon(col.id)}
                      </button>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {isMergedView ? (
              mergedData.length > 0 ? (
                mergedData.flatMap((materialGroup) =>
                  materialGroup.regionGroups.flatMap((regionGroup, regionIdx) =>
                    regionGroup.dcGroups.flatMap((dcGroup, dcIdx) =>
                      dcGroup.customerGroups.flatMap((customerGroup, custIdx) =>
                        customerGroup.allocations.map((item, allocIdx) => {
                          // Compute rowSpan logic
                          const isFirstInMaterialGroup = regionIdx === 0 && dcIdx === 0 && custIdx === 0 && allocIdx === 0;
                          const isFirstInRegionGroup = dcIdx === 0 && custIdx === 0 && allocIdx === 0;
                          const isFirstInDcGroup = custIdx === 0 && allocIdx === 0;
                          const isFirstInCustomerGroup = allocIdx === 0;
                          return (
                            <tr
                              key={`${item.id || allocIdx}-${customerGroup.customerId}-${dcGroup.dcId}-${regionGroup.regionId}-${materialGroup.materialId}`}
                              onClick={() => item.demandForecast && item.demandForecast.length > 0 ? handleRowClick(item) : undefined}
                              className={`hover:bg-gray-50 ${item.demandForecast && item.demandForecast.length > 0 ? 'cursor-pointer' : ''}`}
                            >
                              {activeColumns.map(col => {
                                if (col.id === 'material') {
                                  if (isFirstInMaterialGroup) {
                                    return <td key={col.id} rowSpan={materialGroup.totalRowCount} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 align-top">{materialGroup.materialId}</td>;
                                  }
                                  return null;
                                }
                                if (col.id === 'region') {
                                  if (isFirstInRegionGroup) {
                                    return <td key={col.id} rowSpan={regionGroup.rowCount} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 align-top">{regionGroup.regionId}</td>;
                                  }
                                  return null;
                                }
                                if (col.id === 'dc') {
                                  if (isFirstInDcGroup) {
                                    return <td key={col.id} rowSpan={dcGroup.rowCount} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 align-top">{dcGroup.dcId}</td>;
                                  }
                                  return null;
                                }
                                if (col.id === 'customer') {
                                  if (isFirstInCustomerGroup) {
                                    return <td key={col.id} rowSpan={customerGroup.rowCount} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 align-top">{customerGroup.customerId}</td>;
                                  }
                                  return null;
                                }
                                return (
                                  <td key={col.id} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                    {renderCellContent(item, col)}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })
                      )
                    )
                  )
                )
              ) : (
                <tr>
                  <td colSpan={activeColumns.length} className="px-6 py-4 text-center text-sm text-gray-500">
                    No data available for merged view.
                  </td>
                </tr>
              )
            ) : (
              sortedAndFilteredData.length > 0 ? (
                sortedAndFilteredData.map((item, index) => (
                  <tr key={item.id || index} onClick={() => handleRowClick(item)} className={`hover:bg-gray-50 ${item.demandForecast ? 'cursor-pointer' : ''}`}>
                    {activeColumns.map(col => (
                      <td key={col.id} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {renderCellContent(item, col)}
                      </td>
                    ))}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={activeColumns.length} className="px-6 py-4 text-center text-sm text-gray-500">
                    No allocation data found matching the current filters.
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>


      {/* Reallocation Decision Factors */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <LatestAllocationUpdatesTable />
        <DecisionFactorsTable data={fulfilled2} />
      </div>


      <ForecastModal isOpen={isModalOpen} onClose={handleCloseModal} customerData={selectedCustomerData} />

      {/* Summary */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 mb-1">Total Allocations</h4>
            <p className="text-2xl font-semibold text-gray-900">{formatNumber(sortedAndFilteredData.reduce((sum, item) => sum + item.allocatedQty, 0))}</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 mb-1">Total Consumed</h4>
            <p className="text-2xl font-semibold text-gray-900">{formatNumber(sortedAndFilteredData.reduce((sum, item) => sum + item.consumedQty, 0))}</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 mb-1">Consumption Rate</h4>
            <p className="text-2xl font-semibold text-gray-900">
              {formatPercentage(
                Math.round(
                  (sortedAndFilteredData.reduce((sum, item) => sum + item.consumedQty, 0) / 
                  sortedAndFilteredData.reduce((sum, item) => sum + item.allocatedQty, 0)) * 100
                ) || 0
              )}
            </p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-500 mb-1">Total Revenue</h4>
            <p className="text-2xl font-semibold text-gray-900">{formatCurrency(sortedAndFilteredData.reduce((sum, item) => sum + item.revenue, 0))}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlanningTable;
