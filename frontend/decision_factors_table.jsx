import React from 'react';
import { FaArrowUp, FaArrowDown, FaEquals, FaStar, FaChartLine, FaGlobeAmericas, FaUsers, FaDollarSign } from 'react-icons/fa';

const DecisionFactorsTable = ({ data }) => {
  if (!data?.reallocationDecisionFactors?.metrics) {
    return null;
  }

  const { orderingCustomer, metrics } = data.reallocationDecisionFactors;
  const allCustomers = Object.keys(metrics.forecastConsumption).filter(c => c !== 'target');
  const customers = [orderingCustomer, ...allCustomers.filter(c => c !== orderingCustomer)];
  
  // Group customers by their regional demand values for merging cells
  const regionalDemandGroups = [];
  let currentValue = null;
  let currentGroup = [];
  
  allCustomers.forEach(customer => {
    const value = metrics.regionalDemand[customer];
    if (value === currentValue) {
      currentGroup.push(customer);
    } else {
      if (currentGroup.length > 0) {
        regionalDemandGroups.push({ value: currentValue, customers: [...currentGroup] });
      }
      currentGroup = [customer];
      currentValue = value;
    }
  });
  
  if (currentGroup.length > 0) {
    regionalDemandGroups.push({ value: currentValue, customers: currentGroup });
  }

  const getIcon = (value, target = 0, reverse = false) => {
    if (value === target) return null; // Don't show any icon when values are equal
    const isHigher = reverse ? value < target : value > target;
    return isHigher ? 
      <FaArrowUp className="inline text-green-600" /> : 
      <FaArrowDown className="inline text-red-600" />;
  };

  const getValueColor = (value, target = 0, reverse = false) => {
    if (value === target) return 'text-gray-900';
    const isHigher = reverse ? value < target : value > target;
    return isHigher ? 'text-green-600' : 'text-red-600';
  };

  const metricRows = [
    {
      id: 'forecastConsumption',
      label: 'Allocation vs. Consumption',
      icon: <FaChartLine className="inline mr-2" />,
      format: (val) => `${val > 0 ? '+' : ''}${val}%`,
      compareToTarget: true
    },
    {
      id: 'regionalDemand',
      label: 'Regional Demand',
      icon: <FaGlobeAmericas className="inline mr-2" />,
      format: (val) => `${val > 0 ? '+' : ''}${val}%`,
      compareToTarget: true
    },
    {
      id: 'priority',
      label: 'Priority Level',
      icon: <FaStar className="inline mr-2" />,
      format: (val) => `Level ${val}`,
      reverse: true,
      compareToOrdering: true
    },
    {
      id: 'revenue',
      label: 'Revenue',
      icon: <FaDollarSign className="inline mr-2" />,
      format: (val) => `$${val}`,
      compareToOrdering: true
    }
  ];

  return (
    <div className="mt-4 mb-8 bg-white rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Allocation Impact Analysis</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Metrics
              </th>
              {customers.map((customer, i) => (
                <th key={i} scope="col" className="px-6 py-3 text-center text-xs font-medium tracking-wider border-l">
                  <span className={customer === orderingCustomer ? 'text-indigo-600' : 'text-red-600'}>
                    {customer}
                  </span>
                  <div className="text-xs normal-case font-normal text-gray-500">
                    {customer === orderingCustomer ? '(Ordering)' : '(Reallocating)'}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {metricRows.map((row, i) => (
              <tr key={i} className={`${i % 2 === 0 ? 'bg-gray-50' : 'bg-white'} ${
                row.id === 'forecastConsumption' ? 'border-t-2 border-gray-200' : ''
              }`}>
                <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                  row.id === 'forecastConsumption' ? 'text-gray-900 font-semibold' : 'text-gray-900'
                }`}>
                  {row.icon} {row.label}
                </td>
                {row.id === 'regionalDemand' ? (
                  // Special handling for regional demand to merge cells with same value
                  regionalDemandGroups.map((group, groupIndex) => (
                    <td 
                      key={groupIndex}
                      colSpan={group.customers.length}
                      className="px-6 py-4 text-center text-sm border-l"
                    >
                      <span className={getValueColor(group.value, 0)}>
                        {row.format(group.value)}
                        {getIcon(group.value, 0)}
                      </span>
                    </td>
                  ))
                ) : (
                  // Normal rendering for other rows
                  customers.map((customer, j) => {
                    const value = metrics[row.id][customer];
                    const target = row.compareToOrdering 
                      ? metrics[row.id][orderingCustomer] 
                      : metrics[row.id].target;
                    const showComparison = row.compareToOrdering || row.compareToTarget;
                    const isOrderingCustomer = customer === orderingCustomer;
                    
                    return (
                      <td 
                        key={j} 
                        className={`px-6 py-4 text-center text-sm border-l ${
                          isOrderingCustomer ? 'bg-blue-50' : ''
                        }`}
                      >
                        <span 
                          className={showComparison ? getValueColor(value, target, row.reverse) : 'text-gray-900'}
                        >
                          {row.format(value)} 
                          {showComparison && getIcon(value, target, row.reverse)}
                        </span>
                      </td>
                    );
                  })
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DecisionFactorsTable;
