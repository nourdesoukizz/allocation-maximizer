import React from "react";

// Example data: replace with dynamic data when available
const updates = [
  { customer: "MicroTech", before: 400, after: 600 },
  { customer: "TechGiant", before: 400, after: 250 },
  { customer: "DataCore", before: 200, after: 150 },
];

const typeColor = {
  increase: "text-green-600 bg-green-50 border-green-200",
  decrease: "text-red-600 bg-red-50 border-red-200",
};

export default function LatestAllocationUpdatesTable() {
  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-2">Latest Allocation Updates</h3>
      <table className="min-w-full divide-y divide-gray-200 border rounded-lg overflow-hidden">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Customer</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Before</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">After</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-100">
          {updates.map((row) => (
            <tr key={row.customer}>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.customer}</td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{row.before}</td>
              <td className={`px-6 py-4 whitespace-nowrap text-sm font-semibold rounded ${row.after > row.before ? 'text-green-700 bg-green-50' : 'text-red-700 bg-red-50'}`}>{row.after}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
