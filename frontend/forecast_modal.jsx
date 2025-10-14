import React from 'react';
import { Line } from 'react-chartjs-2';
import { FaTimes } from 'react-icons/fa';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ForecastModal = ({ isOpen, onClose, customerData }) => {
  if (!isOpen || !customerData) return null;

  const { customer, selectedModel, demandForecast } = customerData;

  // Colors for different models
  const modelColors = {
    'Monte Carlo Simulation': 'rgb(255, 99, 132)',
    'Markov Decision Processes (MDP)': 'rgb(255, 206, 86)',
    'ARIMA': 'rgb(255, 115, 0)',
    'Deep Q-Networks': 'rgb(255, 159, 64)',
  };

  const chartData = {
    labels: demandForecast.map(d => {
      const date = new Date(d.date);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      // Actual data points
      {
        label: 'Actual',
        data: demandForecast.map(d => d.actual),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        pointStyle: 'circle',
        pointRadius: 6,
        borderWidth: 2,
        fill: false,
        tension: 0.1
      },
      // Model predictions
      ...Object.keys(modelColors).map(modelName => ({
        label: modelName,
        data: demandForecast.map(d => d.models[modelName]),
        borderColor: modelColors[modelName],
        backgroundColor: modelColors[modelName].replace('rgb', 'rgba').replace(')', ', 0.5)'),
        pointStyle: 'circle',
        pointRadius: modelName === selectedModel ? 4 : 0,
        borderWidth: modelName === selectedModel ? 2 : 1,
        borderDash: modelName === selectedModel ? [] : [5, 5],
        fill: false,
        tension: 0.1,
        hidden: modelName !== selectedModel
      }))
    ]
  };

  // Custom plugin for vertical line at the 8th week
  const verticalLinePlugin = {
    id: 'verticalLinePlugin',
    afterDatasetsDraw(chart, args, pluginOptions) {
      const { ctx, chartArea, scales } = chart;
      // 8th week = index 7
      const xScale = scales.x;
      if (!xScale) return;
      const xValue = xScale.getPixelForValue(chartData.labels[7]);
      if (xValue && xValue > xScale.left && xValue < xScale.right) {
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.moveTo(xValue, chartArea.top);
        ctx.lineTo(xValue, chartArea.bottom);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.9)'; // Red
        ctx.stroke();
        ctx.setLineDash([]);
        // Draw label
        ctx.font = '12px sans-serif';
        ctx.fillStyle = 'rgba(220, 38, 38, 0.9)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Current Month', xValue + 6, chartArea.top + 6);
        ctx.restore();
      }
    }
  };

  const chartOptions = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        onClick: (e, legendItem, legend) => {
          const index = legendItem.datasetIndex;
          const ci = legend.chart;
          
          if (legendItem.text === 'Actual') {
            ci.setDatasetVisibility(index, !ci.isDatasetVisible(index));
          } else {
            // For model datasets, only show one at a time
            chartData.datasets.forEach((dataset, idx) => {
              if (dataset.label !== 'Actual') {
                ci.setDatasetVisibility(idx, dataset.label === legendItem.text);
              }
            });
          }
          ci.update();
        }
      },
      title: {
        display: true,
        text: `${customer} - Weekly Demand Forecast`,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value} units`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Week',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: false
        }
      },
      y: {
        title: {
          display: true,
          text: 'Units',
          font: {
            weight: 'bold'
          }
        },
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };

  return (
    <div className={`fixed inset-0 z-50 overflow-y-auto ${isOpen ? '' : 'hidden'}`}>
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 transition-opacity" aria-hidden="true">
          <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
        </div>

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Demand Forecast Analysis
              </h3>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500"
              >
                <FaTimes className="h-6 w-6" />
              </button>
            </div>
            
            <div className="mt-4">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <Line data={chartData} options={chartOptions} plugins={[verticalLinePlugin]} />
              </div>
              
              <div className="mt-4 bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h4 className="font-semibold text-gray-700 mb-2">Model Analysis</h4>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Selected Model:</span> {selectedModel}
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  <span className="font-medium">Forecast Period:</span> 8 weeks historical + 4 weeks forecast
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  <span className="font-medium">Current Week:</span> Week of {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastModal;
