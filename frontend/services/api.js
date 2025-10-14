/**
 * API Service for Module 4 - Allocation Maximizer
 * Handles communication with the FastAPI backend
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8004';
const TIMEOUT = 30000; // 30 seconds

/**
 * Generic API client with error handling
 */
class ApiClient {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add timeout using AbortController
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);
    config.signal = controller.signal;

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch {
          errorData = { error: 'Unknown error', message: response.statusText };
        }
        
        throw new ApiError(
          errorData.message || 'Request failed',
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        throw new ApiError('Request timeout', 408, { timeout: true });
      }
      
      if (error instanceof ApiError) {
        throw error;
      }
      
      throw new ApiError(error.message || 'Network error', 0, { network: true });
    }
  }

  get(endpoint, options = {}) {
    return this.request(endpoint, { method: 'GET', ...options });
  }

  post(endpoint, data, options = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
      ...options,
    });
  }

  put(endpoint, data, options = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
      ...options,
    });
  }

  delete(endpoint, options = {}) {
    return this.request(endpoint, { method: 'DELETE', ...options });
  }
}

/**
 * Custom API Error class
 */
class ApiError extends Error {
  constructor(message, status, data = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

// Create API client instance
const apiClient = new ApiClient();

/**
 * Health Check API
 */
export const healthApi = {
  /**
   * Basic health check
   */
  check: () => apiClient.get('/health/'),

  /**
   * Detailed health check with component status
   */
  detailed: () => apiClient.get('/health/detailed'),

  /**
   * Service status
   */
  status: () => apiClient.get('/health/status'),
};

/**
 * Optimization API
 */
export const optimizationApi = {
  /**
   * Run optimization with specified strategy
   * @param {Object} request - Optimization request data
   * @param {string} request.strategy - Strategy to use (priority_based, fair_share, hybrid, auto_select)
   * @param {Array} request.allocation_data - Array of allocation records
   * @param {Object} request.constraints - Optional business constraints
   */
  optimize: (request) => {
    // Map frontend field names to backend field names if needed
    const mappedRequest = mapOptimizationRequest(request);
    return apiClient.post('/optimization/optimize', mappedRequest);
  },

  /**
   * Compare multiple optimization strategies
   * @param {Object} request - Strategy comparison request
   * @param {Array} request.allocation_data - Array of allocation records  
   * @param {Array} request.strategies - Strategies to compare
   * @param {Object} request.constraints - Optional business constraints
   */
  compareStrategies: (request) => {
    const mappedRequest = mapComparisonRequest(request);
    return apiClient.post('/optimization/compare-strategies', mappedRequest);
  },

  /**
   * Get strategy recommendation based on data analysis
   * @param {Object} request - Strategy recommendation request
   * @param {Array} request.allocation_data - Array of allocation records
   * @param {Object} request.user_preferences - User preferences
   */
  recommendStrategy: (request) => {
    const mappedRequest = mapRecommendationRequest(request);
    return apiClient.post('/optimization/recommend-strategy', mappedRequest);
  },

  /**
   * Upload allocation data file
   * @param {File} file - File to upload (CSV, Excel, JSON)
   * @param {boolean} validateOnly - Only validate, don't store
   */
  uploadFile: (file, validateOnly = false) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('validate_only', validateOnly);

    return apiClient.request('/optimization/upload-file', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it with boundary
    });
  },

  /**
   * Get available optimization strategies
   */
  getAvailableStrategies: () => apiClient.get('/optimization/available-strategies'),

  /**
   * Get optimization result by request ID
   * @param {string} requestId - Request ID
   */
  getResult: (requestId) => apiClient.get(`/optimization/results/${requestId}`),

  /**
   * Get CSV data preview for form initialization
   * Returns structured data with customers, distribution centers, and products
   */
  getCsvData: () => apiClient.get('/optimization/data/csv-preview'),
};

/**
 * Map frontend optimization request to backend format
 */
function mapOptimizationRequest(request) {
  return {
    strategy: mapStrategyName(request.optimizerType || request.strategy),
    allocation_data: mapAllocationData(request.allocation_data || request.data),
    constraints: mapConstraints(request.constraints),
    strategy_params: request.strategy_params || {},
    priority_weight: request.priority_weight || 0.6,
    fairness_weight: request.fairness_weight || 0.4,
    prefer_efficiency: request.prefer_efficiency ?? true,
    prefer_speed: request.prefer_speed ?? false,
    max_execution_time: request.max_execution_time || 60,
  };
}

/**
 * Map frontend strategy comparison request to backend format
 */
function mapComparisonRequest(request) {
  return {
    allocation_data: mapAllocationData(request.allocation_data || request.data),
    strategies: (request.strategies || []).map(mapStrategyName),
    constraints: mapConstraints(request.constraints),
  };
}

/**
 * Map frontend strategy recommendation request to backend format
 */
function mapRecommendationRequest(request) {
  return {
    allocation_data: mapAllocationData(request.allocation_data || request.data),
    user_preferences: request.user_preferences || {},
  };
}

/**
 * Map strategy names from frontend to backend
 */
function mapStrategyName(strategy) {
  const strategyMap = {
    'fairshare': 'fair_share',
    'fair_share': 'fair_share',
    'priority': 'priority_based',
    'priority_based': 'priority_based',
    'hybrid': 'hybrid',
    'auto': 'auto_select',
    'auto_select': 'auto_select',
  };
  
  return strategyMap[strategy] || strategy;
}

/**
 * Map allocation data from frontend format to backend format
 */
function mapAllocationData(data) {
  if (!Array.isArray(data)) {
    return [];
  }

  return data.map(record => ({
    dc_id: record.dcId || record.dc_id || record.distributionCenter,
    sku_id: record.skuId || record.sku_id || record.productId || record.product_id,
    customer_id: record.customerId || record.customer_id,
    current_inventory: Number(record.currentInventory || record.current_inventory || record.inventory || 0),
    forecasted_demand: Number(record.forecastedDemand || record.forecasted_demand || record.demand || 0),
    dc_priority: Number(record.dcPriority || record.dc_priority || record.priority || 1),
    customer_tier: record.customerTier || record.customer_tier || 'A',
    sla_level: record.slaLevel || record.sla_level || 'Standard',
    min_order_quantity: Number(record.minOrderQuantity || record.min_order_quantity || 1),
    sku_category: record.skuCategory || record.sku_category || 'default',
  }));
}

/**
 * Map constraints from frontend format to backend format
 */
function mapConstraints(constraints) {
  if (!constraints) {
    return null;
  }

  return {
    min_allocation: Number(constraints.minAllocation || constraints.min_allocation || 0),
    max_allocation: constraints.maxAllocation || constraints.max_allocation || null,
    min_order_quantity: Number(constraints.minOrderQuantity || constraints.min_order_quantity || 1),
    safety_stock_buffer: Number(constraints.safetyStockBuffer || constraints.safety_stock_buffer || 0.1),
    allow_substitution: constraints.allowSubstitution ?? constraints.allow_substitution ?? true,
    max_substitution_ratio: Number(constraints.maxSubstitutionRatio || constraints.max_substitution_ratio || 0.3),
    respect_customer_tier: constraints.respectCustomerTier ?? constraints.respect_customer_tier ?? true,
    respect_sla_levels: constraints.respectSlaLevels ?? constraints.respect_sla_levels ?? true,
  };
}

/**
 * Map backend optimization result to frontend format
 */
export function mapOptimizationResult(result) {
  if (!result) return null;

  return {
    success: result.success,
    status: result.success ? 'completed' : 'failed',
    strategy_used: result.strategy_used,
    optimizer_used: result.strategy_used, // Backward compatibility
    model_used: 'optimization_engine', // Default model name
    
    // Summary metrics
    allocation_summary: {
      total_demand: result.total_demand,
      total_allocated: result.total_allocated,
      fulfillment_rate: result.allocation_efficiency,
      unallocated_demand: result.unallocated_demand,
      substitutions_used: result.substitutions_made?.length || 0,
      optimization_time: result.optimization_time,
    },

    // Detailed results
    allocations: result.allocations?.map(allocation => ({
      dc_id: allocation.dc_id,
      sku_id: allocation.sku_id,
      customer_id: allocation.customer_id,
      allocated_quantity: allocation.allocated_quantity,
      forecasted_demand: allocation.forecasted_demand,
      current_inventory: allocation.current_inventory,
      allocation_efficiency: allocation.allocation_efficiency,
      allocation_round: allocation.allocation_round,
    })) || [],

    // Substitutions
    substitutions_made: result.substitutions_made || [],
    
    // Metadata
    constraints_violated: result.constraints_violated || [],
    request_id: result.request_id,
    timestamp: result.timestamp,
    recommendation: result.recommendation,
  };
}

// Export error class
export { ApiError };

// Export default client
export default apiClient;