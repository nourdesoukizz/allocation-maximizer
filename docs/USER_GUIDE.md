# Allocation Maximizer User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Web Interface](#using-the-web-interface)
4. [Understanding Optimization Strategies](#understanding-optimization-strategies)
5. [Data Input Methods](#data-input-methods)
6. [Interpreting Results](#interpreting-results)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Introduction

The Allocation Maximizer is a sophisticated tool designed to optimize inventory allocation across distribution centers (DCs) to customers. It uses advanced algorithms to determine the most efficient allocation strategy based on your specific business needs and constraints.

### Key Features
- **Multiple Optimization Strategies**: Priority-based, fair share, hybrid, and automatic selection
- **Real-time Processing**: Fast allocation calculations with caching for improved performance
- **Flexible Input**: Support for CSV, Excel, and direct data entry
- **Comprehensive Results**: Detailed allocation reports with efficiency metrics
- **Strategy Comparison**: Compare different approaches to find the best fit

## Getting Started

### Prerequisites
- Web browser (Chrome, Firefox, Safari, or Edge)
- Internet connection
- Allocation data in supported format (CSV, Excel, or JSON)

### Accessing the Application
1. Open your web browser
2. Navigate to: `http://localhost:3000` (development) or your deployed URL
3. The application will load the main dashboard

### First-Time Setup
No account creation or login is required. The application is ready to use immediately.

## Using the Web Interface

### Main Dashboard
The main interface consists of several key sections:

#### 1. Strategy Selection
Choose your optimization approach:
- **Priority Based**: Allocates inventory based on DC priority rankings
- **Fair Share**: Distributes inventory proportionally across all DCs
- **Hybrid**: Combines priority and fairness approaches
- **Auto Select**: Automatically chooses the best strategy for your data

#### 2. Data Input Section
Two options for providing allocation data:
- **File Upload**: Upload CSV or Excel files with your allocation data
- **Manual Entry**: Enter data directly through the web form

#### 3. Configuration Options
Adjust optimization parameters:
- **Priority Weight**: How much to favor priority-based allocation (0.0-1.0)
- **Fairness Weight**: How much to favor fair distribution (0.0-1.0)
- **Efficiency Preference**: Prioritize speed vs. thoroughness
- **Constraints**: Set minimum/maximum allocation limits

#### 4. Results Display
View optimization results including:
- Allocation efficiency percentage
- Detailed allocation breakdown
- Strategy performance metrics
- Substitution recommendations (if applicable)

## Understanding Optimization Strategies

### Priority Based Strategy
**Best For**: Organizations with clear DC hierarchy and efficiency focus

**How It Works**:
- Sorts DCs by priority ranking (1 = highest priority)
- Allocates inventory to highest priority DCs first
- Continues until demand is met or inventory is exhausted

**Example Use Case**: 
A company with premium DCs that must be served first, followed by standard DCs.

**Parameters**:
- `respect_customer_tier`: Prioritize premium customers
- `allow_overflow`: Allow allocation beyond forecasted demand

### Fair Share Strategy
**Best For**: Organizations prioritizing equitable distribution

**How It Works**:
- Calculates each DC's proportional share of total demand
- Distributes available inventory based on these ratios
- Ensures balanced allocation across all DCs

**Example Use Case**:
A cooperative where all member DCs should receive proportional allocation based on their demand.

**Parameters**:
- `fairness_weight`: How strictly to enforce proportional distribution
- `rebalancing_iterations`: Number of optimization rounds

### Hybrid Strategy
**Best For**: Organizations balancing efficiency and fairness

**How It Works**:
- Combines priority-based and fair share approaches
- Uses configurable weights to balance the two methods
- Optimizes for both efficiency and equity

**Example Use Case**:
A company that wants to respect DC priorities while ensuring reasonable distribution to all locations.

**Parameters**:
- `priority_weight`: Emphasis on priority-based allocation (0.0-1.0)
- `fairness_weight`: Emphasis on fair distribution (0.0-1.0)

### Auto Select Strategy
**Best For**: Users unsure which strategy to use

**How It Works**:
- Analyzes your data characteristics
- Automatically selects the most appropriate strategy
- Provides reasoning for the recommendation

**Example Use Case**:
First-time users or scenarios with varying data patterns.

**Parameters**:
- `prefer_efficiency`: Favor strategies that maximize allocation efficiency
- `prefer_fairness`: Favor strategies that ensure equitable distribution
- `prefer_speed`: Favor faster algorithms over more complex ones

## Data Input Methods

### Method 1: File Upload

#### Supported Formats
- **CSV**: Comma-separated values (.csv)
- **Excel**: Microsoft Excel files (.xlsx)
- **JSON**: JavaScript Object Notation (.json)

#### Required Columns
Your data file must include these columns:
- `dc_id`: Distribution Center identifier
- `sku_id`: Product/SKU identifier  
- `customer_id`: Customer identifier
- `current_inventory`: Available inventory quantity
- `forecasted_demand`: Expected demand quantity
- `dc_priority`: Priority ranking (1 = highest)

#### Optional Columns
- `customer_tier`: Customer classification (premium, standard, basic)
- `sla_level`: Service level agreement (express, standard, economy)
- `min_order_quantity`: Minimum order size
- `sku_category`: Product category

#### Example CSV Format
```csv
dc_id,sku_id,customer_id,current_inventory,forecasted_demand,dc_priority,customer_tier,sla_level
DC001,SKU001,CUST001,1000,800,1,premium,express
DC002,SKU001,CUST002,500,600,2,standard,standard
DC003,SKU002,CUST001,300,400,1,premium,express
```

#### Upload Process
1. Click "Choose File" button
2. Select your data file
3. Wait for validation (green checkmark indicates success)
4. Proceed with optimization

### Method 2: Manual Entry
1. Click "Add Row" to create new allocation records
2. Fill in required fields for each row
3. Use "Remove Row" to delete unwanted entries
4. Validate data before submitting

## Interpreting Results

### Results Overview
After optimization, you'll receive a comprehensive report including:

#### Key Metrics
- **Total Allocated**: Sum of all allocated quantities
- **Total Demand**: Sum of all forecasted demand
- **Allocation Efficiency**: Percentage of demand fulfilled
- **Optimization Time**: Processing duration in seconds

#### Detailed Allocation Table
For each allocation record:
- **DC ID**: Distribution center identifier
- **SKU ID**: Product identifier
- **Customer ID**: Customer identifier
- **Allocated Quantity**: Amount allocated to this customer
- **Forecasted Demand**: Original demand forecast
- **Allocation Efficiency**: Percentage of demand fulfilled for this record
- **Allocation Round**: Which optimization round this allocation occurred in

#### Substitution Information
If substitutions were made:
- **Original SKU**: The initially requested product
- **Substitute SKU**: The replacement product offered
- **Quantity**: Amount of substitute allocated
- **Reason**: Why substitution was necessary

### Performance Indicators

#### Allocation Efficiency
- **90-100%**: Excellent - Most/all demand fulfilled
- **75-89%**: Good - Strong allocation performance
- **60-74%**: Fair - Room for improvement
- **Below 60%**: Poor - Investigate inventory or demand issues

#### Strategy Comparison
When comparing strategies, look for:
- **Highest Efficiency**: Best overall resource utilization
- **Fastest Processing**: Quickest optimization time
- **Best Balance**: Optimal efficiency/fairness trade-off

## Advanced Features

### Strategy Comparison
Compare multiple strategies simultaneously:
1. Select "Compare Strategies" option
2. Choose which strategies to compare
3. View side-by-side results
4. Receive automated recommendations

### Constraint Configuration
Set advanced constraints:
- **Minimum Allocation**: Don't allocate below this threshold
- **Maximum Allocation**: Cap allocations at this level
- **Safety Stock Buffer**: Reserve percentage of inventory
- **Substitution Rules**: Allow/disallow product substitutions

### Caching and Performance
The system automatically caches results for:
- Faster repeated calculations
- Improved response times
- Reduced server load

Cache is automatically cleared when:
- Input data changes significantly
- 1 hour has elapsed
- Manual cache clearing is requested

### API Integration
Advanced users can integrate directly with the API:
- RESTful endpoints for all functionality
- JSON request/response format
- Optional API key authentication
- Rate limiting protection

## Troubleshooting

### Common Issues

#### File Upload Fails
**Symptoms**: Error message during file upload
**Solutions**:
1. Check file format (CSV, Excel, JSON only)
2. Verify file size (must be under 10MB)
3. Ensure all required columns are present
4. Check for special characters in data

#### Low Allocation Efficiency
**Symptoms**: Efficiency below 60%
**Solutions**:
1. Verify inventory quantities are sufficient
2. Check demand forecasts for accuracy
3. Review DC priority settings
4. Consider enabling substitutions

#### Slow Performance
**Symptoms**: Long processing times
**Solutions**:
1. Reduce data set size for testing
2. Use "prefer_speed" option in auto-select
3. Check internet connection
4. Clear browser cache

#### Unexpected Results
**Symptoms**: Allocations don't match expectations
**Solutions**:
1. Review strategy selection
2. Check constraint settings
3. Verify input data accuracy
4. Compare with other strategies

### Error Messages

#### "Invalid allocation data"
- **Cause**: Data validation failed
- **Solution**: Check required columns and data types

#### "File too large"
- **Cause**: Upload exceeds size limit
- **Solution**: Split large files or remove unnecessary columns

#### "Rate limit exceeded"
- **Cause**: Too many requests in short time
- **Solution**: Wait 60 seconds before retrying

#### "Optimization failed"
- **Cause**: Internal processing error
- **Solution**: Verify data format and try again

### Getting Help

#### Self-Service Options
1. Review this user guide
2. Check the troubleshooting section
3. Verify data format requirements
4. Test with smaller data sets

#### Contact Support
If issues persist:
1. Note the exact error message
2. Save the input data that caused the problem
3. Record the steps that led to the issue
4. Include browser and operating system information

### Best Practices

#### Data Preparation
- Clean data before upload (remove duplicates, fix typos)
- Validate numbers are positive
- Ensure consistent ID formats
- Include all relevant optional fields

#### Strategy Selection
- Start with auto-select for new data sets
- Use priority-based for clear hierarchies
- Choose fair share for cooperative environments
- Try hybrid for balanced approaches

#### Performance Optimization
- Upload data once, run multiple optimizations
- Use caching for repeated similar requests
- Compare strategies to find best fit
- Monitor allocation efficiency trends

#### Result Analysis
- Review detailed allocations, not just summary metrics
- Check for patterns in unallocated demand
- Analyze substitution recommendations
- Compare results across different time periods