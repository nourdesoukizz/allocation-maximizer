# User Acceptance Testing (UAT) Scenarios

## Overview
This document outlines comprehensive User Acceptance Testing scenarios for the Allocation Maximizer application. These tests ensure the system meets business requirements and provides a satisfactory user experience.

## Test Environment Setup
- **Frontend URL**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Test Data**: Use provided sample files in `/data/` directory
- **Browser**: Chrome, Firefox, Safari, Edge
- **Prerequisites**: Backend and frontend services running

## Test Data Sets

### Dataset 1: Basic Allocation (Small)
```csv
dc_id,sku_id,customer_id,current_inventory,forecasted_demand,dc_priority,customer_tier,sla_level
DC001,SKU001,CUST001,1000,800,1,premium,express
DC002,SKU001,CUST002,500,600,2,standard,standard
DC003,SKU002,CUST001,300,400,1,premium,express
```

### Dataset 2: Complex Allocation (Medium)
- 50 records with mixed priorities
- Multiple SKUs and customers
- Varying inventory levels

### Dataset 3: Large Scale (Large)
- 500+ records
- Performance testing dataset
- Stress test scenarios

## Test Scenarios

---

## Scenario 1: Basic Application Access
**Objective**: Verify users can access and navigate the application
**Priority**: Critical
**Estimated Time**: 5 minutes

### Test Steps:
1. **Navigate to Application**
   - Open browser
   - Go to http://localhost:3000
   - **Expected**: Application loads without errors

2. **Interface Verification**
   - Check main dashboard loads
   - Verify all UI components are visible
   - **Expected**: Clean, professional interface with all elements present

3. **Navigation Test**
   - Click through different sections
   - Test responsive design (resize browser)
   - **Expected**: Smooth navigation, responsive layout

### Acceptance Criteria:
- [ ] Application loads within 5 seconds
- [ ] No JavaScript console errors
- [ ] All UI elements render correctly
- [ ] Interface is responsive across screen sizes

---

## Scenario 2: File Upload and Validation
**Objective**: Test file upload functionality with various file types
**Priority**: Critical
**Estimated Time**: 15 minutes

### Test Steps:
1. **Valid CSV Upload**
   - Click "Choose File" button
   - Select valid CSV file (Dataset 1)
   - **Expected**: File uploads successfully, shows green checkmark

2. **File Validation**
   - Upload file with missing columns
   - **Expected**: Clear error message about missing required columns

3. **File Size Validation**
   - Attempt to upload file > 10MB
   - **Expected**: Error message about file size limit

4. **Format Validation**
   - Try uploading TXT file
   - **Expected**: Error message about unsupported format

5. **Data Preview**
   - Upload valid file
   - Check if data preview is shown
   - **Expected**: First few rows displayed for verification

### Acceptance Criteria:
- [ ] CSV, Excel, and JSON files upload successfully
- [ ] Invalid formats are rejected with clear error messages
- [ ] File size limits are enforced
- [ ] Data validation provides specific error details
- [ ] Upload progress is visible to user

---

## Scenario 3: Strategy Selection and Optimization
**Objective**: Test all optimization strategies with different datasets
**Priority**: Critical
**Estimated Time**: 25 minutes

### Test Steps:
1. **Priority-Based Strategy**
   - Upload Dataset 1
   - Select "Priority Based" strategy
   - Click "Optimize Allocation"
   - **Expected**: Results show DC001 gets priority allocation

2. **Fair Share Strategy**
   - Use same dataset
   - Select "Fair Share" strategy
   - Run optimization
   - **Expected**: More balanced allocation across DCs

3. **Hybrid Strategy**
   - Use same dataset
   - Select "Hybrid" strategy
   - Adjust priority/fairness weights
   - **Expected**: Results balance priority and fairness

4. **Auto Select Strategy**
   - Use same dataset
   - Select "Auto Select" strategy
   - **Expected**: System recommends appropriate strategy with reasoning

5. **Parameter Adjustment**
   - Test different weight combinations
   - Enable/disable efficiency preferences
   - **Expected**: Results change based on parameter adjustments

### Acceptance Criteria:
- [ ] All four strategies execute successfully
- [ ] Results differ meaningfully between strategies
- [ ] Strategy recommendations include clear reasoning
- [ ] Parameter changes affect optimization results
- [ ] Execution time is under 5 seconds for small datasets

---

## Scenario 4: Results Interpretation and Analysis
**Objective**: Verify users can understand and analyze optimization results
**Priority**: High
**Estimated Time**: 20 minutes

### Test Steps:
1. **Results Display**
   - Run optimization with Dataset 1
   - Review results table
   - **Expected**: Clear table with allocation details

2. **Efficiency Metrics**
   - Check allocation efficiency percentage
   - Review total allocated vs. total demand
   - **Expected**: Efficiency calculation is accurate

3. **Detailed Allocation View**
   - Examine allocation per DC/SKU/Customer
   - Check for allocation rounds information
   - **Expected**: Detailed breakdown shows allocation logic

4. **Substitution Information**
   - Enable substitutions in constraints
   - Run optimization
   - **Expected**: Substitution details shown if applicable

5. **Export Functionality**
   - Attempt to export/save results
   - **Expected**: Results can be saved or copied

### Acceptance Criteria:
- [ ] Results are clearly formatted and easy to read
- [ ] Efficiency calculations are accurate
- [ ] Allocation details provide sufficient information
- [ ] Substitution logic is transparent
- [ ] Results can be exported or saved

---

## Scenario 5: Strategy Comparison
**Objective**: Test strategy comparison functionality
**Priority**: High
**Estimated Time**: 15 minutes

### Test Steps:
1. **Multi-Strategy Comparison**
   - Upload Dataset 2
   - Select "Compare Strategies" option
   - Choose 3 strategies to compare
   - **Expected**: Side-by-side comparison results

2. **Performance Analysis**
   - Review efficiency metrics for each strategy
   - Check execution times
   - **Expected**: Clear performance comparison

3. **Recommendation Review**
   - Check automated recommendation
   - Review reasoning provided
   - **Expected**: Logical recommendation with clear explanation

4. **Visual Comparison**
   - Review any charts or visual elements
   - **Expected**: Visual aids help understand differences

### Acceptance Criteria:
- [ ] Multiple strategies can be compared simultaneously
- [ ] Comparison results are clearly differentiated
- [ ] Performance metrics are accurate
- [ ] Recommendations are logical and well-reasoned
- [ ] Visual elements enhance understanding

---

## Scenario 6: Advanced Configuration and Constraints
**Objective**: Test advanced configuration options and constraints
**Priority**: Medium
**Estimated Time**: 20 minutes

### Test Steps:
1. **Basic Constraints**
   - Set minimum allocation limits
   - Set maximum allocation limits
   - **Expected**: Allocations respect constraints

2. **Safety Stock Configuration**
   - Set safety stock buffer (e.g., 10%)
   - Run optimization
   - **Expected**: Inventory buffer is maintained

3. **Substitution Rules**
   - Enable SKU substitution
   - Set substitution ratio limits
   - **Expected**: Substitutions suggested when beneficial

4. **Customer Tier Respect**
   - Enable customer tier priority
   - **Expected**: Premium customers get preference

5. **SLA Level Processing**
   - Enable SLA level consideration
   - **Expected**: Express SLA customers prioritized

### Acceptance Criteria:
- [ ] All constraint types are enforced correctly
- [ ] Safety stock buffers are maintained
- [ ] Substitution logic works as expected
- [ ] Customer tiers affect allocation priority
- [ ] SLA levels influence optimization decisions

---

## Scenario 7: Error Handling and Edge Cases
**Objective**: Test system behavior with invalid inputs and edge cases
**Priority**: High
**Estimated Time**: 20 minutes

### Test Steps:
1. **Invalid Data Handling**
   - Upload file with negative inventory
   - Upload file with zero demand
   - **Expected**: Clear error messages, system remains stable

2. **Network Interruption**
   - Start optimization, disconnect network briefly
   - **Expected**: Appropriate error handling, retry options

3. **Large Dataset Handling**
   - Upload Dataset 3 (500+ records)
   - Monitor performance
   - **Expected**: System handles large data gracefully

4. **Concurrent Operations**
   - Start multiple optimizations simultaneously
   - **Expected**: System queues or handles concurrent requests

5. **Browser Refresh During Operation**
   - Start optimization, refresh page
   - **Expected**: Operation status is handled gracefully

### Acceptance Criteria:
- [ ] Invalid data produces helpful error messages
- [ ] Network issues are handled gracefully
- [ ] Large datasets don't crash the system
- [ ] Concurrent operations are managed properly
- [ ] Page refreshes don't cause data loss

---

## Scenario 8: Performance and Scalability
**Objective**: Verify system performance under various loads
**Priority**: Medium
**Estimated Time**: 25 minutes

### Test Steps:
1. **Small Dataset Performance**
   - Use Dataset 1 (3 records)
   - Measure response time
   - **Expected**: Response under 2 seconds

2. **Medium Dataset Performance**
   - Use Dataset 2 (50 records)
   - Measure response time
   - **Expected**: Response under 5 seconds

3. **Large Dataset Performance**
   - Use Dataset 3 (500+ records)
   - Measure response time
   - **Expected**: Response under 30 seconds

4. **Caching Verification**
   - Run same optimization twice
   - Compare response times
   - **Expected**: Second run is faster (cache hit)

5. **Memory Usage**
   - Monitor browser memory during operations
   - **Expected**: No memory leaks, stable usage

### Acceptance Criteria:
- [ ] Small datasets process in under 2 seconds
- [ ] Medium datasets process in under 5 seconds
- [ ] Large datasets process in under 30 seconds
- [ ] Caching improves subsequent request performance
- [ ] Memory usage remains stable during operations

---

## Scenario 9: Security and Data Validation
**Objective**: Test security features and data validation
**Priority**: High
**Estimated Time**: 15 minutes

### Test Steps:
1. **Input Sanitization**
   - Try uploading file with special characters
   - Enter script tags in form fields
   - **Expected**: Input is sanitized, no XSS vulnerabilities

2. **File Upload Security**
   - Try uploading executable files
   - Upload oversized files
   - **Expected**: Security restrictions enforced

3. **Rate Limiting**
   - Make multiple rapid requests
   - **Expected**: Rate limiting prevents abuse

4. **CORS Verification**
   - Test from different origins (if applicable)
   - **Expected**: CORS policy enforced correctly

5. **Error Information**
   - Trigger various errors
   - **Expected**: Error messages don't expose sensitive information

### Acceptance Criteria:
- [ ] Input validation prevents malicious content
- [ ] File upload restrictions are enforced
- [ ] Rate limiting protects against abuse
- [ ] CORS policies are properly configured
- [ ] Error messages are safe and informative

---

## Scenario 10: Mobile and Cross-Browser Compatibility
**Objective**: Verify application works across different devices and browsers
**Priority**: Medium
**Estimated Time**: 20 minutes

### Test Steps:
1. **Chrome Testing**
   - Test all core functionality in Chrome
   - **Expected**: Full functionality works correctly

2. **Firefox Testing**
   - Test core functionality in Firefox
   - **Expected**: Consistent behavior with Chrome

3. **Safari Testing** (if available)
   - Test core functionality in Safari
   - **Expected**: Consistent behavior across browsers

4. **Mobile Responsive**
   - Test on mobile device or resize browser
   - **Expected**: Interface adapts to smaller screens

5. **Tablet View**
   - Test tablet-sized viewport
   - **Expected**: Good usability on tablet devices

### Acceptance Criteria:
- [ ] Functionality works consistently across Chrome, Firefox, Safari
- [ ] Mobile interface is usable and responsive
- [ ] Tablet view provides good user experience
- [ ] No browser-specific bugs or issues
- [ ] Performance is acceptable across platforms

---

## Overall UAT Sign-off Criteria

### Critical Success Factors
- [ ] All optimization strategies work correctly
- [ ] File upload and validation functions properly
- [ ] Results are accurate and clearly presented
- [ ] Performance meets acceptable thresholds
- [ ] Security measures are effective
- [ ] Error handling is user-friendly

### Performance Benchmarks
- [ ] Page load time < 5 seconds
- [ ] Small optimization (< 10 records) < 2 seconds
- [ ] Medium optimization (< 100 records) < 5 seconds
- [ ] Large optimization (< 1000 records) < 30 seconds
- [ ] System remains responsive during operations

### User Experience Standards
- [ ] Interface is intuitive and easy to navigate
- [ ] Error messages are clear and actionable
- [ ] Results are easy to understand and interpret
- [ ] Help documentation is accessible and useful
- [ ] No training required for basic operations

### Business Requirements Validation
- [ ] All required optimization strategies implemented
- [ ] File format support meets requirements
- [ ] Constraint handling is comprehensive
- [ ] Strategy comparison provides business value
- [ ] Results support decision-making processes

## Test Execution Records

### Test Session 1
- **Date**: ___________
- **Tester**: ___________
- **Environment**: ___________
- **Results**: ___________

### Test Session 2
- **Date**: ___________
- **Tester**: ___________
- **Environment**: ___________
- **Results**: ___________

### Test Session 3
- **Date**: ___________
- **Tester**: ___________
- **Environment**: ___________
- **Results**: ___________

## Issues Log

| Issue ID | Scenario | Description | Priority | Status | Resolution |
|----------|----------|-------------|----------|---------|-------------|
| UAT-001 | | | | | |
| UAT-002 | | | | | |
| UAT-003 | | | | | |

## Final Sign-off

### Business Stakeholder Approval
- **Name**: ___________
- **Date**: ___________
- **Signature**: ___________
- **Comments**: ___________

### Technical Lead Approval
- **Name**: ___________
- **Date**: ___________
- **Signature**: ___________
- **Comments**: ___________

### QA Lead Approval
- **Name**: ___________
- **Date**: ___________
- **Signature**: ___________
- **Comments**: ___________