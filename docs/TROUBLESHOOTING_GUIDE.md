# Troubleshooting Guide - Allocation Maximizer

## Table of Contents
1. [Quick Diagnostic Steps](#quick-diagnostic-steps)
2. [Frontend Issues](#frontend-issues)
3. [Backend API Issues](#backend-api-issues)
4. [Data Processing Issues](#data-processing-issues)
5. [Performance Issues](#performance-issues)
6. [Security & Authentication Issues](#security--authentication-issues)
7. [Deployment Issues](#deployment-issues)
8. [Cache & Redis Issues](#cache--redis-issues)
9. [Error Code Reference](#error-code-reference)
10. [Debug Tools & Techniques](#debug-tools--techniques)

## Quick Diagnostic Steps

### System Health Check
```bash
# Check backend health
curl http://localhost:8000/health/

# Check detailed health with component status
curl http://localhost:8000/health/detailed

# Check performance metrics
curl http://localhost:8000/health/metrics
```

### Service Status Verification
```bash
# Backend status
ps aux | grep python | grep run_server

# Frontend status (if using npm)
ps aux | grep npm | grep dev

# Redis status
redis-cli ping
```

### Log Investigation
```bash
# Backend logs (if running in terminal)
tail -f logs/app.log

# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# System logs
journalctl -u your-service-name -f
```

## Frontend Issues

### Issue: Frontend Won't Load
**Symptoms**: White screen, "Cannot connect" error, or 404

**Diagnostic Steps**:
1. Check browser console for errors (F12 → Console)
2. Verify development server is running
3. Check network connectivity

**Solutions**:
```bash
# Restart frontend development server
cd frontend/
npm run dev

# Clear browser cache
# Chrome: Ctrl+Shift+R (hard refresh)
# Firefox: Ctrl+F5

# Check for port conflicts
lsof -i :3000
```

### Issue: API Connection Failed
**Symptoms**: "Network Error", "Failed to fetch", CORS errors

**Diagnostic Steps**:
1. Check browser Network tab (F12 → Network)
2. Verify API URL configuration
3. Test API directly: `curl http://localhost:8000/health/`

**Solutions**:
```bash
# Check frontend environment variables
cat .env.local
# Should contain: VITE_API_URL=http://localhost:8000

# Verify backend is running
curl http://localhost:8000/health/

# Check CORS configuration in backend
# Update ALLOWED_ORIGINS if needed
```

### Issue: File Upload Fails
**Symptoms**: "File upload failed", progress bar stuck

**Diagnostic Steps**:
1. Check file size (must be < 10MB)
2. Verify file format (CSV, Excel, JSON only)
3. Check browser console for errors

**Solutions**:
```bash
# Test file upload via curl
curl -X POST "http://localhost:8000/optimization/upload-file" \
  -F "file=@your_file.csv" \
  -F "validate_only=false"

# Check backend file size limits
grep MAX_FILE_SIZE backend/config.py
```

### Issue: Optimization Results Not Displaying
**Symptoms**: Spinner keeps loading, no results shown

**Diagnostic Steps**:
1. Check browser console for JavaScript errors
2. Verify API response format
3. Check for data validation errors

**Solutions**:
```javascript
// Check API response in browser console
fetch('http://localhost:8000/optimization/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(your_request_data)
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error(error));
```

## Backend API Issues

### Issue: Server Won't Start
**Symptoms**: "Port already in use", import errors, dependency conflicts

**Diagnostic Steps**:
1. Check for port conflicts: `lsof -i :8000`
2. Verify Python environment and dependencies
3. Check configuration files

**Solutions**:
```bash
# Kill processes using port 8000
sudo lsof -t -i:8000 | xargs kill -9

# Recreate virtual environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+

# Verify all required files exist
ls -la main.py run_server.py requirements.txt
```

### Issue: Import/Module Errors
**Symptoms**: "ModuleNotFoundError", "ImportError"

**Diagnostic Steps**:
1. Check Python path and virtual environment
2. Verify all dependencies are installed
3. Check for circular imports

**Solutions**:
```bash
# Verify virtual environment is activated
which python  # Should point to venv/bin/python

# Install missing dependencies
pip install -r requirements.txt

# Check for missing modules
python -c "import fastapi, uvicorn, pandas, redis"

# Reinstall specific problematic packages
pip uninstall package_name
pip install package_name
```

### Issue: Database/Redis Connection Failed
**Symptoms**: "Redis connection failed", caching not working

**Diagnostic Steps**:
1. Check Redis server status: `redis-cli ping`
2. Verify connection string in environment variables
3. Test Redis connectivity

**Solutions**:
```bash
# Start Redis server
redis-server

# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Check environment variables
echo $REDIS_URL

# Fallback to in-memory cache (temporary)
unset REDIS_URL
# Restart backend - it will use in-memory cache
```

## Data Processing Issues

### Issue: CSV Validation Errors
**Symptoms**: "Invalid allocation data", specific column errors

**Required Columns**:
- `dc_id`: Distribution Center ID
- `sku_id`: Product/SKU ID
- `customer_id`: Customer ID
- `current_inventory`: Available inventory (numeric)
- `forecasted_demand`: Expected demand (numeric)
- `dc_priority`: Priority ranking (integer, 1=highest)

**Solutions**:
```bash
# Validate CSV format
head -5 your_file.csv

# Check for required columns
csvcut -n your_file.csv

# Example correct format:
dc_id,sku_id,customer_id,current_inventory,forecasted_demand,dc_priority
DC001,SKU001,CUST001,1000,800,1
DC002,SKU001,CUST002,500,600,2
```

### Issue: Optimization Fails
**Symptoms**: "Optimization failed", 500 error, low efficiency

**Diagnostic Steps**:
1. Check data quality and completeness
2. Verify inventory vs. demand ratios
3. Review constraint configurations

**Solutions**:
```bash
# Test with minimal data set
curl -X POST "http://localhost:8000/optimization/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "priority_based",
    "allocation_data": [
      {
        "dc_id": "DC001",
        "sku_id": "SKU001",
        "customer_id": "CUST001",
        "current_inventory": 1000,
        "forecasted_demand": 800,
        "dc_priority": 1
      }
    ]
  }'

# Check for data quality issues:
# - Negative inventory/demand values
# - Missing or duplicate IDs
# - Invalid priority values
```

### Issue: Low Allocation Efficiency
**Symptoms**: Efficiency below 60%, high unallocated demand

**Common Causes & Solutions**:
1. **Insufficient Inventory**: Total inventory < total demand
   - Review inventory levels
   - Consider enabling substitutions
   
2. **Poor Priority Configuration**: All DCs have same priority
   - Review and adjust DC priority rankings
   - Use fair_share strategy instead
   
3. **Strict Constraints**: Constraints prevent allocation
   - Review min/max allocation limits
   - Adjust safety stock buffer

## Performance Issues

### Issue: Slow Response Times
**Symptoms**: Requests taking > 5 seconds, timeouts

**Diagnostic Steps**:
1. Check performance metrics: `curl http://localhost:8000/health/metrics`
2. Monitor resource usage: `top`, `htop`
3. Check data set size

**Solutions**:
```bash
# Enable caching
export REDIS_URL=redis://localhost:6379

# Reduce data set size for testing
head -100 large_file.csv > test_file.csv

# Use faster strategy
# Set "prefer_speed": true in auto_select

# Increase worker count (production)
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Issue: Memory Usage High
**Symptoms**: Out of memory errors, slow performance

**Solutions**:
```bash
# Monitor memory usage
ps aux --sort=-%mem | head

# Optimize data processing
# - Process files in chunks
# - Clear unused data from memory
# - Use more efficient data types

# Increase system memory or use smaller data sets
```

## Security & Authentication Issues

### Issue: Rate Limiting Errors
**Symptoms**: "429 Too Many Requests"

**Current Limits**:
- Optimization: 100 requests/minute
- Strategy comparison: 50 requests/minute
- File upload: 20 requests/minute

**Solutions**:
```bash
# Wait for rate limit reset (1 minute)
sleep 60

# Use API key for higher limits (if configured)
curl -H "Authorization: Bearer your_api_key" ...

# Adjust rate limits in production (backend config)
```

### Issue: CORS Errors
**Symptoms**: "Access-Control-Allow-Origin" errors

**Solutions**:
```bash
# Update backend CORS configuration
export ALLOWED_ORIGINS="http://localhost:3000,https://yourdomain.com"

# Restart backend after changing CORS settings

# For development, temporarily allow all origins
export ALLOWED_ORIGINS="*"  # Not recommended for production
```

## Deployment Issues

### Issue: Docker Build Failures
**Symptoms**: Build errors, dependency conflicts

**Solutions**:
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker-compose build --no-cache

# Check Dockerfile syntax
docker build -t test-image .

# Verify base image compatibility
```

### Issue: Environment Variables Not Loading
**Symptoms**: Default values being used, config errors

**Solutions**:
```bash
# Check environment variables
printenv | grep -E "(REDIS_URL|ENVIRONMENT|DEBUG)"

# Verify .env file format (no spaces around =)
ENVIRONMENT=production
DEBUG=false

# For Docker, use docker-compose environment section
```

## Cache & Redis Issues

### Issue: Redis Connection Intermittent
**Symptoms**: Sometimes works, sometimes fails

**Solutions**:
```bash
# Check Redis memory usage
redis-cli info memory

# Monitor Redis logs
redis-cli monitor

# Increase Redis memory limit
# Edit redis.conf: maxmemory 512mb

# Check network connectivity
ping redis-host
```

### Issue: Cache Not Working
**Symptoms**: No performance improvement, always calculating

**Diagnostic Steps**:
```bash
# Check cache statistics
curl http://localhost:8000/health/cache-stats

# Monitor cache hit/miss ratio
curl http://localhost:8000/health/metrics
```

**Solutions**:
```bash
# Clear and restart cache
curl -X POST http://localhost:8000/health/cache-clear

# Verify cache TTL settings
# Check if TTL is too short

# Test cache manually
redis-cli set test_key "test_value"
redis-cli get test_key
```

## Error Code Reference

### HTTP Status Codes
- **400 Bad Request**: Invalid input data, missing required fields
- **401 Unauthorized**: Invalid or missing API key
- **413 Payload Too Large**: File size exceeds limit (10MB)
- **422 Unprocessable Entity**: Data validation failed
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server processing error
- **503 Service Unavailable**: System overloaded or maintenance

### Common Error Messages
- **"Invalid allocation data"**: Check CSV format and required columns
- **"Optimization failed"**: Data quality or constraint issues
- **"File upload failed"**: File size, format, or content issues
- **"Cache service unavailable"**: Redis connection problems
- **"Rate limit exceeded"**: Too many requests, wait or use API key

## Debug Tools & Techniques

### Backend Debugging
```bash
# Enable debug mode
export DEBUG=true

# Increase logging verbosity
export LOG_LEVEL=DEBUG

# Use Python debugger
python -m pdb main.py

# Profile performance
python -m cProfile run_server.py
```

### Frontend Debugging
```javascript
// Enable debug mode in browser
localStorage.setItem('debug', 'true');

// Monitor API calls
// Open DevTools → Network tab

// Check Redux state (if using Redux)
window.__REDUX_DEVTOOLS_EXTENSION__?.()
```

### Network Debugging
```bash
# Test API endpoints
curl -v http://localhost:8000/health/

# Monitor network traffic
tcpdump -i any port 8000

# Check DNS resolution
nslookup your-domain.com

# Test connectivity
telnet your-server 8000
```

### Data Debugging
```bash
# Validate CSV structure
csvstat your_file.csv

# Check for encoding issues
file your_file.csv
iconv -f ISO-8859-1 -t UTF-8 your_file.csv > fixed_file.csv

# Sample data for testing
head -10 your_file.csv > sample.csv
```

### Log Analysis
```bash
# Search for specific errors
grep -i "error" logs/app.log

# Monitor logs in real-time
tail -f logs/app.log | grep -i "optimization"

# Analyze request patterns
awk '{print $1}' access.log | sort | uniq -c | sort -nr
```

### Performance Monitoring
```bash
# Monitor system resources
iostat 1
vmstat 1
netstat -an | grep 8000

# Application metrics
curl http://localhost:8000/health/metrics | jq '.'

# Database performance
redis-cli --latency-history
```

### Emergency Recovery
```bash
# Stop all services
docker-compose down
pkill -f "python run_server"
pkill -f "npm"

# Clear all data and restart
rm -rf logs/* cache/*
docker-compose up --build

# Restore from backup
cp backup/allocation_data.csv data/
systemctl restart your-service
```

For additional support:
1. Check the [User Guide](USER_GUIDE.md) for usage instructions
2. Review [API Documentation](API_DOCUMENTATION.md) for endpoint details
3. Consult [Deployment Guide](DEPLOYMENT_GUIDE.md) for configuration help
4. Create an issue in the project repository with detailed error information