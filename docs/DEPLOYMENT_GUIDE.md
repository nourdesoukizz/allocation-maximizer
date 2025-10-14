# Deployment Guide - Allocation Maximizer

## Table of Contents
1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Production Environment Setup](#production-environment-setup)
4. [Docker Deployment](#docker-deployment)
5. [Railway Platform Deployment](#railway-platform-deployment)
6. [Environment Configuration](#environment-configuration)
7. [Database & Cache Setup](#database--cache-setup)
8. [Security Configuration](#security-configuration)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

## Overview

The Allocation Maximizer consists of two main components:
- **Backend API**: FastAPI application with ML models and optimization algorithms
- **Frontend**: React-based web interface

### Architecture
```
Internet → Load Balancer → Frontend (React) → Backend API (FastAPI) → Redis Cache
                                                    ↓
                                              ML Models & Data
```

## Development Environment

### Prerequisites
- Python 3.11+
- Node.js 18+
- Redis Server (optional, falls back to in-memory cache)
- Git

### Backend Setup
```bash
# Navigate to backend directory
cd /path/to/allocation-maximizer/backend/api/module4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENVIRONMENT=development
export DEBUG=true
export HOST=localhost
export PORT=8000
export REDIS_URL=redis://localhost:6379  # Optional

# Run development server
python run_server.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd /path/to/allocation-maximizer/frontend

# Install dependencies
npm install

# Set environment variables
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

### Verification
1. Backend: http://localhost:8000/docs
2. Frontend: http://localhost:3000
3. Health Check: http://localhost:8000/health/

## Production Environment Setup

### System Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB minimum
- **OS**: Ubuntu 20.04+, CentOS 8+, or Docker-compatible

### Backend Production Setup
```bash
# Clone repository
git clone <repository-url>
cd allocation-maximizer/backend/api/module4

# Create production virtual environment
python -m venv prod_venv
source prod_venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
pip install gunicorn

# Set production environment variables
export ENVIRONMENT=production
export DEBUG=false
export HOST=0.0.0.0
export PORT=8000
export REDIS_URL=redis://your-redis-host:6379
export ALLOWED_ORIGINS="https://yourdomain.com"

# Run with Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Production Setup
```bash
# Navigate to frontend directory
cd allocation-maximizer/frontend

# Install dependencies
npm install

# Set production environment variables
echo "VITE_API_URL=https://your-api-domain.com" > .env.production

# Build for production
npm run build

# Serve with static file server (nginx, serve, etc.)
npm install -g serve
serve -s dist -p 3000
```

## Docker Deployment

### Backend Dockerfile
Create `backend/api/module4/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Start application
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Frontend Dockerfile
Create `frontend/Dockerfile`:
```dockerfile
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  backend:
    build: ./backend/api/module4
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - REDIS_URL=redis://redis:6379
      - ALLOWED_ORIGINS=http://localhost:3000
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend

volumes:
  redis_data:
```

### Docker Commands
```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build backend
docker-compose up -d backend
```

## Railway Platform Deployment

### Backend Railway Setup
1. Create new Railway project
2. Connect GitHub repository
3. Set environment variables:
   ```
   ENVIRONMENT=production
   DEBUG=false
   PORT=8000
   ALLOWED_ORIGINS=https://your-frontend-domain.up.railway.app
   ```
4. Add Redis service:
   - Click "Add Service" → "Database" → "Redis"
   - Note the connection URL
5. Update backend environment:
   ```
   REDIS_URL=${{Redis.REDIS_URL}}
   ```

### Frontend Railway Setup
1. Create separate Railway service
2. Connect same GitHub repository
3. Set build settings:
   ```
   Build Command: npm run build
   Start Command: npm run preview
   ```
4. Set environment variables:
   ```
   VITE_API_URL=https://your-backend-domain.up.railway.app
   ```

### Railway Configuration Files

Create `railway.json` in project root:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT",
    "healthcheckPath": "/health/"
  }
}
```

## Environment Configuration

### Backend Environment Variables
```bash
# Application Settings
ENVIRONMENT=production  # development, staging, production
DEBUG=false            # Enable debug mode
HOST=0.0.0.0          # Host to bind to
PORT=8000             # Port to run on

# CORS Settings
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Cache Settings
REDIS_URL=redis://localhost:6379  # Redis connection string
CACHE_TTL=3600                    # Cache TTL in seconds

# File Upload Settings
MAX_FILE_SIZE_MB=10              # Maximum upload size
SUPPORTED_FILE_FORMATS=csv,xlsx,json

# Security Settings
API_RATE_LIMIT=100               # Requests per minute
SECRET_KEY=your-secret-key       # For API key generation

# Data Settings
DATA_DIRECTORY=./data            # Directory for CSV files
```

### Frontend Environment Variables
```bash
# API Configuration
VITE_API_URL=https://api.yourdomain.com  # Backend API URL

# Feature Flags
VITE_ENABLE_FILE_UPLOAD=true            # Enable file upload feature
VITE_ENABLE_STRATEGY_COMPARISON=true    # Enable strategy comparison

# Analytics (optional)
VITE_ANALYTICS_ID=your-analytics-id     # Google Analytics ID
```

## Database & Cache Setup

### Redis Configuration
For production Redis setup:

#### Self-Hosted Redis
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Configure Redis (/etc/redis/redis.conf)
bind 127.0.0.1
port 6379
maxmemory 256mb
maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Redis Cloud Services
- **Redis Cloud**: Managed Redis service
- **AWS ElastiCache**: Amazon's Redis service
- **Railway Redis**: Built-in Redis add-on

### Data Directory Setup
```bash
# Create data directory
mkdir -p /app/data

# Set permissions
chmod 755 /app/data

# Copy sample data
cp allocation_data.csv /app/data/
```

## Security Configuration

### SSL/TLS Setup
For production, enable HTTPS:

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### API Security
```bash
# Generate API keys for production
curl -X POST "https://your-api-domain.com/health/security/add-api-key" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "production_client", "permissions": ["read", "write"]}'
```

### Firewall Configuration
```bash
# Allow HTTP/HTTPS traffic
sudo ufw allow 80
sudo ufw allow 443

# Allow SSH (if needed)
sudo ufw allow 22

# Allow backend port (if directly exposed)
sudo ufw allow 8000

# Enable firewall
sudo ufw enable
```

## Monitoring & Logging

### Application Monitoring
The application includes built-in monitoring endpoints:

- **Health Check**: `/health/`
- **Detailed Health**: `/health/detailed`
- **Performance Metrics**: `/health/metrics`
- **Cache Statistics**: `/health/cache-stats`

### Log Configuration
```python
# logging.conf
[loggers]
keys=root,uvicorn,application

[handlers]
keys=console,file

[formatters]
keys=default

[logger_root]
level=INFO
handlers=console,file

[logger_uvicorn]
level=INFO
handlers=console,file
qualname=uvicorn

[logger_application]
level=INFO
handlers=console,file
qualname=application

[handler_console]
class=StreamHandler
level=INFO
formatter=default
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=INFO
formatter=default
args=('app.log',)

[formatter_default]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### External Monitoring
For production monitoring, consider:
- **Sentry**: Error tracking and performance monitoring
- **DataDog**: Application performance monitoring
- **New Relic**: Full-stack observability
- **Prometheus + Grafana**: Open-source monitoring stack

## Troubleshooting

### Common Deployment Issues

#### Backend Won't Start
```bash
# Check logs
docker-compose logs backend

# Common fixes
1. Verify environment variables
2. Check port availability
3. Ensure Redis is accessible
4. Validate Python dependencies
```

#### Frontend Build Fails
```bash
# Check build logs
npm run build

# Common fixes
1. Verify Node.js version (18+)
2. Clear node_modules and reinstall
3. Check environment variables
4. Verify API URL accessibility
```

#### Redis Connection Issues
```bash
# Test Redis connectivity
redis-cli -h your-redis-host -p 6379 ping

# Common fixes
1. Check Redis server status
2. Verify connection string
3. Check firewall rules
4. Validate authentication
```

#### CORS Errors
```bash
# Update ALLOWED_ORIGINS in backend
export ALLOWED_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"

# Restart backend service
docker-compose restart backend
```

### Health Checks
```bash
# Backend health check
curl https://your-api-domain.com/health/

# Frontend accessibility
curl https://your-frontend-domain.com/

# Full system check
curl https://your-api-domain.com/health/detailed
```

### Performance Optimization
```bash
# Backend optimization
1. Increase worker count: --workers 8
2. Enable Redis caching
3. Optimize data file sizes
4. Use production ASGI server

# Frontend optimization
1. Enable gzip compression
2. Configure CDN
3. Optimize bundle size
4. Enable caching headers
```

### Backup & Recovery
```bash
# Backup Redis data
redis-cli --rdb /backup/dump.rdb

# Backup application data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/

# Recovery process
1. Restore Redis data
2. Restore application files
3. Restart services
4. Verify functionality
```