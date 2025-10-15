# Railway Deployment Guide

## Overview
This project uses a multi-stage Docker build to deploy both the React frontend and FastAPI backend together.

## Configuration

### Required Environment Variables
Set these in your Railway project settings:

```bash
PORT=8000                    # Railway will set this automatically
ENVIRONMENT=production       # Set environment to production
```

### Optional Environment Variables
```bash
CORS_ORIGINS=*              # Allow all origins (default in production)
MODULE4_DEBUG=false          # Disable debug mode in production
MODULE4_LOG_LEVEL=INFO      # Set logging level
REDIS_URL=                  # Optional: Redis URL if using external Redis
```

## Health Check
The app exposes a health check endpoint at `/health` which Railway uses to verify the deployment.

- **Health Check Path**: `/health`
- **Expected Response**: `{"status": "healthy"}`

## Build Process
1. **Frontend Build**: Builds React app with Vite
2. **Backend Setup**: Installs Python dependencies
3. **Combined Deploy**: Serves both frontend and API from single container

## API Endpoints
- **Frontend**: `https://your-app.railway.app/`
- **API**: `https://your-app.railway.app/api/`
- **Health Check**: `https://your-app.railway.app/health`
- **API Docs**: `https://your-app.railway.app/api/docs`

## Troubleshooting

### Deployment Fails with Health Check Timeout
- Check Railway logs for startup errors
- Verify the app is binding to `0.0.0.0:$PORT`
- Ensure all Python dependencies are in `requirements.txt`

### Import Errors
- Check that PYTHONPATH is set correctly (done in Dockerfile)
- Verify backend directory structure is intact

### Frontend Not Loading
- Ensure `frontend/dist` was built successfully
- Check that static files are being served

## Logs
View logs in Railway dashboard to debug issues:
```bash
# Look for these log messages:
# "Backend path: /app/backend/api/module4"
# "Successfully imported backend app"
# "Uvicorn running on http://0.0.0.0:8000"
```

## Manual Build Test (Local)
To test the Docker build locally:
```bash
# Build the image
docker build -t allocation-maximizer .

# Run the container
docker run -p 8000:8000 -e PORT=8000 allocation-maximizer

# Test health check
curl http://localhost:8000/health
```
