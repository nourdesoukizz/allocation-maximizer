"""
Security middleware for Module 4 API
"""

import time
import logging
import uuid
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import re
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)

class SecurityMiddleware:
    """Security middleware for API protection"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
    def add_api_key(self, key: str, name: str, permissions: list = None):
        """Add an API key with optional permissions"""
        self.api_keys[key] = {
            "name": name,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.now(timezone.utc),
            "last_used": None,
            "usage_count": 0
        }
        
    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info"""
        if key in self.api_keys:
            key_info = self.api_keys[key]
            key_info["last_used"] = datetime.now(timezone.utc)
            key_info["usage_count"] += 1
            return key_info
        return None
    
    def sanitize_log_data(self, data: Any) -> Any:
        """Sanitize sensitive data from logs"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret', 'auth']):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, (dict, list)):
                    sanitized[key] = self.sanitize_log_data(value)
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(data, list):
            return [self.sanitize_log_data(item) for item in data]
        else:
            return data
    
    def validate_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data"""
        errors = []
        
        # Check for common injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # XSS
            r'on\w+\s*=',  # Event handlers
            r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL injection
            r'(\.|;|\||&|\$|\`)',  # Command injection
        ]
        
        def check_string_value(value: str, field_name: str):
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Potentially dangerous content detected in field '{field_name}'")
                        return False
                        
                # Length validation
                if len(value) > 10000:  # Max 10KB per string field
                    errors.append(f"Field '{field_name}' exceeds maximum length")
                    return False
            return True
        
        def validate_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, str):
                        check_string_value(value, current_path)
                    elif isinstance(value, (dict, list)):
                        validate_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    if isinstance(item, str):
                        check_string_value(item, current_path)
                    elif isinstance(item, (dict, list)):
                        validate_recursive(item, current_path)
        
        validate_recursive(data)
        
        if errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input validation failed",
                    "validation_errors": errors
                }
            )
        
        return data

# Global security middleware instance
security_middleware = SecurityMiddleware()

# Optional API key security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = None):
    """Optional API key authentication"""
    if not credentials:
        return None  # Allow anonymous access
        
    key_info = security_middleware.validate_api_key(credentials.credentials)
    if not key_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return key_info

class RequestLoggingMiddleware:
    """Middleware for request/response logging and metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        self.request_count += 1
        client_ip = get_remote_address(request)
        
        # Sanitize request data for logging
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", ""),
        }
        
        sanitized_request = security_middleware.sanitize_log_data(request_data)
        
        logger.info(f"Request {request_id}: {request.method} {request.url.path}", extra={
            "request_id": request_id,
            "request_data": sanitized_request
        })
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            self.response_times.append(process_time)
            
            # Add security headers
            for header, value in security_middleware.security_headers.items():
                response.headers[header] = value
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{process_time:.3f}s"
            
            # Log response
            logger.info(f"Response {request_id}: {response.status_code}", extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": process_time
            })
            
            return response
            
        except Exception as e:
            self.error_count += 1
            process_time = time.time() - start_time
            
            logger.error(f"Error {request_id}: {str(e)}", extra={
                "request_id": request_id,
                "error": str(e),
                "response_time": process_time
            })
            
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            "average_response_time": round(avg_response_time, 3),
            "recent_response_times": self.response_times[-100:],  # Last 100 requests
        }

# Global request logging middleware instance
request_logging_middleware = RequestLoggingMiddleware()

def setup_rate_limiting(app):
    """Setup rate limiting for the FastAPI app"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
def get_rate_limiter():
    """Get the rate limiter instance for use in route decorators"""
    return limiter