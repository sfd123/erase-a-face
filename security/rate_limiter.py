"""
Rate limiting functionality for API endpoints.

This module provides rate limiting based on IP addresses and API keys
to prevent abuse and ensure fair usage of the service.
"""

import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Implements rate limiting using token bucket algorithm
    with sliding window for more accurate rate limiting.
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        # Storage for rate limit data: {identifier: {'tokens': int, 'last_refill': float, 'requests': deque}}
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': 0,
            'last_refill': time.time(),
            'requests': deque()
        })
        
        # Rate limit configurations
        self.rate_limits = {
            'upload': {
                'requests_per_minute': 5,
                'requests_per_hour': 20,
                'burst_size': 10
            },
            'status': {
                'requests_per_minute': 30,
                'requests_per_hour': 500,
                'burst_size': 50
            },
            'download': {
                'requests_per_minute': 10,
                'requests_per_hour': 100,
                'burst_size': 20
            },
            'default': {
                'requests_per_minute': 20,
                'requests_per_hour': 200,
                'burst_size': 30
            }
        }
    
    def check_rate_limit(self, identifier: str, endpoint_type: str = 'default') -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (IP address, API key, etc.)
            endpoint_type: Type of endpoint (upload, status, download, default)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        config = self.rate_limits.get(endpoint_type, self.rate_limits['default'])
        
        bucket = self.buckets[identifier]
        
        # Refill tokens based on time passed
        self._refill_tokens(bucket, config, current_time)
        
        # Check sliding window limits
        window_check = self._check_sliding_window(bucket, config, current_time)
        
        if not window_check['allowed']:
            return False, {
                'allowed': False,
                'reason': 'sliding_window_exceeded',
                'retry_after': window_check['retry_after'],
                'limits': config,
                'current_usage': window_check['usage']
            }
        
        # Check token bucket
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            bucket['requests'].append(current_time)
            
            return True, {
                'allowed': True,
                'remaining_tokens': bucket['tokens'],
                'limits': config,
                'current_usage': window_check['usage']
            }
        else:
            # Calculate retry after time
            retry_after = 60 / config['requests_per_minute']
            
            return False, {
                'allowed': False,
                'reason': 'token_bucket_empty',
                'retry_after': retry_after,
                'limits': config,
                'current_usage': window_check['usage']
            }
    
    def _refill_tokens(self, bucket: Dict, config: Dict, current_time: float):
        """Refill tokens in the bucket based on elapsed time."""
        # Initialize bucket if it's new
        if bucket['tokens'] == 0 and not bucket['requests']:
            bucket['tokens'] = config['burst_size']
            bucket['last_refill'] = current_time
            return
        
        time_passed = current_time - bucket['last_refill']
        
        # Calculate tokens to add (based on requests per minute)
        tokens_to_add = time_passed * (config['requests_per_minute'] / 60.0)
        
        # Add tokens but don't exceed burst size
        bucket['tokens'] = min(
            config['burst_size'],
            bucket['tokens'] + tokens_to_add
        )
        
        bucket['last_refill'] = current_time
    
    def _check_sliding_window(self, bucket: Dict, config: Dict, current_time: float) -> Dict[str, any]:
        """Check sliding window rate limits."""
        requests = bucket['requests']
        
        # Remove old requests outside the windows
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        # Clean up old requests
        while requests and requests[0] < hour_ago:
            requests.popleft()
        
        # Count requests in different windows
        requests_last_minute = sum(1 for req_time in requests if req_time >= minute_ago)
        requests_last_hour = len(requests)
        
        # Check limits
        if requests_last_minute >= config['requests_per_minute']:
            return {
                'allowed': False,
                'retry_after': 60 - (current_time - minute_ago),
                'usage': {
                    'requests_last_minute': requests_last_minute,
                    'requests_last_hour': requests_last_hour
                }
            }
        
        if requests_last_hour >= config['requests_per_hour']:
            # Find the oldest request in the hour and calculate retry time
            oldest_request = min(requests) if requests else current_time
            retry_after = 3600 - (current_time - oldest_request)
            
            return {
                'allowed': False,
                'retry_after': retry_after,
                'usage': {
                    'requests_last_minute': requests_last_minute,
                    'requests_last_hour': requests_last_hour
                }
            }
        
        return {
            'allowed': True,
            'usage': {
                'requests_last_minute': requests_last_minute,
                'requests_last_hour': requests_last_hour
            }
        }
    
    def get_client_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Unique identifier string
        """
        # Try to get real IP from headers (for reverse proxy setups)
        real_ip = (
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or
            request.headers.get('X-Real-IP', '') or
            request.client.host if request.client else 'unknown'
        )
        
        # You could also use API keys here if implemented
        # api_key = request.headers.get('X-API-Key')
        # if api_key:
        #     return f"api_key:{api_key}"
        
        return f"ip:{real_ip}"
    
    def cleanup_old_entries(self, max_age_hours: int = 24):
        """Clean up old rate limit entries."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # Find entries to remove
        to_remove = []
        for identifier, bucket in self.buckets.items():
            if bucket['last_refill'] < cutoff_time and not bucket['requests']:
                to_remove.append(identifier)
        
        # Remove old entries
        for identifier in to_remove:
            del self.buckets[identifier]
        
        logger.info(f"Cleaned up {len(to_remove)} old rate limit entries")
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        
        active_clients = 0
        total_requests_last_hour = 0
        
        for bucket in self.buckets.values():
            if bucket['requests']:
                active_clients += 1
                # Count requests in last hour
                hour_ago = current_time - 3600
                total_requests_last_hour += sum(
                    1 for req_time in bucket['requests'] if req_time >= hour_ago
                )
        
        return {
            'total_clients': len(self.buckets),
            'active_clients': active_clients,
            'total_requests_last_hour': total_requests_last_hour,
            'rate_limits': self.rate_limits
        }


# Global rate limiter instance
_rate_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter()
    return _rate_limiter_instance


def rate_limit_middleware(endpoint_type: str = 'default'):
    """
    Rate limiting middleware decorator.
    
    Args:
        endpoint_type: Type of endpoint for rate limiting
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)
            
            rate_limiter = get_rate_limiter()
            client_id = rate_limiter.get_client_identifier(request)
            
            allowed, info = rate_limiter.check_rate_limit(client_id, endpoint_type)
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_id}: {info}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        'error': 'rate_limit_exceeded',
                        'message': f'Rate limit exceeded: {info["reason"]}',
                        'retry_after': info['retry_after'],
                        'limits': info['limits']
                    },
                    headers={'Retry-After': str(int(info['retry_after']))}
                )
            
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit-Minute'] = str(info['limits']['requests_per_minute'])
                response.headers['X-RateLimit-Limit-Hour'] = str(info['limits']['requests_per_hour'])
                response.headers['X-RateLimit-Remaining'] = str(info.get('remaining_tokens', 0))
                response.headers['X-RateLimit-Usage-Minute'] = str(info['current_usage']['requests_last_minute'])
                response.headers['X-RateLimit-Usage-Hour'] = str(info['current_usage']['requests_last_hour'])
            
            return response
        
        return wrapper
    return decorator