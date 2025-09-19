"""
Secure token system for download links with expiration.

This module provides secure, time-limited tokens for file downloads
to prevent unauthorized access and ensure privacy.
"""

import hmac
import hashlib
import base64
import time
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Raised when token validation fails."""
    pass


class SecureTokenManager:
    """
    Manages secure tokens for file downloads with expiration.
    
    Uses HMAC-SHA256 for token generation and validation
    to ensure tokens cannot be forged.
    """
    
    def __init__(self, secret_key: str = None):
        """
        Initialize token manager.
        
        Args:
            secret_key: Secret key for HMAC signing (auto-generated if None)
        """
        if secret_key is None:
            # Generate a random secret key
            import secrets
            secret_key = secrets.token_hex(32)
            logger.warning("Using auto-generated secret key. Set a fixed key for production.")
        
        self.secret_key = secret_key.encode('utf-8')
        
        # Default token settings
        self.default_expiry_hours = 24
        self.max_expiry_hours = 168  # 7 days
    
    def generate_download_token(
        self, 
        job_id: str, 
        file_path: str,
        expiry_hours: Optional[int] = None,
        additional_data: Optional[Dict] = None
    ) -> str:
        """
        Generate a secure download token.
        
        Args:
            job_id: Processing job ID
            file_path: Path to the file
            expiry_hours: Token expiry in hours (default: 24)
            additional_data: Additional data to include in token
            
        Returns:
            Secure token string
        """
        if expiry_hours is None:
            expiry_hours = self.default_expiry_hours
        
        # Limit maximum expiry
        expiry_hours = min(expiry_hours, self.max_expiry_hours)
        
        # Calculate expiry timestamp
        expiry_time = time.time() + (expiry_hours * 3600)
        
        # Create token payload
        payload = {
            'job_id': job_id,
            'file_path': file_path,
            'expiry': expiry_time,
            'issued_at': time.time(),
            'version': '1.0'
        }
        
        # Add additional data if provided
        if additional_data:
            payload['additional'] = additional_data
        
        # Serialize payload
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode('utf-8')).decode('utf-8')
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key,
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode('utf-8')
        
        # Combine payload and signature
        token = f"{payload_b64}.{signature_b64}"
        
        logger.info(f"Generated download token for job {job_id}, expires in {expiry_hours} hours")
        
        return token
    
    def validate_token(self, token: str) -> Dict[str, any]:
        """
        Validate and decode a download token.
        
        Args:
            token: Token to validate
            
        Returns:
            Decoded token payload
            
        Raises:
            TokenValidationError: If token is invalid or expired
        """
        try:
            # Split token into payload and signature
            if '.' not in token:
                raise TokenValidationError("Invalid token format")
            
            payload_b64, signature_b64 = token.rsplit('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key,
                payload_b64.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            provided_signature = base64.urlsafe_b64decode(signature_b64.encode('utf-8'))
            
            if not hmac.compare_digest(expected_signature, provided_signature):
                raise TokenValidationError("Invalid token signature")
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64.encode('utf-8')).decode('utf-8')
            payload = json.loads(payload_json)
            
            # Validate required fields
            required_fields = ['job_id', 'file_path', 'expiry', 'issued_at']
            for field in required_fields:
                if field not in payload:
                    raise TokenValidationError(f"Missing required field: {field}")
            
            # Check expiry
            current_time = time.time()
            if payload['expiry'] < current_time:
                expiry_date = datetime.fromtimestamp(payload['expiry'])
                raise TokenValidationError(f"Token expired at {expiry_date}")
            
            # Check if token was issued in the future (clock skew protection)
            if payload['issued_at'] > current_time + 300:  # 5 minute tolerance
                raise TokenValidationError("Token issued in the future")
            
            logger.info(f"Successfully validated token for job {payload['job_id']}")
            
            return payload
            
        except json.JSONDecodeError as e:
            raise TokenValidationError(f"Invalid token payload: {e}")
        except Exception as e:
            if isinstance(e, TokenValidationError):
                raise
            raise TokenValidationError(f"Token validation failed: {e}")
    
    def generate_secure_download_url(
        self, 
        base_url: str,
        job_id: str,
        file_path: str,
        expiry_hours: Optional[int] = None
    ) -> str:
        """
        Generate a secure download URL with embedded token.
        
        Args:
            base_url: Base URL for downloads
            job_id: Processing job ID
            file_path: Path to the file
            expiry_hours: Token expiry in hours
            
        Returns:
            Secure download URL
        """
        token = self.generate_download_token(job_id, file_path, expiry_hours)
        
        # URL-encode the token
        encoded_token = quote(token, safe='')
        
        # Construct secure URL
        secure_url = f"{base_url.rstrip('/')}/secure-download?token={encoded_token}"
        
        return secure_url
    
    def extract_token_from_url(self, url: str) -> Optional[str]:
        """
        Extract token from a secure download URL.
        
        Args:
            url: URL containing the token
            
        Returns:
            Extracted token or None if not found
        """
        try:
            from urllib.parse import urlparse, parse_qs
            
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            if 'token' in query_params:
                encoded_token = query_params['token'][0]
                return unquote(encoded_token)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract token from URL: {e}")
            return None
    
    def get_token_info(self, token: str) -> Dict[str, any]:
        """
        Get information about a token without full validation.
        
        Args:
            token: Token to analyze
            
        Returns:
            Token information dictionary
        """
        try:
            payload = self.validate_token(token)
            
            current_time = time.time()
            time_to_expiry = payload['expiry'] - current_time
            
            return {
                'valid': True,
                'job_id': payload['job_id'],
                'file_path': payload['file_path'],
                'issued_at': datetime.fromtimestamp(payload['issued_at']),
                'expires_at': datetime.fromtimestamp(payload['expiry']),
                'time_to_expiry_seconds': max(0, time_to_expiry),
                'time_to_expiry_hours': max(0, time_to_expiry / 3600),
                'additional_data': payload.get('additional', {})
            }
            
        except TokenValidationError as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def revoke_tokens_for_job(self, job_id: str):
        """
        Revoke all tokens for a specific job.
        
        Note: This is a placeholder for token revocation.
        In a production system, you'd maintain a revocation list.
        
        Args:
            job_id: Job ID to revoke tokens for
        """
        # In a real implementation, you'd store revoked tokens in a database
        # or cache (Redis) and check against this list during validation
        logger.info(f"Token revocation requested for job {job_id}")
        # TODO: Implement token revocation storage
    
    def cleanup_expired_revocations(self):
        """Clean up expired token revocations."""
        # Placeholder for cleaning up expired revocation entries
        logger.info("Cleaning up expired token revocations")
        # TODO: Implement revocation cleanup


# Global token manager instance
_token_manager_instance: Optional[SecureTokenManager] = None


def get_token_manager(secret_key: str = None) -> SecureTokenManager:
    """Get global token manager instance."""
    global _token_manager_instance
    if _token_manager_instance is None:
        _token_manager_instance = SecureTokenManager(secret_key)
    return _token_manager_instance