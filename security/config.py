"""
Security configuration for Golf Video Anonymizer.

This module contains security-related configuration settings
and constants used throughout the application.
"""

import os
from typing import List, Dict, Any

# Security settings
SECURITY_CONFIG = {
    # Token settings
    'token_secret_key': os.getenv('TOKEN_SECRET_KEY', None),  # Should be set in production
    'default_token_expiry_hours': int(os.getenv('DEFAULT_TOKEN_EXPIRY_HOURS', '24')),
    'max_token_expiry_hours': int(os.getenv('MAX_TOKEN_EXPIRY_HOURS', '168')),  # 7 days
    
    # Rate limiting settings
    'rate_limiting_enabled': os.getenv('RATE_LIMITING_ENABLED', 'true').lower() == 'true',
    'rate_limit_cleanup_interval_hours': int(os.getenv('RATE_LIMIT_CLEANUP_HOURS', '24')),
    
    # Malware scanning settings
    'malware_scanning_enabled': os.getenv('MALWARE_SCANNING_ENABLED', 'true').lower() == 'true',
    'malware_scan_cache_size': int(os.getenv('MALWARE_SCAN_CACHE_SIZE', '1000')),
    'max_entropy_threshold': float(os.getenv('MAX_ENTROPY_THRESHOLD', '7.5')),
    
    # File cleanup settings
    'auto_cleanup_enabled': os.getenv('AUTO_CLEANUP_ENABLED', 'true').lower() == 'true',
    'cleanup_delay_hours': int(os.getenv('CLEANUP_DELAY_HOURS', '48')),
    'temp_file_cleanup_hours': int(os.getenv('TEMP_FILE_CLEANUP_HOURS', '24')),
    
    # Input validation settings
    'strict_filename_validation': os.getenv('STRICT_FILENAME_VALIDATION', 'true').lower() == 'true',
    'max_filename_length': int(os.getenv('MAX_FILENAME_LENGTH', '255')),
    'max_string_length': int(os.getenv('MAX_STRING_LENGTH', '1000')),
    
    # Security headers
    'security_headers_enabled': os.getenv('SECURITY_HEADERS_ENABLED', 'true').lower() == 'true',
    'trusted_hosts': os.getenv('TRUSTED_HOSTS', 'localhost,127.0.0.1').split(','),
    'allowed_origins': os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:8000').split(','),
}

# Rate limiting configurations per endpoint type
RATE_LIMIT_CONFIG = {
    'upload': {
        'requests_per_minute': int(os.getenv('UPLOAD_RATE_LIMIT_MINUTE', '5')),
        'requests_per_hour': int(os.getenv('UPLOAD_RATE_LIMIT_HOUR', '20')),
        'burst_size': int(os.getenv('UPLOAD_BURST_SIZE', '10'))
    },
    'status': {
        'requests_per_minute': int(os.getenv('STATUS_RATE_LIMIT_MINUTE', '30')),
        'requests_per_hour': int(os.getenv('STATUS_RATE_LIMIT_HOUR', '500')),
        'burst_size': int(os.getenv('STATUS_BURST_SIZE', '50'))
    },
    'download': {
        'requests_per_minute': int(os.getenv('DOWNLOAD_RATE_LIMIT_MINUTE', '10')),
        'requests_per_hour': int(os.getenv('DOWNLOAD_RATE_LIMIT_HOUR', '100')),
        'burst_size': int(os.getenv('DOWNLOAD_BURST_SIZE', '20'))
    },
    'default': {
        'requests_per_minute': int(os.getenv('DEFAULT_RATE_LIMIT_MINUTE', '20')),
        'requests_per_hour': int(os.getenv('DEFAULT_RATE_LIMIT_HOUR', '200')),
        'burst_size': int(os.getenv('DEFAULT_BURST_SIZE', '30'))
    }
}

# Security headers configuration
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
}

# Malware detection patterns (can be extended)
ADDITIONAL_MALWARE_SIGNATURES = {
    # Add more signatures as needed
    b'\x50\x4b\x03\x04': 'ZIP_ARCHIVE',  # ZIP file (could contain malware)
    b'\x52\x61\x72\x21': 'RAR_ARCHIVE',  # RAR file
    b'\x1f\x8b\x08': 'GZIP_ARCHIVE',     # GZIP file
}

# Suspicious file extensions that should be blocked
BLOCKED_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
    '.app', '.deb', '.pkg', '.dmg', '.iso', '.msi', '.zip', '.rar', '.7z'
}

# Content-Type validation
ALLOWED_CONTENT_TYPES = {
    'video/mp4',
    'video/quicktime',
    'video/x-msvideo',
    'video/avi',
    'application/octet-stream'  # Generic binary
}

def get_security_config() -> Dict[str, Any]:
    """Get security configuration."""
    return SECURITY_CONFIG.copy()

def get_rate_limit_config() -> Dict[str, Dict[str, int]]:
    """Get rate limiting configuration."""
    return RATE_LIMIT_CONFIG.copy()

def get_security_headers() -> Dict[str, str]:
    """Get security headers configuration."""
    return SECURITY_HEADERS.copy()

def is_security_feature_enabled(feature: str) -> bool:
    """Check if a security feature is enabled."""
    return SECURITY_CONFIG.get(f'{feature}_enabled', True)

def get_trusted_hosts() -> List[str]:
    """Get list of trusted hosts."""
    return SECURITY_CONFIG['trusted_hosts']

def get_allowed_origins() -> List[str]:
    """Get list of allowed CORS origins."""
    return SECURITY_CONFIG['allowed_origins']