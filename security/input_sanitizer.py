"""
Input sanitization and validation for API endpoints.

This module provides comprehensive input sanitization to prevent
injection attacks and ensure data integrity.
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import unicodedata

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass


class InputSanitizer:
    """
    Comprehensive input sanitizer for API endpoints.
    
    Provides sanitization for various input types including
    filenames, job IDs, and general text inputs.
    """
    
    # Regex patterns for validation
    PATTERNS = {
        'job_id': re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'),
        'filename': re.compile(r'^[a-zA-Z0-9._\-\s()]+\.[a-zA-Z0-9]{2,5}$'),
        'safe_string': re.compile(r'^[a-zA-Z0-9._\-\s]+$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
    }
    
    # Dangerous characters and patterns
    DANGEROUS_CHARS = {
        'path_traversal': ['../', '..\\', '.\\', './'],
        'script_injection': ['<script', '</script', 'javascript:', 'vbscript:', 'onload=', 'onerror='],
        'sql_injection': ["'", '"', ';', '--', '/*', '*/', 'union', 'select', 'drop', 'delete'],
        'command_injection': ['|', '&', ';', '`', '$', '(', ')', '{', '}', '[', ']'],
    }
    
    # Maximum lengths for different input types
    MAX_LENGTHS = {
        'filename': 255,
        'job_id': 36,
        'general_string': 1000,
        'description': 5000,
    }
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'sanitizations_applied': 0
        }
    
    def sanitize_filename(self, filename: str, strict: bool = True) -> str:
        """
        Sanitize and validate filename.
        
        Args:
            filename: Original filename
            strict: Whether to apply strict validation
            
        Returns:
            Sanitized filename
            
        Raises:
            InputValidationError: If filename is invalid
        """
        self.validation_stats['total_validations'] += 1
        
        if not filename or not isinstance(filename, str):
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Filename cannot be empty")
        
        # Remove null bytes and control characters
        sanitized = self._remove_control_chars(filename)
        
        # Normalize unicode
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Check length
        if len(sanitized) > self.MAX_LENGTHS['filename']:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError(f"Filename too long (max {self.MAX_LENGTHS['filename']} characters)")
        
        # Check for path traversal attempts
        if self._contains_dangerous_patterns(sanitized, 'path_traversal'):
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Filename contains path traversal patterns")
        
        # Remove or replace dangerous characters
        if strict:
            # Strict mode: only allow safe characters
            if not self.PATTERNS['filename'].match(sanitized):
                self.validation_stats['failed_validations'] += 1
                raise InputValidationError("Filename contains invalid characters")
        else:
            # Permissive mode: sanitize dangerous characters
            sanitized = self._sanitize_dangerous_chars(sanitized)
            self.validation_stats['sanitizations_applied'] += 1
        
        # Ensure filename has valid extension
        path = Path(sanitized)
        if not path.suffix:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Filename must have an extension")
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        if path.stem.upper() in reserved_names:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError(f"Filename uses reserved name: {path.stem}")
        
        logger.debug(f"Sanitized filename: {filename} -> {sanitized}")
        return sanitized
    
    def validate_job_id(self, job_id: str) -> str:
        """
        Validate job ID format.
        
        Args:
            job_id: Job ID to validate
            
        Returns:
            Validated job ID
            
        Raises:
            InputValidationError: If job ID is invalid
        """
        self.validation_stats['total_validations'] += 1
        
        if not job_id or not isinstance(job_id, str):
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Job ID cannot be empty")
        
        # Remove whitespace
        job_id = job_id.strip()
        
        # Check format (UUID)
        if not self.PATTERNS['job_id'].match(job_id):
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Invalid job ID format")
        
        return job_id
    
    def sanitize_string(self, text: str, max_length: Optional[int] = None, allow_html: bool = False) -> str:
        """
        Sanitize general string input.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML (will be escaped)
            
        Returns:
            Sanitized text
            
        Raises:
            InputValidationError: If text is invalid
        """
        self.validation_stats['total_validations'] += 1
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove null bytes and control characters
        sanitized = self._remove_control_chars(text)
        
        # Normalize unicode
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Check length
        max_len = max_length or self.MAX_LENGTHS['general_string']
        if len(sanitized) > max_len:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError(f"Text too long (max {max_len} characters)")
        
        # Handle HTML
        if not allow_html:
            # Check for script injection attempts
            if self._contains_dangerous_patterns(sanitized.lower(), 'script_injection'):
                self.validation_stats['failed_validations'] += 1
                raise InputValidationError("Text contains potentially dangerous script content")
            
            # Escape HTML entities
            sanitized = html.escape(sanitized)
            self.validation_stats['sanitizations_applied'] += 1
        
        # Check for SQL injection patterns
        if self._contains_dangerous_patterns(sanitized.lower(), 'sql_injection'):
            logger.warning(f"Potential SQL injection attempt detected: {sanitized[:100]}")
            # Don't fail, but log the attempt
        
        return sanitized
    
    def sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize HTTP headers.
        
        Args:
            headers: Dictionary of headers
            
        Returns:
            Sanitized headers dictionary
        """
        sanitized_headers = {}
        
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            
            # Sanitize header name
            clean_key = self._remove_control_chars(key).strip()
            if not clean_key or len(clean_key) > 100:
                continue
            
            # Sanitize header value
            clean_value = self._remove_control_chars(value).strip()
            if len(clean_value) > 1000:
                clean_value = clean_value[:1000]
            
            # Check for injection attempts in headers
            if (self._contains_dangerous_patterns(clean_value.lower(), 'script_injection') or
                self._contains_dangerous_patterns(clean_value.lower(), 'command_injection')):
                logger.warning(f"Dangerous pattern in header {clean_key}: {clean_value[:100]}")
                continue
            
            sanitized_headers[clean_key] = clean_value
        
        return sanitized_headers
    
    def validate_file_size(self, size: int, max_size: int) -> bool:
        """
        Validate file size.
        
        Args:
            size: File size in bytes
            max_size: Maximum allowed size in bytes
            
        Returns:
            True if size is valid
            
        Raises:
            InputValidationError: If size is invalid
        """
        self.validation_stats['total_validations'] += 1
        
        if not isinstance(size, int) or size < 0:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("Invalid file size")
        
        if size > max_size:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError(f"File size {size} exceeds maximum {max_size}")
        
        if size == 0:
            self.validation_stats['failed_validations'] += 1
            raise InputValidationError("File cannot be empty")
        
        return True
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters from text."""
        # Remove null bytes and other control characters except newlines and tabs
        return ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    def _contains_dangerous_patterns(self, text: str, pattern_type: str) -> bool:
        """Check if text contains dangerous patterns."""
        patterns = self.DANGEROUS_CHARS.get(pattern_type, [])
        text_lower = text.lower()
        
        for pattern in patterns:
            if pattern.lower() in text_lower:
                return True
        
        return False
    
    def _sanitize_dangerous_chars(self, text: str) -> str:
        """Replace dangerous characters with safe alternatives."""
        # Replace path traversal patterns
        for pattern in self.DANGEROUS_CHARS['path_traversal']:
            text = text.replace(pattern, '_')
        
        # Replace command injection characters
        for char in self.DANGEROUS_CHARS['command_injection']:
            text = text.replace(char, '_')
        
        return text
    
    def validate_ip_address(self, ip: str) -> bool:
        """
        Validate IP address format.
        
        Args:
            ip: IP address string
            
        Returns:
            True if valid IP address
        """
        try:
            import ipaddress
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.validation_stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'sanitizations_applied': 0
        }


# Global sanitizer instance
_sanitizer_instance: Optional[InputSanitizer] = None


def get_input_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = InputSanitizer()
    return _sanitizer_instance