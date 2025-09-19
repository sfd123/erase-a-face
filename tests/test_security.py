"""
Comprehensive security tests for the Golf Video Anonymizer.

Tests malware scanning, rate limiting, secure tokens, input sanitization,
and other security features.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request

from security.malware_scanner import MalwareScanner, MalwareDetectionError
from security.rate_limiter import RateLimiter, RateLimitExceeded
from security.secure_tokens import SecureTokenManager, TokenValidationError
from security.input_sanitizer import InputSanitizer, InputValidationError


class TestMalwareScanner:
    """Test malware scanning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = MalwareScanner()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scan_clean_video_file(self):
        """Test scanning a clean video file."""
        # Create a mock video file with MP4 header
        test_file = self.temp_dir / "clean_video.mp4"
        with open(test_file, 'wb') as f:
            # Write MP4 magic number
            f.write(b'\x00\x00\x00\x18ftypmp4')
            # Add some random video-like data
            f.write(b'A' * 1000)
        
        result = self.scanner.scan_file(test_file)
        
        assert result['is_safe'] is True
        assert len(result['threats_detected']) == 0
        assert 'signatures' in result['scan_details']
        assert 'entropy' in result['scan_details']
    
    def test_scan_malicious_executable(self):
        """Test scanning a file with malicious signatures."""
        # Create a file with PE executable header
        test_file = self.temp_dir / "malicious.mp4"
        with open(test_file, 'wb') as f:
            # Write PE header (malicious signature)
            f.write(b'\x4d\x5a\x90\x00')
            f.write(b'A' * 1000)
        
        result = self.scanner.scan_file(test_file)
        
        assert result['is_safe'] is False
        assert len(result['threats_detected']) > 0
        assert any('PE_EXECUTABLE' in threat['threat'] for threat in result['threats_detected'])
    
    def test_scan_suspicious_strings(self):
        """Test detection of suspicious strings."""
        test_file = self.temp_dir / "suspicious.mp4"
        with open(test_file, 'wb') as f:
            # Write MP4 header first
            f.write(b'\x00\x00\x00\x18ftypmp4')
            # Add suspicious content
            f.write(b'some video data cmd.exe more data')
        
        result = self.scanner.scan_file(test_file)
        
        assert result['is_safe'] is False
        assert any('cmd.exe' in threat['threat'] for threat in result['threats_detected'])
    
    def test_high_entropy_detection(self):
        """Test detection of high entropy (encrypted/packed) content."""
        test_file = self.temp_dir / "high_entropy.mp4"
        with open(test_file, 'wb') as f:
            # Write MP4 header
            f.write(b'\x00\x00\x00\x18ftypmp4')
            # Add high entropy data (random bytes)
            import os
            f.write(os.urandom(10000))
        
        result = self.scanner.scan_file(test_file)
        
        # High entropy should trigger warnings but not necessarily mark as unsafe
        assert len(result['warnings']) > 0
        entropy = result['scan_details']['entropy']['entropy']
        assert entropy > 7.0  # Should be high entropy
    
    def test_scan_nonexistent_file(self):
        """Test scanning a non-existent file."""
        nonexistent_file = self.temp_dir / "nonexistent.mp4"
        
        with pytest.raises(MalwareDetectionError):
            self.scanner.scan_file(nonexistent_file)
    
    def test_cache_functionality(self):
        """Test scan result caching."""
        test_file = self.temp_dir / "cache_test.mp4"
        with open(test_file, 'wb') as f:
            f.write(b'\x00\x00\x00\x18ftypmp4')
            f.write(b'A' * 1000)
        
        # First scan
        result1 = self.scanner.scan_file(test_file)
        
        # Second scan should use cache
        result2 = self.scanner.scan_file(test_file)
        
        assert result1['file_hash'] == result2['file_hash']
        assert result1['is_safe'] == result2['is_safe']
        
        # Check cache stats
        stats = self.scanner.get_cache_stats()
        assert stats['cached_results'] >= 1


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()
        # Use smaller limits for testing
        self.rate_limiter.rate_limits['test'] = {
            'requests_per_minute': 3,
            'requests_per_hour': 10,
            'burst_size': 5
        }
    
    def test_rate_limit_within_limits(self):
        """Test requests within rate limits."""
        client_id = "test_client_1"
        
        # Should allow requests within limits
        for i in range(3):
            allowed, info = self.rate_limiter.check_rate_limit(client_id, 'test')
            assert allowed is True
            assert info['allowed'] is True
    
    def test_rate_limit_exceeded_per_minute(self):
        """Test rate limit exceeded per minute."""
        client_id = "test_client_2"
        
        # Use up the per-minute limit
        for i in range(3):
            allowed, info = self.rate_limiter.check_rate_limit(client_id, 'test')
            assert allowed is True
        
        # Next request should be denied
        allowed, info = self.rate_limiter.check_rate_limit(client_id, 'test')
        assert allowed is False
        assert 'retry_after' in info
    
    def test_rate_limit_token_refill(self):
        """Test token bucket refill over time."""
        client_id = "test_client_3"
        
        # Use up tokens
        for i in range(3):
            self.rate_limiter.check_rate_limit(client_id, 'test')
        
        # Should be denied
        allowed, info = self.rate_limiter.check_rate_limit(client_id, 'test')
        assert allowed is False
        
        # Manually advance time and refill tokens
        bucket = self.rate_limiter.buckets[client_id]
        bucket['last_refill'] = time.time() - 60  # 1 minute ago
        # Also clear the sliding window to simulate time passing
        bucket['requests'].clear()
        
        # Should be allowed again
        allowed, info = self.rate_limiter.check_rate_limit(client_id, 'test')
        assert allowed is True
    
    def test_different_clients_separate_limits(self):
        """Test that different clients have separate rate limits."""
        client1 = "test_client_4"
        client2 = "test_client_5"
        
        # Use up client1's limit
        for i in range(3):
            allowed, info = self.rate_limiter.check_rate_limit(client1, 'test')
            assert allowed is True
        
        # Client1 should be denied
        allowed, info = self.rate_limiter.check_rate_limit(client1, 'test')
        assert allowed is False
        
        # Client2 should still be allowed
        allowed, info = self.rate_limiter.check_rate_limit(client2, 'test')
        assert allowed is True
    
    def test_cleanup_old_entries(self):
        """Test cleanup of old rate limit entries."""
        client_id = "test_client_6"
        
        # Make a request
        self.rate_limiter.check_rate_limit(client_id, 'test')
        
        # Verify entry exists
        assert client_id in self.rate_limiter.buckets
        
        # Simulate old entry
        bucket = self.rate_limiter.buckets[client_id]
        bucket['last_refill'] = time.time() - (25 * 3600)  # 25 hours ago
        bucket['requests'].clear()
        
        # Cleanup
        self.rate_limiter.cleanup_old_entries(24)
        
        # Entry should be removed
        assert client_id not in self.rate_limiter.buckets


class TestSecureTokenManager:
    """Test secure token functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.token_manager = SecureTokenManager("test_secret_key_12345")
    
    def test_generate_and_validate_token(self):
        """Test token generation and validation."""
        job_id = "test-job-123"
        file_path = "/path/to/file.mp4"
        
        # Generate token
        token = self.token_manager.generate_download_token(job_id, file_path)
        
        assert isinstance(token, str)
        assert '.' in token  # Should have payload.signature format
        
        # Validate token
        payload = self.token_manager.validate_token(token)
        
        assert payload['job_id'] == job_id
        assert payload['file_path'] == file_path
        assert 'expiry' in payload
        assert 'issued_at' in payload
    
    def test_token_expiry(self):
        """Test token expiration."""
        job_id = "test-job-456"
        file_path = "/path/to/file.mp4"
        
        # Generate token with very short expiry
        token = self.token_manager.generate_download_token(
            job_id, file_path, expiry_hours=0.001  # ~3.6 seconds
        )
        
        # Should be valid immediately
        payload = self.token_manager.validate_token(token)
        assert payload['job_id'] == job_id
        
        # Wait for expiry
        time.sleep(4)
        
        # Should be expired
        with pytest.raises(TokenValidationError, match="expired"):
            self.token_manager.validate_token(token)
    
    def test_token_tampering_detection(self):
        """Test detection of tampered tokens."""
        job_id = "test-job-789"
        file_path = "/path/to/file.mp4"
        
        # Generate valid token
        token = self.token_manager.generate_download_token(job_id, file_path)
        
        # Tamper with token
        tampered_token = token[:-5] + "XXXXX"
        
        # Should detect tampering
        with pytest.raises(TokenValidationError, match="signature"):
            self.token_manager.validate_token(tampered_token)
    
    def test_invalid_token_format(self):
        """Test handling of invalid token formats."""
        invalid_tokens = [
            "invalid_token_no_dot",
            "",
            "a.b.c",  # Too many parts
            "invalid_base64.invalid_base64"
        ]
        
        for invalid_token in invalid_tokens:
            with pytest.raises(TokenValidationError):
                self.token_manager.validate_token(invalid_token)
    
    def test_secure_download_url_generation(self):
        """Test secure download URL generation."""
        base_url = "https://example.com/api"
        job_id = "test-job-url"
        file_path = "/path/to/file.mp4"
        
        url = self.token_manager.generate_secure_download_url(
            base_url, job_id, file_path
        )
        
        assert url.startswith(base_url)
        assert "secure-download" in url
        assert "token=" in url
        
        # Extract and validate token
        extracted_token = self.token_manager.extract_token_from_url(url)
        assert extracted_token is not None
        
        payload = self.token_manager.validate_token(extracted_token)
        assert payload['job_id'] == job_id
    
    def test_token_info(self):
        """Test token information extraction."""
        job_id = "test-job-info"
        file_path = "/path/to/file.mp4"
        
        token = self.token_manager.generate_download_token(job_id, file_path)
        
        info = self.token_manager.get_token_info(token)
        
        assert info['valid'] is True
        assert info['job_id'] == job_id
        assert info['file_path'] == file_path
        assert 'expires_at' in info
        assert 'time_to_expiry_hours' in info


class TestInputSanitizer:
    """Test input sanitization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_valid_filename(self):
        """Test sanitization of valid filenames."""
        valid_filenames = [
            "video.mp4",
            "my_video_file.mov",
            "test-video (1).avi",
            "Video File 2023.mp4"
        ]
        
        for filename in valid_filenames:
            result = self.sanitizer.sanitize_filename(filename)
            assert result == filename
    
    def test_sanitize_invalid_filename(self):
        """Test handling of invalid filenames."""
        invalid_filenames = [
            "",  # Empty
            "video",  # No extension
            "../../../etc/passwd.mp4",  # Path traversal
            "CON.mp4",  # Reserved name
            "a" * 300 + ".mp4",  # Too long
        ]
        
        for filename in invalid_filenames:
            with pytest.raises(InputValidationError):
                self.sanitizer.sanitize_filename(filename)
    
    def test_validate_job_id(self):
        """Test job ID validation."""
        valid_job_id = "12345678-1234-1234-1234-123456789abc"
        result = self.sanitizer.validate_job_id(valid_job_id)
        assert result == valid_job_id
        
        invalid_job_ids = [
            "",
            "invalid-job-id",
            "12345678-1234-1234-1234",  # Too short
            "not-a-uuid-at-all"
        ]
        
        for job_id in invalid_job_ids:
            with pytest.raises(InputValidationError):
                self.sanitizer.validate_job_id(job_id)
    
    def test_sanitize_string_with_html(self):
        """Test string sanitization with HTML content."""
        dangerous_strings = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ]
        
        for dangerous_string in dangerous_strings:
            with pytest.raises(InputValidationError):
                self.sanitizer.sanitize_string(dangerous_string)
    
    def test_sanitize_string_normal_content(self):
        """Test sanitization of normal string content."""
        normal_strings = [
            "This is a normal string",
            "Video processing completed successfully",
            "File: video.mp4 (size: 10MB)"
        ]
        
        for normal_string in normal_strings:
            result = self.sanitizer.sanitize_string(normal_string)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_sanitize_headers(self):
        """Test header sanitization."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TestClient/1.0",
            "X-Forwarded-For": "192.168.1.1",
            "Malicious-Header": "<script>alert('xss')</script>",
            "": "empty-key",  # Empty key
            "Very-Long-Header": "x" * 2000,  # Too long
        }
        
        sanitized = self.sanitizer.sanitize_headers(headers)
        
        # Should keep safe headers
        assert "Content-Type" in sanitized
        assert "User-Agent" in sanitized
        
        # Should remove dangerous headers
        assert "Malicious-Header" not in sanitized
        assert "" not in sanitized
    
    def test_validate_file_size(self):
        """Test file size validation."""
        max_size = 1000
        
        # Valid sizes
        assert self.sanitizer.validate_file_size(500, max_size) is True
        assert self.sanitizer.validate_file_size(1000, max_size) is True
        
        # Invalid sizes
        with pytest.raises(InputValidationError):
            self.sanitizer.validate_file_size(1001, max_size)  # Too large
        
        with pytest.raises(InputValidationError):
            self.sanitizer.validate_file_size(0, max_size)  # Empty
        
        with pytest.raises(InputValidationError):
            self.sanitizer.validate_file_size(-1, max_size)  # Negative
    
    def test_validation_stats(self):
        """Test validation statistics tracking."""
        initial_stats = self.sanitizer.get_validation_stats()
        
        # Perform some validations
        try:
            self.sanitizer.sanitize_filename("valid.mp4")
        except:
            pass
        
        try:
            self.sanitizer.sanitize_filename("")  # This should fail
        except:
            pass
        
        final_stats = self.sanitizer.get_validation_stats()
        
        assert final_stats['total_validations'] > initial_stats['total_validations']


class TestSecurityIntegration:
    """Test security features integration with API."""
    
    def test_rate_limiting_integration(self):
        """Test rate limiting integration with FastAPI."""
        # This would require a test client setup
        # For now, we'll test the middleware function directly
        from security.rate_limiter import rate_limit_middleware
        
        # Mock function to decorate
        @rate_limit_middleware('test')
        async def mock_endpoint():
            return {"message": "success"}
        
        # This is a simplified test - in practice you'd need proper FastAPI test setup
        assert callable(mock_endpoint)
    
    def test_malware_scanning_integration(self):
        """Test malware scanning integration."""
        from security.malware_scanner import get_malware_scanner
        
        scanner = get_malware_scanner()
        assert isinstance(scanner, MalwareScanner)
        
        # Test singleton behavior
        scanner2 = get_malware_scanner()
        assert scanner is scanner2
    
    def test_token_manager_integration(self):
        """Test token manager integration."""
        from security.secure_tokens import get_token_manager
        
        manager = get_token_manager()
        assert isinstance(manager, SecureTokenManager)
        
        # Test singleton behavior
        manager2 = get_token_manager()
        assert manager is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])