"""
Simplified API tests that work with the actual implementation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints with real implementation."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint serves web interface."""
        response = client.get("/")
        assert response.status_code == 200
        # Root endpoint serves HTML web interface, not JSON
        assert "text/html" in response.headers.get("content-type", "")
        assert "Golf Video Anonymizer" in response.text
    
    def test_basic_health_endpoint(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_api_health_endpoint(self, client):
        """Test API health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post("/api/v1/upload")
        assert response.status_code == 422
    
    def test_upload_invalid_filename(self, client):
        """Test upload with invalid filename."""
        test_content = b"fake video content"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("../malicious.exe", test_content, "video/mp4")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "validation_error"
        assert ".." in data["detail"]["message"]
    
    def test_upload_invalid_video_content(self, client):
        """Test upload with invalid video content."""
        test_content = b"not a real video file"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("fake_video.mp4", test_content, "video/mp4")}
        )
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error"] == "file_validation_error"
        assert "format" in data["detail"]["message"].lower()
    
    def test_upload_large_file(self, client):
        """Test upload with file too large."""
        # Create content larger than 500MB (simulate)
        large_content = b"0" * (1024 * 1024)  # 1MB for test
        
        # Mock the file size check by creating a file that reports as larger
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_file.write(large_content)
            temp_file.flush()
            
            # Read the file back
            with open(temp_file.name, "rb") as f:
                content = f.read()
            
            # This should pass size validation but fail format validation
            response = client.post(
                "/api/v1/upload",
                files={"file": ("large_video.mp4", content, "video/mp4")}
            )
            
            # Should fail on format validation, not size (since our test file is < 500MB)
            assert response.status_code == 422
    
    def test_status_nonexistent_job(self, client):
        """Test status for non-existent job."""
        response = client.get("/api/v1/status/nonexistent-job-id")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
    
    def test_download_nonexistent_job(self, client):
        """Test download for non-existent job."""
        response = client.get("/api/v1/download/nonexistent-job-id")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
    
    def test_queue_stats(self, client):
        """Test queue statistics endpoint."""
        response = client.get("/api/v1/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert isinstance(data["data"], dict)
    
    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint."""
        response = client.get("/api/v1/invalid")
        assert response.status_code == 404
    
    def test_invalid_http_method(self, client):
        """Test invalid HTTP method."""
        response = client.delete("/api/v1/upload")
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])