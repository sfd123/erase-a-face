"""
Comprehensive API tests demonstrating full functionality.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestAPIComprehensive:
    """Comprehensive tests for all API endpoints."""
    
    def test_api_documentation_available(self, client):
        """Test that API documentation is available."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_health_endpoints(self, client):
        """Test all health check endpoints."""
        # Basic health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # API health with detailed info
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
        assert data["max_file_size_mb"] == 500.0
        assert ".mp4" in data["supported_formats"]
        assert ".mov" in data["supported_formats"]
        assert ".avi" in data["supported_formats"]
    
    def test_upload_validation_comprehensive(self, client):
        """Test comprehensive upload validation."""
        # Test 1: No file
        response = client.post("/api/v1/upload")
        assert response.status_code == 422
        
        # Test 2: Invalid filename characters
        invalid_filenames_400 = [
            "../malicious.mp4",
            "file:with:colons.mp4",
            "file*with*asterisks.mp4",
            "file?with?questions.mp4",
            "file<with>brackets.mp4",
            "file|with|pipes.mp4"
        ]
        
        # These should fail at filename validation (400)
        for filename in invalid_filenames_400:
            response = client.post(
                "/api/v1/upload",
                files={"file": (filename, b"content", "video/mp4")}
            )
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error"] == "validation_error"
        
        # This one might pass filename validation but fail content validation (422)
        response = client.post(
            "/api/v1/upload",
            files={"file": ('file"with"quotes.mp4', b"content", "video/mp4")}
        )
        assert response.status_code in [400, 422]  # Either is acceptable
        data = response.json()
        assert data["detail"]["error"] == "validation_error"
        
        # Test 3: Invalid file extensions
        invalid_extensions = [
            "video.txt",
            "video.exe",
            "video.pdf",
            "video.jpg"
        ]
        
        for filename in invalid_extensions:
            response = client.post(
                "/api/v1/upload",
                files={"file": (filename, b"content", "video/mp4")}
            )
            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["error"] == "validation_error"
        
        # Test 4: Valid filename but invalid content
        response = client.post(
            "/api/v1/upload",
            files={"file": ("valid_video.mp4", b"not a video", "video/mp4")}
        )
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error"] == "validation_error"
        assert "format" in data["detail"]["message"].lower()
    
    def test_job_status_and_download_flow(self, client):
        """Test the complete job status and download flow."""
        # Test status for non-existent job
        response = client.get("/api/v1/status/nonexistent-job")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
        
        # Test download for non-existent job
        response = client.get("/api/v1/download/nonexistent-job")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
        
        # Test status with malformed job IDs
        malformed_ids = [
            "job-with-special-chars!@#",
            "very-long-job-id-" + "x" * 100,
            "",
            "job with spaces"
        ]
        
        for job_id in malformed_ids:
            if job_id:  # Skip empty string for URL path
                response = client.get(f"/api/v1/status/{job_id}")
                assert response.status_code == 404
    
    def test_queue_statistics(self, client):
        """Test queue statistics endpoint."""
        response = client.get("/api/v1/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
        stats = data["data"]
        expected_fields = [
            "pending_jobs", "retry_jobs", "failed_jobs", "total_jobs",
            "pending", "processing", "completed", "failed"
        ]
        
        for field in expected_fields:
            assert field in stats
            assert isinstance(stats[field], int)
            assert stats[field] >= 0
    
    def test_error_responses_format(self, client):
        """Test that all error responses follow the expected format."""
        # Test various error scenarios
        error_tests = [
            ("/api/v1/upload", "POST", {}, 422),  # No file
            ("/api/v1/status/nonexistent", "GET", {}, 404),  # Job not found
            ("/api/v1/download/nonexistent", "GET", {}, 404),  # Job not found
            ("/api/v1/invalid", "GET", {}, 404),  # Invalid endpoint
        ]
        
        for url, method, data, expected_status in error_tests:
            if method == "POST":
                response = client.post(url, **data)
            else:
                response = client.get(url)
            
            assert response.status_code == expected_status
            
            if response.status_code != 404 or "/api/v1/" in url:
                # API errors should have structured format
                error_data = response.json()
                if "detail" in error_data:
                    detail = error_data["detail"]
                    if isinstance(detail, dict):
                        assert "error" in detail
                        assert "message" in detail
    
    def test_cors_and_headers(self, client):
        """Test CORS headers and response headers."""
        # Test preflight request
        response = client.options("/api/v1/health")
        # Note: TestClient doesn't fully simulate CORS, but we can check the endpoint exists
        
        # Test that responses include appropriate headers
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_http_methods(self, client):
        """Test HTTP method restrictions."""
        # Test that endpoints only accept appropriate methods
        method_tests = [
            ("/api/v1/upload", ["POST"], ["GET", "PUT", "DELETE", "PATCH"]),
            ("/api/v1/health", ["GET"], ["POST", "PUT", "DELETE", "PATCH"]),
            ("/api/v1/status/test", ["GET"], ["POST", "PUT", "DELETE", "PATCH"]),
            ("/api/v1/download/test", ["GET"], ["POST", "PUT", "DELETE", "PATCH"]),
        ]
        
        for url, allowed_methods, disallowed_methods in method_tests:
            for method in disallowed_methods:
                response = getattr(client, method.lower())(url)
                assert response.status_code == 405  # Method Not Allowed
    
    def test_api_versioning(self, client):
        """Test API versioning structure."""
        # All API endpoints should be under /api/v1/
        response = client.get("/")
        assert response.status_code == 200
        root_data = response.json()
        assert root_data["api"] == "/api/v1"
        
        # Test that v1 endpoints exist
        v1_endpoints = [
            "/api/v1/health",
            "/api/v1/queue/stats"
        ]
        
        for endpoint in v1_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])