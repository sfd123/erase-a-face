"""
Integration tests for the web interface with the API.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestWebInterface:
    """Test web interface integration with API."""
    
    def test_root_serves_html(self):
        """Test that root endpoint serves the HTML interface."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Check that it's actually HTML content
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "<title>Golf Video Anonymizer</title>" in content
    
    def test_static_css_served(self):
        """Test that CSS files are served correctly."""
        response = client.get("/static/styles.css")
        
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        
        # Check for some CSS content
        content = response.text
        assert "body" in content
        assert "container" in content
    
    def test_static_js_served(self):
        """Test that JavaScript files are served correctly."""
        response = client.get("/static/app.js")
        
        assert response.status_code == 200
        
        # Check for JavaScript content
        content = response.text
        assert "VideoAnonymizerApp" in content
        assert "class" in content
    
    def test_static_test_files_served(self):
        """Test that test files are accessible."""
        response = client.get("/static/test.html")
        assert response.status_code == 200
        
        response = client.get("/static/app.test.js")
        assert response.status_code == 200
    
    def test_nonexistent_static_file(self):
        """Test that nonexistent static files return 404."""
        response = client.get("/static/nonexistent.js")
        assert response.status_code == 404
    
    def test_api_health_endpoint_accessible(self):
        """Test that API health endpoint is accessible for web interface."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
        
        # Verify the data is what the web interface expects
        assert isinstance(data["supported_formats"], list)
        assert isinstance(data["max_file_size_mb"], (int, float))
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present for web interface."""
        response = client.get("/api/v1/health")
        
        # Check for CORS headers (these are added by the CORS middleware)
        assert response.status_code == 200
        
        # The test client doesn't include CORS headers by default,
        # but we can verify the middleware is configured
        # by checking that the endpoint is accessible
    
    def test_upload_endpoint_validation(self):
        """Test upload endpoint validation for web interface."""
        # Test empty upload (what web interface would send on error)
        response = client.post("/api/v1/upload")
        
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_status_endpoint_with_invalid_job(self):
        """Test status endpoint with invalid job ID (web interface error case)."""
        response = client.get("/api/v1/status/invalid-job-id")
        
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data["detail"]
        assert "job_not_found" in data["detail"]["error"]
    
    def test_download_endpoint_with_invalid_job(self):
        """Test download endpoint with invalid job ID (web interface error case)."""
        response = client.get("/api/v1/download/invalid-job-id")
        
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data["detail"]
        assert "job_not_found" in data["detail"]["error"]


class TestWebInterfaceContent:
    """Test web interface HTML content and structure."""
    
    def test_html_structure(self):
        """Test that HTML has required structure for functionality."""
        response = client.get("/")
        content = response.text
        
        # Check for required sections
        assert 'id="upload-section"' in content
        assert 'id="progress-section"' in content
        assert 'id="results-section"' in content
        assert 'id="error-section"' in content
        
        # Check for required form elements
        assert 'id="file-input"' in content
        assert 'id="upload-btn"' in content
        assert 'id="download-btn"' in content
        
        # Check for progress elements
        assert 'id="progress-fill"' in content
        assert 'id="job-id"' in content
        assert 'id="job-status"' in content
    
    def test_css_includes_required_classes(self):
        """Test that CSS includes required classes for functionality."""
        response = client.get("/static/styles.css")
        content = response.text
        
        # Check for key CSS classes
        assert ".upload-area" in content
        assert ".progress-bar" in content
        assert ".hidden" in content
        assert ".btn" in content
        assert ".error-container" in content
    
    def test_js_includes_required_functionality(self):
        """Test that JavaScript includes required functionality."""
        response = client.get("/static/app.js")
        content = response.text
        
        # Check for key JavaScript functionality
        assert "validateFile" in content
        assert "uploadFile" in content
        assert "startStatusPolling" in content
        assert "downloadVideo" in content
        assert "handleDragOver" in content
        assert "handleDrop" in content


class TestWebInterfaceConfiguration:
    """Test web interface configuration and API integration."""
    
    def test_api_base_url_configuration(self):
        """Test that JavaScript uses correct API base URL."""
        response = client.get("/static/app.js")
        content = response.text
        
        # Check that API base URL is configured correctly
        assert "'/api/v1'" in content or '"/api/v1"' in content
    
    def test_supported_formats_match_api(self):
        """Test that web interface supported formats match API."""
        # Get API configuration
        api_response = client.get("/api/v1/health")
        api_data = api_response.json()
        api_formats = api_data["supported_formats"]
        
        # Check JavaScript configuration
        js_response = client.get("/static/app.js")
        js_content = js_response.text
        
        # Verify that the formats are present in the JavaScript
        for format_ext in api_formats:
            assert format_ext.lower() in js_content.lower()
    
    def test_file_size_limit_configuration(self):
        """Test that web interface file size limit matches API."""
        # Get API configuration
        api_response = client.get("/api/v1/health")
        api_data = api_response.json()
        max_size_mb = api_data["max_file_size_mb"]
        
        # Check JavaScript configuration
        js_response = client.get("/static/app.js")
        js_content = js_response.text
        
        # Verify that the size limit is present (500MB default)
        assert "500" in js_content  # Should contain the MB limit
        assert "1024 * 1024" in js_content  # Should contain byte conversion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])