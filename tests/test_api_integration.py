"""
Integration tests for the Golf Video Anonymizer REST API.

Tests all API endpoints with various scenarios including success cases,
validation errors, and error handling.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from fastapi.testclient import TestClient
from fastapi import UploadFile
import httpx

from main import app
from models.processing_job import ProcessingJob, JobStatus
from storage.job_queue import JobQueue, JobQueueError
from storage.file_manager import FileManager, FileValidationError


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_job_queue():
    """Mock JobQueue for testing."""
    return Mock(spec=JobQueue)


@pytest.fixture
def mock_file_manager():
    """Mock FileManager for testing."""
    return Mock(spec=FileManager)


@pytest.fixture
def sample_job():
    """Create a sample processing job for testing."""
    return ProcessingJob(
        job_id="test-job-123",
        original_filename="test_video.mp4",
        file_path="/tmp/test_video.mp4",
        status=JobStatus.PENDING,
        created_at=datetime.now()
    )


@pytest.fixture
def completed_job():
    """Create a completed processing job for testing."""
    job = ProcessingJob(
        job_id="completed-job-456",
        original_filename="golf_swing.mp4",
        file_path="/tmp/golf_swing.mp4",
        status=JobStatus.COMPLETED,
        created_at=datetime.now(),
        completed_at=datetime.now(),
        output_file_path="/tmp/processed_golf_swing.mp4",
        faces_detected=2
    )
    return job


@pytest.fixture
def create_test_video_file():
    """Create a temporary test video file."""
    def _create_file(filename="test_video.mp4", content=b"fake video content", size=None):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        if size:
            # Create file of specific size
            temp_file.write(b"0" * size)
        else:
            temp_file.write(content)
        temp_file.close()
        return temp_file.name
    return _create_file


class TestVideoUploadEndpoint:
    """Test cases for video upload endpoint."""
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_successful_upload(self, mock_get_file_manager, mock_get_job_queue, 
                              client, mock_job_queue, mock_file_manager, sample_job):
        """Test successful video upload."""
        # Setup mocks
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        
        mock_file_manager.MAX_FILE_SIZE = 500 * 1024 * 1024
        mock_file_manager.create_temp_file.return_value = Path("/tmp/temp_file.mp4")
        mock_file_manager.store_uploaded_file.return_value = Path("/tmp/stored_file.mp4")
        mock_file_manager.cleanup_temp_files.return_value = 1
        mock_job_queue.enqueue_job.return_value = True
        
        # Create test file
        test_content = b"fake video content"
        
        # Make request
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test_video.mp4", test_content, "video/mp4")}
        )
        
        # Assertions
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["original_filename"] == "test_video.mp4"
        assert data["status"] == "pending"
        assert "message" in data
        
        # Verify mocks were called
        mock_file_manager.create_temp_file.assert_called_once()
        mock_file_manager.store_uploaded_file.assert_called_once()
        mock_job_queue.enqueue_job.assert_called_once()
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post("/api/v1/upload")
        assert response.status_code == 422  # Validation error
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_upload_invalid_filename(self, mock_get_file_manager, mock_get_job_queue, 
                                   client, mock_job_queue, mock_file_manager):
        """Test upload with invalid filename."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        
        test_content = b"fake video content"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("../malicious.exe", test_content, "video/mp4")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "validation_error"
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_upload_file_too_large(self, mock_get_file_manager, mock_get_job_queue, 
                                  client, mock_job_queue, mock_file_manager):
        """Test upload with file too large."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        
        mock_file_manager.MAX_FILE_SIZE = 1024  # 1KB limit for test
        
        large_content = b"0" * 2048  # 2KB file
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("large_video.mp4", large_content, "video/mp4")}
        )
        
        assert response.status_code == 413
        data = response.json()
        assert data["detail"]["error"] == "file_too_large"
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_upload_validation_error(self, mock_get_file_manager, mock_get_job_queue, 
                                   client, mock_job_queue, mock_file_manager):
        """Test upload with file validation error."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        
        mock_file_manager.MAX_FILE_SIZE = 500 * 1024 * 1024
        mock_file_manager.create_temp_file.return_value = Path("/tmp/temp_file.mp4")
        mock_file_manager.store_uploaded_file.side_effect = FileValidationError("Invalid video format")
        
        test_content = b"not a video file"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("fake_video.mp4", test_content, "video/mp4")}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error"] == "validation_error"
        # The actual error message will be from the real validation, so let's check for validation error
        assert "video" in data["detail"]["message"].lower() or "format" in data["detail"]["message"].lower()
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_upload_queue_error(self, mock_get_file_manager, mock_get_job_queue, 
                               client, mock_job_queue, mock_file_manager):
        """Test upload with job queue error."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        
        mock_file_manager.MAX_FILE_SIZE = 500 * 1024 * 1024
        mock_file_manager.create_temp_file.return_value = Path("/tmp/temp_file.mp4")
        mock_file_manager.store_uploaded_file.return_value = Path("/tmp/stored_file.mp4")
        mock_job_queue.enqueue_job.side_effect = JobQueueError("Redis connection failed")
        
        test_content = b"fake video content"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test_video.mp4", test_content, "video/mp4")}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "service_unavailable"


class TestProcessingStatusEndpoint:
    """Test cases for processing status endpoint."""
    
    @patch('api.routes.get_job_queue')
    def test_get_status_success(self, mock_get_job_queue, client, mock_job_queue, sample_job):
        """Test successful status retrieval."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_job_status.return_value = sample_job
        
        response = client.get(f"/api/v1/status/{sample_job.job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == sample_job.job_id
        assert data["status"] == sample_job.status.value
        assert data["original_filename"] == sample_job.original_filename
        assert "created_at" in data
    
    @patch('api.routes.get_job_queue')
    def test_get_status_completed_job(self, mock_get_job_queue, client, mock_job_queue, completed_job):
        """Test status retrieval for completed job."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_job_status.return_value = completed_job
        
        response = client.get(f"/api/v1/status/{completed_job.job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == completed_job.job_id
        assert data["status"] == "completed"
        assert data["faces_detected"] == 2
        assert data["completed_at"] is not None
        assert data["processing_duration"] is not None
    
    @patch('api.routes.get_job_queue')
    def test_get_status_job_not_found(self, mock_get_job_queue, client, mock_job_queue):
        """Test status retrieval for non-existent job."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_job_status.return_value = None
        
        response = client.get("/api/v1/status/nonexistent-job")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
    
    @patch('api.routes.get_job_queue')
    def test_get_status_queue_error(self, mock_get_job_queue, client, mock_job_queue):
        """Test status retrieval with queue error."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_job_status.side_effect = JobQueueError("Redis connection failed")
        
        response = client.get("/api/v1/status/test-job")
        
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "service_unavailable"


class TestVideoDownloadEndpoint:
    """Test cases for video download endpoint."""
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_download_success(self, mock_get_file_manager, mock_get_job_queue, 
                             client, mock_job_queue, mock_file_manager, completed_job, 
                             create_test_video_file):
        """Test successful video download."""
        # Create actual test file
        test_file_path = create_test_video_file("processed_video.mp4")
        completed_job.output_file_path = test_file_path
        
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        mock_job_queue.get_job_status.return_value = completed_job
        
        try:
            response = client.get(f"/api/v1/download/{completed_job.job_id}")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
            assert "anonymized_golf_swing.mp4" in response.headers.get("content-disposition", "")
            assert response.headers.get("x-job-id") == completed_job.job_id
            assert response.headers.get("x-faces-detected") == "2"
        finally:
            # Clean up test file
            os.unlink(test_file_path)
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_download_job_not_found(self, mock_get_file_manager, mock_get_job_queue, 
                                   client, mock_job_queue, mock_file_manager):
        """Test download for non-existent job."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        mock_job_queue.get_job_status.return_value = None
        
        response = client.get("/api/v1/download/nonexistent-job")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_download_job_not_completed(self, mock_get_file_manager, mock_get_job_queue, 
                                       client, mock_job_queue, mock_file_manager, sample_job):
        """Test download for incomplete job."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        mock_job_queue.get_job_status.return_value = sample_job
        
        response = client.get(f"/api/v1/download/{sample_job.job_id}")
        
        assert response.status_code == 202
        data = response.json()
        assert data["detail"]["error"] == "processing_incomplete"
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_download_failed_job(self, mock_get_file_manager, mock_get_job_queue, 
                                client, mock_job_queue, mock_file_manager):
        """Test download for failed job."""
        failed_job = ProcessingJob(
            job_id="failed-job-789",
            original_filename="failed_video.mp4",
            file_path="/tmp/failed_video.mp4",
            status=JobStatus.FAILED,
            created_at=datetime.now(),
            completed_at=datetime.now(),
            error_message="Processing failed due to corrupted video"
        )
        
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        mock_job_queue.get_job_status.return_value = failed_job
        
        response = client.get(f"/api/v1/download/{failed_job.job_id}")
        
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error"] == "processing_failed"
        assert "Processing failed due to corrupted video" in data["detail"]["message"]
    
    @patch('api.routes.get_job_queue')
    @patch('api.routes.get_file_manager')
    def test_download_file_not_found(self, mock_get_file_manager, mock_get_job_queue, 
                                    client, mock_job_queue, mock_file_manager, completed_job):
        """Test download when output file is missing."""
        completed_job.output_file_path = "/nonexistent/path/video.mp4"
        
        mock_get_job_queue.return_value = mock_job_queue
        mock_get_file_manager.return_value = mock_file_manager
        mock_job_queue.get_job_status.return_value = completed_job
        
        response = client.get(f"/api/v1/download/{completed_job.job_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "file_not_found"


class TestHealthCheckEndpoint:
    """Test cases for health check endpoint."""
    
    @patch('api.routes.get_job_queue')
    def test_health_check_healthy(self, mock_get_job_queue, client, mock_job_queue):
        """Test health check when service is healthy."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.health_check.return_value = True
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "supported_formats" in data
        assert "max_file_size_mb" in data
    
    @patch('api.routes.get_job_queue')
    def test_health_check_degraded(self, mock_get_job_queue, client, mock_job_queue):
        """Test health check when service is degraded."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.health_check.return_value = False
        
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


class TestQueueStatsEndpoint:
    """Test cases for queue statistics endpoint."""
    
    @patch('api.routes.get_job_queue')
    def test_queue_stats_success(self, mock_get_job_queue, client, mock_job_queue):
        """Test successful queue stats retrieval."""
        mock_stats = {
            "pending_jobs": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1,
            "total_jobs": 18
        }
        
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_queue_stats.return_value = mock_stats
        
        response = client.get("/api/v1/queue/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == mock_stats
    
    @patch('api.routes.get_job_queue')
    def test_queue_stats_error(self, mock_get_job_queue, client, mock_job_queue):
        """Test queue stats with error."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_queue_stats.side_effect = JobQueueError("Redis connection failed")
        
        response = client.get("/api/v1/queue/stats")
        
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "service_unavailable"


class TestRootEndpoints:
    """Test cases for root and basic endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Golf Video Anonymizer"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_basic_health_endpoint(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint."""
        response = client.get("/api/v1/invalid")
        assert response.status_code == 404
    
    def test_invalid_http_method(self, client):
        """Test invalid HTTP method."""
        response = client.delete("/api/v1/upload")
        assert response.status_code == 405
    
    @patch('api.routes.get_job_queue')
    def test_malformed_job_id(self, mock_get_job_queue, client, mock_job_queue):
        """Test status endpoint with malformed job ID."""
        mock_get_job_queue.return_value = mock_job_queue
        mock_job_queue.get_job_status.return_value = None
        
        response = client.get("/api/v1/status/malformed-id-with-special-chars!@#")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "job_not_found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])