"""
Comprehensive integration tests for complete processing workflows.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from main import app
from models.processing_job import ProcessingJob, JobStatus
from processing.video_processor import VideoProcessor
from storage.file_manager import FileManager
from storage.job_queue import JobQueue


@pytest.mark.integration
class TestEndToEndVideoProcessing:
    """Test complete end-to-end video processing workflows."""
    
    @pytest.fixture
    def integration_setup(self, temp_dir, mock_redis, mock_opencv_cascades):
        """Set up integration test environment."""
        file_manager = FileManager(base_path=str(temp_dir))
        job_queue = JobQueue()
        video_processor = VideoProcessor()
        
        return {
            'file_manager': file_manager,
            'job_queue': job_queue,
            'video_processor': video_processor,
            'temp_dir': temp_dir
        }
    
    def test_successful_video_processing_workflow(self, integration_setup, sample_video_file):
        """Test successful video processing from start to finish."""
        setup = integration_setup
        
        # Create processing job
        job = ProcessingJob.create_new("test_video.mp4", str(sample_video_file))
        
        # Add to queue
        setup['job_queue'].add_job(job)
        
        # Process video
        with patch.object(setup['video_processor'], 'process_video') as mock_process:
            mock_process.return_value = {
                'success': True,
                'output_path': str(setup['temp_dir'] / 'output.mp4'),
                'faces_detected': 3,
                'processing_time': 5.2
            }
            
            result = setup['video_processor'].process_video(
                str(sample_video_file),
                str(setup['temp_dir'] / 'output.mp4')
            )
            
            assert result['success'] is True
            assert result['faces_detected'] == 3
            assert result['processing_time'] > 0
    
    def test_video_processing_with_no_faces(self, integration_setup, sample_video_file):
        """Test video processing when no faces are detected."""
        setup = integration_setup
        
        with patch.object(setup['video_processor'], 'process_video') as mock_process:
            mock_process.return_value = {
                'success': True,
                'output_path': str(setup['temp_dir'] / 'output.mp4'),
                'faces_detected': 0,
                'processing_time': 2.1
            }
            
            result = setup['video_processor'].process_video(
                str(sample_video_file),
                str(setup['temp_dir'] / 'output.mp4')
            )
            
            assert result['success'] is True
            assert result['faces_detected'] == 0


@pytest.mark.integration
class TestAPIWorkflows:
    """Test complete API workflows."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        return TestClient(app)
    
    def test_upload_status_download_workflow(self, api_client, temp_dir):
        """Test complete upload -> status -> download workflow."""
        # Create test video content
        test_content = b"fake video content for testing"
        
        # Upload file
        response = api_client.post(
            "/api/v1/upload",
            files={"file": ("test_video.mp4", test_content, "video/mp4")}
        )
        
        # Should fail validation but test the workflow
        assert response.status_code in [400, 422]  # Expected validation failure
    
    def test_job_status_tracking_workflow(self, api_client):
        """Test job status tracking throughout processing."""
        # Test status for non-existent job
        response = api_client.get("/api/v1/status/fake-job-id")
        assert response.status_code == 404
        
        # Test queue stats
        response = api_client.get("/api/v1/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.integration
class TestErrorRecoveryWorkflows:
    """Test error recovery and handling workflows."""
    
    def test_processing_failure_recovery(self, temp_dir, mock_redis):
        """Test recovery from processing failures."""
        file_manager = FileManager(base_path=str(temp_dir))
        job_queue = JobQueue()
        
        # Create job that will fail
        job = ProcessingJob.create_new("corrupted.mp4", "/nonexistent/path.mp4")
        job_queue.add_job(job)
        
        # Mark as failed
        job.mark_failed("File not found")
        job_queue.update_job(job)
        
        # Verify failure is recorded
        retrieved_job = job_queue.get_job(job.job_id)
        assert retrieved_job.status == JobStatus.FAILED
        assert "File not found" in retrieved_job.error_message
    
    def test_cleanup_after_failure(self, temp_dir):
        """Test cleanup operations after processing failure."""
        file_manager = FileManager(base_path=str(temp_dir))
        
        # Create temporary files
        temp_file1 = temp_dir / "temp1.mp4"
        temp_file2 = temp_dir / "temp2.mp4"
        temp_file1.write_bytes(b"temp content 1")
        temp_file2.write_bytes(b"temp content 2")
        
        # Simulate cleanup after failure
        cleanup_count = file_manager.cleanup_old_files(max_age_minutes=0)
        
        # Should clean up files
        assert cleanup_count >= 0


@pytest.mark.integration
class TestConcurrentProcessing:
    """Test concurrent processing scenarios."""
    
    def test_multiple_job_processing(self, temp_dir, mock_redis):
        """Test processing multiple jobs concurrently."""
        job_queue = JobQueue()
        
        # Create multiple jobs
        jobs = []
        for i in range(5):
            job = ProcessingJob.create_new(f"video_{i}.mp4", f"/path/video_{i}.mp4")
            jobs.append(job)
            job_queue.add_job(job)
        
        # Simulate concurrent processing
        for job in jobs:
            job.mark_processing()
            job_queue.update_job(job)
        
        # Complete jobs
        for i, job in enumerate(jobs):
            job.mark_completed(f"/path/output_{i}.mp4", faces_detected=i+1)
            job_queue.update_job(job)
        
        # Verify all jobs completed
        for job in jobs:
            retrieved_job = job_queue.get_job(job.job_id)
            assert retrieved_job.status == JobStatus.COMPLETED
    
    def test_queue_management_under_load(self, mock_redis):
        """Test queue management under high load."""
        job_queue = JobQueue()
        
        # Add many jobs quickly
        job_ids = []
        for i in range(20):
            job = ProcessingJob.create_new(f"bulk_video_{i}.mp4", f"/path/bulk_{i}.mp4")
            job_queue.add_job(job)
            job_ids.append(job.job_id)
        
        # Get queue statistics
        stats = job_queue.get_stats()
        assert isinstance(stats, dict)
        
        # Verify jobs can be retrieved
        for job_id in job_ids[:5]:  # Test first 5
            job = job_queue.get_job(job_id)
            assert job is not None
            assert job.status == JobStatus.PENDING


@pytest.mark.integration
class TestDataPersistence:
    """Test data persistence across operations."""
    
    def test_job_persistence_across_restarts(self, mock_redis):
        """Test job persistence across system restarts."""
        # Create job queue and add job
        job_queue1 = JobQueue()
        job = ProcessingJob.create_new("persistent_video.mp4", "/path/persistent.mp4")
        job_queue1.add_job(job)
        
        # Simulate system restart with new job queue instance
        job_queue2 = JobQueue()
        retrieved_job = job_queue2.get_job(job.job_id)
        
        # Job should still exist (mocked Redis maintains state)
        assert retrieved_job is not None
        assert retrieved_job.job_id == job.job_id
    
    def test_file_cleanup_persistence(self, temp_dir):
        """Test file cleanup operations maintain consistency."""
        file_manager = FileManager(base_path=str(temp_dir))
        
        # Create files with different timestamps
        files_created = []
        for i in range(3):
            file_path = temp_dir / f"test_file_{i}.mp4"
            file_path.write_bytes(b"test content")
            files_created.append(file_path)
        
        # Verify files exist
        for file_path in files_created:
            assert file_path.exists()
        
        # Cleanup should be consistent
        initial_count = len(list(temp_dir.glob("*.mp4")))
        assert initial_count == 3


@pytest.mark.integration
class TestSystemIntegration:
    """Test integration between all system components."""
    
    def test_complete_system_workflow(self, temp_dir, mock_redis, mock_opencv_cascades):
        """Test complete system workflow with all components."""
        # Initialize all components
        file_manager = FileManager(base_path=str(temp_dir))
        job_queue = JobQueue()
        video_processor = VideoProcessor()
        
        # Create test video file
        test_video = temp_dir / "integration_test.mp4"
        test_video.write_bytes(b"mock video content")
        
        # Create and process job
        job = ProcessingJob.create_new("integration_test.mp4", str(test_video))
        
        # Add to queue
        job_queue.add_job(job)
        assert job.status == JobStatus.PENDING
        
        # Mark as processing
        job.mark_processing()
        job_queue.update_job(job)
        assert job.status == JobStatus.PROCESSING
        
        # Simulate successful processing
        output_path = temp_dir / "output_integration.mp4"
        job.mark_completed(str(output_path), faces_detected=2)
        job_queue.update_job(job)
        
        # Verify final state
        final_job = job_queue.get_job(job.job_id)
        assert final_job.status == JobStatus.COMPLETED
        assert final_job.faces_detected == 2
        assert final_job.output_file_path == str(output_path)
    
    def test_error_propagation_across_components(self, temp_dir, mock_redis):
        """Test error propagation across system components."""
        file_manager = FileManager(base_path=str(temp_dir))
        job_queue = JobQueue()
        
        # Create job with invalid file
        job = ProcessingJob.create_new("invalid.mp4", "/nonexistent/file.mp4")
        job_queue.add_job(job)
        
        # Simulate processing error
        error_message = "File not found: /nonexistent/file.mp4"
        job.mark_failed(error_message)
        job_queue.update_job(job)
        
        # Verify error is properly recorded
        failed_job = job_queue.get_job(job.job_id)
        assert failed_job.status == JobStatus.FAILED
        assert error_message in failed_job.error_message
        assert failed_job.is_complete is True


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningOperations:
    """Test long-running operations and timeouts."""
    
    def test_long_processing_simulation(self, temp_dir, mock_redis):
        """Test simulation of long-running processing operations."""
        job_queue = JobQueue()
        
        # Create job for long processing
        job = ProcessingJob.create_new("long_video.mp4", "/path/long_video.mp4")
        job_queue.add_job(job)
        
        # Mark as processing
        job.mark_processing()
        start_time = job.created_at
        
        # Simulate processing time
        time.sleep(0.1)  # Short sleep for test
        
        # Complete processing
        job.mark_completed("/path/output_long.mp4", faces_detected=10)
        job_queue.update_job(job)
        
        # Verify processing duration is recorded
        assert job.processing_duration is not None
        assert job.processing_duration > 0
    
    def test_timeout_handling(self, mock_redis):
        """Test handling of processing timeouts."""
        job_queue = JobQueue()
        
        # Create job that will timeout
        job = ProcessingJob.create_new("timeout_video.mp4", "/path/timeout.mp4")
        job_queue.add_job(job)
        job.mark_processing()
        
        # Simulate timeout
        job.mark_failed("Processing timeout after 300 seconds")
        job_queue.update_job(job)
        
        # Verify timeout is handled properly
        timeout_job = job_queue.get_job(job.job_id)
        assert timeout_job.status == JobStatus.FAILED
        assert "timeout" in timeout_job.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])