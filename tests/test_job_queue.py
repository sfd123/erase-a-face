"""Unit tests for JobQueue class."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from storage.job_queue import JobQueue, JobQueueError
from models.processing_job import ProcessingJob, JobStatus


class TestJobQueue:
    """Test cases for JobQueue class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        
        # Create a separate mock for pipeline
        mock_pipeline = Mock()
        mock_pipeline.set.return_value = None
        mock_pipeline.lpush.return_value = None
        mock_pipeline.incr.return_value = None
        mock_pipeline.expire.return_value = None
        mock_pipeline.delete.return_value = None
        mock_pipeline.execute.return_value = [True, 1]
        
        mock_client.pipeline.return_value = mock_pipeline
        return mock_client
    
    @pytest.fixture
    def job_queue(self, mock_redis):
        """Create JobQueue instance with mocked Redis."""
        with patch('redis.from_url', return_value=mock_redis):
            return JobQueue("redis://localhost:6379/0")
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample ProcessingJob for testing."""
        return ProcessingJob.create_new("test_video.mp4", "/tmp/test_video.mp4")
    
    def test_init_successful_connection(self, mock_redis):
        """Test successful JobQueue initialization."""
        with patch('redis.from_url', return_value=mock_redis):
            queue = JobQueue("redis://localhost:6379/0", max_retries=5)
            assert queue.max_retries == 5
            assert queue.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    def test_init_connection_failure(self):
        """Test JobQueue initialization with Redis connection failure."""
        mock_redis = Mock()
        mock_redis.ping.side_effect = RedisConnectionError("Connection failed")
        
        with patch('redis.from_url', return_value=mock_redis):
            with pytest.raises(JobQueueError, match="Redis connection failed"):
                JobQueue("redis://localhost:6379/0")
    
    def test_serialize_job(self, job_queue, sample_job):
        """Test job serialization to JSON."""
        sample_job.status = JobStatus.COMPLETED
        sample_job.completed_at = datetime.now()
        sample_job.output_file_path = "/tmp/output.mp4"
        sample_job.faces_detected = 2
        
        serialized = job_queue._serialize_job(sample_job)
        job_dict = json.loads(serialized)
        
        assert job_dict["job_id"] == sample_job.job_id
        assert job_dict["original_filename"] == sample_job.original_filename
        assert job_dict["status"] == "completed"
        assert job_dict["faces_detected"] == 2
    
    def test_deserialize_job(self, job_queue, sample_job):
        """Test job deserialization from JSON."""
        # First serialize the job
        serialized = job_queue._serialize_job(sample_job)
        
        # Then deserialize it
        deserialized = job_queue._deserialize_job(serialized)
        
        assert deserialized.job_id == sample_job.job_id
        assert deserialized.original_filename == sample_job.original_filename
        assert deserialized.status == sample_job.status
        assert deserialized.created_at == sample_job.created_at
    
    def test_deserialize_invalid_json(self, job_queue):
        """Test deserialization with invalid JSON."""
        with pytest.raises(JobQueueError, match="Invalid job data format"):
            job_queue._deserialize_job("invalid json")
    
    def test_enqueue_job_success(self, job_queue, sample_job, mock_redis):
        """Test successful job enqueuing."""
        result = job_queue.enqueue_job(sample_job)
        
        assert result is True
        # Check that pipeline methods were called
        pipeline = mock_redis.pipeline.return_value
        pipeline.set.assert_called_once()
        pipeline.lpush.assert_called_once_with("processing_queue", sample_job.job_id)
        pipeline.execute.assert_called_once()
    
    def test_enqueue_job_redis_error(self, job_queue, sample_job, mock_redis):
        """Test job enqueuing with Redis error."""
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.side_effect = RedisError("Redis error")
        
        with pytest.raises(JobQueueError, match="Failed to enqueue job"):
            job_queue.enqueue_job(sample_job)
    
    def test_dequeue_job_success(self, job_queue, sample_job, mock_redis):
        """Test successful job dequeuing."""
        # Mock Redis responses
        mock_redis.brpop.return_value = ("processing_queue", sample_job.job_id)
        mock_redis.get.return_value = job_queue._serialize_job(sample_job)
        
        result = job_queue.dequeue_job(timeout=10)
        
        assert result is not None
        assert result.job_id == sample_job.job_id
        mock_redis.brpop.assert_called_once_with("processing_queue", timeout=10)
    
    def test_dequeue_job_timeout(self, job_queue, mock_redis):
        """Test job dequeuing with timeout."""
        mock_redis.brpop.return_value = None
        
        result = job_queue.dequeue_job(timeout=5)
        
        assert result is None
    
    def test_dequeue_job_missing_data(self, job_queue, sample_job, mock_redis):
        """Test job dequeuing when job data is missing."""
        mock_redis.brpop.return_value = ("processing_queue", sample_job.job_id)
        mock_redis.get.return_value = None
        
        result = job_queue.dequeue_job()
        
        assert result is None    

    def test_update_job_status_success(self, job_queue, sample_job, mock_redis):
        """Test successful job status update."""
        mock_redis.get.return_value = job_queue._serialize_job(sample_job)
        
        result = job_queue.update_job_status(
            sample_job.job_id, 
            JobStatus.COMPLETED,
            output_file_path="/tmp/output.mp4",
            faces_detected=3
        )
        
        assert result is True
        mock_redis.set.assert_called()
    
    def test_update_job_status_not_found(self, job_queue, mock_redis):
        """Test job status update for non-existent job."""
        mock_redis.get.return_value = None
        
        result = job_queue.update_job_status("nonexistent", JobStatus.FAILED)
        
        assert result is False
    
    def test_get_job_status_success(self, job_queue, sample_job, mock_redis):
        """Test successful job status retrieval."""
        mock_redis.get.return_value = job_queue._serialize_job(sample_job)
        
        result = job_queue.get_job_status(sample_job.job_id)
        
        assert result is not None
        assert result.job_id == sample_job.job_id
    
    def test_get_job_status_not_found(self, job_queue, mock_redis):
        """Test job status retrieval for non-existent job."""
        mock_redis.get.return_value = None
        
        result = job_queue.get_job_status("nonexistent")
        
        assert result is None
    
    def test_retry_failed_job_success(self, job_queue, sample_job, mock_redis):
        """Test successful job retry."""
        # Set up failed job
        sample_job.status = JobStatus.FAILED
        sample_job.error_message = "Processing failed"
        
        mock_redis.get.side_effect = [
            job_queue._serialize_job(sample_job),  # First call for get_job_status
            "0"  # Retry count
        ]
        
        result = job_queue.retry_failed_job(sample_job.job_id)
        
        assert result is True
        mock_redis.incr.assert_called_once()
        mock_redis.lpush.assert_called_with("retry_queue", sample_job.job_id)
    
    def test_retry_failed_job_max_retries_exceeded(self, job_queue, sample_job, mock_redis):
        """Test job retry when max retries exceeded."""
        sample_job.status = JobStatus.FAILED
        
        mock_redis.get.side_effect = [
            job_queue._serialize_job(sample_job),  # First call for get_job_status
            "3"  # Retry count equals max_retries
        ]
        
        result = job_queue.retry_failed_job(sample_job.job_id)
        
        assert result is False
        mock_redis.lpush.assert_called_with("failed_queue", sample_job.job_id)
    
    def test_retry_failed_job_wrong_status(self, job_queue, sample_job, mock_redis):
        """Test job retry for job that's not failed."""
        sample_job.status = JobStatus.COMPLETED
        mock_redis.get.return_value = job_queue._serialize_job(sample_job)
        
        result = job_queue.retry_failed_job(sample_job.job_id)
        
        assert result is False
    
    def test_get_next_retry_job_success(self, job_queue, sample_job, mock_redis):
        """Test getting next retry job."""
        mock_redis.brpop.return_value = ("retry_queue", sample_job.job_id)
        mock_redis.get.return_value = job_queue._serialize_job(sample_job)
        
        result = job_queue.get_next_retry_job(timeout=5)
        
        assert result is not None
        assert result.job_id == sample_job.job_id
    
    def test_get_next_retry_job_timeout(self, job_queue, mock_redis):
        """Test getting retry job with timeout."""
        mock_redis.brpop.return_value = None
        
        result = job_queue.get_next_retry_job(timeout=1)
        
        assert result is None
    
    def test_get_queue_stats(self, job_queue, mock_redis):
        """Test getting queue statistics."""
        # Mock queue lengths
        mock_redis.llen.side_effect = [5, 2, 1]  # pending, retry, failed
        
        # Mock job keys and data
        mock_redis.keys.return_value = ["job:1", "job:2", "job:3"]
        
        job1 = ProcessingJob.create_new("test1.mp4", "/tmp/test1.mp4")
        job1.status = JobStatus.PENDING
        job2 = ProcessingJob.create_new("test2.mp4", "/tmp/test2.mp4")
        job2.status = JobStatus.COMPLETED
        job3 = ProcessingJob.create_new("test3.mp4", "/tmp/test3.mp4")
        job3.status = JobStatus.FAILED
        
        mock_redis.get.side_effect = [
            job_queue._serialize_job(job1),
            job_queue._serialize_job(job2),
            job_queue._serialize_job(job3)
        ]
        
        stats = job_queue.get_queue_stats()
        
        assert stats["pending_jobs"] == 5
        assert stats["retry_jobs"] == 2
        assert stats["failed_jobs"] == 1
        assert stats["pending"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["total_jobs"] == 3
    
    def test_cleanup_completed_jobs(self, job_queue, mock_redis):
        """Test cleanup of old completed jobs."""
        # Create old completed job
        old_job = ProcessingJob.create_new("old.mp4", "/tmp/old.mp4")
        old_job.status = JobStatus.COMPLETED
        old_job.completed_at = datetime.now() - timedelta(hours=25)
        
        # Create recent completed job
        recent_job = ProcessingJob.create_new("recent.mp4", "/tmp/recent.mp4")
        recent_job.status = JobStatus.COMPLETED
        recent_job.completed_at = datetime.now() - timedelta(hours=1)
        
        mock_redis.keys.return_value = [f"job:{old_job.job_id}", f"job:{recent_job.job_id}"]
        mock_redis.get.side_effect = [
            job_queue._serialize_job(old_job),
            job_queue._serialize_job(recent_job)
        ]
        
        cleaned_count = job_queue.cleanup_completed_jobs(older_than_hours=24)
        
        assert cleaned_count == 1
        mock_redis.delete.assert_called()
    
    def test_health_check_success(self, job_queue, mock_redis):
        """Test successful health check."""
        mock_redis.ping.return_value = True
        
        result = job_queue.health_check()
        
        assert result is True
    
    def test_health_check_failure(self, job_queue, mock_redis):
        """Test health check with Redis error."""
        mock_redis.ping.side_effect = RedisError("Connection lost")
        
        result = job_queue.health_check()
        
        assert result is False
    
    def test_redis_error_handling(self, job_queue, mock_redis):
        """Test general Redis error handling."""
        mock_redis.get.side_effect = RedisError("Redis error")
        
        with pytest.raises(JobQueueError, match="Failed to retrieve job status"):
            job_queue.get_job_status("test_job")


class TestJobQueueIntegration:
    """Integration tests for JobQueue with real Redis operations."""
    
    @pytest.fixture
    def redis_available(self):
        """Check if Redis is available for integration tests."""
        try:
            client = redis.Redis(host='localhost', port=6379, db=15)  # Use test DB
            client.ping()
            client.flushdb()  # Clean test database
            return client
        except (RedisConnectionError, ConnectionError):
            pytest.skip("Redis not available for integration tests")
    
    @pytest.fixture
    def integration_queue(self, redis_available):
        """Create JobQueue for integration testing."""
        return JobQueue("redis://localhost:6379/15")  # Use test database
    
    def test_full_job_lifecycle(self, integration_queue):
        """Test complete job lifecycle with real Redis."""
        # Create and enqueue job
        job = ProcessingJob.create_new("integration_test.mp4", "/tmp/integration_test.mp4")
        
        # Enqueue
        assert integration_queue.enqueue_job(job) is True
        
        # Dequeue
        dequeued_job = integration_queue.dequeue_job(timeout=1)
        assert dequeued_job is not None
        assert dequeued_job.job_id == job.job_id
        
        # Update status
        assert integration_queue.update_job_status(
            job.job_id, 
            JobStatus.PROCESSING
        ) is True
        
        # Complete job
        assert integration_queue.update_job_status(
            job.job_id,
            JobStatus.COMPLETED,
            output_file_path="/tmp/output.mp4",
            faces_detected=2
        ) is True
        
        # Verify final status
        final_job = integration_queue.get_job_status(job.job_id)
        assert final_job.status == JobStatus.COMPLETED
        assert final_job.output_file_path == "/tmp/output.mp4"
        assert final_job.faces_detected == 2
    
    def test_retry_mechanism(self, integration_queue):
        """Test job retry mechanism with real Redis."""
        # Create and fail a job
        job = ProcessingJob.create_new("retry_test.mp4", "/tmp/retry_test.mp4")
        integration_queue.enqueue_job(job)
        
        # Mark as failed
        integration_queue.update_job_status(
            job.job_id,
            JobStatus.FAILED,
            error_message="Test failure"
        )
        
        # Retry the job
        assert integration_queue.retry_failed_job(job.job_id) is True
        
        # Get from retry queue
        retry_job = integration_queue.get_next_retry_job(timeout=1)
        assert retry_job is not None
        assert retry_job.job_id == job.job_id
        assert retry_job.status == JobStatus.PENDING