"""Job queue implementation using Redis for background processing."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from models.processing_job import ProcessingJob, JobStatus


logger = logging.getLogger(__name__)


class JobQueueError(Exception):
    """Base exception for job queue operations."""
    pass


class JobQueue:
    """Redis-based job queue for managing video processing jobs."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", max_retries: int = 3):
        """
        Initialize the job queue with Redis connection.
        
        Args:
            redis_url: Redis connection URL
            max_retries: Maximum number of retry attempts for failed jobs
        """
        self.max_retries = max_retries
        self.redis_client = None
        self._connect(redis_url)
        
        # Redis key patterns
        self.job_key_prefix = "job:"
        self.queue_key = "processing_queue"
        self.retry_queue_key = "retry_queue"
        self.failed_queue_key = "failed_queue"
        
    def _connect(self, redis_url: str) -> None:
        """Establish Redis connection with error handling."""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise JobQueueError(f"Redis connection failed: {e}")
    
    def _serialize_job(self, job: ProcessingJob) -> str:
        """Serialize ProcessingJob to JSON string."""
        job_dict = {
            "job_id": job.job_id,
            "original_filename": job.original_filename,
            "file_path": job.file_path,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "output_file_path": job.output_file_path,
            "faces_detected": job.faces_detected
        }
        return json.dumps(job_dict)
    
    def _deserialize_job(self, job_data: str) -> ProcessingJob:
        """Deserialize JSON string to ProcessingJob."""
        try:
            job_dict = json.loads(job_data)
            return ProcessingJob(
                job_id=job_dict["job_id"],
                original_filename=job_dict["original_filename"],
                file_path=job_dict["file_path"],
                status=JobStatus(job_dict["status"]),
                created_at=datetime.fromisoformat(job_dict["created_at"]),
                completed_at=datetime.fromisoformat(job_dict["completed_at"]) if job_dict["completed_at"] else None,
                error_message=job_dict["error_message"],
                output_file_path=job_dict["output_file_path"],
                faces_detected=job_dict["faces_detected"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to deserialize job data: {e}")
            raise JobQueueError(f"Invalid job data format: {e}")  
  
    def enqueue_job(self, job: ProcessingJob) -> bool:
        """
        Add a job to the processing queue.
        
        Args:
            job: ProcessingJob to enqueue
            
        Returns:
            bool: True if job was successfully enqueued
            
        Raises:
            JobQueueError: If enqueue operation fails
        """
        try:
            # Store job data
            job_key = f"{self.job_key_prefix}{job.job_id}"
            job_data = self._serialize_job(job)
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.set(job_key, job_data)
            pipe.lpush(self.queue_key, job.job_id)
            pipe.execute()
            
            logger.info(f"Job {job.job_id} enqueued successfully")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to enqueue job {job.job_id}: {e}")
            raise JobQueueError(f"Failed to enqueue job: {e}")
    
    def dequeue_job(self, timeout: int = 10) -> Optional[ProcessingJob]:
        """
        Remove and return the next job from the queue.
        
        Args:
            timeout: Timeout in seconds for blocking pop operation
            
        Returns:
            ProcessingJob or None if no job available within timeout
            
        Raises:
            JobQueueError: If dequeue operation fails
        """
        try:
            # Blocking pop from queue
            result = self.redis_client.brpop(self.queue_key, timeout=timeout)
            if not result:
                return None
            
            _, job_id = result
            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                logger.warning(f"Job data not found for job_id: {job_id}")
                return None
            
            job = self._deserialize_job(job_data)
            logger.info(f"Job {job_id} dequeued successfully")
            return job
            
        except RedisError as e:
            logger.error(f"Failed to dequeue job: {e}")
            raise JobQueueError(f"Failed to dequeue job: {e}")
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         error_message: Optional[str] = None,
                         output_file_path: Optional[str] = None,
                         faces_detected: Optional[int] = None) -> bool:
        """
        Update job status and related fields.
        
        Args:
            job_id: Job identifier
            status: New job status
            error_message: Error message for failed jobs
            output_file_path: Path to processed video file
            faces_detected: Number of faces detected in video
            
        Returns:
            bool: True if update was successful
            
        Raises:
            JobQueueError: If update operation fails
        """
        try:
            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                logger.warning(f"Job not found for update: {job_id}")
                return False
            
            job = self._deserialize_job(job_data)
            
            # Update job fields
            job.status = status
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now()
            
            if error_message:
                job.error_message = error_message
            if output_file_path:
                job.output_file_path = output_file_path
            if faces_detected is not None:
                job.faces_detected = faces_detected
            
            # Save updated job
            updated_data = self._serialize_job(job)
            self.redis_client.set(job_key, updated_data)
            
            logger.info(f"Job {job_id} status updated to {status.value}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise JobQueueError(f"Failed to update job status: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Retrieve job information by job ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProcessingJob or None if job not found
            
        Raises:
            JobQueueError: If retrieval operation fails
        """
        try:
            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                return None
            
            return self._deserialize_job(job_data)
            
        except RedisError as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            raise JobQueueError(f"Failed to retrieve job status: {e}")
    
    def retry_failed_job(self, job_id: str) -> bool:
        """
        Retry a failed job if it hasn't exceeded max retry attempts.
        
        Args:
            job_id: Job identifier to retry
            
        Returns:
            bool: True if job was queued for retry, False if max retries exceeded
            
        Raises:
            JobQueueError: If retry operation fails
        """
        try:
            job = self.get_job_status(job_id)
            if not job:
                logger.warning(f"Cannot retry job {job_id}: job not found")
                return False
            
            if job.status != JobStatus.FAILED:
                logger.warning(f"Cannot retry job {job_id}: job status is {job.status.value}")
                return False
            
            # Check retry count
            retry_count_key = f"retry_count:{job_id}"
            retry_count = int(self.redis_client.get(retry_count_key) or 0)
            
            if retry_count >= self.max_retries:
                logger.info(f"Job {job_id} exceeded max retries ({self.max_retries})")
                # Move to failed queue permanently
                self.redis_client.lpush(self.failed_queue_key, job_id)
                return False
            
            # Increment retry count and reset job status
            pipe = self.redis_client.pipeline()
            pipe.incr(retry_count_key)
            pipe.expire(retry_count_key, 86400)  # Expire after 24 hours
            
            # Reset job status and clear error message
            job.status = JobStatus.PENDING
            job.error_message = None
            job.completed_at = None
            
            job_key = f"{self.job_key_prefix}{job_id}"
            job_data = self._serialize_job(job)
            pipe.set(job_key, job_data)
            
            # Add back to processing queue
            pipe.lpush(self.retry_queue_key, job_id)
            pipe.execute()
            
            logger.info(f"Job {job_id} queued for retry (attempt {retry_count + 1})")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to retry job {job_id}: {e}")
            raise JobQueueError(f"Failed to retry job: {e}")
    
    def get_next_retry_job(self, timeout: int = 5) -> Optional[ProcessingJob]:
        """
        Get the next job from the retry queue.
        
        Args:
            timeout: Timeout in seconds for blocking pop operation
            
        Returns:
            ProcessingJob or None if no retry job available
        """
        try:
            result = self.redis_client.brpop(self.retry_queue_key, timeout=timeout)
            if not result:
                return None
            
            _, job_id = result
            return self.get_job_status(job_id)
            
        except RedisError as e:
            logger.error(f"Failed to get retry job: {e}")
            raise JobQueueError(f"Failed to get retry job: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the job queues.
        
        Returns:
            Dict containing queue statistics
        """
        try:
            stats = {
                "pending_jobs": self.redis_client.llen(self.queue_key),
                "retry_jobs": self.redis_client.llen(self.retry_queue_key),
                "failed_jobs": self.redis_client.llen(self.failed_queue_key),
                "total_jobs": 0
            }
            
            # Count jobs by status
            job_keys = self.redis_client.keys(f"{self.job_key_prefix}*")
            status_counts = {status.value: 0 for status in JobStatus}
            
            for job_key in job_keys:
                job_data = self.redis_client.get(job_key)
                if job_data:
                    try:
                        job = self._deserialize_job(job_data)
                        status_counts[job.status.value] += 1
                    except JobQueueError:
                        continue
            
            stats.update(status_counts)
            stats["total_jobs"] = sum(status_counts.values())
            
            return stats
            
        except RedisError as e:
            logger.error(f"Failed to get queue stats: {e}")
            raise JobQueueError(f"Failed to get queue statistics: {e}")
    
    def cleanup_completed_jobs(self, older_than_hours: int = 24) -> int:
        """
        Clean up completed jobs older than specified hours.
        
        Args:
            older_than_hours: Remove jobs completed more than this many hours ago
            
        Returns:
            int: Number of jobs cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            job_keys = self.redis_client.keys(f"{self.job_key_prefix}*")
            cleaned_count = 0
            
            for job_key in job_keys:
                job_data = self.redis_client.get(job_key)
                if not job_data:
                    continue
                
                try:
                    job = self._deserialize_job(job_data)
                    if (job.is_complete and job.completed_at and 
                        job.completed_at < cutoff_time):
                        
                        # Remove job data and retry count
                        pipe = self.redis_client.pipeline()
                        pipe.delete(job_key)
                        pipe.delete(f"retry_count:{job.job_id}")
                        pipe.execute()
                        
                        cleaned_count += 1
                        
                except JobQueueError:
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} completed jobs")
            return cleaned_count
            
        except RedisError as e:
            logger.error(f"Failed to cleanup jobs: {e}")
            raise JobQueueError(f"Failed to cleanup completed jobs: {e}")
    
    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            bool: True if Redis is accessible
        """
        try:
            self.redis_client.ping()
            return True
        except RedisError:
            return False