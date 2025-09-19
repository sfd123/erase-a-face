"""
Background cleanup service for failed processing jobs.

This module provides automatic cleanup of failed jobs, temporary files,
and expired processed videos to maintain system health.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

from models.processing_job import ProcessingJob, JobStatus
from storage.job_queue import JobQueue, JobQueueError
from storage.file_manager import FileManager
from api.error_handlers import ErrorHandler

logger = logging.getLogger(__name__)


class CleanupService:
    """
    Background service for automatic cleanup of failed jobs and temporary files.
    
    This service runs periodically to:
    - Clean up files from failed processing jobs
    - Remove expired temporary files
    - Clean up old completed job data
    - Maintain storage health
    """
    
    def __init__(self, job_queue: JobQueue, file_manager: FileManager, 
                 cleanup_interval_minutes: int = 30):
        """
        Initialize cleanup service.
        
        Args:
            job_queue: Job queue for accessing job data
            file_manager: File manager for file operations
            cleanup_interval_minutes: How often to run cleanup (in minutes)
        """
        self.job_queue = job_queue
        self.file_manager = file_manager
        self.error_handler = ErrorHandler(file_manager, job_queue)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        # Cleanup configuration
        self.failed_job_retention_hours = 24  # Keep failed jobs for 24 hours
        self.temp_file_max_age_hours = 2      # Clean temp files after 2 hours
        self.completed_job_retention_hours = 48  # Keep completed jobs for 48 hours
        
        # Service state
        self.is_running = False
        self.last_cleanup = None
        self.cleanup_stats = {
            'total_cleanups': 0,
            'failed_jobs_cleaned': 0,
            'temp_files_cleaned': 0,
            'completed_jobs_cleaned': 0,
            'errors_encountered': 0
        }
    
    async def start(self) -> None:
        """Start the cleanup service."""
        if self.is_running:
            logger.warning("Cleanup service is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting cleanup service with {self.cleanup_interval.total_seconds()/60:.1f} minute intervals")
        
        try:
            while self.is_running:
                await self._run_cleanup_cycle()
                await asyncio.sleep(self.cleanup_interval.total_seconds())
        except asyncio.CancelledError:
            logger.info("Cleanup service was cancelled")
        except Exception as e:
            logger.error(f"Cleanup service encountered an error: {e}")
        finally:
            self.is_running = False
    
    def stop(self) -> None:
        """Stop the cleanup service."""
        self.is_running = False
        logger.info("Cleanup service stop requested")
    
    async def _run_cleanup_cycle(self) -> None:
        """Run a single cleanup cycle."""
        try:
            logger.debug("Starting cleanup cycle")
            cycle_start = datetime.now()
            
            # Clean up failed jobs
            failed_cleaned = await self._cleanup_failed_jobs()
            
            # Clean up temporary files
            temp_cleaned = await self._cleanup_temp_files()
            
            # Clean up old completed jobs
            completed_cleaned = await self._cleanup_completed_jobs()
            
            # Update statistics
            self.cleanup_stats['total_cleanups'] += 1
            self.cleanup_stats['failed_jobs_cleaned'] += failed_cleaned
            self.cleanup_stats['temp_files_cleaned'] += temp_cleaned
            self.cleanup_stats['completed_jobs_cleaned'] += completed_cleaned
            self.last_cleanup = cycle_start
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(
                f"Cleanup cycle completed in {cycle_duration:.2f}s: "
                f"{failed_cleaned} failed jobs, {temp_cleaned} temp files, "
                f"{completed_cleaned} completed jobs cleaned"
            )
            
        except Exception as e:
            self.cleanup_stats['errors_encountered'] += 1
            logger.error(f"Error during cleanup cycle: {e}")
    
    async def _cleanup_failed_jobs(self) -> int:
        """
        Clean up files from failed processing jobs.
        
        Returns:
            Number of failed jobs cleaned up
        """
        try:
            # Get queue statistics to find failed jobs
            queue_stats = self.job_queue.get_queue_stats()
            failed_job_count = queue_stats.get('failed', 0)
            
            if failed_job_count == 0:
                return 0
            
            # Get all job keys to find failed jobs
            # Note: This is a simplified approach. In production, you might want
            # to maintain a separate index of failed jobs for efficiency
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=self.failed_job_retention_hours)
            
            # This would need to be implemented based on your Redis key structure
            # For now, we'll use a placeholder approach
            failed_jobs = await self._get_expired_failed_jobs(cutoff_time)
            
            for job in failed_jobs:
                if self.error_handler.cleanup_failed_job(job.job_id):
                    cleaned_count += 1
                    logger.debug(f"Cleaned up failed job: {job.job_id}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up failed jobs: {e}")
            return 0
    
    async def _cleanup_temp_files(self) -> int:
        """
        Clean up expired temporary files.
        
        Returns:
            Number of temporary files cleaned up
        """
        try:
            cleaned_count = self.file_manager.cleanup_temp_files(
                max_age_hours=self.temp_file_max_age_hours
            )
            
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} temporary files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
            return 0
    
    async def _cleanup_completed_jobs(self) -> int:
        """
        Clean up old completed job data and associated files.
        
        Returns:
            Number of completed jobs cleaned up
        """
        try:
            # Clean up completed jobs from the queue
            cleaned_count = self.job_queue.cleanup_completed_jobs(
                older_than_hours=self.completed_job_retention_hours
            )
            
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} completed job records")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up completed jobs: {e}")
            return 0
    
    async def _get_expired_failed_jobs(self, cutoff_time: datetime) -> List[ProcessingJob]:
        """
        Get failed jobs that are older than the cutoff time.
        
        Args:
            cutoff_time: Jobs older than this time will be returned
            
        Returns:
            List of expired failed jobs
        """
        # This is a placeholder implementation
        # In a real implementation, you would query Redis for failed jobs
        # and filter by completion time
        
        try:
            # For now, return empty list as this requires more complex Redis queries
            # In production, you might maintain separate indexes for different job states
            return []
            
        except Exception as e:
            logger.error(f"Error getting expired failed jobs: {e}")
            return []
    
    def force_cleanup(self) -> Dict[str, int]:
        """
        Force an immediate cleanup cycle (synchronous).
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            logger.info("Running forced cleanup cycle")
            
            # Clean up failed jobs (simplified synchronous version)
            failed_cleaned = 0
            try:
                # This would need proper implementation
                pass
            except Exception as e:
                logger.error(f"Error in forced failed job cleanup: {e}")
            
            # Clean up temporary files
            temp_cleaned = self.file_manager.cleanup_temp_files(
                max_age_hours=self.temp_file_max_age_hours
            )
            
            # Clean up completed jobs
            completed_cleaned = 0
            try:
                completed_cleaned = self.job_queue.cleanup_completed_jobs(
                    older_than_hours=self.completed_job_retention_hours
                )
            except Exception as e:
                logger.error(f"Error in forced completed job cleanup: {e}")
            
            results = {
                'failed_jobs_cleaned': failed_cleaned,
                'temp_files_cleaned': temp_cleaned,
                'completed_jobs_cleaned': completed_cleaned,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Forced cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during forced cleanup: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """
        Get cleanup service statistics.
        
        Returns:
            Dictionary with cleanup statistics
        """
        return {
            'is_running': self.is_running,
            'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None,
            'cleanup_interval_minutes': self.cleanup_interval.total_seconds() / 60,
            'retention_settings': {
                'failed_job_retention_hours': self.failed_job_retention_hours,
                'temp_file_max_age_hours': self.temp_file_max_age_hours,
                'completed_job_retention_hours': self.completed_job_retention_hours
            },
            'statistics': self.cleanup_stats.copy()
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Update cleanup service configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if 'failed_job_retention_hours' in kwargs:
            self.failed_job_retention_hours = kwargs['failed_job_retention_hours']
        
        if 'temp_file_max_age_hours' in kwargs:
            self.temp_file_max_age_hours = kwargs['temp_file_max_age_hours']
        
        if 'completed_job_retention_hours' in kwargs:
            self.completed_job_retention_hours = kwargs['completed_job_retention_hours']
        
        if 'cleanup_interval_minutes' in kwargs:
            self.cleanup_interval = timedelta(minutes=kwargs['cleanup_interval_minutes'])
        
        logger.info(f"Updated cleanup service configuration: {kwargs}")


# Global cleanup service instance
_cleanup_service: CleanupService = None


def get_cleanup_service() -> CleanupService:
    """Get the global cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        raise RuntimeError("Cleanup service not initialized. Call initialize_cleanup_service() first.")
    return _cleanup_service


def initialize_cleanup_service(job_queue: JobQueue, file_manager: FileManager, 
                             cleanup_interval_minutes: int = 30) -> CleanupService:
    """
    Initialize the global cleanup service.
    
    Args:
        job_queue: Job queue instance
        file_manager: File manager instance
        cleanup_interval_minutes: Cleanup interval in minutes
        
    Returns:
        Initialized cleanup service
    """
    global _cleanup_service
    _cleanup_service = CleanupService(job_queue, file_manager, cleanup_interval_minutes)
    return _cleanup_service


async def start_cleanup_service() -> None:
    """Start the global cleanup service."""
    service = get_cleanup_service()
    await service.start()


def stop_cleanup_service() -> None:
    """Stop the global cleanup service."""
    global _cleanup_service
    if _cleanup_service:
        _cleanup_service.stop()