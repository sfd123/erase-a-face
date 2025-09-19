"""
Batch processing service for handling multiple video processing jobs efficiently.

This module provides batch processing capabilities to handle multiple video
processing jobs concurrently while managing system resources and providing
progress updates for long-running operations.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from models.processing_job import ProcessingJob, JobStatus
from processing.video_processor import VideoProcessor, VideoProcessorConfig
from storage.job_queue import JobQueue
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessorConfig:
    """Configuration for batch processing operations."""
    
    # Concurrency settings
    max_concurrent_jobs: int = 2  # Maximum number of jobs to process simultaneously
    max_worker_threads: int = 4   # Maximum worker threads for the executor
    
    # Resource management
    memory_limit_per_job_mb: int = 512  # Memory limit per job in MB
    cpu_usage_threshold: float = 0.8    # CPU usage threshold to throttle processing
    
    # Progress reporting
    progress_update_interval: int = 5   # Update progress every N seconds
    enable_detailed_logging: bool = True
    
    # Job prioritization
    enable_job_prioritization: bool = True
    priority_by_file_size: bool = True  # Prioritize smaller files first
    
    # Error handling
    max_retry_attempts: int = 2
    retry_delay_seconds: int = 30


class BatchProgressTracker:
    """Tracks progress across multiple concurrent jobs."""
    
    def __init__(self):
        self.jobs_progress: Dict[str, float] = {}
        self.jobs_status: Dict[str, str] = {}
        self.start_time = time.time()
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.total_jobs = 0
        self._lock = threading.Lock()
    
    def update_job_progress(self, job_id: str, progress: float, status: str = "") -> None:
        """Update progress for a specific job."""
        with self._lock:
            self.jobs_progress[job_id] = progress
            if status:
                self.jobs_status[job_id] = status
    
    def mark_job_completed(self, job_id: str, success: bool = True) -> None:
        """Mark a job as completed."""
        with self._lock:
            self.jobs_progress[job_id] = 1.0
            if success:
                self.completed_jobs += 1
                self.jobs_status[job_id] = "Completed successfully"
            else:
                self.failed_jobs += 1
                self.jobs_status[job_id] = "Failed"
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall batch processing progress."""
        with self._lock:
            total_progress = sum(self.jobs_progress.values())
            avg_progress = total_progress / len(self.jobs_progress) if self.jobs_progress else 0.0
            
            elapsed_time = time.time() - self.start_time
            
            return {
                'overall_progress': avg_progress,
                'completed_jobs': self.completed_jobs,
                'failed_jobs': self.failed_jobs,
                'total_jobs': self.total_jobs,
                'elapsed_time': elapsed_time,
                'jobs_in_progress': len([p for p in self.jobs_progress.values() if 0 < p < 1.0]),
                'individual_progress': dict(self.jobs_progress),
                'job_statuses': dict(self.jobs_status)
            }


class BatchProcessor:
    """
    Batch processor for handling multiple video processing jobs efficiently.
    
    This class manages concurrent processing of multiple video jobs while
    monitoring system resources and providing progress updates.
    """
    
    def __init__(self, config: Optional[BatchProcessorConfig] = None,
                 video_processor_config: Optional[VideoProcessorConfig] = None,
                 job_queue: Optional[JobQueue] = None,
                 file_manager: Optional[FileManager] = None):
        """
        Initialize batch processor.
        
        Args:
            config: Batch processing configuration
            video_processor_config: Video processor configuration
            job_queue: Job queue for managing jobs
            file_manager: File manager for handling files
        """
        self.config = config or BatchProcessorConfig()
        self.video_processor_config = video_processor_config or VideoProcessorConfig()
        self.job_queue = job_queue or JobQueue()
        self.file_manager = file_manager or FileManager()
        
        # Processing state
        self.is_processing = False
        self.progress_tracker = BatchProgressTracker()
        self._shutdown_requested = False
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
    
    def process_jobs_batch(self, jobs: List[ProcessingJob],
                          progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Process a batch of jobs concurrently.
        
        Args:
            jobs: List of processing jobs to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch processing results
        """
        if self.is_processing:
            raise RuntimeError("Batch processor is already running")
        
        self.is_processing = True
        self._shutdown_requested = False
        
        try:
            # Initialize progress tracking
            self.progress_tracker = BatchProgressTracker()
            self.progress_tracker.total_jobs = len(jobs)
            
            # Prioritize jobs if enabled
            if self.config.enable_job_prioritization:
                jobs = self._prioritize_jobs(jobs)
            
            logger.info(f"Starting batch processing of {len(jobs)} jobs")
            
            # Start progress reporting thread
            progress_thread = None
            if progress_callback:
                progress_thread = threading.Thread(
                    target=self._progress_reporter,
                    args=(progress_callback,),
                    daemon=True
                )
                progress_thread.start()
            
            # Process jobs with controlled concurrency
            results = self._process_jobs_concurrent(jobs)
            
            # Wait for progress thread to finish
            if progress_thread:
                progress_thread.join(timeout=5)
            
            # Generate final results
            final_results = self._generate_batch_results(jobs, results)
            
            logger.info(f"Batch processing completed. Success: {self.progress_tracker.completed_jobs}, "
                       f"Failed: {self.progress_tracker.failed_jobs}")
            
            return final_results
            
        finally:
            self.is_processing = False
            self._shutdown_requested = True
    
    def _prioritize_jobs(self, jobs: List[ProcessingJob]) -> List[ProcessingJob]:
        """
        Prioritize jobs based on configuration settings.
        
        Args:
            jobs: List of jobs to prioritize
            
        Returns:
            Prioritized list of jobs
        """
        if not self.config.priority_by_file_size:
            return jobs
        
        # Sort by file size (smaller files first for faster feedback)
        def get_file_size(job: ProcessingJob) -> int:
            try:
                return Path(job.file_path).stat().st_size
            except Exception:
                return 0
        
        prioritized = sorted(jobs, key=get_file_size)
        logger.info("Jobs prioritized by file size (smallest first)")
        return prioritized
    
    def _process_jobs_concurrent(self, jobs: List[ProcessingJob]) -> Dict[str, Any]:
        """
        Process jobs with controlled concurrency.
        
        Args:
            jobs: List of jobs to process
            
        Returns:
            Dictionary mapping job IDs to results
        """
        results = {}
        active_futures = {}
        
        # Submit initial batch of jobs
        job_iter = iter(jobs)
        for _ in range(min(self.config.max_concurrent_jobs, len(jobs))):
            try:
                job = next(job_iter)
                future = self.executor.submit(self._process_single_job, job)
                active_futures[future] = job
            except StopIteration:
                break
        
        # Process completed jobs and submit new ones
        while active_futures and not self._shutdown_requested:
            # Wait for at least one job to complete
            completed_futures = as_completed(active_futures.keys(), timeout=1.0)
            
            for future in completed_futures:
                job = active_futures.pop(future)
                
                try:
                    result = future.result()
                    results[job.job_id] = result
                    self.progress_tracker.mark_job_completed(job.job_id, success=result['success'])
                    
                    if result['success']:
                        logger.info(f"Job {job.job_id} completed successfully")
                    else:
                        logger.error(f"Job {job.job_id} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Job {job.job_id} failed with exception: {e}")
                    results[job.job_id] = {'success': False, 'error': str(e)}
                    self.progress_tracker.mark_job_completed(job.job_id, success=False)
                
                # Submit next job if available
                try:
                    next_job = next(job_iter)
                    future = self.executor.submit(self._process_single_job, next_job)
                    active_futures[future] = next_job
                except StopIteration:
                    pass  # No more jobs to submit
        
        return results
    
    def _process_single_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """
        Process a single job with error handling and progress tracking.
        
        Args:
            job: Processing job to execute
            
        Returns:
            Dictionary with job result information
        """
        job_start_time = time.time()
        
        try:
            # Create video processor for this job
            processor = VideoProcessor(
                config=self.video_processor_config,
                file_manager=self.file_manager
            )
            
            # Create progress callback for this job
            def job_progress_callback(progress: float, message: str):
                self.progress_tracker.update_job_progress(job.job_id, progress, message)
            
            # Process the video
            output_path = processor.process_video(job, job_progress_callback)
            
            # Get processing statistics
            stats = processor.get_processing_stats()
            
            processing_time = time.time() - job_start_time
            
            return {
                'success': True,
                'output_path': output_path,
                'processing_time': processing_time,
                'faces_detected': job.faces_detected,
                'stats': stats
            }
            
        except Exception as e:
            processing_time = time.time() - job_start_time
            error_msg = str(e)
            
            logger.error(f"Job {job.job_id} failed after {processing_time:.2f}s: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time
            }
    
    def _progress_reporter(self, progress_callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Background thread for reporting progress updates.
        
        Args:
            progress_callback: Callback function for progress updates
        """
        while not self._shutdown_requested and self.is_processing:
            try:
                progress_data = self.progress_tracker.get_overall_progress()
                progress_callback(progress_data)
                time.sleep(self.config.progress_update_interval)
            except Exception as e:
                logger.error(f"Error in progress reporting: {e}")
                break
    
    def _generate_batch_results(self, jobs: List[ProcessingJob], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing results.
        
        Args:
            jobs: Original list of jobs
            results: Processing results for each job
            
        Returns:
            Comprehensive batch results dictionary
        """
        successful_jobs = [job_id for job_id, result in results.items() if result.get('success', False)]
        failed_jobs = [job_id for job_id, result in results.items() if not result.get('success', False)]
        
        total_processing_time = sum(result.get('processing_time', 0) for result in results.values())
        total_faces_detected = sum(result.get('faces_detected', 0) for result in results.values())
        
        batch_results = {
            'batch_summary': {
                'total_jobs': len(jobs),
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs),
                'success_rate': len(successful_jobs) / len(jobs) if jobs else 0,
                'total_processing_time': total_processing_time,
                'total_faces_detected': total_faces_detected,
                'batch_start_time': self.progress_tracker.start_time,
                'batch_duration': time.time() - self.progress_tracker.start_time
            },
            'job_results': results,
            'successful_job_ids': successful_jobs,
            'failed_job_ids': failed_jobs
        }
        
        return batch_results
    
    def process_queue_continuously(self, max_jobs_per_batch: int = 10,
                                 batch_timeout_seconds: int = 60,
                                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        Continuously process jobs from the queue in batches.
        
        Args:
            max_jobs_per_batch: Maximum number of jobs to process in one batch
            batch_timeout_seconds: Maximum time to wait for a full batch
            progress_callback: Optional callback for progress updates
        """
        logger.info("Starting continuous batch processing from queue")
        
        while not self._shutdown_requested:
            try:
                # Collect jobs for batch processing
                jobs = self._collect_jobs_for_batch(max_jobs_per_batch, batch_timeout_seconds)
                
                if not jobs:
                    time.sleep(5)  # Wait before checking for more jobs
                    continue
                
                # Process the batch
                batch_results = self.process_jobs_batch(jobs, progress_callback)
                
                # Update job statuses in queue
                self._update_job_statuses_in_queue(batch_results)
                
            except Exception as e:
                logger.error(f"Error in continuous batch processing: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _collect_jobs_for_batch(self, max_jobs: int, timeout_seconds: int) -> List[ProcessingJob]:
        """
        Collect jobs from the queue for batch processing.
        
        Args:
            max_jobs: Maximum number of jobs to collect
            timeout_seconds: Maximum time to wait for jobs
            
        Returns:
            List of jobs for batch processing
        """
        jobs = []
        start_time = time.time()
        
        while len(jobs) < max_jobs and (time.time() - start_time) < timeout_seconds:
            job = self.job_queue.dequeue_job(timeout=5)
            if job:
                jobs.append(job)
            else:
                break  # No more jobs available
        
        return jobs
    
    def _update_job_statuses_in_queue(self, batch_results: Dict[str, Any]) -> None:
        """
        Update job statuses in the queue based on batch results.
        
        Args:
            batch_results: Results from batch processing
        """
        job_results = batch_results.get('job_results', {})
        
        for job_id, result in job_results.items():
            try:
                if result.get('success', False):
                    self.job_queue.update_job_status(
                        job_id,
                        JobStatus.COMPLETED,
                        output_file_path=result.get('output_path'),
                        faces_detected=result.get('faces_detected', 0)
                    )
                else:
                    self.job_queue.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message=result.get('error', 'Unknown error')
                    )
            except Exception as e:
                logger.error(f"Failed to update job status for {job_id}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the batch processor gracefully."""
        logger.info("Shutting down batch processor")
        self._shutdown_requested = True
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("Batch processor shutdown complete")
    
    def get_system_resources(self) -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with system resource information
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return {}