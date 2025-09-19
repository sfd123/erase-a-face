"""
Centralized error handling for the Golf Video Anonymizer API.

This module provides comprehensive error handling, cleanup mechanisms,
and user-friendly error messages for various failure scenarios.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from models.processing_job import ProcessingJob, JobStatus
from models.validation import ValidationError
from storage.file_manager import FileManager, FileValidationError
from storage.job_queue import JobQueue, JobQueueError
from processing.video_processor import VideoProcessingError

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling and cleanup for the video processing service."""
    
    def __init__(self, file_manager: FileManager, job_queue: JobQueue):
        """
        Initialize error handler with required services.
        
        Args:
            file_manager: File manager for cleanup operations
            job_queue: Job queue for job status updates
        """
        self.file_manager = file_manager
        self.job_queue = job_queue
    
    def handle_upload_error(self, error: Exception, job: Optional[ProcessingJob] = None, 
                          temp_files: Optional[List[Path]] = None) -> HTTPException:
        """
        Handle errors during file upload with appropriate cleanup.
        
        Args:
            error: The exception that occurred
            job: Processing job if created
            temp_files: List of temporary files to clean up
            
        Returns:
            HTTPException with appropriate status code and message
        """
        # Clean up temporary files
        if temp_files:
            for temp_file in temp_files:
                self._safe_cleanup_file(temp_file)
        
        # Update job status if job was created
        if job:
            try:
                self.job_queue.update_job_status(
                    job.job_id, 
                    JobStatus.FAILED, 
                    error_message=f"Upload failed: {str(error)}"
                )
            except Exception as update_error:
                logger.error(f"Failed to update job status after upload error: {update_error}")
        
        # Return appropriate HTTP exception based on error type
        if isinstance(error, ValidationError):
            return HTTPException(
                status_code=400,
                detail={
                    "error": "validation_error",
                    "message": f"File validation failed: {str(error)}",
                    "help": "Please check that your file is a valid video in a supported format.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        elif isinstance(error, FileValidationError):
            return HTTPException(
                status_code=422,
                detail={
                    "error": "file_validation_error", 
                    "message": f"File content validation failed: {str(error)}",
                    "help": "The file appears to be corrupted or not a valid video file.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        elif isinstance(error, JobQueueError):
            return HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Processing queue is currently unavailable. Please try again later.",
                    "help": "This is a temporary issue. Please wait a few minutes and try again.",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        else:
            logger.error(f"Unexpected upload error: {error}", exc_info=True)
            return HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "An unexpected error occurred during file upload.",
                    "help": "Please try again. If the problem persists, contact support.",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def handle_processing_error(self, error: Exception, job: ProcessingJob) -> None:
        """
        Handle errors during video processing with cleanup and job status update.
        
        Args:
            error: The exception that occurred during processing
            job: The processing job that failed
        """
        error_message = self._get_user_friendly_error_message(error)
        
        try:
            # Update job status to failed
            job.mark_failed(error_message)
            self.job_queue.update_job_status(
                job.job_id,
                JobStatus.FAILED,
                error_message=error_message
            )
            
            # Clean up any partial output files
            if job.output_file_path:
                self._safe_cleanup_file(Path(job.output_file_path))
            
            # Clean up input file if it exists
            if job.file_path:
                self._safe_cleanup_file(Path(job.file_path))
            
            # Clean up any temporary files for this job
            self.file_manager.cleanup_temp_files(job.job_id)
            
            logger.info(f"Cleaned up failed job {job.job_id}: {error_message}")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up after processing error for job {job.job_id}: {cleanup_error}")
    
    def handle_no_faces_detected(self, job: ProcessingJob, input_path: str, output_path: str) -> None:
        """
        Handle the case where no faces were detected in the video.
        
        Args:
            job: The processing job
            input_path: Path to the input video
            output_path: Path where output should be saved
        """
        try:
            # Copy original file to output location (no processing needed)
            import shutil
            shutil.copy2(input_path, output_path)
            
            # Mark job as completed with 0 faces detected
            job.mark_completed(output_path, faces_detected=0)
            self.job_queue.update_job_status(
                job.job_id,
                JobStatus.COMPLETED,
                output_file_path=output_path,
                faces_detected=0
            )
            
            logger.info(f"Job {job.job_id} completed with no faces detected - original video preserved")
            
        except Exception as e:
            error_msg = f"Failed to handle no-faces-detected case: {str(e)}"
            self.handle_processing_error(Exception(error_msg), job)
    
    def handle_corrupted_video_error(self, job: ProcessingJob) -> None:
        """
        Handle corrupted video file errors with specific messaging.
        
        Args:
            job: The processing job with corrupted video
        """
        error_message = (
            "The uploaded video file appears to be corrupted or in an unsupported format. "
            "Please try uploading a different video file."
        )
        
        try:
            job.mark_failed(error_message)
            self.job_queue.update_job_status(
                job.job_id,
                JobStatus.FAILED,
                error_message=error_message
            )
            
            # Clean up the corrupted file
            if job.file_path:
                self._safe_cleanup_file(Path(job.file_path))
            
            self.file_manager.cleanup_temp_files(job.job_id)
            
            logger.warning(f"Job {job.job_id} failed due to corrupted video file")
            
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up corrupted video job {job.job_id}: {cleanup_error}")
    
    def cleanup_failed_job(self, job_id: str) -> bool:
        """
        Perform comprehensive cleanup for a failed job.
        
        Args:
            job_id: ID of the failed job to clean up
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Get job details
            job = self.job_queue.get_job_status(job_id)
            if not job:
                logger.warning(f"Cannot cleanup job {job_id}: job not found")
                return False
            
            cleanup_count = 0
            
            # Clean up input file
            if job.file_path:
                if self._safe_cleanup_file(Path(job.file_path)):
                    cleanup_count += 1
            
            # Clean up output file if it exists
            if job.output_file_path:
                if self._safe_cleanup_file(Path(job.output_file_path)):
                    cleanup_count += 1
            
            # Clean up temporary files
            temp_cleanup_count = self.file_manager.cleanup_temp_files(job_id)
            cleanup_count += temp_cleanup_count
            
            logger.info(f"Cleaned up {cleanup_count} files for failed job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            return False
    
    def _safe_cleanup_file(self, file_path: Path) -> bool:
        """
        Safely delete a file with error handling.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if file was deleted or didn't exist, False if deletion failed
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
                return True
            return True  # File doesn't exist, consider it cleaned up
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
            return False
    
    def _get_user_friendly_error_message(self, error: Exception) -> str:
        """
        Convert technical error messages to user-friendly messages.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_str = str(error).lower()
        
        if isinstance(error, VideoProcessingError):
            if "could not open" in error_str or "metadata" in error_str:
                return "The video file could not be processed. It may be corrupted or in an unsupported format."
            elif "no faces" in error_str:
                return "No faces were detected in the video. The original video will be returned unchanged."
            elif "memory" in error_str or "size" in error_str:
                return "The video file is too large or complex to process. Please try a smaller video."
            elif "codec" in error_str or "encoding" in error_str:
                return "There was an issue with the video encoding. Please try a different video format."
            else:
                return "Video processing failed due to a technical issue. Please try again with a different video."
        
        elif "redis" in error_str or "connection" in error_str:
            return "The processing service is temporarily unavailable. Please try again in a few minutes."
        
        elif "disk" in error_str or "space" in error_str:
            return "Insufficient storage space to process the video. Please try again later."
        
        elif "timeout" in error_str:
            return "Video processing timed out. The video may be too long or complex. Please try a shorter video."
        
        else:
            return "An unexpected error occurred during video processing. Please try again."
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors and cleanup operations.
        
        Returns:
            Dictionary with error statistics
        """
        try:
            queue_stats = self.job_queue.get_queue_stats()
            storage_stats = self.file_manager.get_storage_stats()
            
            return {
                "failed_jobs": queue_stats.get("failed", 0),
                "retry_jobs": queue_stats.get("retry_jobs", 0),
                "temp_files": storage_stats.get("temp_files", 0),
                "temp_size_mb": storage_stats.get("temp_size_mb", 0),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {
                "error": "Failed to retrieve error statistics",
                "last_updated": datetime.now().isoformat()
            }


def create_error_response(error_type: str, message: str, status_code: int = 500, 
                         help_text: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        error_type: Type of error (e.g., "validation_error", "processing_error")
        message: Error message
        status_code: HTTP status code
        help_text: Optional help text for the user
        details: Optional additional details
        
    Returns:
        JSONResponse with standardized error format
    """
    response_data = {
        "error": error_type,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if help_text:
        response_data["help"] = help_text
    
    if details:
        response_data["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


def setup_error_handlers(app):
    """
    Setup global error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    import logging
    
    logger = logging.getLogger(__name__)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    logger.info("Error handlers setup complete")