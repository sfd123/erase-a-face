"""
REST API handlers for the Golf Video Anonymizer service.

This module contains the FastAPI route handlers for video upload,
processing status queries, and video downloads.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from models.processing_job import ProcessingJob, JobStatus
from models.validation import ValidationError, validate_filename, get_supported_formats, get_max_file_size_mb
from storage.job_queue import JobQueue, JobQueueError
from storage.file_manager import FileManager, FileValidationError

logger = logging.getLogger(__name__)


# Response models
class UploadResponse(BaseModel):
    """Response model for video upload."""
    job_id: str
    message: str
    original_filename: str
    status: str


class StatusResponse(BaseModel):
    """Response model for job status queries."""
    job_id: str
    status: str
    original_filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    faces_detected: Optional[int] = None
    processing_duration: Optional[float] = None


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str
    supported_formats: list
    max_file_size_mb: float


# Dependency injection
def get_job_queue() -> JobQueue:
    """Get JobQueue instance."""
    return JobQueue()


def get_file_manager() -> FileManager:
    """Get FileManager instance."""
    return FileManager()


class VideoUploadHandler:
    """Handler for video file uploads."""
    
    def __init__(self, job_queue: JobQueue, file_manager: FileManager):
        self.job_queue = job_queue
        self.file_manager = file_manager
    
    async def upload_video(
        self, 
        background_tasks: BackgroundTasks,
        file: UploadFile,
        job_queue: JobQueue,
        file_manager: FileManager
    ) -> UploadResponse:
        """
        Handle video file upload with validation and job creation.
        
        Args:
            background_tasks: FastAPI background tasks
            file: Uploaded video file
            job_queue: Job queue instance
            file_manager: File manager instance
            
        Returns:
            UploadResponse with job details
            
        Raises:
            HTTPException: For various validation and processing errors
        """
        try:
            # Validate filename
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "validation_error",
                        "message": "No filename provided"
                    }
                )
            
            validate_filename(file.filename)
            
            # Check file size
            if file.size and file.size > file_manager.MAX_FILE_SIZE:
                max_size_mb = get_max_file_size_mb()
                current_size_mb = file.size / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": "file_too_large",
                        "message": f"File size {current_size_mb:.1f} MB exceeds maximum allowed size {max_size_mb:.1f} MB"
                    }
                )
            
            # Create processing job
            job = ProcessingJob.create_new(
                original_filename=file.filename,
                file_path=""  # Will be set after file storage
            )
            
            # Create temporary file for upload
            temp_file = file_manager.create_temp_file(job.job_id, Path(file.filename).suffix)
            
            # Save uploaded file
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Store file securely with validation
            stored_file_path = file_manager.store_uploaded_file(
                temp_file, job.job_id, file.filename
            )
            
            # Update job with stored file path
            job.file_path = str(stored_file_path)
            
            # Enqueue job for processing
            job_queue.enqueue_job(job)
            
            # Schedule cleanup of temp file
            background_tasks.add_task(
                file_manager.cleanup_temp_files, 
                job.job_id
            )
            
            logger.info(f"Successfully uploaded and queued job {job.job_id} for file {file.filename}")
            
            return UploadResponse(
                job_id=job.job_id,
                message="Video uploaded successfully and queued for processing",
                original_filename=file.filename,
                status=job.status.value
            )
            
        except ValidationError as e:
            # Clean up temp file on validation error
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup temp file after validation error: {cleanup_error}")
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "validation_error",
                    "message": f"File validation failed: {str(e)}",
                    "supported_formats": get_supported_formats(),
                    "max_file_size_mb": get_max_file_size_mb(),
                    "help": "Please ensure your file is a valid video in one of the supported formats and under the size limit."
                }
            )
        
        except FileValidationError as e:
            # Clean up temp file on validation error
            if 'temp_file' in locals() and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup temp file after validation error: {cleanup_error}")
            
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "file_validation_error",
                    "message": f"File content validation failed: {str(e)}",
                    "supported_formats": get_supported_formats(),
                    "max_file_size_mb": get_max_file_size_mb(),
                    "help": "The file appears to be corrupted or not a valid video file. Please try uploading a different file."
                }
            )
        
        except JobQueueError as e:
            logger.error(f"Job queue error during upload: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Processing queue is currently unavailable. Please try again later."
                }
            )
        
        except HTTPException:
            # Re-raise HTTP exceptions (like 400, 422 for validation errors)
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "An unexpected error occurred during file upload"
                }
            )


class ProcessingStatusHandler:
    """Handler for processing status queries."""
    
    def __init__(self, job_queue: JobQueue):
        self.job_queue = job_queue
    
    async def get_job_status(
        self, 
        job_id: str,
        job_queue: JobQueue
    ) -> StatusResponse:
        """
        Get processing status for a job.
        
        Args:
            job_id: Job identifier
            job_queue: Job queue instance
            
        Returns:
            StatusResponse with job status details
            
        Raises:
            HTTPException: If job not found or queue error
        """
        try:
            job = job_queue.get_job_status(job_id)
            
            if not job:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "job_not_found",
                        "message": f"Job with ID {job_id} not found"
                    }
                )
            
            return StatusResponse(
                job_id=job.job_id,
                status=job.status.value,
                original_filename=job.original_filename,
                created_at=job.created_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
                faces_detected=job.faces_detected if job.status == JobStatus.COMPLETED else None,
                processing_duration=job.processing_duration
            )
            
        except JobQueueError as e:
            logger.error(f"Job queue error during status query: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Status service is currently unavailable. Please try again later."
                }
            )
        
        except HTTPException:
            # Re-raise HTTP exceptions (like 404 for job not found)
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during status query: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "An unexpected error occurred while retrieving job status"
                }
            )


class VideoDownloadHandler:
    """Handler for processed video downloads."""
    
    def __init__(self, job_queue: JobQueue, file_manager: FileManager):
        self.job_queue = job_queue
        self.file_manager = file_manager
    
    async def download_video(
        self, 
        job_id: str,
        job_queue: JobQueue,
        file_manager: FileManager
    ) -> FileResponse:
        """
        Download processed video file.
        
        Args:
            job_id: Job identifier
            job_queue: Job queue instance
            file_manager: File manager instance
            
        Returns:
            FileResponse with the processed video file
            
        Raises:
            HTTPException: For various error conditions
        """
        try:
            job = job_queue.get_job_status(job_id)
            
            if not job:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "job_not_found",
                        "message": f"Job with ID {job_id} not found"
                    }
                )
            
            if job.status != JobStatus.COMPLETED:
                if job.status == JobStatus.FAILED:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "processing_failed",
                            "message": f"Video processing failed: {job.error_message or 'Unknown error'}"
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=202,
                        detail={
                            "error": "processing_incomplete",
                            "message": f"Video is still being processed. Current status: {job.status.value}"
                        }
                    )
            
            if not job.output_file_path:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "missing_output",
                        "message": "Processed video file path is missing"
                    }
                )
            
            output_path = Path(job.output_file_path)
            
            if not output_path.exists():
                logger.error(f"Output file missing for job {job_id}: {output_path}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "file_not_found",
                        "message": "Processed video file not found. It may have been cleaned up."
                    }
                )
            
            # Create download filename
            original_name = Path(job.original_filename)
            download_filename = f"anonymized_{original_name.stem}{original_name.suffix}"
            
            # Return file response
            return FileResponse(
                path=str(output_path),
                filename=download_filename,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={download_filename}",
                    "X-Job-ID": job_id,
                    "X-Faces-Detected": str(job.faces_detected or 0)
                }
            )
            
        except JobQueueError as e:
            logger.error(f"Job queue error during download: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Download service is currently unavailable. Please try again later."
                }
            )
        
        except HTTPException:
            # Re-raise HTTP exceptions (like 404 for job not found)
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "An unexpected error occurred during file download"
                }
            )


class HealthCheckHandler:
    """Handler for health check and service information."""
    
    def __init__(self, job_queue: JobQueue):
        self.job_queue = job_queue
    
    async def health_check(
        self,
        job_queue: JobQueue
    ) -> HealthResponse:
        """
        Perform health check and return service information.
        
        Args:
            job_queue: Job queue instance
            
        Returns:
            HealthResponse with service health information
        """
        # Check Redis connection
        redis_healthy = job_queue.health_check()
        
        status = "healthy" if redis_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version="1.0.0",
            supported_formats=get_supported_formats(),
            max_file_size_mb=get_max_file_size_mb()
        )