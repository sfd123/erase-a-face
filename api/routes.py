"""
FastAPI routes for the Golf Video Anonymizer API.

This module defines the REST API endpoints and integrates them
with the handler classes.
"""

import logging
from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

from .handlers import (
    VideoUploadHandler, 
    ProcessingStatusHandler, 
    VideoDownloadHandler,
    HealthCheckHandler,
    UploadResponse,
    StatusResponse,
    HealthResponse,
    get_job_queue,
    get_file_manager
)
from storage.job_queue import JobQueue
from storage.file_manager import FileManager

# Create API router
api_router = APIRouter(prefix="/api/v1", tags=["video-anonymizer"])


@api_router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload video for processing",
    description="Upload a video file to be processed for face anonymization. "
                "Supported formats: MP4, MOV, AVI. Maximum file size: 500MB."
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process"),
    job_queue: JobQueue = Depends(get_job_queue),
    file_manager: FileManager = Depends(get_file_manager)
) -> UploadResponse:
    """Upload video file for face anonymization processing."""
    handler = VideoUploadHandler(job_queue, file_manager)
    return await handler.upload_video(background_tasks, file, job_queue, file_manager)


@api_router.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Get processing status",
    description="Get the current processing status of a video anonymization job."
)
async def get_processing_status(
    job_id: str,
    job_queue: JobQueue = Depends(get_job_queue)
) -> StatusResponse:
    """Get processing status for a specific job."""
    handler = ProcessingStatusHandler(job_queue)
    return await handler.get_job_status(job_id, job_queue)


@api_router.get(
    "/download/{job_id}",
    response_class=FileResponse,
    summary="Download processed video",
    description="Download the processed video file with anonymized faces. "
                "Only available for completed jobs."
)
async def download_processed_video(
    job_id: str,
    job_queue: JobQueue = Depends(get_job_queue),
    file_manager: FileManager = Depends(get_file_manager)
) -> FileResponse:
    """Download processed video file."""
    handler = VideoDownloadHandler(job_queue, file_manager)
    return await handler.download_video(job_id, job_queue, file_manager)


@api_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and get configuration information."
)
async def health_check(
    job_queue: JobQueue = Depends(get_job_queue)
) -> HealthResponse:
    """Perform health check and return service information."""
    handler = HealthCheckHandler(job_queue)
    return await handler.health_check(job_queue)


# Additional utility endpoints
@api_router.get(
    "/queue/stats",
    summary="Get queue statistics",
    description="Get statistics about the processing queue (admin endpoint)."
)
async def get_queue_stats(
    job_queue: JobQueue = Depends(get_job_queue)
):
    """Get processing queue statistics."""
    try:
        stats = job_queue.get_queue_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "Queue statistics are currently unavailable"
            }
        )


@api_router.post(
    "/admin/cleanup",
    summary="Force cleanup operation",
    description="Force immediate cleanup of failed jobs and temporary files (admin endpoint)."
)
async def force_cleanup(
    job_queue: JobQueue = Depends(get_job_queue),
    file_manager: FileManager = Depends(get_file_manager)
):
    """Force immediate cleanup of failed jobs and temporary files."""
    try:
        from processing.cleanup_service import get_cleanup_service
        cleanup_service = get_cleanup_service()
        results = cleanup_service.force_cleanup()
        
        return {
            "status": "success",
            "message": "Cleanup operation completed",
            "data": results
        }
    except Exception as e:
        from fastapi import HTTPException
        logger.error(f"Failed to perform cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "cleanup_failed",
                "message": "Failed to perform cleanup operation"
            }
        )


@api_router.get(
    "/admin/cleanup/stats",
    summary="Get cleanup statistics",
    description="Get statistics about cleanup operations (admin endpoint)."
)
async def get_cleanup_stats():
    """Get cleanup service statistics."""
    try:
        from processing.cleanup_service import get_cleanup_service
        cleanup_service = get_cleanup_service()
        stats = cleanup_service.get_cleanup_stats()
        
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        from fastapi import HTTPException
        logger.error(f"Failed to get cleanup stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "stats_unavailable",
                "message": "Cleanup statistics are currently unavailable"
            }
        )