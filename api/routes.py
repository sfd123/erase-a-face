"""
FastAPI routes for the Golf Video Anonymizer API.

This module defines the REST API endpoints and integrates them
with the handler classes.
"""

from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse

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