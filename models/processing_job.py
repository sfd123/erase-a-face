"""ProcessingJob data model for tracking video processing jobs."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class JobStatus(Enum):
    """Status enumeration for processing jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    """Data model for video processing jobs."""
    
    job_id: str
    original_filename: str
    file_path: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_file_path: Optional[str] = None
    faces_detected: int = 0
    
    @classmethod
    def create_new(cls, original_filename: str, file_path: str) -> 'ProcessingJob':
        """Create a new processing job with generated ID and current timestamp."""
        return cls(
            job_id=str(uuid.uuid4()),
            original_filename=original_filename,
            file_path=file_path,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
    
    def mark_processing(self) -> None:
        """Mark the job as currently processing."""
        self.status = JobStatus.PROCESSING
    
    def mark_completed(self, output_file_path: str, faces_detected: int = 0) -> None:
        """Mark the job as completed successfully."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output_file_path = output_file_path
        self.faces_detected = faces_detected
    
    def mark_failed(self, error_message: str) -> None:
        """Mark the job as failed with error message."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    @property
    def is_complete(self) -> bool:
        """Check if the job is in a terminal state (completed or failed)."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds if job is complete."""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None