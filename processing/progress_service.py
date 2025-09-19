"""
Progress reporting service for long-running video processing operations.

This module provides comprehensive progress tracking and reporting capabilities
for video processing jobs, including real-time updates, ETA calculations,
and detailed status information.
"""

import time
import threading
import logging
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Status enumeration for progress tracking."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressMetrics:
    """Metrics for progress tracking."""
    
    # Basic progress
    progress_percent: float = 0.0
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    current_step: str = ""
    
    # Timing information
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    
    # Processing details
    frames_processed: int = 0
    total_frames: int = 0
    current_frame: int = 0
    
    # Performance metrics
    processing_speed_fps: float = 0.0
    memory_usage_mb: int = 0
    cpu_usage_percent: float = 0.0
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'progress_percent': self.progress_percent,
            'status': self.status.value,
            'current_step': self.current_step,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'current_frame': self.current_frame,
            'processing_speed_fps': self.processing_speed_fps,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'error_message': self.error_message,
            'warnings': self.warnings.copy(),
            'metadata': self.metadata.copy()
        }


class ProgressTracker:
    """
    Individual progress tracker for a single operation.
    
    This class tracks progress for a single long-running operation,
    calculating ETAs, monitoring performance, and providing detailed status.
    """
    
    def __init__(self, operation_id: str, total_frames: int = 0):
        """
        Initialize progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            total_frames: Total number of frames to process (if known)
        """
        self.operation_id = operation_id
        self.metrics = ProgressMetrics(total_frames=total_frames)
        self._lock = threading.Lock()
        
        # Performance tracking
        self._frame_times: List[float] = []
        self._max_frame_times = 100  # Keep last 100 frame times for averaging
        
        # ETA calculation
        self._progress_history: List[tuple] = []  # (timestamp, progress) pairs
        self._max_history_points = 50
    
    def start(self, step: str = "Starting processing") -> None:
        """Mark the operation as started."""
        with self._lock:
            self.metrics.start_time = datetime.now()
            self.metrics.last_update_time = self.metrics.start_time
            self.metrics.status = ProgressStatus.INITIALIZING
            self.metrics.current_step = step
            self.metrics.progress_percent = 0.0
            
            logger.info(f"Progress tracking started for operation {self.operation_id}")
    
    def update_progress(self, progress_percent: float, step: str = "", 
                       current_frame: int = None, **kwargs) -> None:
        """
        Update progress information.
        
        Args:
            progress_percent: Progress as percentage (0.0 to 100.0)
            step: Current processing step description
            current_frame: Current frame number being processed
            **kwargs: Additional metadata to store
        """
        with self._lock:
            now = datetime.now()
            
            # Update basic progress
            self.metrics.progress_percent = max(0.0, min(100.0, progress_percent))
            self.metrics.last_update_time = now
            
            if step:
                self.metrics.current_step = step
            
            if current_frame is not None:
                # Calculate frames processed and performance
                if self.metrics.current_frame > 0:
                    frames_delta = current_frame - self.metrics.current_frame
                    time_delta = (now - self.metrics.last_update_time).total_seconds()
                    
                    if time_delta > 0 and frames_delta > 0:
                        frame_time = time_delta / frames_delta
                        self._frame_times.append(frame_time)
                        
                        # Keep only recent frame times
                        if len(self._frame_times) > self._max_frame_times:
                            self._frame_times = self._frame_times[-self._max_frame_times:]
                        
                        # Calculate average FPS
                        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
                        self.metrics.processing_speed_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                
                self.metrics.current_frame = current_frame
                self.metrics.frames_processed = current_frame
            
            # Update status based on progress
            if self.metrics.progress_percent >= 100.0:
                self.metrics.status = ProgressStatus.COMPLETED
            elif self.metrics.progress_percent > 0.0:
                self.metrics.status = ProgressStatus.PROCESSING
            
            # Store additional metadata
            for key, value in kwargs.items():
                self.metrics.metadata[key] = value
            
            # Update progress history for ETA calculation
            self._progress_history.append((now.timestamp(), progress_percent))
            if len(self._progress_history) > self._max_history_points:
                self._progress_history = self._progress_history[-self._max_history_points:]
            
            # Calculate ETA
            self._calculate_eta()
    
    def update_system_metrics(self, memory_mb: int = 0, cpu_percent: float = 0.0) -> None:
        """
        Update system resource metrics.
        
        Args:
            memory_mb: Current memory usage in MB
            cpu_percent: Current CPU usage percentage
        """
        with self._lock:
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.cpu_usage_percent = cpu_percent
    
    def add_warning(self, warning: str) -> None:
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add
        """
        with self._lock:
            self.metrics.warnings.append(warning)
            logger.warning(f"Operation {self.operation_id}: {warning}")
    
    def mark_completed(self, step: str = "Processing completed") -> None:
        """Mark the operation as completed successfully."""
        with self._lock:
            self.metrics.status = ProgressStatus.COMPLETED
            self.metrics.progress_percent = 100.0
            self.metrics.current_step = step
            self.metrics.last_update_time = datetime.now()
            
            if self.metrics.start_time:
                total_time = (self.metrics.last_update_time - self.metrics.start_time).total_seconds()
                logger.info(f"Operation {self.operation_id} completed in {total_time:.2f} seconds")
    
    def mark_failed(self, error_message: str) -> None:
        """
        Mark the operation as failed.
        
        Args:
            error_message: Error message describing the failure
        """
        with self._lock:
            self.metrics.status = ProgressStatus.FAILED
            self.metrics.error_message = error_message
            self.metrics.last_update_time = datetime.now()
            
            logger.error(f"Operation {self.operation_id} failed: {error_message}")
    
    def mark_cancelled(self) -> None:
        """Mark the operation as cancelled."""
        with self._lock:
            self.metrics.status = ProgressStatus.CANCELLED
            self.metrics.current_step = "Operation cancelled"
            self.metrics.last_update_time = datetime.now()
            
            logger.info(f"Operation {self.operation_id} was cancelled")
    
    def _calculate_eta(self) -> None:
        """Calculate estimated time of completion based on progress history."""
        if len(self._progress_history) < 2 or self.metrics.progress_percent >= 100.0:
            return
        
        # Use linear regression on recent progress points
        recent_points = self._progress_history[-10:]  # Use last 10 points
        
        if len(recent_points) < 2:
            return
        
        # Calculate progress rate (percent per second)
        time_span = recent_points[-1][0] - recent_points[0][0]
        progress_span = recent_points[-1][1] - recent_points[0][1]
        
        if time_span <= 0 or progress_span <= 0:
            return
        
        progress_rate = progress_span / time_span  # percent per second
        
        if progress_rate > 0:
            remaining_progress = 100.0 - self.metrics.progress_percent
            eta_seconds = remaining_progress / progress_rate
            
            self.metrics.estimated_completion_time = datetime.now() + timedelta(seconds=eta_seconds)
    
    def get_metrics(self) -> ProgressMetrics:
        """
        Get current progress metrics.
        
        Returns:
            Copy of current progress metrics
        """
        with self._lock:
            # Create a copy to avoid race conditions
            return ProgressMetrics(
                progress_percent=self.metrics.progress_percent,
                status=self.metrics.status,
                current_step=self.metrics.current_step,
                start_time=self.metrics.start_time,
                last_update_time=self.metrics.last_update_time,
                estimated_completion_time=self.metrics.estimated_completion_time,
                frames_processed=self.metrics.frames_processed,
                total_frames=self.metrics.total_frames,
                current_frame=self.metrics.current_frame,
                processing_speed_fps=self.metrics.processing_speed_fps,
                memory_usage_mb=self.metrics.memory_usage_mb,
                cpu_usage_percent=self.metrics.cpu_usage_percent,
                error_message=self.metrics.error_message,
                warnings=self.metrics.warnings.copy(),
                metadata=self.metrics.metadata.copy()
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the operation progress.
        
        Returns:
            Dictionary with progress summary
        """
        metrics = self.get_metrics()
        
        summary = {
            'operation_id': self.operation_id,
            'status': metrics.status.value,
            'progress_percent': metrics.progress_percent,
            'current_step': metrics.current_step,
            'processing_speed_fps': metrics.processing_speed_fps,
            'is_complete': metrics.status in (ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED)
        }
        
        # Add timing information
        if metrics.start_time:
            summary['elapsed_time'] = (datetime.now() - metrics.start_time).total_seconds()
        
        if metrics.estimated_completion_time and metrics.status == ProgressStatus.PROCESSING:
            summary['eta_seconds'] = (metrics.estimated_completion_time - datetime.now()).total_seconds()
        
        # Add frame information if available
        if metrics.total_frames > 0:
            summary['frames_progress'] = f"{metrics.frames_processed}/{metrics.total_frames}"
            summary['frames_remaining'] = metrics.total_frames - metrics.frames_processed
        
        return summary


class ProgressService:
    """
    Service for managing multiple progress trackers and providing updates.
    
    This service manages progress tracking for multiple concurrent operations
    and provides callbacks for real-time updates.
    """
    
    def __init__(self):
        """Initialize the progress service."""
        self.trackers: Dict[str, ProgressTracker] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        
        # Background thread for periodic updates
        self._update_thread = None
        self._shutdown_requested = False
        self._update_interval = 2.0  # Update every 2 seconds
    
    def create_tracker(self, operation_id: str, total_frames: int = 0) -> ProgressTracker:
        """
        Create a new progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            total_frames: Total number of frames to process
            
        Returns:
            New progress tracker instance
        """
        with self._lock:
            if operation_id in self.trackers:
                logger.warning(f"Progress tracker for {operation_id} already exists")
                return self.trackers[operation_id]
            
            tracker = ProgressTracker(operation_id, total_frames)
            self.trackers[operation_id] = tracker
            self.callbacks[operation_id] = []
            
            logger.info(f"Created progress tracker for operation {operation_id}")
            return tracker
    
    def get_tracker(self, operation_id: str) -> Optional[ProgressTracker]:
        """
        Get an existing progress tracker.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Progress tracker or None if not found
        """
        with self._lock:
            return self.trackers.get(operation_id)
    
    def remove_tracker(self, operation_id: str) -> bool:
        """
        Remove a progress tracker.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            True if tracker was removed, False if not found
        """
        with self._lock:
            if operation_id in self.trackers:
                del self.trackers[operation_id]
                del self.callbacks[operation_id]
                logger.info(f"Removed progress tracker for operation {operation_id}")
                return True
            return False
    
    def add_callback(self, operation_id: str, callback: Callable[[ProgressMetrics], None]) -> bool:
        """
        Add a callback for progress updates.
        
        Args:
            operation_id: Operation identifier
            callback: Callback function to receive progress updates
            
        Returns:
            True if callback was added, False if operation not found
        """
        with self._lock:
            if operation_id in self.callbacks:
                self.callbacks[operation_id].append(callback)
                return True
            return False
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Get progress summary for all tracked operations.
        
        Returns:
            Dictionary mapping operation IDs to progress summaries
        """
        with self._lock:
            return {
                op_id: tracker.get_summary()
                for op_id, tracker in self.trackers.items()
            }
    
    def get_active_operations(self) -> List[str]:
        """
        Get list of active (not completed) operation IDs.
        
        Returns:
            List of active operation IDs
        """
        with self._lock:
            active_ops = []
            for op_id, tracker in self.trackers.items():
                metrics = tracker.get_metrics()
                if metrics.status not in (ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED):
                    active_ops.append(op_id)
            return active_ops
    
    def cleanup_completed_operations(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed operations older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for completed operations
            
        Returns:
            Number of operations cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            operations_to_remove = []
            
            for op_id, tracker in self.trackers.items():
                metrics = tracker.get_metrics()
                
                # Check if operation is completed and old enough
                if (metrics.status in (ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED) and
                    metrics.last_update_time and metrics.last_update_time < cutoff_time):
                    operations_to_remove.append(op_id)
            
            # Remove old operations
            for op_id in operations_to_remove:
                del self.trackers[op_id]
                del self.callbacks[op_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} completed operations")
        
        return cleaned_count
    
    def start_background_updates(self) -> None:
        """Start background thread for periodic progress updates."""
        if self._update_thread and self._update_thread.is_alive():
            return
        
        self._shutdown_requested = False
        self._update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self._update_thread.start()
        
        logger.info("Started background progress update thread")
    
    def stop_background_updates(self) -> None:
        """Stop background thread for progress updates."""
        self._shutdown_requested = True
        
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5)
        
        logger.info("Stopped background progress update thread")
    
    def _background_update_loop(self) -> None:
        """Background loop for sending progress updates to callbacks."""
        while not self._shutdown_requested:
            try:
                with self._lock:
                    # Send updates to all registered callbacks
                    for op_id, tracker in self.trackers.items():
                        metrics = tracker.get_metrics()
                        callbacks = self.callbacks.get(op_id, [])
                        
                        for callback in callbacks:
                            try:
                                callback(metrics)
                            except Exception as e:
                                logger.error(f"Error in progress callback for {op_id}: {e}")
                
                time.sleep(self._update_interval)
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(self._update_interval)


# Global progress service instance
progress_service = ProgressService()