"""
Video processing pipeline that orchestrates face detection, blurring, and encoding.

This module provides the main VideoProcessor class that coordinates the complete
video anonymization pipeline including face detection, tracking, blurring, and
video encoding with quality preservation and audio track handling.
"""

import cv2
import numpy as np
import logging
import gc
import psutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.processing_job import ProcessingJob, JobStatus
from models.video_metadata import VideoMetadata
from models.face_detection import FaceDetection
from processing.face_detector import FaceDetector, FaceDetectorConfig
from processing.face_blurrer import FaceBlurrer, FaceBlurrerConfig
from processing.progress_service import progress_service, ProgressTracker
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessorConfig:
    """Configuration parameters for video processing pipeline."""
    
    # Quality preservation settings
    output_quality: float = 0.95  # Video quality factor (0.0 to 1.0)
    preserve_audio: bool = True   # Whether to preserve audio track
    
    # Processing settings
    max_frames_per_batch: int = 30  # Process frames in batches for memory efficiency
    progress_callback_interval: int = 10  # Report progress every N frames
    
    # Performance optimization settings
    enable_frame_skipping: bool = False  # Skip frames for very large videos
    frame_skip_ratio: int = 2  # Process every Nth frame when skipping enabled
    max_resolution: Optional[Tuple[int, int]] = None  # Downscale if larger (width, height)
    memory_limit_mb: int = 1024  # Memory limit in MB for processing
    enable_gpu_acceleration: bool = False  # Use GPU if available
    
    # Large file handling
    chunk_size_mb: int = 100  # Process video in chunks for large files
    temp_cleanup_interval: int = 50  # Clean temp files every N frames
    
    # Face detection and blurring configs
    face_detector_config: Optional[FaceDetectorConfig] = None
    face_blurrer_config: Optional[FaceBlurrerConfig] = None
    
    # Output encoding settings
    output_codec: str = 'mp4v'  # Output video codec
    output_fps: Optional[int] = None  # Output FPS (None = preserve original)
    
    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.face_detector_config is None:
            self.face_detector_config = FaceDetectorConfig()
        if self.face_blurrer_config is None:
            self.face_blurrer_config = FaceBlurrerConfig()


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


class VideoProcessor:
    """
    Main video processing pipeline that orchestrates face detection and blurring.
    
    This class coordinates the complete video anonymization process:
    1. Load and validate input video
    2. Extract video metadata
    3. Process frames with face detection and blurring
    4. Encode output video with quality preservation
    5. Handle audio track preservation
    6. Provide progress updates and error handling
    """
    
    def __init__(self, config: Optional[VideoProcessorConfig] = None, 
                 file_manager: Optional[FileManager] = None):
        """
        Initialize video processor with configuration.
        
        Args:
            config: Video processing configuration
            file_manager: File manager for handling temporary files
        """
        self.config = config or VideoProcessorConfig()
        self.file_manager = file_manager or FileManager()
        
        # Initialize face detection and blurring components
        self.face_detector = FaceDetector(self.config.face_detector_config)
        self.face_blurrer = FaceBlurrer(self.config.face_blurrer_config)
        
        # Processing state
        self.current_job: Optional[ProcessingJob] = None
        self.video_metadata: Optional[VideoMetadata] = None
        self.total_faces_detected = 0
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Performance monitoring
        self.processing_stats = {
            'start_time': None,
            'frames_processed': 0,
            'memory_usage_mb': 0,
            'avg_frame_time': 0.0,
            'peak_memory_mb': 0,
            'frames_skipped': 0
        }
        
    def process_video(self, job: ProcessingJob, 
                     progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
        """
        Process a video file to blur faces and return path to output file.
        
        Args:
            job: Processing job containing input file information
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Path to the processed output video file
            
        Raises:
            VideoProcessingError: If processing fails
        """
        self.current_job = job
        self.total_faces_detected = 0
        output_path = None
        
        try:
            # Create progress tracker for detailed progress reporting
            self.progress_tracker = progress_service.create_tracker(
                job.job_id, 
                total_frames=0  # Will be updated after metadata extraction
            )
            self.progress_tracker.start("Initializing video processing")
            
            # Update job status
            job.mark_processing()
            
            if progress_callback:
                progress_callback(0.0, "Starting video processing...")
            
            # Load and validate input video
            input_path = Path(job.file_path)
            if not input_path.exists():
                raise VideoProcessingError(f"Input video file not found: {input_path}")
            
            # Validate video file integrity
            self.progress_tracker.update_progress(2.0, "Validating video file")
            if not self._validate_video_integrity(input_path):
                raise VideoProcessingError("Video file appears to be corrupted or unreadable")
            
            # Extract video metadata
            self.progress_tracker.update_progress(5.0, "Extracting video metadata")
            self.video_metadata = self._extract_video_metadata(input_path)
            
            # Update progress tracker with total frames
            if self.video_metadata:
                total_frames = int(self.video_metadata.fps * self.video_metadata.duration)
                self.progress_tracker.metrics.total_frames = total_frames
            
            if progress_callback:
                progress_callback(0.1, "Video metadata extracted")
            
            # Create output file path
            output_path = self._create_output_path(job.job_id, input_path)
            
            # Process video frames
            self._process_video_frames(input_path, output_path, progress_callback)
            
            # Handle edge case: no faces detected
            if self.total_faces_detected == 0:
                logger.info(f"No faces detected in video {job.job_id}. Returning original video.")
                if progress_callback:
                    progress_callback(0.95, "No faces detected - preserving original video")
                
                # Copy original to output location if different
                if str(input_path) != str(output_path):
                    import shutil
                    shutil.copy2(input_path, output_path)
            
            # Verify output file was created successfully
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise VideoProcessingError("Output video file was not created successfully")
            
            # Mark job as completed
            job.mark_completed(str(output_path), self.total_faces_detected)
            
            # Update progress tracker
            if self.progress_tracker:
                completion_message = (f"Processing complete. {self.total_faces_detected} faces detected and blurred." 
                                    if self.total_faces_detected > 0 
                                    else "Processing complete. No faces detected - original video preserved.")
                self.progress_tracker.mark_completed(completion_message)
            
            if progress_callback:
                if self.total_faces_detected > 0:
                    progress_callback(1.0, f"Processing complete. {self.total_faces_detected} faces detected and blurred.")
                else:
                    progress_callback(1.0, "Processing complete. No faces detected - original video preserved.")
            
            logger.info(f"Video processing completed for job {job.job_id}. Output: {output_path}, Faces: {self.total_faces_detected}")
            return str(output_path)
            
        except Exception as e:
            error_msg = self._categorize_processing_error(e)
            logger.error(f"Job {job.job_id}: {error_msg}")
            job.mark_failed(error_msg)
            
            # Update progress tracker
            if self.progress_tracker:
                self.progress_tracker.mark_failed(error_msg)
            
            # Clean up any partial output files
            if output_path and Path(output_path).exists():
                try:
                    Path(output_path).unlink()
                    logger.info(f"Cleaned up partial output file: {output_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up partial output file: {cleanup_error}")
            
            raise VideoProcessingError(error_msg) from e
        
        finally:
            # Reset face blurrer tracking for next video
            self.face_blurrer.reset_tracking()
    
    def _extract_video_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object with video information
            
        Raises:
            VideoProcessingError: If metadata extraction fails
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise VideoProcessingError(f"Could not open video file: {video_path}")
            
            # Extract basic video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0.0
            
            # Get file size
            file_size = video_path.stat().st_size
            
            # Determine format from file extension
            format_name = video_path.suffix.lower().lstrip('.')
            
            cap.release()
            
            metadata = VideoMetadata(
                duration=duration,
                fps=fps,
                resolution=(width, height),
                format=format_name,
                file_size=file_size
            )
            
            logger.info(f"Video metadata: {width}x{height}, {fps}fps, {duration:.1f}s, {format_name}")
            return metadata
            
        except Exception as e:
            raise VideoProcessingError(f"Failed to extract video metadata: {e}") from e
    
    def _create_output_path(self, job_id: str, input_path: Path) -> Path:
        """
        Create output file path for processed video.
        
        Args:
            job_id: Unique job identifier
            input_path: Path to input video file
            
        Returns:
            Path for output video file
        """
        # Create output filename with job ID and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{job_id}_anonymized_{timestamp}.mp4"
        
        # Use file manager to create temp file path
        output_path = self.file_manager.temp_dir / output_filename
        
        return output_path
    
    def _process_video_frames(self, input_path: Path, output_path: Path, 
                            progress_callback: Optional[Callable[[float, str], None]] = None) -> None:
        """
        Process video frames with face detection and blurring with performance optimizations.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional progress callback
            
        Raises:
            VideoProcessingError: If frame processing fails
        """
        input_cap = None
        output_writer = None
        
        try:
            # Initialize performance monitoring
            self.processing_stats['start_time'] = time.time()
            self.processing_stats['frames_processed'] = 0
            
            # Open input video
            input_cap = cv2.VideoCapture(str(input_path))
            if not input_cap.isOpened():
                raise VideoProcessingError(f"Could not open input video: {input_path}")
            
            # Get video properties
            fps = self.config.output_fps or int(input_cap.get(cv2.CAP_PROP_FPS))
            width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Apply resolution optimization if needed
            output_width, output_height = self._optimize_resolution(width, height)
            scale_factor = output_width / width if width > 0 else 1.0
            
            # Determine if frame skipping should be enabled for large videos
            should_skip_frames = self._should_enable_frame_skipping(total_frames, width, height)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
            output_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (output_width, output_height)
            )
            
            if not output_writer.isOpened():
                raise VideoProcessingError(f"Could not create output video writer: {output_path}")
            
            # Process frames with optimizations
            if self.config.max_frames_per_batch > 1:
                self._process_frames_batch_mode(
                    input_cap, output_writer, total_frames, scale_factor, 
                    should_skip_frames, progress_callback
                )
            else:
                self._process_frames_sequential(
                    input_cap, output_writer, total_frames, scale_factor,
                    should_skip_frames, progress_callback
                )
            
            # Handle audio preservation if enabled
            if self.config.preserve_audio:
                self._preserve_audio_track(input_path, output_path, progress_callback)
            
            # Log performance statistics
            self._log_performance_stats()
            
        except Exception as e:
            raise VideoProcessingError(f"Frame processing failed: {e}") from e
        
        finally:
            # Clean up resources
            if input_cap:
                input_cap.release()
            if output_writer:
                output_writer.release()
            
            # Force garbage collection
            gc.collect()
    
    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Process a single frame for face detection and blurring.
        
        Args:
            frame: Input frame as numpy array
            frame_number: Frame number for tracking
            
        Returns:
            Processed frame with faces blurred
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Frame {frame_number} is empty or invalid")
            return frame
        
        try:
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Frame {frame_number} has unexpected dimensions: {frame.shape}")
                return frame
            
            # Detect faces in the frame
            face_detections = self.face_detector.detect_faces(frame, frame_number)
            
            # Update total face count
            self.total_faces_detected += len(face_detections)
            
            # Apply blur to detected faces
            if face_detections:
                blurred_frame = self.face_blurrer.blur_faces(frame, face_detections)
                return blurred_frame
            else:
                # No faces detected, return original frame
                return frame
            
        except Exception as e:
            logger.warning(f"Error processing frame {frame_number}: {e}")
            # Return original frame if processing fails
            return frame
    
    def _preserve_audio_track(self, input_path: Path, output_path: Path,
                            progress_callback: Optional[Callable[[float, str], None]] = None) -> None:
        """
        Preserve audio track from input video to output video.
        
        Args:
            input_path: Path to input video with audio
            output_path: Path to output video (video only)
            progress_callback: Optional progress callback
        """
        try:
            if progress_callback:
                progress_callback(0.9, "Preserving audio track...")
            
            # Create temporary path for video with audio
            temp_output_with_audio = output_path.with_suffix('.temp.mp4')
            
            # Use ffmpeg to combine processed video with original audio
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', str(output_path),  # Video input (no audio)
                '-i', str(input_path),   # Audio input (original video)
                '-c:v', 'copy',          # Copy video stream
                '-c:a', 'aac',           # Encode audio as AAC
                '-map', '0:v:0',         # Map video from first input
                '-map', '1:a:0',         # Map audio from second input
                '-shortest',             # End when shortest stream ends
                str(temp_output_with_audio)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Replace original output with version that has audio
                output_path.unlink()
                temp_output_with_audio.rename(output_path)
                logger.info("Audio track preserved successfully")
            else:
                logger.warning(f"Failed to preserve audio track: {result.stderr}")
                # Clean up temp file if it exists
                if temp_output_with_audio.exists():
                    temp_output_with_audio.unlink()
        
        except subprocess.TimeoutExpired:
            logger.warning("Audio preservation timed out")
        except FileNotFoundError:
            logger.warning("ffmpeg not found - audio track will not be preserved")
        except Exception as e:
            logger.warning(f"Audio preservation failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current or last processing job.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'job_id': self.current_job.job_id if self.current_job else None,
            'total_faces_detected': self.total_faces_detected,
            'active_face_tracks': self.face_blurrer.get_active_tracks_count(),
            'face_track_info': self.face_blurrer.get_track_info(),
        }
        
        # Add performance statistics
        stats.update(self.processing_stats)
        
        if self.video_metadata:
            stats.update({
                'video_duration': self.video_metadata.duration,
                'video_fps': self.video_metadata.fps,
                'video_resolution': self.video_metadata.resolution,
                'video_format': self.video_metadata.format,
                'total_frames': getattr(self.video_metadata, 'total_frames', 
                                      int(self.video_metadata.fps * self.video_metadata.duration) if self.video_metadata.fps and self.video_metadata.duration else 0),
            })
        
        return stats
    
    def update_config(self, **kwargs) -> None:
        """
        Update processor configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Update sub-component configs if they were modified
        if hasattr(self.config, 'face_detector_config'):
            self.face_detector = FaceDetector(self.config.face_detector_config)
        if hasattr(self.config, 'face_blurrer_config'):
            self.face_blurrer = FaceBlurrer(self.config.face_blurrer_config)
    
    def _validate_video_integrity(self, video_path: Path) -> bool:
        """
        Validate that the video file can be opened and read.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is readable, False otherwise
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            
            # Try to read the first frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None and frame.size > 0
            
        except Exception as e:
            logger.error(f"Video integrity check failed: {e}")
            return False
    
    def _categorize_processing_error(self, error: Exception) -> str:
        """
        Categorize processing errors into user-friendly messages.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Categorized error message
        """
        error_str = str(error).lower()
        
        if "could not open" in error_str or "not found" in error_str:
            return "Video file could not be opened. The file may be corrupted or in an unsupported format."
        
        elif "metadata" in error_str or "extract" in error_str:
            return "Unable to read video properties. The file may be corrupted."
        
        elif "codec" in error_str or "fourcc" in error_str:
            return "Video encoding format is not supported or corrupted."
        
        elif "memory" in error_str or "allocation" in error_str:
            return "Insufficient memory to process this video. Please try a smaller file."
        
        elif "disk" in error_str or "space" in error_str:
            return "Insufficient disk space to process the video."
        
        elif "timeout" in error_str:
            return "Video processing timed out. The video may be too long or complex."
        
        elif "frame" in error_str and "processing" in error_str:
            return "Error processing video frames. The video may contain corrupted data."
        
        elif "audio" in error_str and "preserve" in error_str:
            return "Video processed successfully but audio preservation failed. Video-only output provided."
        
        else:
            return f"Video processing failed: {str(error)}"
    
    def _optimize_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """
        Optimize video resolution for processing performance.
        
        Args:
            width: Original video width
            height: Original video height
            
        Returns:
            Optimized (width, height) tuple
        """
        if self.config.max_resolution is None:
            return width, height
        
        max_width, max_height = self.config.max_resolution
        
        # Calculate scale factor to fit within max resolution
        scale_w = max_width / width if width > max_width else 1.0
        scale_h = max_height / height if height > max_height else 1.0
        scale_factor = min(scale_w, scale_h)
        
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Ensure dimensions are even (required by some codecs)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            logger.info(f"Downscaling video from {width}x{height} to {new_width}x{new_height}")
            return new_width, new_height
        
        return width, height
    
    def _should_enable_frame_skipping(self, total_frames: int, width: int, height: int) -> bool:
        """
        Determine if frame skipping should be enabled based on video characteristics.
        
        Args:
            total_frames: Total number of frames
            width: Video width
            height: Video height
            
        Returns:
            True if frame skipping should be enabled
        """
        if not self.config.enable_frame_skipping:
            return False
        
        # Calculate video size metrics
        pixels_per_frame = width * height
        total_pixels = pixels_per_frame * total_frames
        
        # Enable frame skipping for very large videos
        large_video_threshold = 1920 * 1080 * 1800  # ~30 minutes of 1080p at 30fps
        
        return total_pixels > large_video_threshold
    
    def _process_frames_batch_mode(self, input_cap, output_writer, total_frames: int, 
                                 scale_factor: float, should_skip_frames: bool,
                                 progress_callback: Optional[Callable[[float, str], None]] = None) -> None:
        """
        Process frames in batch mode for better performance.
        
        Args:
            input_cap: Input video capture
            output_writer: Output video writer
            total_frames: Total number of frames
            scale_factor: Resolution scale factor
            should_skip_frames: Whether to skip frames
            progress_callback: Optional progress callback
        """
        frame_number = 0
        frames_batch = []
        
        while True:
            ret, frame = input_cap.read()
            if not ret:
                # Process remaining frames in batch
                if frames_batch:
                    self._process_frame_batch(frames_batch, output_writer, scale_factor)
                break
            
            # Skip frames if enabled
            if should_skip_frames and frame_number % self.config.frame_skip_ratio != 0:
                # For skipped frames, duplicate the last processed frame
                if frames_batch:
                    output_writer.write(frames_batch[-1]['processed'])
                else:
                    # If no previous frame, process this one
                    processed_frame = self._process_single_frame_optimized(frame, frame_number, scale_factor)
                    output_writer.write(processed_frame)
                
                self.processing_stats['frames_skipped'] += 1
                frame_number += 1
                continue
            
            # Add frame to batch
            frames_batch.append({
                'original': frame,
                'frame_number': frame_number,
                'processed': None
            })
            
            # Process batch when full
            if len(frames_batch) >= self.config.max_frames_per_batch:
                self._process_frame_batch(frames_batch, output_writer, scale_factor)
                frames_batch.clear()
                
                # Memory management
                self._manage_memory()
            
            frame_number += 1
            
            # Report progress
            if frame_number % self.config.progress_callback_interval == 0:
                progress = 10.0 + (80.0 * frame_number / total_frames)
                memory_mb = self._get_memory_usage()
                message = f"Processing frame {frame_number}/{total_frames} (Memory: {memory_mb}MB)"
                
                # Update progress tracker
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        progress, message, frame_number,
                        memory_mb=memory_mb
                    )
                
                # Call external callback
                if progress_callback:
                    progress_callback(progress / 100.0, message)
    
    def _process_frames_sequential(self, input_cap, output_writer, total_frames: int,
                                 scale_factor: float, should_skip_frames: bool,
                                 progress_callback: Optional[Callable[[float, str], None]] = None) -> None:
        """
        Process frames sequentially (original method with optimizations).
        
        Args:
            input_cap: Input video capture
            output_writer: Output video writer
            total_frames: Total number of frames
            scale_factor: Resolution scale factor
            should_skip_frames: Whether to skip frames
            progress_callback: Optional progress callback
        """
        frame_number = 0
        last_processed_frame = None
        
        while True:
            ret, frame = input_cap.read()
            if not ret:
                break
            
            # Skip frames if enabled
            if should_skip_frames and frame_number % self.config.frame_skip_ratio != 0:
                if last_processed_frame is not None:
                    output_writer.write(last_processed_frame)
                else:
                    processed_frame = self._process_single_frame_optimized(frame, frame_number, scale_factor)
                    output_writer.write(processed_frame)
                    last_processed_frame = processed_frame
                
                self.processing_stats['frames_skipped'] += 1
                frame_number += 1
                continue
            
            # Process frame
            processed_frame = self._process_single_frame_optimized(frame, frame_number, scale_factor)
            output_writer.write(processed_frame)
            last_processed_frame = processed_frame
            
            frame_number += 1
            self.processing_stats['frames_processed'] += 1
            
            # Memory management
            if frame_number % self.config.temp_cleanup_interval == 0:
                self._manage_memory()
            
            # Report progress
            if frame_number % self.config.progress_callback_interval == 0:
                progress = 10.0 + (80.0 * frame_number / total_frames)
                memory_mb = self._get_memory_usage()
                message = f"Processing frame {frame_number}/{total_frames} (Memory: {memory_mb}MB)"
                
                # Update progress tracker
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        progress, message, frame_number,
                        memory_mb=memory_mb
                    )
                
                # Call external callback
                if progress_callback:
                    progress_callback(progress / 100.0, message)
    
    def _process_frame_batch(self, frames_batch: List[Dict], output_writer, scale_factor: float) -> None:
        """
        Process a batch of frames for better performance.
        
        Args:
            frames_batch: List of frame dictionaries
            output_writer: Output video writer
            scale_factor: Resolution scale factor
        """
        frame_start_time = time.time()
        
        # Process frames in parallel if beneficial
        if len(frames_batch) > 4:  # Only use threading for larger batches
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for frame_data in frames_batch:
                    future = executor.submit(
                        self._process_single_frame_optimized,
                        frame_data['original'],
                        frame_data['frame_number'],
                        scale_factor
                    )
                    futures.append((future, frame_data))
                
                # Collect results in order
                for future, frame_data in futures:
                    frame_data['processed'] = future.result()
        else:
            # Process sequentially for small batches
            for frame_data in frames_batch:
                frame_data['processed'] = self._process_single_frame_optimized(
                    frame_data['original'],
                    frame_data['frame_number'],
                    scale_factor
                )
        
        # Write all processed frames
        for frame_data in frames_batch:
            output_writer.write(frame_data['processed'])
            self.processing_stats['frames_processed'] += 1
        
        # Update timing statistics
        batch_time = time.time() - frame_start_time
        frames_count = len(frames_batch)
        if frames_count > 0:
            avg_time = batch_time / frames_count
            self.processing_stats['avg_frame_time'] = (
                (self.processing_stats['avg_frame_time'] * 0.9) + (avg_time * 0.1)
            )
    
    def _process_single_frame_optimized(self, frame: np.ndarray, frame_number: int, scale_factor: float) -> np.ndarray:
        """
        Process a single frame with performance optimizations.
        
        Args:
            frame: Input frame
            frame_number: Frame number
            scale_factor: Resolution scale factor
            
        Returns:
            Processed frame
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Frame {frame_number} is empty or invalid")
            return frame
        
        # Apply resolution scaling if needed
        if scale_factor != 1.0:
            new_height = int(frame.shape[0] * scale_factor)
            new_width = int(frame.shape[1] * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Use the original processing method
        return self._process_single_frame(frame, frame_number)
    
    def _manage_memory(self) -> None:
        """
        Manage memory usage during processing.
        """
        current_memory = self._get_memory_usage()
        self.processing_stats['memory_usage_mb'] = current_memory
        
        if current_memory > self.processing_stats['peak_memory_mb']:
            self.processing_stats['peak_memory_mb'] = current_memory
        
        # Force garbage collection if memory usage is high
        if current_memory > self.config.memory_limit_mb:
            logger.warning(f"High memory usage detected: {current_memory}MB. Running garbage collection.")
            gc.collect()
            
            # Check memory again after GC
            new_memory = self._get_memory_usage()
            logger.info(f"Memory after GC: {new_memory}MB (freed {current_memory - new_memory}MB)")
    
    def _get_memory_usage(self) -> int:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return int(memory_info.rss / 1024 / 1024)  # Convert to MB
        except Exception:
            return 0
    
    def _log_performance_stats(self) -> None:
        """
        Log performance statistics after processing.
        """
        if self.processing_stats['start_time']:
            total_time = time.time() - self.processing_stats['start_time']
            frames_processed = self.processing_stats['frames_processed']
            frames_skipped = self.processing_stats['frames_skipped']
            
            fps = frames_processed / total_time if total_time > 0 else 0
            
            logger.info(f"Processing completed in {total_time:.2f}s")
            logger.info(f"Frames processed: {frames_processed}, skipped: {frames_skipped}")
            logger.info(f"Average FPS: {fps:.2f}")
            logger.info(f"Peak memory usage: {self.processing_stats['peak_memory_mb']}MB")
            logger.info(f"Average frame processing time: {self.processing_stats['avg_frame_time']:.4f}s")
    
    def cleanup_resources(self) -> None:
        """Clean up any resources used by the processor."""
        self.face_blurrer.reset_tracking()
        self.current_job = None
        self.video_metadata = None
        self.total_faces_detected = 0
        
        # Clean up progress tracker
        if self.progress_tracker:
            progress_service.remove_tracker(self.progress_tracker.operation_id)
            self.progress_tracker = None
        
        # Reset performance stats
        self.processing_stats = {
            'start_time': None,
            'frames_processed': 0,
            'memory_usage_mb': 0,
            'avg_frame_time': 0.0,
            'peak_memory_mb': 0,
            'frames_skipped': 0
        }
        
        # Force garbage collection
        gc.collect()