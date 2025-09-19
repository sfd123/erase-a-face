"""
Video processing pipeline that orchestrates face detection, blurring, and encoding.

This module provides the main VideoProcessor class that coordinates the complete
video anonymization pipeline including face detection, tracking, blurring, and
video encoding with quality preservation and audio track handling.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime

from models.processing_job import ProcessingJob, JobStatus
from models.video_metadata import VideoMetadata
from models.face_detection import FaceDetection
from processing.face_detector import FaceDetector, FaceDetectorConfig
from processing.face_blurrer import FaceBlurrer, FaceBlurrerConfig
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
        
        try:
            # Update job status
            job.mark_processing()
            
            if progress_callback:
                progress_callback(0.0, "Starting video processing...")
            
            # Load and validate input video
            input_path = Path(job.file_path)
            if not input_path.exists():
                raise VideoProcessingError(f"Input video file not found: {input_path}")
            
            # Extract video metadata
            self.video_metadata = self._extract_video_metadata(input_path)
            if progress_callback:
                progress_callback(0.1, "Video metadata extracted")
            
            # Create output file path
            output_path = self._create_output_path(job.job_id, input_path)
            
            # Process video frames
            self._process_video_frames(input_path, output_path, progress_callback)
            
            # Mark job as completed
            job.mark_completed(str(output_path), self.total_faces_detected)
            
            if progress_callback:
                progress_callback(1.0, f"Processing complete. {self.total_faces_detected} faces detected and blurred.")
            
            logger.info(f"Video processing completed for job {job.job_id}. Output: {output_path}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Video processing failed: {str(e)}"
            logger.error(f"Job {job.job_id}: {error_msg}")
            job.mark_failed(error_msg)
            
            # Clean up any partial output files
            if 'output_path' in locals() and Path(output_path).exists():
                try:
                    Path(output_path).unlink()
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
        Process video frames with face detection and blurring.
        
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
            # Open input video
            input_cap = cv2.VideoCapture(str(input_path))
            if not input_cap.isOpened():
                raise VideoProcessingError(f"Could not open input video: {input_path}")
            
            # Get video properties
            fps = self.config.output_fps or int(input_cap.get(cv2.CAP_PROP_FPS))
            width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
            output_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            if not output_writer.isOpened():
                raise VideoProcessingError(f"Could not create output video writer: {output_path}")
            
            # Process frames
            frame_number = 0
            frames_batch = []
            
            while True:
                ret, frame = input_cap.read()
                if not ret:
                    break
                
                # Process frame for face detection and blurring
                processed_frame = self._process_single_frame(frame, frame_number)
                
                # Write processed frame
                output_writer.write(processed_frame)
                
                frame_number += 1
                
                # Report progress
                if progress_callback and frame_number % self.config.progress_callback_interval == 0:
                    progress = 0.1 + (0.8 * frame_number / total_frames)  # 10% to 90%
                    message = f"Processing frame {frame_number}/{total_frames}"
                    progress_callback(progress, message)
            
            # Handle audio preservation if enabled
            if self.config.preserve_audio:
                self._preserve_audio_track(input_path, output_path, progress_callback)
            
            logger.info(f"Processed {frame_number} frames, detected {self.total_faces_detected} faces total")
            
        except Exception as e:
            raise VideoProcessingError(f"Frame processing failed: {e}") from e
        
        finally:
            # Clean up resources
            if input_cap:
                input_cap.release()
            if output_writer:
                output_writer.release()
    
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
            return frame
        
        try:
            # Detect faces in the frame
            face_detections = self.face_detector.detect_faces(frame, frame_number)
            
            # Update total face count
            self.total_faces_detected += len(face_detections)
            
            # Apply blur to detected faces
            blurred_frame = self.face_blurrer.blur_faces(frame, face_detections)
            
            return blurred_frame
            
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
        
        if self.video_metadata:
            stats.update({
                'video_duration': self.video_metadata.duration,
                'video_fps': self.video_metadata.fps,
                'video_resolution': self.video_metadata.resolution,
                'video_format': self.video_metadata.format,
                'total_frames': self.video_metadata.total_frames,
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
    
    def cleanup_resources(self) -> None:
        """Clean up any resources used by the processor."""
        self.face_blurrer.reset_tracking()
        self.current_job = None
        self.video_metadata = None
        self.total_faces_detected = 0