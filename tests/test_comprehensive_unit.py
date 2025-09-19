"""
Comprehensive unit tests for all core components with high coverage.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta

# Import all core components
from models.processing_job import ProcessingJob, JobStatus
from models.video_metadata import VideoMetadata
from models.face_detection import FaceDetection
from models.validation import ValidationError
from processing.face_detector import FaceDetector, FaceDetectorConfig
from processing.face_blurrer import FaceBlurrer, FaceBlurrerConfig
from processing.video_processor import VideoProcessor
from storage.file_manager import FileManager
from storage.job_queue import JobQueue


@pytest.mark.unit
class TestProcessingJobComprehensive:
    """Comprehensive tests for ProcessingJob model."""
    
    def test_job_lifecycle_complete(self):
        """Test complete job lifecycle from creation to completion."""
        # Create job
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        assert job.status == JobStatus.PENDING
        assert job.processing_duration is None
        
        # Start processing
        job.mark_processing()
        assert job.status == JobStatus.PROCESSING
        
        # Complete successfully
        job.mark_completed("/path/output.mp4", faces_detected=5)
        assert job.status == JobStatus.COMPLETED
        assert job.output_file_path == "/path/output.mp4"
        assert job.faces_detected == 5
        assert job.is_complete is True
        assert job.processing_duration is not None
    
    def test_job_lifecycle_failure(self):
        """Test job lifecycle with failure."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        job.mark_processing()
        
        # Fail the job
        error_msg = "Video corrupted"
        job.mark_failed(error_msg)
        
        assert job.status == JobStatus.FAILED
        assert job.error_message == error_msg
        assert job.is_complete is True
        assert job.processing_duration is not None
    
    def test_job_serialization(self):
        """Test job serialization to dictionary."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        job.mark_completed("/path/output.mp4", faces_detected=3)
        
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == job.job_id
        assert job_dict["original_filename"] == "test.mp4"
        assert job_dict["status"] == "completed"
        assert job_dict["faces_detected"] == 3
        assert "created_at" in job_dict
        assert "completed_at" in job_dict
    
    def test_job_from_dict(self):
        """Test job creation from dictionary."""
        job_data = {
            "job_id": "test-job-id",
            "original_filename": "test.mp4",
            "file_path": "/path/test.mp4",
            "status": "completed",
            "created_at": "2023-01-01T12:00:00",
            "completed_at": "2023-01-01T12:05:00",
            "output_file_path": "/path/output.mp4",
            "faces_detected": 2,
            "error_message": None
        }
        
        job = ProcessingJob.from_dict(job_data)
        
        assert job.job_id == "test-job-id"
        assert job.original_filename == "test.mp4"
        assert job.status == JobStatus.COMPLETED
        assert job.faces_detected == 2


@pytest.mark.unit
class TestVideoMetadataComprehensive:
    """Comprehensive tests for VideoMetadata model."""
    
    def test_metadata_calculations(self):
        """Test all metadata calculations."""
        metadata = VideoMetadata(
            duration=300.5,  # 5 minutes
            fps=29.97,
            resolution=(1920, 1080),
            format="mp4",
            file_size=100 * 1024 * 1024  # 100 MB
        )
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.aspect_ratio == pytest.approx(16/9, rel=1e-2)
        assert metadata.total_frames == int(300.5 * 29.97)
        assert metadata.file_size_mb == pytest.approx(100.0)
        assert metadata.is_hd() is True
        assert metadata.is_4k() is False
        assert metadata.duration_minutes == pytest.approx(5.008, rel=1e-2)
    
    def test_video_quality_detection(self):
        """Test video quality detection methods."""
        # SD video
        sd_metadata = VideoMetadata(60.0, 30, (640, 480), "mp4", 1024*1024)
        assert sd_metadata.is_hd() is False
        assert sd_metadata.is_4k() is False
        
        # HD video
        hd_metadata = VideoMetadata(60.0, 30, (1280, 720), "mp4", 1024*1024)
        assert hd_metadata.is_hd() is True
        assert hd_metadata.is_4k() is False
        
        # 4K video
        uhd_metadata = VideoMetadata(60.0, 30, (3840, 2160), "mp4", 1024*1024)
        assert uhd_metadata.is_hd() is True
        assert uhd_metadata.is_4k() is True
    
    def test_edge_cases(self):
        """Test edge cases in metadata calculations."""
        # Zero duration
        zero_duration = VideoMetadata(0.0, 30, (1920, 1080), "mp4", 1024)
        assert zero_duration.total_frames == 0
        assert zero_duration.duration_minutes == 0.0
        
        # Zero height (should not crash)
        zero_height = VideoMetadata(60.0, 30, (1920, 0), "mp4", 1024)
        assert zero_height.aspect_ratio == 0.0


@pytest.mark.unit
class TestFaceDetectionComprehensive:
    """Comprehensive tests for FaceDetection model."""
    
    def test_face_properties(self):
        """Test all face detection properties."""
        detection = FaceDetection(
            frame_number=42,
            bounding_box=(100, 150, 80, 120),
            confidence=0.85
        )
        
        assert detection.frame_number == 42
        assert detection.x == 100
        assert detection.y == 150
        assert detection.width == 80
        assert detection.height == 120
        assert detection.center == (140, 210)  # (100 + 80/2, 150 + 120/2)
        assert detection.area == 9600  # 80 * 120
        assert detection.confidence == 0.85
    
    def test_face_overlap_calculations(self):
        """Test face overlap calculations with various scenarios."""
        face1 = FaceDetection(1, (0, 0, 100, 100), 0.9)
        
        # Complete overlap (same face)
        face2 = FaceDetection(1, (0, 0, 100, 100), 0.8)
        assert face1.overlaps_with(face2, threshold=0.9) is True
        
        # Partial overlap
        face3 = FaceDetection(1, (50, 50, 100, 100), 0.7)
        iou = face1._calculate_iou(face3)
        assert 0.1 < iou < 0.3  # Should be around 0.14
        
        # No overlap
        face4 = FaceDetection(1, (200, 200, 100, 100), 0.6)
        assert face1.overlaps_with(face4) is False
        
        # Adjacent faces (touching but not overlapping)
        face5 = FaceDetection(1, (100, 0, 100, 100), 0.5)
        assert face1.overlaps_with(face5) is False
    
    def test_face_comparison(self):
        """Test face comparison and sorting."""
        faces = [
            FaceDetection(1, (0, 0, 50, 50), 0.6),    # Small, medium confidence
            FaceDetection(1, (0, 0, 100, 100), 0.9),  # Large, high confidence
            FaceDetection(1, (0, 0, 75, 75), 0.8),    # Medium, high confidence
        ]
        
        # Sort by confidence (descending)
        sorted_faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        assert sorted_faces[0].confidence == 0.9
        assert sorted_faces[1].confidence == 0.8
        assert sorted_faces[2].confidence == 0.6
        
        # Sort by area (descending)
        sorted_by_area = sorted(faces, key=lambda f: f.area, reverse=True)
        assert sorted_by_area[0].area == 10000  # 100x100
        assert sorted_by_area[1].area == 5625   # 75x75
        assert sorted_by_area[2].area == 2500   # 50x50


@pytest.mark.unit
@pytest.mark.requires_opencv
class TestFaceDetectorComprehensive:
    """Comprehensive tests for FaceDetector."""
    
    @patch('cv2.CascadeClassifier')
    def test_detector_configuration_validation(self, mock_cascade_class):
        """Test detector configuration validation."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade_class.return_value = mock_cascade
        
        # Valid configuration
        config = FaceDetectorConfig(
            scale_factor=1.1,
            min_neighbors=3,
            min_size=(20, 20),
            max_size=(500, 500)
        )
        detector = FaceDetector(config)
        assert detector.config.scale_factor == 1.1
        
        # Invalid scale factor
        with pytest.raises(ValueError):
            FaceDetectorConfig(scale_factor=0.9)  # Must be > 1.0
        
        # Invalid min_neighbors
        with pytest.raises(ValueError):
            FaceDetectorConfig(min_neighbors=0)  # Must be > 0
    
    @patch('cv2.CascadeClassifier')
    def test_detection_with_different_image_formats(self, mock_cascade_class):
        """Test face detection with different image formats."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_cascade_class.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # BGR image (standard OpenCV format)
        bgr_image = np.zeros((200, 200, 3), dtype=np.uint8)
        detections_bgr = detector.detect_faces(bgr_image)
        assert len(detections_bgr) == 1
        
        # RGB image (should be converted internally)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        detections_rgb = detector.detect_faces(rgb_image)
        assert len(detections_rgb) == 1
        
        # Grayscale image
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        detections_gray = detector.detect_faces(gray_image)
        assert len(detections_gray) == 1
    
    @patch('cv2.CascadeClassifier')
    def test_batch_processing_performance(self, mock_cascade_class):
        """Test batch processing performance and consistency."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_cascade_class.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Create batch of identical images
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        images = [image.copy() for _ in range(10)]
        
        # Process batch
        batch_results = detector.detect_faces_batch(images, start_frame=0)
        
        assert len(batch_results) == 10
        for i, detections in enumerate(batch_results):
            assert len(detections) == 1
            assert detections[0].frame_number == i
            assert detections[0].bounding_box == (50, 50, 100, 100)


@pytest.mark.unit
class TestFaceBlurrerComprehensive:
    """Comprehensive tests for FaceBlurrer."""
    
    def test_blur_configuration(self):
        """Test blur configuration options."""
        # Default configuration
        default_blurrer = FaceBlurrer()
        assert default_blurrer.config.blur_kernel_size == 99
        assert default_blurrer.config.blur_sigma == 30.0
        
        # Custom configuration
        custom_config = FaceBlurrerConfig(
            blur_kernel_size=51,
            blur_sigma=20.0,
            blur_padding_factor=0.2
        )
        custom_blurrer = FaceBlurrer(custom_config)
        assert custom_blurrer.config.blur_kernel_size == 51
        assert custom_blurrer.config.blur_sigma == 20.0
    
    def test_different_blur_configurations(self):
        """Test different blur configurations."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detection = FaceDetection(0, (50, 50, 100, 100), 0.9)
        
        # Light blur
        light_blurrer = FaceBlurrer(FaceBlurrerConfig(blur_kernel_size=31, blur_sigma=10.0))
        light_result = light_blurrer.blur_faces(image, [detection])
        assert light_result.shape == image.shape
        
        # Heavy blur
        heavy_blurrer = FaceBlurrer(FaceBlurrerConfig(blur_kernel_size=99, blur_sigma=50.0))
        heavy_result = heavy_blurrer.blur_faces(image, [detection])
        assert heavy_result.shape == image.shape
        
        # Results should be different
        assert not np.array_equal(light_result, heavy_result)
    
    def test_blur_with_padding(self):
        """Test blur application with different padding factors."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detection = FaceDetection(0, (60, 60, 80, 80), 0.9)
        
        # No padding
        no_padding_blurrer = FaceBlurrer(FaceBlurrerConfig(blur_padding_factor=0.0))
        result_no_padding = no_padding_blurrer.blur_faces(image, [detection])
        
        # With padding
        padding_blurrer = FaceBlurrer(FaceBlurrerConfig(blur_padding_factor=0.3))
        result_with_padding = padding_blurrer.blur_faces(image, [detection])
        
        # Both should have same shape but different blur areas
        assert result_no_padding.shape == result_with_padding.shape
        assert not np.array_equal(result_no_padding, result_with_padding)
    
    def test_edge_case_detections(self):
        """Test blur application with edge case detections."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        blurrer = FaceBlurrer()
        
        # Face at image edge
        edge_detection = FaceDetection(0, (0, 0, 50, 50), 0.9)
        edge_result = blurrer.blur_faces(image, [edge_detection])
        assert edge_result.shape == image.shape
        
        # Face extending beyond image
        beyond_detection = FaceDetection(0, (150, 150, 100, 100), 0.9)
        beyond_result = blurrer.blur_faces(image, [beyond_detection])
        assert beyond_result.shape == image.shape
        
        # Very small face
        tiny_detection = FaceDetection(0, (100, 100, 5, 5), 0.9)
        tiny_result = blurrer.blur_faces(image, [tiny_detection])
        assert tiny_result.shape == image.shape


@pytest.mark.unit
class TestFileManagerComprehensive:
    """Comprehensive tests for FileManager."""
    
    def test_file_manager_initialization(self, temp_dir):
        """Test FileManager initialization."""
        fm = FileManager(base_path=str(temp_dir))
        assert fm.base_path == Path(temp_dir)
        assert fm.base_path.exists()
    
    def test_secure_filename_generation(self, temp_dir):
        """Test secure filename generation."""
        fm = FileManager(base_path=str(temp_dir))
        
        # Normal filename
        secure_name = fm.get_secure_filename("test_video.mp4")
        assert secure_name.endswith(".mp4")
        assert "test_video" in secure_name
        
        # Dangerous filename
        dangerous_name = "../../../etc/passwd"
        secure_dangerous = fm.get_secure_filename(dangerous_name)
        assert ".." not in secure_dangerous
        assert "/" not in secure_dangerous
    
    def test_file_operations(self, temp_dir):
        """Test file save, load, and delete operations."""
        fm = FileManager(base_path=str(temp_dir))
        
        # Save file
        test_content = b"test video content"
        saved_path = fm.save_uploaded_file(test_content, "test.mp4")
        assert saved_path.exists()
        assert saved_path.read_bytes() == test_content
        
        # Load file
        loaded_content = fm.load_file(saved_path)
        assert loaded_content == test_content
        
        # Delete file
        fm.delete_file(saved_path)
        assert not saved_path.exists()
    
    def test_cleanup_operations(self, temp_dir):
        """Test cleanup operations."""
        fm = FileManager(base_path=str(temp_dir))
        
        # Create test files with different ages
        old_file = temp_dir / "old_file.mp4"
        recent_file = temp_dir / "recent_file.mp4"
        
        old_file.write_bytes(b"old content")
        recent_file.write_bytes(b"recent content")
        
        # Modify timestamps
        old_time = datetime.now().timestamp() - 3600  # 1 hour ago
        os.utime(old_file, (old_time, old_time))
        
        # Cleanup old files (older than 30 minutes)
        cleaned_count = fm.cleanup_old_files(max_age_minutes=30)
        
        assert cleaned_count >= 1
        assert not old_file.exists()
        assert recent_file.exists()


@pytest.mark.unit
@pytest.mark.requires_redis
class TestJobQueueComprehensive:
    """Comprehensive tests for JobQueue."""
    
    def test_job_queue_operations(self, mock_redis):
        """Test basic job queue operations."""
        queue = JobQueue()
        
        # Create and add job
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        
        # Mock Redis responses
        mock_redis.hset.return_value = True
        mock_redis.hget.return_value = job.to_json()
        mock_redis.hgetall.return_value = {job.job_id: job.to_json()}
        
        # Add job to queue
        queue.add_job(job)
        mock_redis.hset.assert_called()
        
        # Get job from queue
        retrieved_job = queue.get_job(job.job_id)
        assert retrieved_job is not None
        assert retrieved_job.job_id == job.job_id
        
        # Update job status
        job.mark_completed("/path/output.mp4")
        queue.update_job(job)
        
        # Get queue stats
        stats = queue.get_stats()
        assert isinstance(stats, dict)
    
    def test_job_queue_error_handling(self, mock_redis):
        """Test job queue error handling."""
        queue = JobQueue()
        
        # Test Redis connection error
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        with pytest.raises(Exception):
            queue._check_connection()
        
        # Test job not found
        mock_redis.hget.return_value = None
        job = queue.get_job("nonexistent-job")
        assert job is None


@pytest.mark.unit
class TestValidationComprehensive:
    """Comprehensive validation tests."""
    
    @patch('magic.from_file')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_complete_file_validation(self, mock_getsize, mock_exists, mock_magic):
        """Test complete file validation workflow."""
        from models.validation import validate_video_file, validate_file_size, validate_filename
        
        # Setup mocks for valid file
        mock_exists.return_value = True
        mock_getsize.return_value = 50 * 1024 * 1024  # 50 MB
        mock_magic.return_value = 'video/mp4'
        
        # Should pass all validations
        validate_filename("golf_swing.mp4")
        validate_file_size("/path/golf_swing.mp4")
        validate_video_file("/path/golf_swing.mp4")
    
    def test_filename_validation_edge_cases(self):
        """Test filename validation edge cases."""
        from models.validation import validate_filename, ValidationError
        
        # Test various invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in invalid_chars:
            with pytest.raises(ValidationError):
                validate_filename(f"test{char}file.mp4")
        
        # Test path traversal attempts
        path_traversal_attempts = [
            "../video.mp4",
            "..\\video.mp4",
            "folder/../video.mp4",
            "folder\\..\\video.mp4"
        ]
        for attempt in path_traversal_attempts:
            with pytest.raises(ValidationError):
                validate_filename(attempt)
        
        # Test valid filenames
        valid_names = [
            "video.mp4",
            "golf_swing_2023.mov",
            "test-video-123.avi",
            "My Video (1).mp4"
        ]
        for name in valid_names:
            validate_filename(name)  # Should not raise


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling across all components."""
    
    def test_processing_job_error_states(self):
        """Test processing job error state handling."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        
        # Test various error scenarios
        error_messages = [
            "File not found",
            "Corrupted video file",
            "Insufficient memory",
            "Processing timeout",
            "Unknown error occurred"
        ]
        
        for error_msg in error_messages:
            job_copy = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
            job_copy.mark_failed(error_msg)
            
            assert job_copy.status == JobStatus.FAILED
            assert job_copy.error_message == error_msg
            assert job_copy.is_complete is True
    
    @patch('cv2.CascadeClassifier')
    def test_face_detector_error_recovery(self, mock_cascade_class):
        """Test face detector error recovery."""
        # Test with failing cascade
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = True  # Failed to load
        mock_cascade_class.return_value = mock_cascade
        
        with pytest.raises(RuntimeError):
            FaceDetector()
    
    def test_file_manager_error_handling(self, temp_dir):
        """Test file manager error handling."""
        fm = FileManager(base_path=str(temp_dir))
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            fm.load_file(temp_dir / "nonexistent.mp4")
        
        # Test saving to invalid path
        with pytest.raises(Exception):
            fm.save_uploaded_file(b"content", "/invalid/path/file.mp4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])