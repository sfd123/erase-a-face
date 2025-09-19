"""Unit tests for data models and validation functions."""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, mock_open

from models.processing_job import ProcessingJob, JobStatus
from models.video_metadata import VideoMetadata
from models.face_detection import FaceDetection
from models.validation import (
    validate_video_file, 
    validate_file_size, 
    validate_filename,
    ValidationError,
    get_supported_formats,
    get_max_file_size_mb
)


class TestProcessingJob:
    """Test cases for ProcessingJob data model."""
    
    def test_create_new_job(self):
        """Test creating a new processing job."""
        job = ProcessingJob.create_new("test_video.mp4", "/path/to/video.mp4")
        
        assert job.original_filename == "test_video.mp4"
        assert job.file_path == "/path/to/video.mp4"
        assert job.status == JobStatus.PENDING
        assert job.job_id is not None
        assert len(job.job_id) == 36  # UUID4 length
        assert isinstance(job.created_at, datetime)
        assert job.completed_at is None
        assert job.error_message is None
        assert job.output_file_path is None
        assert job.faces_detected == 0
    
    def test_mark_processing(self):
        """Test marking job as processing."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        job.mark_processing()
        
        assert job.status == JobStatus.PROCESSING
    
    def test_mark_completed(self):
        """Test marking job as completed."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        output_path = "/path/output.mp4"
        faces_count = 3
        
        job.mark_completed(output_path, faces_count)
        
        assert job.status == JobStatus.COMPLETED
        assert job.output_file_path == output_path
        assert job.faces_detected == faces_count
        assert job.completed_at is not None
        assert job.is_complete is True
    
    def test_mark_failed(self):
        """Test marking job as failed."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        error_msg = "Processing failed due to corrupted video"
        
        job.mark_failed(error_msg)
        
        assert job.status == JobStatus.FAILED
        assert job.error_message == error_msg
        assert job.completed_at is not None
        assert job.is_complete is True
    
    def test_processing_duration(self):
        """Test processing duration calculation."""
        job = ProcessingJob.create_new("test.mp4", "/path/test.mp4")
        
        # Before completion, duration should be None
        assert job.processing_duration is None
        
        # After completion, should calculate duration
        job.mark_completed("/path/output.mp4")
        duration = job.processing_duration
        
        assert duration is not None
        assert duration >= 0


class TestVideoMetadata:
    """Test cases for VideoMetadata data model."""
    
    def test_video_metadata_properties(self):
        """Test VideoMetadata properties and calculations."""
        metadata = VideoMetadata(
            duration=120.5,
            fps=30,
            resolution=(1920, 1080),
            format="mp4",
            file_size=50 * 1024 * 1024  # 50 MB
        )
        
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.aspect_ratio == pytest.approx(1920/1080)
        assert metadata.total_frames == int(120.5 * 30)
        assert metadata.file_size_mb == pytest.approx(50.0)
        assert metadata.is_hd() is True
        assert metadata.is_4k() is False
    
    def test_4k_video_detection(self):
        """Test 4K video detection."""
        metadata = VideoMetadata(
            duration=60.0,
            fps=24,
            resolution=(3840, 2160),
            format="mp4",
            file_size=100 * 1024 * 1024
        )
        
        assert metadata.is_4k() is True
        assert metadata.is_hd() is True
    
    def test_aspect_ratio_edge_case(self):
        """Test aspect ratio calculation with zero height."""
        metadata = VideoMetadata(
            duration=10.0,
            fps=30,
            resolution=(1920, 0),
            format="mp4",
            file_size=1024
        )
        
        assert metadata.aspect_ratio == 0.0


class TestFaceDetection:
    """Test cases for FaceDetection data model."""
    
    def test_face_detection_properties(self):
        """Test FaceDetection properties and calculations."""
        detection = FaceDetection(
            frame_number=100,
            bounding_box=(50, 75, 100, 120),
            confidence=0.95
        )
        
        assert detection.x == 50
        assert detection.y == 75
        assert detection.width == 100
        assert detection.height == 120
        assert detection.center == (100, 135)  # (50 + 100/2, 75 + 120/2)
        assert detection.area == 12000  # 100 * 120
    
    def test_face_overlap_detection(self):
        """Test face overlap detection using IoU."""
        face1 = FaceDetection(1, (0, 0, 100, 100), 0.9)
        face2 = FaceDetection(1, (50, 50, 100, 100), 0.8)  # ~14.3% IoU overlap
        face3 = FaceDetection(1, (200, 200, 100, 100), 0.7)  # No overlap
        
        # Test overlap with default threshold (0.5) - should be False since IoU is ~0.143
        assert face1.overlaps_with(face2) is False
        assert face1.overlaps_with(face3) is False
        
        # Test with custom threshold - should be True with low threshold
        assert face1.overlaps_with(face2, threshold=0.1) is True
        assert face1.overlaps_with(face2, threshold=0.2) is False
    
    def test_no_intersection_overlap(self):
        """Test overlap detection when faces don't intersect."""
        face1 = FaceDetection(1, (0, 0, 50, 50), 0.9)
        face2 = FaceDetection(1, (100, 100, 50, 50), 0.8)
        
        assert face1.overlaps_with(face2) is False


class TestValidation:
    """Test cases for validation functions."""
    
    @patch('magic.from_file')
    @patch('os.path.exists')
    def test_validate_video_file_success(self, mock_exists, mock_magic):
        """Test successful video file validation."""
        mock_exists.return_value = True
        mock_magic.return_value = 'video/mp4'
        
        # Should not raise exception for valid MP4 file
        validate_video_file('/path/to/video.mp4')
    
    @patch('os.path.exists')
    def test_validate_video_file_not_found(self, mock_exists):
        """Test validation with non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            validate_video_file('/path/to/nonexistent.mp4')
    
    @patch('magic.from_file')
    @patch('os.path.exists')
    def test_validate_video_file_unsupported_format(self, mock_exists, mock_magic):
        """Test validation with unsupported file format."""
        mock_exists.return_value = True
        mock_magic.return_value = 'image/jpeg'
        
        with pytest.raises(ValidationError) as exc_info:
            validate_video_file('/path/to/image.jpg')
        
        assert "Unsupported video format" in str(exc_info.value)
    
    @patch('magic.from_file')
    @patch('os.path.exists')
    def test_validate_video_file_extension_mismatch(self, mock_exists, mock_magic):
        """Test validation with mismatched extension and MIME type."""
        mock_exists.return_value = True
        mock_magic.return_value = 'video/mp4'
        
        with pytest.raises(ValidationError) as exc_info:
            validate_video_file('/path/to/video.avi')  # MP4 content with AVI extension
        
        assert "does not match detected format" in str(exc_info.value)
    
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_validate_file_size_success(self, mock_exists, mock_getsize):
        """Test successful file size validation."""
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024 * 1024  # 10 MB
        
        # Should not raise exception for valid file size
        validate_file_size('/path/to/video.mp4')
    
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_validate_file_size_too_large(self, mock_exists, mock_getsize):
        """Test validation with file too large."""
        mock_exists.return_value = True
        mock_getsize.return_value = 600 * 1024 * 1024  # 600 MB (over 500 MB limit)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_size('/path/to/large_video.mp4')
        
        assert "File too large" in str(exc_info.value)
    
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_validate_file_size_too_small(self, mock_exists, mock_getsize):
        """Test validation with file too small."""
        mock_exists.return_value = True
        mock_getsize.return_value = 500  # 500 bytes (under 1 KB limit)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_size('/path/to/tiny_video.mp4')
        
        assert "File too small" in str(exc_info.value)
    
    def test_validate_filename_success(self):
        """Test successful filename validation."""
        # Should not raise exception for valid filenames
        validate_filename('video.mp4')
        validate_filename('my_golf_swing.mov')
        validate_filename('test-video-123.avi')
    
    def test_validate_filename_empty(self):
        """Test validation with empty filename."""
        with pytest.raises(ValidationError) as exc_info:
            validate_filename('')
        
        assert "Filename cannot be empty" in str(exc_info.value)
    
    def test_validate_filename_dangerous_characters(self):
        """Test validation with dangerous characters."""
        dangerous_filenames = [
            '../video.mp4',
            'video/file.mp4',
            'video\\file.mp4',
            'video:file.mp4',
            'video*.mp4',
            'video?.mp4',
            'video"file.mp4',
            'video<file.mp4',
            'video>file.mp4',
            'video|file.mp4'
        ]
        
        for filename in dangerous_filenames:
            with pytest.raises(ValidationError) as exc_info:
                validate_filename(filename)
            assert "invalid character" in str(exc_info.value)
    
    def test_validate_filename_too_long(self):
        """Test validation with filename too long."""
        long_filename = 'a' * 252 + '.mp4'  # 256 characters total (over 255 limit)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_filename(long_filename)
        
        assert "Filename too long" in str(exc_info.value)
    
    def test_validate_filename_invalid_extension(self):
        """Test validation with invalid file extension."""
        with pytest.raises(ValidationError) as exc_info:
            validate_filename('video.txt')
        
        assert "Invalid file extension" in str(exc_info.value)
    
    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = get_supported_formats()
        
        assert '.mp4' in formats
        assert '.mov' in formats
        assert '.avi' in formats
        assert len(formats) >= 3
    
    def test_get_max_file_size_mb(self):
        """Test getting maximum file size in MB."""
        max_size = get_max_file_size_mb()
        
        assert max_size == 500.0