"""
Integration tests for the VideoProcessor class.

Tests the complete video processing pipeline including face detection,
blurring, video encoding, and audio preservation.
"""

import pytest
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from processing.video_processor import VideoProcessor, VideoProcessorConfig, VideoProcessingError
from models.processing_job import ProcessingJob, JobStatus
from models.video_metadata import VideoMetadata
from models.face_detection import FaceDetection
from storage.file_manager import FileManager


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create FileManager instance for testing."""
        return FileManager(base_storage_path=str(temp_dir / 'storage'), 
                          temp_dir=str(temp_dir / 'temp'))
    
    @pytest.fixture
    def config(self):
        """Create VideoProcessorConfig for testing."""
        return VideoProcessorConfig(
            output_quality=0.9,
            preserve_audio=False,  # Disable for most tests
            max_frames_per_batch=10,
            progress_callback_interval=5
        )
    
    @pytest.fixture
    def processor(self, config, file_manager):
        """Create VideoProcessor instance for testing."""
        return VideoProcessor(config, file_manager)
    
    @pytest.fixture
    def sample_job(self, temp_dir):
        """Create sample processing job."""
        input_file = temp_dir / 'input.mp4'
        input_file.touch()  # Create empty file
        
        return ProcessingJob.create_new(
            original_filename='test_video.mp4',
            file_path=str(input_file)
        )
    
    @pytest.fixture
    def mock_video_file(self, temp_dir):
        """Create a mock video file for testing."""
        video_path = temp_dir / 'test_video.mp4'
        
        # Create a simple test video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # Create 30 frames (1 second at 30fps)
        for i in range(30):
            # Create frame with some content
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some variation to make it interesting
            frame[100:200, 100:200] = [50 + i, 100 + i, 150 + i]
            out.write(frame)
        
        out.release()
        return video_path
    
    def test_processor_initialization(self, config, file_manager):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor(config, file_manager)
        
        assert processor.config == config
        assert processor.file_manager == file_manager
        assert processor.face_detector is not None
        assert processor.face_blurrer is not None
        assert processor.current_job is None
        assert processor.video_metadata is None
        assert processor.total_faces_detected == 0
    
    def test_processor_initialization_with_defaults(self):
        """Test VideoProcessor initialization with default parameters."""
        processor = VideoProcessor()
        
        assert processor.config is not None
        assert processor.file_manager is not None
        assert processor.face_detector is not None
        assert processor.face_blurrer is not None
    
    def test_extract_video_metadata(self, processor, mock_video_file):
        """Test video metadata extraction."""
        metadata = processor._extract_video_metadata(mock_video_file)
        
        assert isinstance(metadata, VideoMetadata)
        assert metadata.fps == 30
        assert metadata.resolution == (640, 480)
        assert metadata.duration > 0
        assert metadata.file_size > 0
        assert metadata.format == 'mp4'
    
    def test_extract_video_metadata_invalid_file(self, processor, temp_dir):
        """Test metadata extraction with invalid video file."""
        invalid_file = temp_dir / 'invalid.mp4'
        invalid_file.write_text('not a video file')
        
        with pytest.raises(VideoProcessingError, match="Failed to extract video metadata"):
            processor._extract_video_metadata(invalid_file)
    
    def test_extract_video_metadata_nonexistent_file(self, processor, temp_dir):
        """Test metadata extraction with nonexistent file."""
        nonexistent_file = temp_dir / 'nonexistent.mp4'
        
        with pytest.raises(VideoProcessingError, match="Failed to extract video metadata"):
            processor._extract_video_metadata(nonexistent_file)
    
    def test_create_output_path(self, processor):
        """Test output path creation."""
        job_id = 'test-job-123'
        input_path = Path('input.mp4')
        
        output_path = processor._create_output_path(job_id, input_path)
        
        assert output_path.name.startswith(job_id)
        assert output_path.name.endswith('.mp4')
        assert 'anonymized' in output_path.name
        assert output_path.parent == processor.file_manager.temp_dir
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_process_single_frame(self, mock_writer, mock_capture, processor):
        """Test single frame processing."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_number = 5
        
        # Mock face detection to return one face
        mock_detection = FaceDetection(
            frame_number=frame_number,
            bounding_box=(100, 100, 50, 50),
            confidence=0.8
        )
        
        with patch.object(processor.face_detector, 'detect_faces', return_value=[mock_detection]):
            with patch.object(processor.face_blurrer, 'blur_faces', return_value=frame) as mock_blur:
                result = processor._process_single_frame(frame, frame_number)
                
                # Verify face detection was called
                processor.face_detector.detect_faces.assert_called_once_with(frame, frame_number)
                
                # Verify blurring was called
                mock_blur.assert_called_once_with(frame, [mock_detection])
                
                # Verify face count was updated
                assert processor.total_faces_detected == 1
                
                # Verify result
                assert np.array_equal(result, frame)
    
    def test_process_single_frame_no_faces(self, processor):
        """Test single frame processing with no faces detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_number = 0
        
        with patch.object(processor.face_detector, 'detect_faces', return_value=[]):
            with patch.object(processor.face_blurrer, 'blur_faces', return_value=frame):
                result = processor._process_single_frame(frame, frame_number)
                
                assert processor.total_faces_detected == 0
                assert np.array_equal(result, frame)
    
    def test_process_single_frame_error_handling(self, processor):
        """Test single frame processing with error."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_number = 0
        
        with patch.object(processor.face_detector, 'detect_faces', side_effect=Exception("Detection error")):
            result = processor._process_single_frame(frame, frame_number)
            
            # Should return original frame on error
            assert np.array_equal(result, frame)
    
    def test_process_single_frame_empty_frame(self, processor):
        """Test single frame processing with empty frame."""
        empty_frame = np.array([])
        result = processor._process_single_frame(empty_frame, 0)
        
        assert np.array_equal(result, empty_frame)
    
    @patch('subprocess.run')
    def test_preserve_audio_track_success(self, mock_subprocess, processor, temp_dir):
        """Test successful audio track preservation."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        input_path.touch()
        output_path.touch()
        
        # Mock successful ffmpeg execution
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        progress_callback = Mock()
        
        processor._preserve_audio_track(input_path, output_path, progress_callback)
        
        # Verify ffmpeg was called
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert 'ffmpeg' in args
        assert str(input_path) in args
        
        # Verify progress callback was called
        progress_callback.assert_called_once_with(0.9, "Preserving audio track...")
    
    @patch('subprocess.run')
    def test_preserve_audio_track_failure(self, mock_subprocess, processor, temp_dir):
        """Test audio track preservation failure."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        input_path.touch()
        output_path.touch()
        
        # Mock failed ffmpeg execution
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "ffmpeg error"
        
        # Should not raise exception, just log warning
        processor._preserve_audio_track(input_path, output_path)
        
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.run')
    def test_preserve_audio_track_ffmpeg_not_found(self, mock_subprocess, processor, temp_dir):
        """Test audio track preservation when ffmpeg is not found."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        input_path.touch()
        output_path.touch()
        
        # Mock FileNotFoundError (ffmpeg not installed)
        mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")
        
        # Should not raise exception, just log warning
        processor._preserve_audio_track(input_path, output_path)
        
        mock_subprocess.assert_called_once()
    
    def test_get_processing_stats_no_job(self, processor):
        """Test getting processing stats with no current job."""
        stats = processor.get_processing_stats()
        
        assert stats['job_id'] is None
        assert stats['total_faces_detected'] == 0
        assert stats['active_face_tracks'] == 0
        assert isinstance(stats['face_track_info'], list)
    
    def test_get_processing_stats_with_job_and_metadata(self, processor, sample_job):
        """Test getting processing stats with job and metadata."""
        processor.current_job = sample_job
        processor.total_faces_detected = 5
        processor.video_metadata = VideoMetadata(
            duration=10.0,
            fps=30,
            resolution=(640, 480),
            format='mp4',
            file_size=1024000
        )
        
        stats = processor.get_processing_stats()
        
        assert stats['job_id'] == sample_job.job_id
        assert stats['total_faces_detected'] == 5
        assert stats['video_duration'] == 10.0
        assert stats['video_fps'] == 30
        assert stats['video_resolution'] == (640, 480)
        assert stats['video_format'] == 'mp4'
        assert stats['total_frames'] == 300
    
    def test_update_config(self, processor):
        """Test configuration updates."""
        original_quality = processor.config.output_quality
        
        processor.update_config(output_quality=0.8, preserve_audio=True)
        
        assert processor.config.output_quality == 0.8
        assert processor.config.preserve_audio is True
        assert processor.config.output_quality != original_quality
    
    def test_update_config_invalid_parameter(self, processor):
        """Test configuration update with invalid parameter."""
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            processor.update_config(invalid_parameter=123)
    
    def test_cleanup_resources(self, processor, sample_job):
        """Test resource cleanup."""
        # Set up some state
        processor.current_job = sample_job
        processor.total_faces_detected = 10
        processor.video_metadata = VideoMetadata(
            duration=5.0, fps=30, resolution=(640, 480), format='mp4', file_size=1000
        )
        
        processor.cleanup_resources()
        
        assert processor.current_job is None
        assert processor.video_metadata is None
        assert processor.total_faces_detected == 0
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_process_video_frames_basic(self, mock_writer_class, mock_capture_class, processor, temp_dir):
        """Test basic video frame processing."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        
        # Mock video capture
        mock_capture = Mock()
        mock_capture_class.return_value = mock_capture
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FRAME_COUNT: 10
        }.get(prop, 0)
        
        # Mock frame reading - return 10 frames then end (to trigger progress callback)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.read.side_effect = (
            [(True, test_frame)] * 10 + [(False, None)]  # 10 frames then end
        )
        
        # Mock video writer
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer.isOpened.return_value = True
        
        # Mock frame processing
        with patch.object(processor, '_process_single_frame', return_value=test_frame) as mock_process:
            progress_callback = Mock()
            
            processor._process_video_frames(input_path, output_path, progress_callback)
            
            # Verify video capture was opened
            mock_capture_class.assert_called_once_with(str(input_path))
            
            # Verify video writer was created
            mock_writer_class.assert_called_once()
            
            # Verify frames were processed
            assert mock_process.call_count == 10
            
            # Verify frames were written
            assert mock_writer.write.call_count == 10
            
            # Verify progress callback was called
            assert progress_callback.call_count > 0
    
    @patch('cv2.VideoCapture')
    def test_process_video_frames_capture_failure(self, mock_capture_class, processor, temp_dir):
        """Test video frame processing with capture failure."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        
        # Mock failed video capture
        mock_capture = Mock()
        mock_capture_class.return_value = mock_capture
        mock_capture.isOpened.return_value = False
        
        with pytest.raises(VideoProcessingError, match="Could not open input video"):
            processor._process_video_frames(input_path, output_path)
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_process_video_frames_writer_failure(self, mock_writer_class, mock_capture_class, processor, temp_dir):
        """Test video frame processing with writer failure."""
        input_path = temp_dir / 'input.mp4'
        output_path = temp_dir / 'output.mp4'
        
        # Mock successful video capture
        mock_capture = Mock()
        mock_capture_class.return_value = mock_capture
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FRAME_COUNT: 10
        }.get(prop, 0)
        
        # Mock failed video writer
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        mock_writer.isOpened.return_value = False
        
        with pytest.raises(VideoProcessingError, match="Could not create output video writer"):
            processor._process_video_frames(input_path, output_path)


class TestVideoProcessorIntegration:
    """Integration tests for complete video processing workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def processor(self, temp_dir):
        """Create VideoProcessor for integration testing."""
        file_manager = FileManager(
            base_storage_path=str(temp_dir / 'storage'),
            temp_dir=str(temp_dir / 'temp')
        )
        config = VideoProcessorConfig(preserve_audio=False)  # Disable audio for testing
        return VideoProcessor(config, file_manager)
    
    def test_complete_processing_workflow_mock(self, processor, temp_dir):
        """Test complete video processing workflow with mocked components."""
        # Create job
        input_file = temp_dir / 'input.mp4'
        input_file.touch()
        job = ProcessingJob.create_new('test.mp4', str(input_file))
        
        # Mock all the heavy lifting
        mock_metadata = VideoMetadata(
            duration=2.0, fps=30, resolution=(640, 480), format='mp4', file_size=1000
        )
        
        with patch.object(processor, '_extract_video_metadata', return_value=mock_metadata):
            with patch.object(processor, '_process_video_frames') as mock_process_frames:
                with patch.object(processor, '_create_output_path') as mock_create_path:
                    output_path = temp_dir / 'output.mp4'
                    output_path.touch()  # Create the output file
                    mock_create_path.return_value = output_path
                    
                    progress_callback = Mock()
                    
                    result = processor.process_video(job, progress_callback)
                    
                    # Verify job status progression
                    assert job.status == JobStatus.COMPLETED
                    assert job.output_file_path == str(output_path)
                    
                    # Verify methods were called
                    processor._extract_video_metadata.assert_called_once()
                    mock_process_frames.assert_called_once()
                    
                    # Verify progress callbacks
                    assert progress_callback.call_count >= 2
                    
                    # Verify result
                    assert result == str(output_path)
    
    def test_processing_workflow_with_missing_input(self, processor, temp_dir):
        """Test processing workflow with missing input file."""
        # Create job with non-existent file
        job = ProcessingJob.create_new('missing.mp4', str(temp_dir / 'missing.mp4'))
        
        with pytest.raises(VideoProcessingError, match="Input video file not found"):
            processor.process_video(job)
        
        # Verify job was marked as failed
        assert job.status == JobStatus.FAILED
        assert "Input video file not found" in job.error_message
    
    def test_processing_workflow_with_metadata_error(self, processor, temp_dir):
        """Test processing workflow with metadata extraction error."""
        # Create job
        input_file = temp_dir / 'input.mp4'
        input_file.touch()
        job = ProcessingJob.create_new('test.mp4', str(input_file))
        
        with patch.object(processor, '_extract_video_metadata', side_effect=VideoProcessingError("Metadata error")):
            with pytest.raises(VideoProcessingError, match="Video processing failed"):
                processor.process_video(job)
            
            # Verify job was marked as failed
            assert job.status == JobStatus.FAILED
            assert "Metadata error" in job.error_message
    
    def test_processing_workflow_cleanup_on_error(self, processor, temp_dir):
        """Test that partial output files are cleaned up on error."""
        # Create job
        input_file = temp_dir / 'input.mp4'
        input_file.touch()
        job = ProcessingJob.create_new('test.mp4', str(input_file))
        
        mock_metadata = VideoMetadata(
            duration=2.0, fps=30, resolution=(640, 480), format='mp4', file_size=1000
        )
        
        with patch.object(processor, '_extract_video_metadata', return_value=mock_metadata):
            with patch.object(processor, '_create_output_path') as mock_create_path:
                with patch.object(processor, '_process_video_frames', side_effect=Exception("Processing error")):
                    output_path = temp_dir / 'output.mp4'
                    output_path.touch()  # Create the output file
                    mock_create_path.return_value = output_path
                    
                    with pytest.raises(VideoProcessingError):
                        processor.process_video(job)
                    
                    # Verify output file was cleaned up
                    assert not output_path.exists()
                    
                    # Verify job was marked as failed
                    assert job.status == JobStatus.FAILED