"""
Pytest configuration and shared fixtures for the Golf Video Anonymizer test suite.
"""

import pytest
import tempfile
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import redis
from fastapi.testclient import TestClient

from main import app
from storage.job_queue import JobQueue
from storage.file_manager import FileManager
from processing.face_detector import FaceDetector
from processing.face_blurrer import FaceBlurrer
from processing.video_processor import VideoProcessor


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample test image with a face-like rectangle."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Add a face-like rectangle
    cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
    # Add some facial features
    cv2.circle(image, (80, 80), 5, (255, 255, 255), -1)  # Left eye
    cv2.circle(image, (120, 80), 5, (255, 255, 255), -1)  # Right eye
    cv2.rectangle(image, (95, 110), (105, 120), (255, 255, 255), -1)  # Nose
    cv2.rectangle(image, (85, 130), (115, 140), (255, 255, 255), -1)  # Mouth
    return image


@pytest.fixture
def sample_video_frames():
    """Create a sequence of sample video frames."""
    frames = []
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving face across frames
        x_pos = 50 + i * 10
        cv2.rectangle(frame, (x_pos, 50), (x_pos + 80, 130), (128, 128, 128), -1)
        # Add facial features
        cv2.circle(frame, (x_pos + 20, 70), 3, (255, 255, 255), -1)  # Left eye
        cv2.circle(frame, (x_pos + 60, 70), 3, (255, 255, 255), -1)  # Right eye
        cv2.rectangle(frame, (x_pos + 35, 90), (x_pos + 45, 100), (255, 255, 255), -1)  # Nose
        cv2.rectangle(frame, (x_pos + 25, 110), (x_pos + 55, 120), (255, 255, 255), -1)  # Mouth
        frames.append(frame)
    return frames


@pytest.fixture
def sample_video_file(temp_dir, sample_video_frames):
    """Create a sample video file for testing."""
    video_path = temp_dir / "sample_video.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = sample_video_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    
    # Write frames
    for frame in sample_video_frames:
        writer.write(frame)
    
    writer.release()
    return video_path


@pytest.fixture
def large_video_file(temp_dir):
    """Create a large video file for performance testing."""
    video_path = temp_dir / "large_video.mp4"
    
    # Create a larger video (100 frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    for i in range(100):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a moving face
        x_pos = (i * 5) % 500
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 100, 200), (128, 128, 128), -1)
        writer.write(frame)
    
    writer.release()
    return video_path


@pytest.fixture
def corrupted_video_file(temp_dir):
    """Create a corrupted video file for error testing."""
    video_path = temp_dir / "corrupted_video.mp4"
    # Write invalid video data
    with open(video_path, 'wb') as f:
        f.write(b"This is not a valid video file content")
    return video_path


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing."""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        
        # Mock Redis operations
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.hset.return_value = True
        mock_redis_instance.hget.return_value = None
        mock_redis_instance.hgetall.return_value = {}
        mock_redis_instance.delete.return_value = 1
        mock_redis_instance.keys.return_value = []
        
        yield mock_redis_instance


@pytest.fixture
def mock_face_detector():
    """Mock FaceDetector for testing."""
    with patch('processing.face_detector.FaceDetector') as mock_detector_class:
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        
        # Default behavior: detect one face
        from models.face_detection import FaceDetection
        mock_detector.detect_faces.return_value = [
            FaceDetection(0, (50, 50, 100, 100), 0.9)
        ]
        
        yield mock_detector


@pytest.fixture
def mock_opencv_cascades():
    """Mock OpenCV cascade classifiers."""
    with patch('cv2.CascadeClassifier') as mock_cascade_class:
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_cascade_class.return_value = mock_cascade
        yield mock_cascade


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def file_manager(temp_dir):
    """Create FileManager instance with temporary directory."""
    return FileManager(base_path=str(temp_dir))


@pytest.fixture
def job_queue(mock_redis):
    """Create JobQueue instance with mocked Redis."""
    return JobQueue()


@pytest.fixture
def face_detector(mock_opencv_cascades):
    """Create FaceDetector instance with mocked cascades."""
    return FaceDetector()


@pytest.fixture
def face_blurrer():
    """Create FaceBlurrer instance."""
    return FaceBlurrer()


@pytest.fixture
def video_processor(mock_face_detector, face_blurrer, file_manager):
    """Create VideoProcessor instance with mocked dependencies."""
    return VideoProcessor()


# Test data creation helpers
def create_test_video_with_faces(output_path: Path, num_frames: int = 30, fps: int = 30):
    """Create a test video with moving faces."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (320, 240))
    
    for i in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add background
        frame[:] = (50, 50, 50)
        
        # Add moving face
        x_pos = 50 + (i * 5) % 200
        y_pos = 50 + (i * 2) % 100
        
        # Face rectangle
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 60, y_pos + 80), (150, 150, 150), -1)
        
        # Facial features
        cv2.circle(frame, (x_pos + 15, y_pos + 20), 3, (255, 255, 255), -1)  # Left eye
        cv2.circle(frame, (x_pos + 45, y_pos + 20), 3, (255, 255, 255), -1)  # Right eye
        cv2.rectangle(frame, (x_pos + 25, y_pos + 35), (x_pos + 35, y_pos + 45), (255, 255, 255), -1)  # Nose
        cv2.rectangle(frame, (x_pos + 20, y_pos + 55), (x_pos + 40, y_pos + 65), (255, 255, 255), -1)  # Mouth
        
        writer.write(frame)
    
    writer.release()


def create_test_video_no_faces(output_path: Path, num_frames: int = 30, fps: int = 30):
    """Create a test video without faces."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (320, 240))
    
    for i in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add some geometric shapes (no faces)
        cv2.rectangle(frame, (50, 50), (100, 100), (100, 100, 100), -1)
        cv2.circle(frame, (200, 150), 30, (150, 150, 150), -1)
        
        writer.write(frame)
    
    writer.release()


def create_test_video_multiple_faces(output_path: Path, num_frames: int = 30, fps: int = 30):
    """Create a test video with multiple faces."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))
    
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        
        # Face 1 - moving left to right
        x1 = 50 + (i * 3) % 400
        cv2.rectangle(frame, (x1, 100), (x1 + 60, 180), (150, 150, 150), -1)
        cv2.circle(frame, (x1 + 15, 120), 3, (255, 255, 255), -1)
        cv2.circle(frame, (x1 + 45, 120), 3, (255, 255, 255), -1)
        
        # Face 2 - stationary
        x2, y2 = 300, 250
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 100), (140, 140, 140), -1)
        cv2.circle(frame, (x2 + 20, y2 + 25), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x2 + 60, y2 + 25), 4, (255, 255, 255), -1)
        
        writer.write(frame)
    
    writer.release()


# Pytest hooks for test organization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for complete workflows")
    config.addinivalue_line("markers", "performance: Performance tests with various conditions")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "requires_redis: Tests that require Redis connection")
    config.addinivalue_line("markers", "requires_opencv: Tests that require OpenCV functionality")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file names
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add markers based on test function names
        if "redis" in item.name.lower():
            item.add_marker(pytest.mark.requires_redis)
        
        if any(opencv_term in item.name.lower() for opencv_term in ["face", "video", "opencv", "detector", "blurrer"]):
            item.add_marker(pytest.mark.requires_opencv)
        
        # Mark slow tests
        if any(slow_term in item.name.lower() for slow_term in ["large", "batch", "performance", "stress"]):
            item.add_marker(pytest.mark.slow)


# Test data setup for session
@pytest.fixture(scope="session", autouse=True)
def setup_test_data(test_data_dir):
    """Set up test data files for the entire test session."""
    # Create sample videos if they don't exist
    sample_video_path = test_data_dir / "sample_golf_swing.mp4"
    if not sample_video_path.exists():
        create_test_video_with_faces(sample_video_path, num_frames=60, fps=30)
    
    no_faces_video_path = test_data_dir / "no_faces_video.mp4"
    if not no_faces_video_path.exists():
        create_test_video_no_faces(no_faces_video_path, num_frames=30, fps=30)
    
    multiple_faces_video_path = test_data_dir / "multiple_faces_video.mp4"
    if not multiple_faces_video_path.exists():
        create_test_video_multiple_faces(multiple_faces_video_path, num_frames=45, fps=30)
    
    yield test_data_dir
    
    # Cleanup is handled by pytest automatically for session-scoped fixtures


# Skip conditions for optional dependencies
def pytest_runtest_setup(item):
    """Skip tests based on available dependencies."""
    # Skip Redis tests if Redis is not available
    if item.get_closest_marker("requires_redis"):
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
        except (ImportError, redis.ConnectionError):
            pytest.skip("Redis not available")
    
    # Skip OpenCV tests if OpenCV is not properly installed
    if item.get_closest_marker("requires_opencv"):
        try:
            import cv2
            # Try to load a basic cascade to ensure OpenCV is working
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            if not os.path.exists(cascade_path):
                pytest.skip("OpenCV Haar cascades not available")
        except ImportError:
            pytest.skip("OpenCV not available")