"""Unit tests for face blurring and tracking functionality."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from processing.face_blurrer import FaceBlurrer, FaceBlurrerConfig, TrackedFace
from models.face_detection import FaceDetection


class TestFaceBlurrerConfig:
    """Test FaceBlurrerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FaceBlurrerConfig()
        
        assert config.blur_kernel_size == 99
        assert config.blur_sigma == 30.0
        assert config.max_tracking_distance == 50.0
        assert config.tracking_confidence_threshold == 0.3
        assert config.max_frames_without_detection == 5
        assert config.blur_padding_factor == 0.2
        assert config.min_blur_size == (30, 30)
        assert config.max_blur_size == (400, 400)
        assert config.lk_params is not None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FaceBlurrerConfig(
            blur_kernel_size=51,
            blur_sigma=20.0,
            max_tracking_distance=30.0
        )
        
        assert config.blur_kernel_size == 51
        assert config.blur_sigma == 20.0
        assert config.max_tracking_distance == 30.0


class TestTrackedFace:
    """Test TrackedFace class."""
    
    def test_tracked_face_creation(self):
        """Test TrackedFace creation."""
        detection = FaceDetection(0, (10, 10, 50, 50), 0.8)
        tracking_points = np.array([[10, 10], [20, 20]], dtype=np.float32).reshape(-1, 1, 2)
        
        tracked_face = TrackedFace(1, detection, tracking_points)
        
        assert tracked_face.track_id == 1
        assert tracked_face.last_detection == detection
        assert np.array_equal(tracked_face.tracking_points, tracking_points)
        assert tracked_face.frames_without_detection == 0
        assert tracked_face.is_active is True
    
    def test_update_detection(self):
        """Test updating tracked face with new detection."""
        detection1 = FaceDetection(0, (10, 10, 50, 50), 0.8)
        detection2 = FaceDetection(1, (15, 15, 50, 50), 0.9)
        tracking_points1 = np.array([[10, 10]], dtype=np.float32).reshape(-1, 1, 2)
        tracking_points2 = np.array([[15, 15]], dtype=np.float32).reshape(-1, 1, 2)
        
        tracked_face = TrackedFace(1, detection1, tracking_points1)
        tracked_face.frames_without_detection = 2
        
        tracked_face.update_detection(detection2, tracking_points2)
        
        assert tracked_face.last_detection == detection2
        assert np.array_equal(tracked_face.tracking_points, tracking_points2)
        assert tracked_face.frames_without_detection == 0
    
    def test_increment_missed_frames(self):
        """Test incrementing missed frames counter."""
        detection = FaceDetection(0, (10, 10, 50, 50), 0.8)
        tracking_points = np.array([[10, 10]], dtype=np.float32).reshape(-1, 1, 2)
        
        tracked_face = TrackedFace(1, detection, tracking_points)
        
        assert tracked_face.frames_without_detection == 0
        tracked_face.increment_missed_frames()
        assert tracked_face.frames_without_detection == 1
        tracked_face.increment_missed_frames()
        assert tracked_face.frames_without_detection == 2
    
    def test_deactivate(self):
        """Test deactivating tracked face."""
        detection = FaceDetection(0, (10, 10, 50, 50), 0.8)
        tracking_points = np.array([[10, 10]], dtype=np.float32).reshape(-1, 1, 2)
        
        tracked_face = TrackedFace(1, detection, tracking_points)
        
        assert tracked_face.is_active is True
        tracked_face.deactivate()
        assert tracked_face.is_active is False


class TestFaceBlurrer:
    """Test FaceBlurrer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FaceBlurrerConfig(blur_kernel_size=21, blur_sigma=10.0)
        self.blurrer = FaceBlurrer(self.config)
        
        # Create test frame (100x100 RGB) with some variation for blur to be visible
        self.test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add some patterns to make blur effects visible
        self.test_frame[20:80, 20:80] = [200, 200, 200]  # Bright square
        self.test_frame[30:70, 30:70] = [50, 50, 50]     # Dark square inside
        self.test_frame[40:60, 40:60] = [255, 0, 0]      # Red square in center
        
        # Create test detection
        self.test_detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
    
    def test_blurrer_initialization(self):
        """Test FaceBlurrer initialization."""
        blurrer = FaceBlurrer()
        
        assert blurrer.config is not None
        assert blurrer.tracked_faces == []
        assert blurrer.next_track_id == 1
        assert blurrer.previous_gray is None
    
    def test_blurrer_with_custom_config(self):
        """Test FaceBlurrer with custom configuration."""
        config = FaceBlurrerConfig(blur_kernel_size=31)
        blurrer = FaceBlurrer(config)
        
        assert blurrer.config.blur_kernel_size == 31
    
    def test_blur_kernel_size_validation(self):
        """Test that blur kernel size is made odd if even."""
        config = FaceBlurrerConfig(blur_kernel_size=20)  # Even number
        blurrer = FaceBlurrer(config)
        
        assert blurrer.config.blur_kernel_size == 21  # Should be made odd
    
    def test_blur_faces_empty_frame(self):
        """Test blurring with empty frame."""
        empty_frame = np.array([])
        result = self.blurrer.blur_faces(empty_frame, [])
        
        assert np.array_equal(result, empty_frame)
    
    def test_blur_faces_no_detections(self):
        """Test blurring with no face detections."""
        result = self.blurrer.blur_faces(self.test_frame, [])
        
        # Frame should be unchanged (except for possible tracking updates)
        assert result.shape == self.test_frame.shape
    
    def test_blur_faces_single_detection(self):
        """Test blurring with single face detection."""
        result = self.blurrer.blur_faces(self.test_frame, [self.test_detection])
        
        assert result.shape == self.test_frame.shape
        assert not np.array_equal(result, self.test_frame)  # Should be different due to blur
        
        # Check that blur was applied to face region
        x, y, w, h = self.test_detection.bounding_box
        face_region_original = self.test_frame[y:y+h, x:x+w]
        face_region_blurred = result[y:y+h, x:x+w]
        
        # Blurred region should be different from original
        assert not np.array_equal(face_region_original, face_region_blurred)
    
    def test_blur_faces_multiple_detections(self):
        """Test blurring with multiple face detections."""
        detection1 = FaceDetection(0, (10, 10, 30, 30), 0.8)
        detection2 = FaceDetection(0, (60, 60, 30, 30), 0.9)
        
        result = self.blurrer.blur_faces(self.test_frame, [detection1, detection2])
        
        assert result.shape == self.test_frame.shape
        assert not np.array_equal(result, self.test_frame)
    
    @patch('cv2.goodFeaturesToTrack')
    def test_extract_tracking_points_success(self, mock_good_features):
        """Test successful extraction of tracking points."""
        # Mock successful corner detection
        mock_corners = np.array([[[30, 30]], [[40, 40]], [[50, 50]]], dtype=np.float32)
        mock_good_features.return_value = mock_corners
        
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        
        points = self.blurrer._extract_tracking_points(gray, detection)
        
        assert points is not None
        assert len(points) == 3
        mock_good_features.assert_called_once()
    
    @patch('cv2.goodFeaturesToTrack')
    def test_extract_tracking_points_failure(self, mock_good_features):
        """Test fallback tracking points when corner detection fails."""
        # Mock failed corner detection
        mock_good_features.return_value = None
        
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        
        points = self.blurrer._extract_tracking_points(gray, detection)
        
        # Should return fallback points, not None
        assert points is not None
        assert len(points) == 5  # Default fallback points
        mock_good_features.assert_called_once()
    
    def test_calculate_face_distance(self):
        """Test face distance calculation."""
        face1 = FaceDetection(0, (10, 10, 20, 20), 0.8)  # Center at (20, 20)
        face2 = FaceDetection(0, (30, 40, 20, 20), 0.9)  # Center at (40, 50)
        
        distance = self.blurrer._calculate_face_distance(face1, face2)
        
        # Distance should be sqrt((40-20)^2 + (50-20)^2) = sqrt(400 + 900) = sqrt(1300) ≈ 36.06
        expected_distance = np.sqrt(20**2 + 30**2)
        assert abs(distance - expected_distance) < 0.01
    
    def test_calculate_blur_area_normal(self):
        """Test blur area calculation with normal face."""
        frame_shape = (100, 100, 3)
        x, y, w, h = 25, 25, 50, 50
        
        blur_area = self.blurrer._calculate_blur_area(x, y, w, h, frame_shape)
        bx, by, bw, bh = blur_area
        
        # Should have padding
        expected_padding_w = int(50 * 0.2)  # 10 pixels
        expected_padding_h = int(50 * 0.2)  # 10 pixels
        
        assert bx == max(0, 25 - expected_padding_w)
        assert by == max(0, 25 - expected_padding_h)
        assert bw <= 50 + 2 * expected_padding_w
        assert bh <= 50 + 2 * expected_padding_h
    
    def test_calculate_blur_area_edge_case(self):
        """Test blur area calculation at frame edges."""
        frame_shape = (100, 100, 3)
        x, y, w, h = 0, 0, 20, 20  # Face at top-left corner
        
        blur_area = self.blurrer._calculate_blur_area(x, y, w, h, frame_shape)
        bx, by, bw, bh = blur_area
        
        # Should be clamped to frame boundaries
        assert bx >= 0
        assert by >= 0
        assert bx + bw <= 100
        assert by + bh <= 100
    
    def test_calculate_blur_area_minimum_size(self):
        """Test blur area respects minimum size."""
        frame_shape = (100, 100, 3)
        x, y, w, h = 40, 40, 5, 5  # Very small face
        
        blur_area = self.blurrer._calculate_blur_area(x, y, w, h, frame_shape)
        bx, by, bw, bh = blur_area
        
        # Should respect minimum blur size
        assert bw >= self.config.min_blur_size[0]
        assert bh >= self.config.min_blur_size[1]
    
    def test_apply_blur_to_face(self):
        """Test applying blur to a specific face."""
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        
        result = self.blurrer._apply_blur_to_face(self.test_frame, detection)
        
        assert result.shape == self.test_frame.shape
        
        # Face region should be blurred (different from original)
        x, y, w, h = detection.bounding_box
        original_region = self.test_frame[y:y+h, x:x+w]
        blurred_region = result[y:y+h, x:x+w]
        
        assert not np.array_equal(original_region, blurred_region)
    
    def test_reset_tracking(self):
        """Test resetting tracking state."""
        # Add some tracked faces
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        tracking_points = np.array([[30, 30]], dtype=np.float32).reshape(-1, 1, 2)
        tracked_face = TrackedFace(1, detection, tracking_points)
        
        self.blurrer.tracked_faces.append(tracked_face)
        self.blurrer.next_track_id = 5
        self.blurrer.previous_gray = np.ones((100, 100), dtype=np.uint8)
        
        self.blurrer.reset_tracking()
        
        assert len(self.blurrer.tracked_faces) == 0
        assert self.blurrer.next_track_id == 1
        assert self.blurrer.previous_gray is None
    
    def test_get_active_tracks_count(self):
        """Test getting active tracks count."""
        # Initially no tracks
        assert self.blurrer.get_active_tracks_count() == 0
        
        # Add active track
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        tracking_points = np.array([[30, 30]], dtype=np.float32).reshape(-1, 1, 2)
        active_face = TrackedFace(1, detection, tracking_points)
        self.blurrer.tracked_faces.append(active_face)
        
        assert self.blurrer.get_active_tracks_count() == 1
        
        # Add inactive track
        inactive_face = TrackedFace(2, detection, tracking_points)
        inactive_face.deactivate()
        self.blurrer.tracked_faces.append(inactive_face)
        
        assert self.blurrer.get_active_tracks_count() == 1  # Still 1 active
    
    def test_get_track_info(self):
        """Test getting track information."""
        # Add tracked face
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        tracking_points = np.array([[30, 30]], dtype=np.float32).reshape(-1, 1, 2)
        tracked_face = TrackedFace(1, detection, tracking_points)
        tracked_face.frames_without_detection = 2
        self.blurrer.tracked_faces.append(tracked_face)
        
        track_info = self.blurrer.get_track_info()
        
        assert len(track_info) == 1
        info = track_info[0]
        assert info['track_id'] == 1
        assert info['bounding_box'] == (25, 25, 50, 50)
        assert info['confidence'] == 0.8
        assert info['frames_without_detection'] == 2
    
    @patch('cv2.calcOpticalFlowPyrLK')
    def test_update_existing_tracks_success(self, mock_optical_flow):
        """Test successful update of existing tracks."""
        # Setup mock optical flow
        new_points = np.array([[[35, 35]], [[45, 45]], [[55, 55]]], dtype=np.float32)
        status = np.array([[1], [1], [1]], dtype=np.uint8)
        error = np.array([[0.1], [0.1], [0.1]], dtype=np.float32)
        mock_optical_flow.return_value = (new_points, status, error)
        
        # Add tracked face
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        tracking_points = np.array([[[30, 30]], [[40, 40]], [[50, 50]]], dtype=np.float32)
        tracked_face = TrackedFace(1, detection, tracking_points)
        self.blurrer.tracked_faces.append(tracked_face)
        
        # Set previous gray frame
        self.blurrer.previous_gray = np.ones((100, 100), dtype=np.uint8) * 128
        
        # Update tracks
        gray = np.ones((100, 100), dtype=np.uint8) * 130
        self.blurrer._update_existing_tracks(gray)
        
        # Check that optical flow was called
        mock_optical_flow.assert_called_once()
        
        # Check that tracking points were updated
        assert tracked_face.is_active
        assert len(tracked_face.tracking_points) == 3
    
    @patch('cv2.calcOpticalFlowPyrLK')
    def test_update_existing_tracks_failure(self, mock_optical_flow):
        """Test failed update of existing tracks."""
        # Setup mock optical flow with poor tracking
        new_points = np.array([[[35, 35]], [[45, 45]]], dtype=np.float32)
        status = np.array([[0], [0]], dtype=np.uint8)  # All points lost
        error = np.array([[10.0], [10.0]], dtype=np.float32)
        mock_optical_flow.return_value = (new_points, status, error)
        
        # Add tracked face
        detection = FaceDetection(0, (25, 25, 50, 50), 0.8)
        tracking_points = np.array([[[30, 30]], [[40, 40]]], dtype=np.float32)
        tracked_face = TrackedFace(1, detection, tracking_points)
        self.blurrer.tracked_faces.append(tracked_face)
        
        # Set previous gray frame
        self.blurrer.previous_gray = np.ones((100, 100), dtype=np.uint8) * 128
        
        # Update tracks
        gray = np.ones((100, 100), dtype=np.uint8) * 130
        self.blurrer._update_existing_tracks(gray)
        
        # Check that frames without detection was incremented
        assert tracked_face.frames_without_detection == 1


class TestBlurQuality:
    """Test blur quality and visual effects."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blurrer = FaceBlurrer()
        
        # Create test frame with distinct patterns
        self.test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some patterns to make blur effects visible
        self.test_frame[50:150, 50:150] = [255, 255, 255]  # White square
        self.test_frame[75:125, 75:125] = [0, 0, 0]        # Black square inside
    
    def test_blur_intensity(self):
        """Test that blur actually reduces image sharpness."""
        detection = FaceDetection(0, (60, 60, 80, 80), 0.8)
        
        # Calculate original sharpness (Laplacian variance)
        x, y, w, h = detection.bounding_box
        original_region = self.test_frame[y:y+h, x:x+w]
        original_gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY)
        original_sharpness = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        
        # Apply blur
        result = self.blurrer.blur_faces(self.test_frame, [detection])
        
        # Calculate blurred sharpness
        blurred_region = result[y:y+h, x:x+w]
        blurred_gray = cv2.cvtColor(blurred_region, cv2.COLOR_BGR2GRAY)
        blurred_sharpness = cv2.Laplacian(blurred_gray, cv2.CV_64F).var()
        
        # Blurred region should be less sharp
        assert blurred_sharpness < original_sharpness
    
    def test_blur_consistency(self):
        """Test that blur is applied consistently across multiple frames."""
        detection = FaceDetection(0, (60, 60, 80, 80), 0.8)
        
        # Apply blur to multiple identical frames
        results = []
        for i in range(3):
            result = self.blurrer.blur_faces(self.test_frame.copy(), [detection])
            results.append(result)
        
        # Results should be similar (accounting for tracking variations)
        x, y, w, h = detection.bounding_box
        for i in range(1, len(results)):
            region1 = results[0][y:y+h, x:x+w]
            region2 = results[i][y:y+h, x:x+w]
            
            # Calculate similarity (should be high)
            diff = np.mean(np.abs(region1.astype(float) - region2.astype(float)))
            assert diff < 50  # Allow some variation due to tracking
    
    def test_blur_area_coverage(self):
        """Test that blur covers the entire face area plus padding."""
        detection = FaceDetection(0, (60, 60, 80, 80), 0.8)
        
        result = self.blurrer.blur_faces(self.test_frame, [detection])
        
        # Check that area around face is blurred
        x, y, w, h = detection.bounding_box
        
        # Calculate expected blur area
        blur_area = self.blurrer._calculate_blur_area(x, y, w, h, self.test_frame.shape)
        bx, by, bw, bh = blur_area
        
        # Check that blur was applied to calculated area
        original_blur_region = self.test_frame[by:by+bh, bx:bx+bw]
        result_blur_region = result[by:by+bh, bx:bx+bw]
        
        assert not np.array_equal(original_blur_region, result_blur_region)


if __name__ == '__main__':
    pytest.main([__file__])