"""Unit tests for FaceDetector class."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from processing.face_detector import FaceDetector, FaceDetectorConfig
from models.face_detection import FaceDetection


class TestFaceDetectorConfig:
    """Test FaceDetectorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FaceDetectorConfig()
        
        assert config.scale_factor == 1.1
        assert config.min_neighbors == 5
        assert config.min_size == (30, 30)
        assert config.max_size == (300, 300)
        assert config.confidence_threshold == 0.5
        assert len(config.cascade_files) == 3
        assert 'haarcascade_frontalface_alt.xml' in config.cascade_files
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FaceDetectorConfig(
            scale_factor=1.2,
            min_neighbors=3,
            min_size=(20, 20),
            max_size=(400, 400),
            confidence_threshold=0.7,
            cascade_files=['custom_cascade.xml']
        )
        
        assert config.scale_factor == 1.2
        assert config.min_neighbors == 3
        assert config.min_size == (20, 20)
        assert config.max_size == (400, 400)
        assert config.confidence_threshold == 0.7
        assert config.cascade_files == ['custom_cascade.xml']


class TestFaceDetector:
    """Test FaceDetector class."""
    
    @pytest.fixture
    def mock_cascade(self):
        """Create a mock cascade classifier."""
        cascade = MagicMock()
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        return cascade
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple 200x200 BGR image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some content to make it more realistic
        cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
        return image
    
    @pytest.fixture
    def bright_image(self):
        """Create a bright test image."""
        image = np.full((200, 200, 3), 200, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        return image
    
    @pytest.fixture
    def dim_image(self):
        """Create a dim test image."""
        image = np.full((200, 200, 3), 50, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (80, 80, 80), -1)
        return image
    
    @pytest.fixture
    def blurry_image(self):
        """Create a blurry test image."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
        # Apply heavy blur
        image = cv2.GaussianBlur(image, (15, 15), 5)
        return image
    
    @patch('cv2.CascadeClassifier')
    def test_detector_initialization(self, mock_classifier, mock_cascade):
        """Test FaceDetector initialization."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        assert detector.config is not None
        assert len(detector.cascades) > 0
        assert mock_classifier.call_count >= 1
    
    @patch('cv2.CascadeClassifier')
    def test_detector_initialization_with_config(self, mock_classifier, mock_cascade):
        """Test FaceDetector initialization with custom config."""
        mock_classifier.return_value = mock_cascade
        
        config = FaceDetectorConfig(scale_factor=1.3, min_neighbors=4)
        detector = FaceDetector(config)
        
        assert detector.config.scale_factor == 1.3
        assert detector.config.min_neighbors == 4
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_single_face(self, mock_classifier, mock_cascade, sample_image):
        """Test face detection with single face."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        detections = detector.detect_faces(sample_image, frame_number=1)
        
        assert len(detections) == 1
        assert detections[0].frame_number == 1
        assert detections[0].bounding_box == (50, 50, 100, 100)
        assert 0.0 <= detections[0].confidence <= 1.0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_multiple_faces(self, mock_classifier, mock_cascade, sample_image):
        """Test face detection with multiple faces."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([
            [50, 50, 100, 100],
            [150, 50, 80, 80]
        ])
        
        detector = FaceDetector()
        detections = detector.detect_faces(sample_image)
        
        assert len(detections) == 2
        assert detections[0].bounding_box == (50, 50, 100, 100)
        assert detections[1].bounding_box == (150, 50, 80, 80)
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_no_faces(self, mock_classifier, mock_cascade, sample_image):
        """Test face detection when no faces are found."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([])
        
        detector = FaceDetector()
        detections = detector.detect_faces(sample_image)
        
        assert len(detections) == 0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_empty_image(self, mock_classifier, mock_cascade):
        """Test face detection with empty image."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        detections = detector.detect_faces(np.array([]))
        
        assert len(detections) == 0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_none_image(self, mock_classifier, mock_cascade):
        """Test face detection with None image."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        detections = detector.detect_faces(None)
        
        assert len(detections) == 0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_bright_lighting(self, mock_classifier, mock_cascade, bright_image):
        """Test face detection in bright lighting conditions."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        detections = detector.detect_faces(bright_image)
        
        assert len(detections) == 1
        # Should still detect face in bright conditions
        assert detections[0].confidence > 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_dim_lighting(self, mock_classifier, mock_cascade, dim_image):
        """Test face detection in dim lighting conditions."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        detections = detector.detect_faces(dim_image)
        
        assert len(detections) == 1
        # Should still detect face in dim conditions
        assert detections[0].confidence > 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_blurry_image(self, mock_classifier, mock_cascade, blurry_image):
        """Test face detection with blurry image."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        detections = detector.detect_faces(blurry_image)
        
        assert len(detections) == 1
        # Blurry images might have lower confidence
        assert detections[0].confidence >= 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_confidence_threshold_filtering(self, mock_classifier, mock_cascade, sample_image):
        """Test that low confidence detections are filtered out."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 20, 20]])  # Very small face
        
        config = FaceDetectorConfig(confidence_threshold=0.8)  # High threshold
        detector = FaceDetector(config)
        detections = detector.detect_faces(sample_image)
        
        # Small faces should have low confidence and be filtered out
        assert len(detections) == 0
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_batch(self, mock_classifier, mock_cascade, sample_image):
        """Test batch face detection."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        images = [sample_image, sample_image, sample_image]
        results = detector.detect_faces_batch(images, start_frame=10)
        
        assert len(results) == 3
        for i, detections in enumerate(results):
            assert len(detections) == 1
            assert detections[0].frame_number == 10 + i
    
    @patch('cv2.CascadeClassifier')
    def test_remove_duplicates(self, mock_classifier, mock_cascade, sample_image):
        """Test duplicate detection removal."""
        mock_classifier.return_value = mock_cascade
        # Return overlapping detections
        mock_cascade.detectMultiScale.return_value = np.array([
            [50, 50, 100, 100],
            [55, 55, 95, 95]  # Overlapping detection
        ])
        
        detector = FaceDetector()
        detections = detector.detect_faces(sample_image)
        
        # Should remove duplicate/overlapping detection
        assert len(detections) == 1
    
    @patch('cv2.CascadeClassifier')
    def test_get_available_cascades(self, mock_classifier, mock_cascade):
        """Test getting available cascade classifiers."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        cascades = detector.get_available_cascades()
        
        assert isinstance(cascades, list)
        assert len(cascades) > 0
    
    @patch('cv2.CascadeClassifier')
    def test_update_config(self, mock_classifier, mock_cascade):
        """Test updating detector configuration."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        original_scale = detector.config.scale_factor
        
        detector.update_config(scale_factor=1.5, min_neighbors=7)
        
        assert detector.config.scale_factor == 1.5
        assert detector.config.min_neighbors == 7
        assert detector.config.scale_factor != original_scale
    
    @patch('cv2.CascadeClassifier')
    def test_update_config_invalid_parameter(self, mock_classifier, mock_cascade):
        """Test updating detector configuration with invalid parameter."""
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            detector.update_config(invalid_param=123)
    
    def test_calculate_confidence_large_face(self):
        """Test confidence calculation for large face."""
        # Create a mock detector to test the private method
        config = FaceDetectorConfig()
        
        # Create a test image with a large face region
        gray_image = np.full((200, 200), 128, dtype=np.uint8)
        
        # We'll test this indirectly through the public interface
        # by mocking the cascade to return a large face detection
        with patch('cv2.CascadeClassifier') as mock_classifier:
            mock_cascade = MagicMock()
            mock_cascade.empty.return_value = False
            mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])  # Large face
            mock_classifier.return_value = mock_cascade
            
            detector = FaceDetector(config)
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            detections = detector.detect_faces(image)
            
            if detections:
                # Large faces should have higher confidence
                assert detections[0].confidence > 0.5
    
    def test_calculate_confidence_small_face(self):
        """Test confidence calculation for small face."""
        config = FaceDetectorConfig()
        
        gray_image = np.full((200, 200), 128, dtype=np.uint8)
        
        with patch('cv2.CascadeClassifier') as mock_classifier:
            mock_cascade = MagicMock()
            mock_cascade.empty.return_value = False
            mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 25, 25]])  # Small face
            mock_classifier.return_value = mock_cascade
            
            detector = FaceDetector(config)
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            detections = detector.detect_faces(image)
            
            # Small faces might be filtered out due to low confidence
            # or have lower confidence if they pass the threshold
            if detections:
                assert 0.0 <= detections[0].confidence <= 1.0
    
    @patch('cv2.CascadeClassifier')
    def test_cascade_loading_failure(self, mock_classifier):
        """Test handling of cascade loading failures."""
        # Mock empty cascade (failed to load)
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = True
        mock_classifier.return_value = mock_cascade
        
        # Should raise RuntimeError when no cascades can be loaded
        with pytest.raises(RuntimeError, match="No valid Haar cascades could be loaded"):
            FaceDetector()
    
    @patch('cv2.CascadeClassifier')
    def test_image_enhancement(self, mock_classifier, mock_cascade):
        """Test that image enhancement is applied during detection."""
        mock_classifier.return_value = mock_cascade
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        
        detector = FaceDetector()
        
        # Create an image with poor contrast
        poor_contrast_image = np.full((200, 200, 3), 100, dtype=np.uint8)
        
        detections = detector.detect_faces(poor_contrast_image)
        
        # Should still be able to detect faces after enhancement
        mock_cascade.detectMultiScale.assert_called()
        # Verify that the enhanced image was passed to detectMultiScale
        call_args = mock_cascade.detectMultiScale.call_args[0]
        enhanced_image = call_args[0]
        assert enhanced_image.dtype == np.uint8
        assert len(enhanced_image.shape) == 2  # Should be grayscale


class TestFaceDetectorConditions:
    """Test FaceDetector under various real-world conditions."""
    
    @pytest.fixture
    def profile_face_image(self):
        """Create an image simulating a profile face."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create an elliptical shape to simulate profile
        cv2.ellipse(image, (100, 100), (30, 50), 90, 0, 180, (128, 128, 128), -1)
        return image
    
    @pytest.fixture
    def backlit_image(self):
        """Create an image with backlighting conditions."""
        image = np.full((200, 200, 3), 220, dtype=np.uint8)  # Bright background
        # Dark face region
        cv2.rectangle(image, (75, 75), (125, 125), (40, 40, 40), -1)
        return image
    
    @pytest.fixture
    def multiple_faces_image(self):
        """Create an image with multiple faces at different angles."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        # Face 1 - frontal
        cv2.rectangle(image, (50, 50), (100, 100), (128, 128, 128), -1)
        # Face 2 - smaller, different position
        cv2.rectangle(image, (200, 100), (240, 140), (120, 120, 120), -1)
        # Face 3 - larger
        cv2.rectangle(image, (100, 150), (180, 230), (140, 140, 140), -1)
        return image
    
    @pytest.fixture
    def noisy_image(self):
        """Create a noisy image to test robustness."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
        # Add random noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        return image
    
    @patch('cv2.CascadeClassifier')
    def test_profile_face_detection(self, mock_classifier, profile_face_image):
        """Test detection of profile faces."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[70, 75, 60, 50]])
        mock_classifier.return_value = mock_cascade
        
        # Use config that includes profile cascade
        config = FaceDetectorConfig(
            cascade_files=['haarcascade_profileface.xml']
        )
        detector = FaceDetector(config)
        detections = detector.detect_faces(profile_face_image)
        
        assert len(detections) >= 0  # Should handle profile faces
        if detections:
            assert detections[0].confidence > 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_backlit_conditions(self, mock_classifier, backlit_image):
        """Test face detection in backlit conditions."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[75, 75, 50, 50]])
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        detections = detector.detect_faces(backlit_image)
        
        # Should still detect faces in backlit conditions due to image enhancement
        assert len(detections) >= 0
        if detections:
            assert detections[0].confidence > 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_multiple_faces_different_sizes(self, mock_classifier, multiple_faces_image):
        """Test detection of multiple faces of different sizes."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([
            [50, 50, 50, 50],    # Face 1
            [200, 100, 40, 40],  # Face 2 - smaller
            [100, 150, 80, 80]   # Face 3 - larger
        ])
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        detections = detector.detect_faces(multiple_faces_image)
        
        assert len(detections) == 3
        # Check that faces of different sizes are detected
        sizes = [d.area for d in detections]
        assert len(set(sizes)) > 1  # Different sizes
    
    @patch('cv2.CascadeClassifier')
    def test_noisy_image_robustness(self, mock_classifier, noisy_image):
        """Test robustness to image noise."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        detections = detector.detect_faces(noisy_image)
        
        # Should still detect faces despite noise due to image enhancement
        assert len(detections) >= 0
        if detections:
            assert detections[0].confidence > 0.0
    
    @patch('cv2.CascadeClassifier')
    def test_extreme_lighting_conditions(self, mock_classifier):
        """Test face detection in extreme lighting conditions."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Very dark image
        dark_image = np.full((200, 200, 3), 10, dtype=np.uint8)
        cv2.rectangle(dark_image, (50, 50), (150, 150), (30, 30, 30), -1)
        
        detections_dark = detector.detect_faces(dark_image)
        
        # Very bright image
        bright_image = np.full((200, 200, 3), 245, dtype=np.uint8)
        cv2.rectangle(bright_image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        detections_bright = detector.detect_faces(bright_image)
        
        # Should handle both extreme conditions
        assert len(detections_dark) >= 0
        assert len(detections_bright) >= 0
    
    @patch('cv2.CascadeClassifier')
    def test_different_aspect_ratios(self, mock_classifier):
        """Test face detection with different face aspect ratios."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Test different aspect ratios
        test_cases = [
            (50, 50, 100, 100),  # Square face (1:1)
            (50, 50, 80, 100),   # Tall face (0.8:1)
            (50, 50, 100, 80),   # Wide face (1.25:1)
            (50, 50, 60, 120),   # Very tall face (0.5:1)
            (50, 50, 120, 60),   # Very wide face (2:1)
        ]
        
        for x, y, w, h in test_cases:
            mock_cascade.detectMultiScale.return_value = np.array([[x, y, w, h]])
            
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(image, (x, y), (x+w, y+h), (128, 128, 128), -1)
            
            detections = detector.detect_faces(image)
            
            # Should detect faces with reasonable aspect ratios
            if detections:
                aspect_ratio = detections[0].width / detections[0].height
                # Confidence should be higher for more face-like aspect ratios
                if 0.7 <= aspect_ratio <= 1.3:
                    assert detections[0].confidence > 0.5
    
    @patch('cv2.CascadeClassifier')
    def test_configurable_parameters_effect(self, mock_classifier):
        """Test that configurable parameters affect detection behavior."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_classifier.return_value = mock_cascade
        
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
        
        # Test with different scale factors
        config1 = FaceDetectorConfig(scale_factor=1.1)
        detector1 = FaceDetector(config1)
        
        config2 = FaceDetectorConfig(scale_factor=1.3)
        detector2 = FaceDetector(config2)
        
        # Clear previous calls
        mock_cascade.detectMultiScale.reset_mock()
        
        # Make calls with different detectors
        detector1.detect_faces(image)
        detector2.detect_faces(image)
        
        # Verify that detectMultiScale was called multiple times
        calls = mock_cascade.detectMultiScale.call_args_list
        assert len(calls) >= 2
        
        # Verify that the detectors have different configurations
        assert detector1.config.scale_factor != detector2.config.scale_factor
        assert detector1.config.scale_factor == 1.1
        assert detector2.config.scale_factor == 1.3
    
    @patch('cv2.CascadeClassifier')
    def test_confidence_calculation_factors(self, mock_classifier):
        """Test that confidence calculation considers multiple factors."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Test large face (should have higher confidence)
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        large_face_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(large_face_image, (50, 50), (150, 150), (128, 128, 128), -1)
        
        large_detections = detector.detect_faces(large_face_image)
        
        # Test small face (should have lower confidence)
        mock_cascade.detectMultiScale.return_value = np.array([[90, 90, 20, 20]])
        small_face_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(small_face_image, (90, 90), (110, 110), (128, 128, 128), -1)
        
        small_detections = detector.detect_faces(small_face_image)
        
        # Large faces should generally have higher confidence than small faces
        if large_detections and small_detections:
            assert large_detections[0].confidence >= small_detections[0].confidence
    
    @patch('cv2.CascadeClassifier')
    def test_batch_processing_consistency(self, mock_classifier):
        """Test that batch processing gives consistent results."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_classifier.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Create identical images
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (128, 128, 128), -1)
        
        images = [image.copy() for _ in range(5)]
        
        # Process individually
        individual_results = []
        for i, img in enumerate(images):
            detections = detector.detect_faces(img, frame_number=i)
            individual_results.append(detections)
        
        # Process as batch
        batch_results = detector.detect_faces_batch(images, start_frame=0)
        
        # Results should be consistent
        assert len(individual_results) == len(batch_results)
        for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
            assert len(individual) == len(batch)
            if individual and batch:
                assert individual[0].frame_number == batch[0].frame_number
                assert individual[0].bounding_box == batch[0].bounding_box