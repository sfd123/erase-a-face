"""Face detection implementation using OpenCV Haar Cascades."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from models.face_detection import FaceDetection


@dataclass
class FaceDetectorConfig:
    """Configuration parameters for face detection."""
    
    # Haar cascade parameters
    scale_factor: float = 1.1  # How much the image size is reduced at each scale
    min_neighbors: int = 5     # How many neighbors each candidate rectangle should retain
    min_size: Tuple[int, int] = (30, 30)  # Minimum possible face size
    max_size: Tuple[int, int] = (300, 300)  # Maximum possible face size
    
    # Detection confidence threshold
    confidence_threshold: float = 0.5
    
    # Multiple cascade files for different conditions
    cascade_files: List[str] = None
    
    def __post_init__(self):
        """Set default cascade files if none provided."""
        if self.cascade_files is None:
            self.cascade_files = [
                'haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml'
            ]


class FaceDetector:
    """Face detector using OpenCV Haar Cascades with configurable parameters."""
    
    def __init__(self, config: Optional[FaceDetectorConfig] = None):
        """Initialize face detector with configuration.
        
        Args:
            config: Face detection configuration. Uses defaults if None.
        """
        self.config = config or FaceDetectorConfig()
        self.cascades = []
        self._load_cascades()
    
    def _load_cascades(self) -> None:
        """Load Haar cascade classifiers."""
        for cascade_file in self.config.cascade_files:
            try:
                cascade_path = cv2.data.haarcascades + cascade_file
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.cascades.append((cascade_file, cascade))
                else:
                    print(f"Warning: Could not load cascade {cascade_file}")
            except Exception as e:
                print(f"Error loading cascade {cascade_file}: {e}")
        
        if not self.cascades:
            raise RuntimeError("No valid Haar cascades could be loaded")
    
    def detect_faces(self, image: np.ndarray, frame_number: int = 0) -> List[FaceDetection]:
        """Detect faces in a single image/frame.
        
        Args:
            image: Input image as numpy array (BGR format)
            frame_number: Frame number for tracking purposes
            
        Returns:
            List of FaceDetection objects for detected faces
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        gray = self._enhance_image(gray)
        
        all_detections = []
        
        # Try each cascade classifier
        for cascade_name, cascade in self.cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=self.config.min_size,
                maxSize=self.config.max_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert detections to FaceDetection objects
            for (x, y, w, h) in faces:
                # Calculate confidence based on detection quality
                confidence = self._calculate_confidence(gray, x, y, w, h)
                
                if confidence >= self.config.confidence_threshold:
                    detection = FaceDetection(
                        frame_number=frame_number,
                        bounding_box=(x, y, w, h),
                        confidence=confidence
                    )
                    all_detections.append(detection)
        
        # Remove duplicate detections (non-maximum suppression)
        filtered_detections = self._remove_duplicates(all_detections)
        
        return filtered_detections
    
    def _enhance_image(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhance image for better face detection.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Enhanced grayscale image
        """
        # Apply histogram equalization to improve contrast
        enhanced = cv2.equalizeHist(gray_image)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _calculate_confidence(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Calculate confidence score for a face detection.
        
        Args:
            gray_image: Grayscale image
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Extract face region
        face_region = gray_image[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return 0.0
        
        # Calculate confidence based on various factors
        confidence = 0.5  # Base confidence
        
        # Factor 1: Size - larger faces are generally more reliable
        face_area = w * h
        if face_area > 2500:  # 50x50 pixels
            confidence += 0.2
        elif face_area < 900:  # 30x30 pixels
            confidence -= 0.1
        
        # Factor 2: Aspect ratio - faces should be roughly rectangular
        aspect_ratio = w / h if h > 0 else 0
        if 0.7 <= aspect_ratio <= 1.3:  # Good face aspect ratio
            confidence += 0.1
        else:
            confidence -= 0.1
        
        # Factor 3: Image quality - check for blur/noise
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        if laplacian_var > 100:  # Good sharpness
            confidence += 0.1
        elif laplacian_var < 50:  # Too blurry
            confidence -= 0.1
        
        # Clamp confidence to valid range
        return max(0.0, min(1.0, confidence))
    
    def _remove_duplicates(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Remove duplicate face detections using non-maximum suppression.
        
        Args:
            detections: List of face detections
            
        Returns:
            Filtered list with duplicates removed
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            # Check if this detection overlaps significantly with any already accepted
            is_duplicate = False
            for accepted in filtered:
                if detection.overlaps_with(accepted, threshold=0.3):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def detect_faces_batch(self, images: List[np.ndarray], start_frame: int = 0) -> List[List[FaceDetection]]:
        """Detect faces in multiple images/frames.
        
        Args:
            images: List of input images as numpy arrays
            start_frame: Starting frame number for tracking
            
        Returns:
            List of face detection lists, one per input image
        """
        results = []
        for i, image in enumerate(images):
            frame_detections = self.detect_faces(image, start_frame + i)
            results.append(frame_detections)
        
        return results
    
    def get_available_cascades(self) -> List[str]:
        """Get list of successfully loaded cascade classifiers.
        
        Returns:
            List of cascade file names that were loaded successfully
        """
        return [name for name, _ in self.cascades]
    
    def update_config(self, **kwargs) -> None:
        """Update detector configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")