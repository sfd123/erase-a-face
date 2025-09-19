"""Face blurring and tracking implementation using OpenCV."""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from models.face_detection import FaceDetection


@dataclass
class FaceBlurrerConfig:
    """Configuration parameters for face blurring and tracking."""
    
    # Blur parameters
    blur_kernel_size: int = 99  # Must be odd number
    blur_sigma: float = 30.0    # Gaussian blur standard deviation
    
    # Tracking parameters
    max_tracking_distance: float = 50.0  # Maximum distance to consider same face
    tracking_confidence_threshold: float = 0.3  # Minimum confidence for tracking
    max_frames_without_detection: int = 5  # Max frames to track without detection
    
    # Blur area sizing
    blur_padding_factor: float = 0.2  # Extra padding around face (20%)
    min_blur_size: Tuple[int, int] = (30, 30)  # Minimum blur area
    max_blur_size: Tuple[int, int] = (400, 400)  # Maximum blur area
    
    # Optical flow parameters
    lk_params: Dict = None
    
    def __post_init__(self):
        """Set default Lucas-Kanade optical flow parameters."""
        if self.lk_params is None:
            self.lk_params = {
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }


@dataclass
class TrackedFace:
    """Represents a face being tracked across frames."""
    
    track_id: int
    last_detection: FaceDetection
    tracking_points: np.ndarray  # Feature points for optical flow
    frames_without_detection: int = 0
    is_active: bool = True
    
    def update_detection(self, detection: FaceDetection, tracking_points: np.ndarray):
        """Update the tracked face with new detection."""
        self.last_detection = detection
        self.tracking_points = tracking_points
        self.frames_without_detection = 0
    
    def increment_missed_frames(self):
        """Increment counter for frames without detection."""
        self.frames_without_detection += 1
    
    def deactivate(self):
        """Mark this tracked face as inactive."""
        self.is_active = False


class FaceBlurrer:
    """Face blurrer with tracking capabilities using optical flow."""
    
    def __init__(self, config: Optional[FaceBlurrerConfig] = None):
        """Initialize face blurrer with configuration.
        
        Args:
            config: Face blurring configuration. Uses defaults if None.
        """
        self.config = config or FaceBlurrerConfig()
        self.tracked_faces: List[TrackedFace] = []
        self.next_track_id = 1
        self.previous_gray = None
        
        # Validate blur kernel size (must be odd)
        if self.config.blur_kernel_size % 2 == 0:
            self.config.blur_kernel_size += 1
    
    def blur_faces(self, frame: np.ndarray, detections: List[FaceDetection]) -> np.ndarray:
        """Apply blur to detected faces in a frame with tracking.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            detections: List of face detections for this frame
            
        Returns:
            Frame with faces blurred
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Convert to grayscale for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update tracking
        self._update_tracking(gray, detections)
        
        # Apply blur to all active tracked faces
        blurred_frame = frame.copy()
        for tracked_face in self.tracked_faces:
            if tracked_face.is_active:
                blurred_frame = self._apply_blur_to_face(blurred_frame, tracked_face.last_detection)
        
        # Store current frame for next iteration
        self.previous_gray = gray.copy()
        
        return blurred_frame
    
    def _update_tracking(self, gray: np.ndarray, detections: List[FaceDetection]) -> None:
        """Update face tracking with new detections.
        
        Args:
            gray: Current frame in grayscale
            detections: New face detections
        """
        # If this is the first frame, initialize tracking
        if self.previous_gray is None:
            self._initialize_tracking(gray, detections)
            return
        
        # Update existing tracks using optical flow
        self._update_existing_tracks(gray)
        
        # Match new detections with existing tracks
        self._match_detections_to_tracks(detections, gray)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(detections, gray)
        
        # Clean up inactive tracks
        self._cleanup_inactive_tracks()
    
    def _initialize_tracking(self, gray: np.ndarray, detections: List[FaceDetection]) -> None:
        """Initialize tracking for the first frame.
        
        Args:
            gray: First frame in grayscale
            detections: Face detections in first frame
        """
        for detection in detections:
            tracking_points = self._extract_tracking_points(gray, detection)
            if tracking_points is not None:
                tracked_face = TrackedFace(
                    track_id=self.next_track_id,
                    last_detection=detection,
                    tracking_points=tracking_points
                )
                self.tracked_faces.append(tracked_face)
                self.next_track_id += 1
    
    def _update_existing_tracks(self, gray: np.ndarray) -> None:
        """Update existing tracks using optical flow.
        
        Args:
            gray: Current frame in grayscale
        """
        for tracked_face in self.tracked_faces:
            if not tracked_face.is_active or tracked_face.tracking_points is None:
                continue
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                tracked_face.tracking_points,
                None,
                **self.config.lk_params
            )
            
            # Filter good points
            good_points = new_points[status == 1]
            
            if len(good_points) < 3:  # Not enough points to track reliably
                tracked_face.increment_missed_frames()
                if tracked_face.frames_without_detection > self.config.max_frames_without_detection:
                    tracked_face.deactivate()
                continue
            
            # Update tracking points
            tracked_face.tracking_points = good_points.reshape(-1, 1, 2)
            
            # Estimate new bounding box from tracking points
            self._update_bounding_box_from_tracking(tracked_face)
    
    def _match_detections_to_tracks(self, detections: List[FaceDetection], gray: np.ndarray) -> None:
        """Match new detections with existing tracks.
        
        Args:
            detections: New face detections
            gray: Current frame in grayscale
        """
        matched_detections = set()
        
        for tracked_face in self.tracked_faces:
            if not tracked_face.is_active:
                continue
            
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                
                # Calculate distance between tracked face and detection
                distance = self._calculate_face_distance(tracked_face.last_detection, detection)
                
                if distance < self.config.max_tracking_distance and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                # Update tracked face with new detection
                detection = detections[best_match]
                tracking_points = self._extract_tracking_points(gray, detection)
                if tracking_points is not None:
                    tracked_face.update_detection(detection, tracking_points)
                    matched_detections.add(best_match)
        
        # Mark remaining detections as unmatched for new track creation
        self.unmatched_detections = [
            detection for i, detection in enumerate(detections)
            if i not in matched_detections
        ]
    
    def _create_new_tracks(self, detections: List[FaceDetection], gray: np.ndarray) -> None:
        """Create new tracks for unmatched detections.
        
        Args:
            detections: All detections (will use unmatched ones)
            gray: Current frame in grayscale
        """
        for detection in getattr(self, 'unmatched_detections', []):
            tracking_points = self._extract_tracking_points(gray, detection)
            if tracking_points is not None:
                tracked_face = TrackedFace(
                    track_id=self.next_track_id,
                    last_detection=detection,
                    tracking_points=tracking_points
                )
                self.tracked_faces.append(tracked_face)
                self.next_track_id += 1
    
    def _cleanup_inactive_tracks(self) -> None:
        """Remove inactive tracks from the tracking list."""
        self.tracked_faces = [face for face in self.tracked_faces if face.is_active]
    
    def _extract_tracking_points(self, gray: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """Extract feature points from face region for tracking.
        
        Args:
            gray: Grayscale frame
            detection: Face detection
            
        Returns:
            Array of tracking points or None if extraction fails
        """
        x, y, w, h = detection.bounding_box
        
        # Ensure bounding box is within image bounds
        h_img, w_img = gray.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return None
        
        # Extract face region
        face_region = gray[y:y+h, x:x+w]
        
        # Detect corners in face region
        corners = cv2.goodFeaturesToTrack(
            face_region,
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )
        
        if corners is None or len(corners) < 3:
            # If corner detection fails, create some default tracking points
            # This ensures we can still track and blur faces even in uniform regions
            center_x = w // 2
            center_y = h // 2
            corners = np.array([
                [[center_x - 10, center_y - 10]],
                [[center_x + 10, center_y - 10]],
                [[center_x, center_y + 10]],
                [[center_x - 5, center_y]],
                [[center_x + 5, center_y]]
            ], dtype=np.float32)
        
        # Convert coordinates to global frame coordinates
        corners[:, :, 0] += x
        corners[:, :, 1] += y
        
        return corners
    
    def _update_bounding_box_from_tracking(self, tracked_face: TrackedFace) -> None:
        """Update bounding box based on tracking points movement.
        
        Args:
            tracked_face: Tracked face to update
        """
        if tracked_face.tracking_points is None or len(tracked_face.tracking_points) == 0:
            return
        
        # Calculate bounding box from tracking points
        points = tracked_face.tracking_points.reshape(-1, 2)
        x_min, y_min = np.min(points, axis=0).astype(int)
        x_max, y_max = np.max(points, axis=0).astype(int)
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        w = x_max - x_min + padding
        h = y_max - y_min + padding
        
        # Update detection with new bounding box
        tracked_face.last_detection = FaceDetection(
            frame_number=tracked_face.last_detection.frame_number,
            bounding_box=(x_min, y_min, w, h),
            confidence=tracked_face.last_detection.confidence * 0.9  # Slightly reduce confidence
        )
    
    def _calculate_face_distance(self, face1: FaceDetection, face2: FaceDetection) -> float:
        """Calculate distance between two face detections.
        
        Args:
            face1: First face detection
            face2: Second face detection
            
        Returns:
            Distance between face centers
        """
        center1 = face1.center
        center2 = face2.center
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _apply_blur_to_face(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Apply Gaussian blur to a specific face region.
        
        Args:
            frame: Input frame
            detection: Face detection to blur
            
        Returns:
            Frame with face blurred
        """
        x, y, w, h = detection.bounding_box
        
        # Calculate blur area with padding
        blur_area = self._calculate_blur_area(x, y, w, h, frame.shape)
        bx, by, bw, bh = blur_area
        
        # Extract face region
        face_region = frame[by:by+bh, bx:bx+bw]
        
        if face_region.size == 0:
            return frame
        
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(
            face_region,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            self.config.blur_sigma
        )
        
        # Replace original region with blurred version
        result_frame = frame.copy()
        result_frame[by:by+bh, bx:bx+bw] = blurred_region
        
        return result_frame
    
    def _calculate_blur_area(self, x: int, y: int, w: int, h: int, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Calculate the blur area with padding based on face dimensions.
        
        Args:
            x, y, w, h: Face bounding box
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Blur area as (x, y, width, height)
        """
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate padding
        padding_w = int(w * self.config.blur_padding_factor)
        padding_h = int(h * self.config.blur_padding_factor)
        
        # Calculate blur area
        blur_x = max(0, x - padding_w)
        blur_y = max(0, y - padding_h)
        blur_w = min(frame_width - blur_x, w + 2 * padding_w)
        blur_h = min(frame_height - blur_y, h + 2 * padding_h)
        
        # Ensure minimum and maximum blur size
        blur_w = max(self.config.min_blur_size[0], min(self.config.max_blur_size[0], blur_w))
        blur_h = max(self.config.min_blur_size[1], min(self.config.max_blur_size[1], blur_h))
        
        return (blur_x, blur_y, blur_w, blur_h)
    
    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self.tracked_faces.clear()
        self.next_track_id = 1
        self.previous_gray = None
    
    def get_active_tracks_count(self) -> int:
        """Get number of currently active face tracks.
        
        Returns:
            Number of active tracks
        """
        return len([face for face in self.tracked_faces if face.is_active])
    
    def get_track_info(self) -> List[Dict]:
        """Get information about all active tracks.
        
        Returns:
            List of dictionaries with track information
        """
        track_info = []
        for face in self.tracked_faces:
            if face.is_active:
                info = {
                    'track_id': face.track_id,
                    'bounding_box': face.last_detection.bounding_box,
                    'confidence': face.last_detection.confidence,
                    'frames_without_detection': face.frames_without_detection
                }
                track_info.append(info)
        
        return track_info