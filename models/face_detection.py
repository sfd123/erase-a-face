"""FaceDetection data model for storing face detection results."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FaceDetection:
    """Data model for face detection results in video frames."""
    
    frame_number: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Detection confidence score (0.0 to 1.0)
    
    @property
    def x(self) -> int:
        """Get x coordinate of bounding box."""
        return self.bounding_box[0]
    
    @property
    def y(self) -> int:
        """Get y coordinate of bounding box."""
        return self.bounding_box[1]
    
    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bounding_box[2]
    
    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bounding_box[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of the face bounding box."""
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        return (center_x, center_y)
    
    @property
    def area(self) -> int:
        """Get area of the face bounding box."""
        return self.width * self.height
    
    def overlaps_with(self, other: 'FaceDetection', threshold: float = 0.5) -> bool:
        """Check if this face detection overlaps with another face detection."""
        # Calculate intersection area
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return False  # No intersection
        
        intersection_area = (x2 - x1) * (y2 - y1)
        union_area = self.area + other.area - intersection_area
        
        # Calculate Intersection over Union (IoU)
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou >= threshold