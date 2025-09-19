"""VideoMetadata data model for storing video file information."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VideoMetadata:
    """Data model for video file metadata."""
    
    duration: float  # Duration in seconds
    fps: int  # Frames per second
    resolution: Tuple[int, int]  # (width, height)
    format: str  # Video format/codec
    file_size: int  # File size in bytes
    
    @property
    def width(self) -> int:
        """Get video width."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get video height."""
        return self.resolution[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0.0
    
    @property
    def total_frames(self) -> int:
        """Calculate total number of frames."""
        return int(self.duration * self.fps)
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
    def is_hd(self) -> bool:
        """Check if video is HD quality (720p or higher)."""
        return self.height >= 720
    
    def is_4k(self) -> bool:
        """Check if video is 4K quality."""
        return self.height >= 2160