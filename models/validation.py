"""Validation functions for video files and processing parameters."""

import os
import magic
from pathlib import Path
from typing import List, Optional


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


# Supported video formats and their MIME types
SUPPORTED_VIDEO_FORMATS = {
    'video/mp4': ['.mp4'],
    'video/quicktime': ['.mov'],
    'video/x-msvideo': ['.avi'],
    'video/avi': ['.avi']  # Alternative MIME type for AVI
}

# File size limits
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MIN_FILE_SIZE_BYTES = 1024  # 1 KB


def validate_video_file(file_path: str) -> None:
    """
    Validate video file format using magic numbers.
    
    Args:
        file_path: Path to the video file
        
    Raises:
        ValidationError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get MIME type using magic numbers (more secure than file extension)
    try:
        mime_type = magic.from_file(file_path, mime=True)
    except Exception as e:
        raise ValidationError(f"Could not determine file type: {str(e)}")
    
    # Check if MIME type is supported
    if mime_type not in SUPPORTED_VIDEO_FORMATS:
        supported_extensions = []
        for extensions in SUPPORTED_VIDEO_FORMATS.values():
            supported_extensions.extend(extensions)
        
        raise ValidationError(
            f"Unsupported video format: {mime_type}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )
    
    # Additional check: verify file extension matches MIME type
    file_extension = Path(file_path).suffix.lower()
    expected_extensions = SUPPORTED_VIDEO_FORMATS[mime_type]
    
    if file_extension not in expected_extensions:
        raise ValidationError(
            f"File extension '{file_extension}' does not match detected format '{mime_type}'"
        )


def validate_file_size(file_path: str) -> None:
    """
    Validate video file size is within acceptable limits.
    
    Args:
        file_path: Path to the video file
        
    Raises:
        ValidationError: If file size is outside acceptable range
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    
    if file_size < MIN_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File too small: {file_size} bytes. Minimum size: {MIN_FILE_SIZE_BYTES} bytes"
        )
    
    if file_size > MAX_FILE_SIZE_BYTES:
        max_size_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        current_size_mb = file_size / (1024 * 1024)
        raise ValidationError(
            f"File too large: {current_size_mb:.1f} MB. Maximum size: {max_size_mb:.1f} MB"
        )


def validate_filename(filename: str) -> None:
    """
    Validate filename for security and compatibility.
    
    Args:
        filename: Original filename
        
    Raises:
        ValidationError: If filename contains invalid characters
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    # Check for dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        if char in filename:
            raise ValidationError(f"Filename contains invalid character: '{char}'")
    
    # Check filename length
    if len(filename) > 255:
        raise ValidationError("Filename too long (maximum 255 characters)")
    
    # Check for valid extension
    file_extension = Path(filename).suffix.lower()
    valid_extensions = []
    for extensions in SUPPORTED_VIDEO_FORMATS.values():
        valid_extensions.extend(extensions)
    
    if file_extension not in valid_extensions:
        raise ValidationError(
            f"Invalid file extension: {file_extension}. "
            f"Supported extensions: {', '.join(valid_extensions)}"
        )


def get_supported_formats() -> List[str]:
    """Get list of supported video file extensions."""
    extensions = []
    for ext_list in SUPPORTED_VIDEO_FORMATS.values():
        extensions.extend(ext_list)
    return sorted(list(set(extensions)))


def get_max_file_size_mb() -> float:
    """Get maximum file size in megabytes."""
    return MAX_FILE_SIZE_BYTES / (1024 * 1024)