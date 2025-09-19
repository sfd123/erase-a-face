"""
File storage and management system for the Golf Video Anonymizer.

This module provides secure file I/O operations, temporary file handling,
and file validation using magic numbers for security.
"""

import os
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class FileManager:
    """
    Manages secure file operations for video processing.
    
    Handles file uploads, validation, temporary storage, and cleanup
    with security considerations for video file processing.
    """
    
    # Magic numbers for supported video formats
    VIDEO_MAGIC_NUMBERS = {
        b'\x00\x00\x00\x18ftypmp4': 'mp4',  # MP4
        b'\x00\x00\x00\x20ftypmp4': 'mp4',  # MP4 variant
        b'\x00\x00\x00\x1cftypisom': 'mp4', # MP4 ISO
        b'\x00\x00\x00\x20ftypisom': 'mp4', # MP4 ISO variant
        b'\x00\x00\x00\x18ftypM4V': 'm4v',  # M4V
        b'\x00\x00\x00\x1cftypqt': 'mov',   # QuickTime MOV
        b'RIFF': 'avi',                      # AVI (partial check)
    }
    
    # Maximum file size (500MB as per requirements)
    MAX_FILE_SIZE = 500 * 1024 * 1024
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.m4v'}
    
    def __init__(self, base_storage_path: str = None, temp_dir: str = None):
        """
        Initialize FileManager with storage paths.
        
        Args:
            base_storage_path: Base directory for file storage
            temp_dir: Directory for temporary files
        """
        self.base_storage_path = Path(base_storage_path or 'storage/files')
        self.temp_dir = Path(temp_dir or tempfile.gettempdir()) / 'golf_video_anonymizer'
        
        # Create directories if they don't exist
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track temporary files for cleanup
        self._temp_files: Dict[str, Path] = {}
        
    def validate_file(self, file_path: Path) -> Dict[str, any]:
        """
        Validate uploaded video file using magic numbers and size checks.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dict containing validation results and file metadata
            
        Raises:
            FileValidationError: If file validation fails
        """
        if not file_path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise FileValidationError(
                f"File size {file_size} bytes exceeds maximum allowed size "
                f"{self.MAX_FILE_SIZE} bytes"
            )
        
        if file_size == 0:
            raise FileValidationError("File is empty")
        
        # Check file extension
        file_extension = file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise FileValidationError(
                f"Unsupported file extension: {file_extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Validate using magic numbers
        detected_format = self._detect_file_format(file_path)
        if not detected_format:
            raise FileValidationError(
                "File format could not be determined from file content. "
                "File may be corrupted or not a valid video file."
            )
        
        # Additional security check - ensure detected format matches extension
        if not self._format_matches_extension(detected_format, file_extension):
            logger.warning(
                f"File extension {file_extension} doesn't match detected format {detected_format}"
            )
        
        return {
            'file_size': file_size,
            'file_extension': file_extension,
            'detected_format': detected_format,
            'is_valid': True,
            'validation_timestamp': datetime.now()
        }
    
    def _detect_file_format(self, file_path: Path) -> Optional[str]:
        """
        Detect file format using magic numbers.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Detected format string or None if not recognized
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 32 bytes for magic number detection
                header = f.read(32)
                
                # Check for video magic numbers
                for magic_bytes, format_name in self.VIDEO_MAGIC_NUMBERS.items():
                    if header.startswith(magic_bytes):
                        return format_name
                
                # Special handling for AVI files (RIFF format)
                if header.startswith(b'RIFF') and b'AVI ' in header:
                    return 'avi'
                
                return None
                
        except Exception as e:
            logger.error(f"Error reading file for format detection: {e}")
            return None
    
    def _format_matches_extension(self, detected_format: str, extension: str) -> bool:
        """
        Check if detected format matches file extension.
        
        Args:
            detected_format: Format detected from magic numbers
            extension: File extension (with dot)
            
        Returns:
            True if format and extension are compatible
        """
        extension = extension.lower().lstrip('.')
        format_extension_map = {
            'mp4': ['mp4', 'm4v'],
            'mov': ['mov'],
            'avi': ['avi'],
            'm4v': ['m4v', 'mp4']
        }
        
        return extension in format_extension_map.get(detected_format, [])  
  
    def create_temp_file(self, job_id: str, suffix: str = '') -> Path:
        """
        Create a temporary file for processing.
        
        Args:
            job_id: Unique identifier for the processing job
            suffix: Optional file suffix
            
        Returns:
            Path to the created temporary file
        """
        temp_file = self.temp_dir / f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
        
        # Ensure the file doesn't already exist
        counter = 1
        original_temp_file = temp_file
        while temp_file.exists():
            temp_file = original_temp_file.with_name(f"{original_temp_file.stem}_{counter}{original_temp_file.suffix}")
            counter += 1
        
        # Create empty file
        temp_file.touch()
        
        # Track for cleanup
        self._temp_files[job_id] = temp_file
        
        logger.info(f"Created temporary file: {temp_file}")
        return temp_file
    
    def store_uploaded_file(self, source_path: Path, job_id: str, original_filename: str) -> Path:
        """
        Store uploaded file securely with validation.
        
        Args:
            source_path: Path to the uploaded file
            job_id: Unique job identifier
            original_filename: Original name of the uploaded file
            
        Returns:
            Path to the stored file
            
        Raises:
            FileValidationError: If file validation fails
        """
        # Validate the file first
        validation_result = self.validate_file(source_path)
        
        # Create secure filename
        safe_filename = self._create_safe_filename(original_filename, job_id)
        storage_path = self.base_storage_path / safe_filename
        
        # Copy file to secure storage
        try:
            shutil.copy2(source_path, storage_path)
            logger.info(f"Stored file: {original_filename} -> {storage_path}")
            
            # Verify the copied file
            if not storage_path.exists() or storage_path.stat().st_size != source_path.stat().st_size:
                raise FileValidationError("File copy verification failed")
            
            return storage_path
            
        except Exception as e:
            # Clean up on failure
            if storage_path.exists():
                storage_path.unlink()
            raise FileValidationError(f"Failed to store file: {e}")
    
    def _create_safe_filename(self, original_filename: str, job_id: str) -> str:
        """
        Create a safe filename for storage.
        
        Args:
            original_filename: Original filename from upload
            job_id: Unique job identifier
            
        Returns:
            Safe filename for storage
        """
        # Extract extension
        original_path = Path(original_filename)
        extension = original_path.suffix.lower()
        
        # Create hash of original filename for uniqueness
        filename_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        # Create safe filename: jobid_hash_timestamp.ext
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{job_id}_{filename_hash}_{timestamp}{extension}"
        
        return safe_filename
    
    def cleanup_temp_files(self, job_id: str = None, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files.
        
        Args:
            job_id: Specific job ID to clean up, or None for all old files
            max_age_hours: Maximum age in hours for files to keep
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        if job_id and job_id in self._temp_files:
            # Clean up specific job's temp file
            temp_file = self._temp_files[job_id]
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up temp file for job {job_id}: {temp_file}")
                except Exception as e:
                    logger.error(f"Failed to clean up temp file {temp_file}: {e}")
            
            del self._temp_files[job_id]
        
        else:
            # Clean up old temporary files
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for temp_file in self.temp_dir.glob('*'):
                if temp_file.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            temp_file.unlink()
                            cleaned_count += 1
                            logger.info(f"Cleaned up old temp file: {temp_file}")
                    except Exception as e:
                        logger.error(f"Failed to clean up temp file {temp_file}: {e}")
            
            # Clean up tracking dict for old entries
            if not job_id:
                self._temp_files.clear()
        
        return cleaned_count
    
    def cleanup_processed_files(self, file_paths: List[Path], max_age_hours: int = 48) -> int:
        """
        Clean up processed files after they've been downloaded or expired.
        
        Args:
            file_paths: List of file paths to clean up
            max_age_hours: Maximum age in hours before automatic cleanup
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for file_path in file_paths:
            if file_path.exists():
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up processed file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up processed file {file_path}: {e}")
        
        return cleaned_count
    
    def schedule_automatic_cleanup(self, job_id: str, file_paths: List[Path], delay_hours: int = 48):
        """
        Schedule automatic cleanup of files after processing completion.
        
        Args:
            job_id: Job ID for tracking
            file_paths: List of file paths to clean up
            delay_hours: Hours to wait before cleanup
        """
        cleanup_time = datetime.now() + timedelta(hours=delay_hours)
        
        # Store cleanup schedule (in production, this would be in a database or queue)
        if not hasattr(self, '_cleanup_schedule'):
            self._cleanup_schedule = {}
        
        self._cleanup_schedule[job_id] = {
            'file_paths': [str(path) for path in file_paths],
            'cleanup_time': cleanup_time,
            'scheduled_at': datetime.now()
        }
        
        logger.info(f"Scheduled cleanup for job {job_id} at {cleanup_time}")
    
    def execute_scheduled_cleanup(self) -> int:
        """
        Execute scheduled cleanup tasks that are due.
        
        Returns:
            Number of files cleaned up
        """
        if not hasattr(self, '_cleanup_schedule'):
            return 0
        
        current_time = datetime.now()
        cleaned_count = 0
        completed_jobs = []
        
        for job_id, schedule in self._cleanup_schedule.items():
            if current_time >= schedule['cleanup_time']:
                file_paths = [Path(path) for path in schedule['file_paths']]
                
                for file_path in file_paths:
                    if file_path.exists():
                        try:
                            self.secure_delete(file_path)
                            cleaned_count += 1
                            logger.info(f"Automatically cleaned up file for job {job_id}: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to clean up scheduled file {file_path}: {e}")
                
                completed_jobs.append(job_id)
        
        # Remove completed cleanup tasks
        for job_id in completed_jobs:
            del self._cleanup_schedule[job_id]
        
        return cleaned_count
    
    def get_file_info(self, file_path: Path) -> Dict[str, any]:
        """
        Get detailed information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {'exists': False}
        
        stat = file_path.stat()
        
        return {
            'exists': True,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix.lower(),
            'name': file_path.name,
            'path': str(file_path)
        }
    
    def secure_delete(self, file_path: Path, passes: int = 3) -> bool:
        """
        Securely delete a file by overwriting it multiple times.
        
        Args:
            file_path: Path to the file to delete
            passes: Number of overwrite passes
            
        Returns:
            True if deletion was successful
        """
        if not file_path.exists():
            return True
        
        try:
            file_size = file_path.stat().st_size
            
            # Overwrite file content multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            file_path.unlink()
            logger.info(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete file {file_path}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        file_count = 0
        temp_size = 0
        temp_count = 0
        
        # Calculate main storage stats
        for file_path in self.base_storage_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        # Calculate temp storage stats
        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file():
                temp_size += file_path.stat().st_size
                temp_count += 1
        
        return {
            'storage_path': str(self.base_storage_path),
            'temp_path': str(self.temp_dir),
            'total_files': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'temp_files': temp_count,
            'temp_size_bytes': temp_size,
            'temp_size_mb': round(temp_size / (1024 * 1024), 2),
            'tracked_temp_files': len(self._temp_files)
        }
    
    def get_storage_info(self) -> Dict[str, any]:
        """
        Get storage information for health checks.
        
        Returns:
            Dictionary with storage information
        """
        stats = self.get_storage_stats()
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_storage_path)
            
            stats.update({
                'disk_total_bytes': total,
                'disk_used_bytes': used,
                'disk_free_bytes': free,
                'disk_free_mb': round(free / (1024 * 1024), 2),
                'disk_usage_percent': round((used / total) * 100, 2)
            })
        except Exception as e:
            logger.warning(f"Could not get disk usage info: {e}")
            stats['disk_info_error'] = str(e)
        
        return stats
    
    def ensure_directories(self):
        """
        Ensure all required directories exist.
        
        Creates the base storage and temp directories if they don't exist.
        """
        try:
            self.base_storage_path.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directories exist: {self.base_storage_path}, {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise

# Global file manager instance
_file_manager = None

def get_file_manager() -> FileManager:
    """
    Get the global FileManager instance.
    
    Returns:
        FileManager instance
    """
    global _file_manager
    if _file_manager is None:
        import config
        _file_manager = FileManager(
            base_storage_path=str(config.UPLOAD_DIR),
            temp_dir=str(config.TEMP_DIR)
        )
    return _file_manager