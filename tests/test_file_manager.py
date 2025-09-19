"""
Unit tests for the FileManager class.

Tests file validation, temporary file handling, secure operations,
and cleanup functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, mock_open

from storage.file_manager import FileManager, FileValidationError


@pytest.fixture
def temp_storage():
    """Create temporary storage directories for testing."""
    temp_dir = tempfile.mkdtemp()
    storage_dir = Path(temp_dir) / 'storage'
    temp_files_dir = Path(temp_dir) / 'temp'
    
    yield storage_dir, temp_files_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def file_manager(temp_storage):
    """Create FileManager instance with temporary storage."""
    storage_dir, temp_files_dir = temp_storage
    return FileManager(str(storage_dir), str(temp_files_dir))


@pytest.fixture
def sample_video_files(temp_storage):
    """Create sample video files with different formats."""
    storage_dir, _ = temp_storage
    files = {}
    
    # Create MP4 file with proper magic number
    mp4_file = storage_dir / 'test.mp4'
    mp4_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mp4_file, 'wb') as f:
        f.write(b'\x00\x00\x00\x18ftypmp4\x00' + b'fake video data' * 100)
    files['mp4'] = mp4_file
    
    # Create MOV file with proper magic number
    mov_file = storage_dir / 'test.mov'
    with open(mov_file, 'wb') as f:
        f.write(b'\x00\x00\x00\x1cftypqt\x00\x00' + b'fake video data' * 100)
    files['mov'] = mov_file
    
    # Create AVI file with proper magic number
    avi_file = storage_dir / 'test.avi'
    with open(avi_file, 'wb') as f:
        f.write(b'RIFF\x00\x00\x00\x00AVI LIST' + b'fake video data' * 100)
    files['avi'] = avi_file
    
    # Create invalid file
    invalid_file = storage_dir / 'test.txt'
    with open(invalid_file, 'wb') as f:
        f.write(b'This is not a video file')
    files['invalid'] = invalid_file
    
    # Create empty file
    empty_file = storage_dir / 'empty.mp4'
    empty_file.touch()
    files['empty'] = empty_file
    
    return files


class TestFileManager:
    """Test suite for FileManager class."""
    pass


class TestFileValidation:
    """Test file validation functionality."""
    
    def test_validate_mp4_file(self, file_manager, sample_video_files):
        """Test validation of MP4 file."""
        result = file_manager.validate_file(sample_video_files['mp4'])
        
        assert result['is_valid'] is True
        assert result['detected_format'] == 'mp4'
        assert result['file_extension'] == '.mp4'
        assert result['file_size'] > 0
        assert 'validation_timestamp' in result
    
    def test_validate_mov_file(self, file_manager, sample_video_files):
        """Test validation of MOV file."""
        result = file_manager.validate_file(sample_video_files['mov'])
        
        assert result['is_valid'] is True
        assert result['detected_format'] == 'mov'
        assert result['file_extension'] == '.mov'
    
    def test_validate_avi_file(self, file_manager, sample_video_files):
        """Test validation of AVI file."""
        result = file_manager.validate_file(sample_video_files['avi'])
        
        assert result['is_valid'] is True
        assert result['detected_format'] == 'avi'
        assert result['file_extension'] == '.avi'
    
    def test_validate_nonexistent_file(self, file_manager):
        """Test validation of non-existent file."""
        with pytest.raises(FileValidationError, match="File does not exist"):
            file_manager.validate_file(Path('/nonexistent/file.mp4'))
    
    def test_validate_empty_file(self, file_manager, sample_video_files):
        """Test validation of empty file."""
        with pytest.raises(FileValidationError, match="File is empty"):
            file_manager.validate_file(sample_video_files['empty'])
    
    def test_validate_unsupported_extension(self, file_manager, sample_video_files):
        """Test validation of file with unsupported extension."""
        with pytest.raises(FileValidationError, match="Unsupported file extension"):
            file_manager.validate_file(sample_video_files['invalid'])
    
    def test_validate_invalid_format(self, file_manager, temp_storage):
        """Test validation of file with valid extension but invalid content."""
        storage_dir, _ = temp_storage
        fake_mp4 = storage_dir / 'fake.mp4'
        fake_mp4.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fake_mp4, 'wb') as f:
            f.write(b'This is not a real MP4 file content')
        
        with pytest.raises(FileValidationError, match="File format could not be determined"):
            file_manager.validate_file(fake_mp4)
    
    def test_validate_oversized_file(self, file_manager, temp_storage):
        """Test validation of file exceeding size limit."""
        storage_dir, _ = temp_storage
        large_file = storage_dir / 'large.mp4'
        large_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file larger than MAX_FILE_SIZE
        with open(large_file, 'wb') as f:
            f.write(b'\x00\x00\x00\x18ftypmp4\x00')
            # Write data to exceed the limit
            chunk_size = 1024 * 1024  # 1MB chunks
            for _ in range(FileManager.MAX_FILE_SIZE // chunk_size + 1):
                f.write(b'x' * chunk_size)
        
        with pytest.raises(FileValidationError, match="exceeds maximum allowed size"):
            file_manager.validate_file(large_file)


class TestTemporaryFileHandling:
    """Test temporary file operations."""
    
    def test_create_temp_file(self, file_manager):
        """Test creation of temporary file."""
        job_id = 'test_job_123'
        temp_file = file_manager.create_temp_file(job_id, '.mp4')
        
        assert temp_file.exists()
        assert job_id in str(temp_file)
        assert temp_file.suffix == '.mp4'
        assert job_id in file_manager._temp_files
        assert file_manager._temp_files[job_id] == temp_file
    
    def test_create_temp_file_unique_names(self, file_manager):
        """Test that temporary files have unique names."""
        job_id1 = 'job1'
        job_id2 = 'job2'
        
        temp_file1 = file_manager.create_temp_file(job_id1, '.mp4')
        temp_file2 = file_manager.create_temp_file(job_id2, '.mp4')
        
        assert temp_file1 != temp_file2
        assert temp_file1.exists()
        assert temp_file2.exists()
    
    def test_cleanup_specific_temp_file(self, file_manager):
        """Test cleanup of specific job's temporary file."""
        job_id = 'cleanup_test'
        temp_file = file_manager.create_temp_file(job_id)
        
        assert temp_file.exists()
        
        cleaned_count = file_manager.cleanup_temp_files(job_id)
        
        assert cleaned_count == 1
        assert not temp_file.exists()
        assert job_id not in file_manager._temp_files
    
    def test_cleanup_old_temp_files(self, file_manager):
        """Test cleanup of old temporary files."""
        # Create some temp files
        job1 = file_manager.create_temp_file('job1')
        job2 = file_manager.create_temp_file('job2')
        
        # Modify file times to make them appear old
        old_time = (datetime.now() - timedelta(hours=25)).timestamp()
        os.utime(job1, (old_time, old_time))
        os.utime(job2, (old_time, old_time))
        
        cleaned_count = file_manager.cleanup_temp_files(max_age_hours=24)
        
        assert cleaned_count == 2
        assert not job1.exists()
        assert not job2.exists()


class TestFileStorage:
    """Test file storage operations."""
    
    def test_store_uploaded_file(self, file_manager, sample_video_files):
        """Test storing an uploaded file."""
        source_file = sample_video_files['mp4']
        job_id = 'upload_test'
        original_filename = 'user_video.mp4'
        
        stored_path = file_manager.store_uploaded_file(source_file, job_id, original_filename)
        
        assert stored_path.exists()
        assert job_id in stored_path.name
        assert stored_path.suffix == '.mp4'
        assert stored_path.stat().st_size == source_file.stat().st_size
    
    def test_store_invalid_file(self, file_manager, sample_video_files):
        """Test storing an invalid file raises error."""
        invalid_file = sample_video_files['invalid']
        
        with pytest.raises(FileValidationError):
            file_manager.store_uploaded_file(invalid_file, 'test_job', 'invalid.txt')
    
    def test_create_safe_filename(self, file_manager):
        """Test safe filename creation."""
        original_filename = 'My Video File!@#$.mp4'
        job_id = 'test123'
        
        safe_filename = file_manager._create_safe_filename(original_filename, job_id)
        
        assert job_id in safe_filename
        assert safe_filename.endswith('.mp4')
        assert len(safe_filename.split('_')) >= 3  # job_id, hash, timestamp


class TestFileInfo:
    """Test file information retrieval."""
    
    def test_get_file_info_existing(self, file_manager, sample_video_files):
        """Test getting info for existing file."""
        file_path = sample_video_files['mp4']
        info = file_manager.get_file_info(file_path)
        
        assert info['exists'] is True
        assert info['size'] > 0
        assert info['extension'] == '.mp4'
        assert info['name'] == file_path.name
        assert 'created' in info
        assert 'modified' in info
    
    def test_get_file_info_nonexistent(self, file_manager):
        """Test getting info for non-existent file."""
        info = file_manager.get_file_info(Path('/nonexistent/file.mp4'))
        
        assert info['exists'] is False
        assert len(info) == 1


class TestSecureOperations:
    """Test secure file operations."""
    
    def test_secure_delete(self, file_manager, temp_storage):
        """Test secure file deletion."""
        storage_dir, _ = temp_storage
        test_file = storage_dir / 'delete_test.txt'
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with some content
        with open(test_file, 'w') as f:
            f.write('sensitive data that should be securely deleted')
        
        assert test_file.exists()
        
        result = file_manager.secure_delete(test_file)
        
        assert result is True
        assert not test_file.exists()
    
    def test_secure_delete_nonexistent(self, file_manager):
        """Test secure deletion of non-existent file."""
        result = file_manager.secure_delete(Path('/nonexistent/file.txt'))
        assert result is True


class TestStorageStats:
    """Test storage statistics."""
    
    def test_get_storage_stats(self, file_manager, sample_video_files):
        """Test getting storage statistics."""
        # Store a file to have some data
        file_manager.store_uploaded_file(
            sample_video_files['mp4'], 
            'stats_test', 
            'test.mp4'
        )
        
        stats = file_manager.get_storage_stats()
        
        assert 'storage_path' in stats
        assert 'temp_path' in stats
        assert stats['total_files'] >= 1
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] > 0
        assert 'temp_files' in stats
        assert 'tracked_temp_files' in stats


class TestMagicNumberDetection:
    """Test magic number detection for different formats."""
    
    def test_detect_mp4_format(self, file_manager, temp_storage):
        """Test detection of MP4 format."""
        storage_dir, _ = temp_storage
        mp4_file = storage_dir / 'test.mp4'
        mp4_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mp4_file, 'wb') as f:
            f.write(b'\x00\x00\x00\x18ftypmp4\x00test data')
        
        detected = file_manager._detect_file_format(mp4_file)
        assert detected == 'mp4'
    
    def test_detect_avi_format(self, file_manager, temp_storage):
        """Test detection of AVI format."""
        storage_dir, _ = temp_storage
        avi_file = storage_dir / 'test.avi'
        avi_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(avi_file, 'wb') as f:
            f.write(b'RIFF\x00\x00\x00\x00AVI LISTtest data')
        
        detected = file_manager._detect_file_format(avi_file)
        assert detected == 'avi'
    
    def test_detect_unknown_format(self, file_manager, temp_storage):
        """Test detection of unknown format."""
        storage_dir, _ = temp_storage
        unknown_file = storage_dir / 'test.unknown'
        unknown_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(unknown_file, 'wb') as f:
            f.write(b'unknown file format data')
        
        detected = file_manager._detect_file_format(unknown_file)
        assert detected is None


class TestFormatExtensionMatching:
    """Test format and extension matching."""
    
    def test_mp4_extension_matching(self, file_manager):
        """Test MP4 format extension matching."""
        assert file_manager._format_matches_extension('mp4', '.mp4') is True
        assert file_manager._format_matches_extension('mp4', '.m4v') is True
        assert file_manager._format_matches_extension('mp4', '.mov') is False
    
    def test_mov_extension_matching(self, file_manager):
        """Test MOV format extension matching."""
        assert file_manager._format_matches_extension('mov', '.mov') is True
        assert file_manager._format_matches_extension('mov', '.mp4') is False
    
    def test_avi_extension_matching(self, file_manager):
        """Test AVI format extension matching."""
        assert file_manager._format_matches_extension('avi', '.avi') is True
        assert file_manager._format_matches_extension('avi', '.mp4') is False