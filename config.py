"""
Configuration settings for Golf Video Anonymizer

This module provides comprehensive configuration management for different
environments (development, testing, production) with environment variable
support and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Base configuration class."""
    
    def __init__(self):
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration settings."""
        # Validate required directories
        for directory in [self.UPLOAD_DIR, self.PROCESSED_DIR, self.TEMP_DIR]:
            try:
                directory.mkdir(exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
        
        # Validate Redis connection settings
        if not self.REDIS_HOST:
            raise ValueError("REDIS_HOST cannot be empty")
        
        if not (1 <= self.REDIS_PORT <= 65535):
            raise ValueError("REDIS_PORT must be between 1 and 65535")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for logging/debugging."""
        return {
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG,
            "api_host": self.API_HOST,
            "api_port": self.API_PORT,
            "redis_host": self.REDIS_HOST,
            "redis_port": self.REDIS_PORT,
            "max_file_size": self.MAX_FILE_SIZE,
            "upload_dir": str(self.UPLOAD_DIR),
            "processed_dir": str(self.PROCESSED_DIR),
            "temp_dir": str(self.TEMP_DIR)
        }

class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    # Environment
    ENVIRONMENT = "development"
    DEBUG = True
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
    PROCESSED_DIR = BASE_DIR / "storage" / "processed"
    TEMP_DIR = BASE_DIR / "storage" / "temp"
    
    # File settings
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
    ALLOWED_MIME_TYPES = {
        "video/mp4",
        "video/quicktime", 
        "video/x-msvideo"
    }
    
    # Redis settings
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None
    
    # Processing settings
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 5
    BLUR_KERNEL_SIZE = (99, 99)
    BLUR_SIGMA = 30
    MAX_CONCURRENT_JOBS = 2
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Security settings
    SECRET_KEY = "dev-secret-key-change-in-production"
    TOKEN_EXPIRY_HOURS = 24
    
    # Cleanup settings
    CLEANUP_INTERVAL_MINUTES = 30
    FILE_RETENTION_HOURS = 24
    
    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_FILE = "app.log"

class ProductionConfig(Config):
    """Production environment configuration."""
    
    # Environment
    ENVIRONMENT = "production"
    DEBUG = False
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
    PROCESSED_DIR = BASE_DIR / "storage" / "processed"
    TEMP_DIR = BASE_DIR / "storage" / "temp"
    
    # File settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
    ALLOWED_MIME_TYPES = {
        "video/mp4",
        "video/quicktime", 
        "video/x-msvideo"
    }
    
    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    # Processing settings
    FACE_DETECTION_SCALE_FACTOR = float(os.getenv("FACE_DETECTION_SCALE_FACTOR", 1.1))
    FACE_DETECTION_MIN_NEIGHBORS = int(os.getenv("FACE_DETECTION_MIN_NEIGHBORS", 5))
    BLUR_KERNEL_SIZE = (99, 99)
    BLUR_SIGMA = int(os.getenv("BLUR_SIGMA", 30))
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", 4))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY")
    TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", 24))
    
    # Cleanup settings
    CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", 60))
    FILE_RETENTION_HOURS = int(os.getenv("FILE_RETENTION_HOURS", 48))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "/var/log/golf-video-anonymizer.log")
    
    def validate_config(self):
        """Additional validation for production."""
        super().validate_config()
        
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set in production")
        
        if len(self.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")

class TestingConfig(Config):
    """Testing environment configuration."""
    
    # Environment
    ENVIRONMENT = "testing"
    DEBUG = True
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "test_storage" / "uploads"
    PROCESSED_DIR = BASE_DIR / "test_storage" / "processed"
    TEMP_DIR = BASE_DIR / "test_storage" / "temp"
    
    # File settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for testing
    ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
    ALLOWED_MIME_TYPES = {
        "video/mp4",
        "video/quicktime", 
        "video/x-msvideo"
    }
    
    # Redis settings (use different DB for testing)
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 1
    REDIS_PASSWORD = None
    
    # Processing settings
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_DETECTION_MIN_NEIGHBORS = 3  # Lower for faster testing
    BLUR_KERNEL_SIZE = (51, 51)  # Smaller for faster testing
    BLUR_SIGMA = 15
    MAX_CONCURRENT_JOBS = 1
    
    # API settings
    API_HOST = "127.0.0.1"
    API_PORT = 8001
    
    # Security settings
    SECRET_KEY = "test-secret-key-not-for-production"
    TOKEN_EXPIRY_HOURS = 1
    
    # Cleanup settings
    CLEANUP_INTERVAL_MINUTES = 5
    FILE_RETENTION_HOURS = 1
    
    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_FILE = "test.log"

# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()

# Global configuration instance
_config = get_config()

# Export configuration attributes for backward compatibility
ENVIRONMENT = _config.ENVIRONMENT
DEBUG = _config.DEBUG
BASE_DIR = _config.BASE_DIR
UPLOAD_DIR = _config.UPLOAD_DIR
PROCESSED_DIR = _config.PROCESSED_DIR
TEMP_DIR = _config.TEMP_DIR
MAX_FILE_SIZE = _config.MAX_FILE_SIZE
ALLOWED_EXTENSIONS = _config.ALLOWED_EXTENSIONS
ALLOWED_MIME_TYPES = _config.ALLOWED_MIME_TYPES
REDIS_HOST = _config.REDIS_HOST
REDIS_PORT = _config.REDIS_PORT
REDIS_DB = _config.REDIS_DB
REDIS_PASSWORD = getattr(_config, 'REDIS_PASSWORD', None)
FACE_DETECTION_SCALE_FACTOR = _config.FACE_DETECTION_SCALE_FACTOR
FACE_DETECTION_MIN_NEIGHBORS = _config.FACE_DETECTION_MIN_NEIGHBORS
BLUR_KERNEL_SIZE = _config.BLUR_KERNEL_SIZE
BLUR_SIGMA = _config.BLUR_SIGMA
MAX_CONCURRENT_JOBS = _config.MAX_CONCURRENT_JOBS
API_HOST = _config.API_HOST
API_PORT = _config.API_PORT
SECRET_KEY = _config.SECRET_KEY
TOKEN_EXPIRY_HOURS = _config.TOKEN_EXPIRY_HOURS
CLEANUP_INTERVAL_MINUTES = _config.CLEANUP_INTERVAL_MINUTES
FILE_RETENTION_HOURS = _config.FILE_RETENTION_HOURS
LOG_LEVEL = _config.LOG_LEVEL
LOG_FILE = _config.LOG_FILE

def get_config_instance() -> Config:
    """Get the current configuration instance."""
    return _config