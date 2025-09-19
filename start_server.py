#!/usr/bin/env python3
"""
Production startup script for Golf Video Anonymizer service.

This script provides a production-ready way to start the service with
proper configuration, logging, and error handling.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

def setup_production_logging():
    """Setup production logging configuration."""
    import config
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE) if hasattr(config, 'LOG_FILE') else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

def validate_environment():
    """Validate that the environment is properly configured."""
    import config
    
    logger = logging.getLogger(__name__)
    
    # Check required directories
    required_dirs = [config.UPLOAD_DIR, config.PROCESSED_DIR, config.TEMP_DIR]
    for directory in required_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Directory ready: {directory}")
        except Exception as e:
            logger.error(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    # Check Redis connection (if available)
    try:
        from storage.job_queue import get_job_queue
        job_queue = get_job_queue()
        if job_queue.health_check():
            logger.info("✓ Redis connection healthy")
        else:
            logger.warning("⚠ Redis connection unhealthy - job queue may not work")
    except Exception as e:
        logger.warning(f"⚠ Redis check failed: {e}")
    
    # Check configuration
    if config.ENVIRONMENT == "production":
        if not hasattr(config, 'SECRET_KEY') or not config.SECRET_KEY:
            logger.error("✗ SECRET_KEY not set for production")
            return False
        
        if len(config.SECRET_KEY) < 32:
            logger.error("✗ SECRET_KEY too short for production")
            return False
        
        logger.info("✓ Production configuration validated")
    
    return True

def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Golf Video Anonymizer Service")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    parser.add_argument("--log-level", default=None, help="Log level (debug, info, warning, error)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_production_logging()
    
    # Import config after logging is setup
    import config
    
    logger.info(f"Starting Golf Video Anonymizer Service")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation complete")
        sys.exit(0)
    
    # Determine server configuration
    host = args.host or config.API_HOST
    port = args.port or config.API_PORT
    reload = args.reload or config.DEBUG
    log_level = args.log_level or config.LOG_LEVEL.lower()
    
    # Production safety checks
    if config.ENVIRONMENT == "production" and reload:
        logger.warning("Auto-reload disabled in production")
        reload = False
    
    if config.ENVIRONMENT == "production" and args.workers == 1:
        logger.info("Consider using multiple workers in production (--workers N)")
    
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Reload: {reload}")
    logger.info(f"  Log level: {log_level}")
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=args.workers if not reload else 1,  # Workers don't work with reload
            reload=reload,
            log_level=log_level,
            access_log=True,
            server_header=False,  # Security: don't expose server info
            date_header=False     # Security: don't expose date info
        )
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()