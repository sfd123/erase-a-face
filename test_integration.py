#!/usr/bin/env python3
"""
Test script to verify the main application integration.

This script tests that all components can be imported and initialized
without errors.
"""

import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test config import
        import config
        logger.info(f"✓ Config loaded - Environment: {config.ENVIRONMENT}")
        
        # Test storage components
        from storage.job_queue import get_job_queue
        from storage.file_manager import get_file_manager
        logger.info("✓ Storage components imported")
        
        # Test processing components
        from processing.batch_processor import get_batch_processor
        from processing.cleanup_service import get_cleanup_service
        logger.info("✓ Processing components imported")
        
        # Test API components
        from api.routes import api_router
        from api.handlers import VideoUploadHandler
        logger.info("✓ API components imported")
        
        # Test main app
        from main import create_app
        logger.info("✓ Main application imported")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_component_initialization():
    """Test that components can be initialized."""
    logger.info("Testing component initialization...")
    
    try:
        # Test file manager
        from storage.file_manager import get_file_manager
        file_manager = get_file_manager()
        file_manager.ensure_directories()
        logger.info("✓ File manager initialized")
        
        # Test job queue (this might fail if Redis is not running)
        try:
            from storage.job_queue import get_job_queue
            job_queue = get_job_queue()
            if job_queue.health_check():
                logger.info("✓ Job queue initialized and Redis is healthy")
            else:
                logger.warning("⚠ Job queue initialized but Redis is not healthy")
        except Exception as e:
            logger.warning(f"⚠ Job queue initialization failed (Redis may not be running): {e}")
        
        # Test batch processor
        from processing.batch_processor import get_batch_processor
        batch_processor = get_batch_processor()
        logger.info("✓ Batch processor initialized")
        
        # Test cleanup service
        from processing.cleanup_service import get_cleanup_service
        cleanup_service = get_cleanup_service()
        logger.info("✓ Cleanup service initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component initialization failed: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    logger.info("Testing FastAPI app creation...")
    
    try:
        from main import create_app
        app = create_app()
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/v1/upload", "/api/v1/status/{job_id}"]
        
        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                logger.info(f"✓ Route found: {expected_route}")
            else:
                logger.warning(f"⚠ Route not found: {expected_route}")
        
        logger.info("✓ FastAPI app created successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ App creation failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting integration tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Component Initialization", test_component_initialization),
        ("App Creation", test_app_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n--- Test Summary ---")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("✓ All integration tests passed!")
        return 0
    else:
        logger.error("✗ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())