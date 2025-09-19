#!/usr/bin/env python3
"""
Main entry point for Golf Video Anonymizer service

This module creates and configures the FastAPI application, integrating all
components including API routes, processing services, storage, and security.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.routes import api_router
from api.error_handlers import setup_error_handlers
from security.input_sanitizer import get_input_sanitizer
from storage.job_queue import get_job_queue
from storage.file_manager import get_file_manager
from processing.cleanup_service import get_cleanup_service
from processing.batch_processor import get_batch_processor
import config

# Configure logging based on environment
def setup_logging():
    """Configure logging for the application."""
    log_level = logging.DEBUG if config.DEBUG else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log') if not config.DEBUG else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Global state for graceful shutdown
shutdown_event = asyncio.Event()
background_tasks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown procedures."""
    logger.info("Starting Golf Video Anonymizer service...")
    
    # Startup procedures
    try:
        # Initialize core services
        logger.info("Initializing core services...")
        
        # Initialize job queue
        job_queue = get_job_queue()
        await job_queue.initialize()
        logger.info("Job queue initialized")
        
        # Initialize file manager
        file_manager = get_file_manager()
        file_manager.ensure_directories()
        logger.info("File manager initialized")
        
        # Initialize cleanup service
        cleanup_service = get_cleanup_service()
        cleanup_task = asyncio.create_task(cleanup_service.start_periodic_cleanup())
        background_tasks.append(cleanup_task)
        logger.info("Cleanup service started")
        
        # Initialize batch processor
        batch_processor = get_batch_processor()
        batch_task = asyncio.create_task(batch_processor.start_processing())
        background_tasks.append(batch_task)
        logger.info("Batch processor started")
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    finally:
        # Shutdown procedures
        logger.info("Shutting down Golf Video Anonymizer service...")
        
        # Signal shutdown to background tasks
        shutdown_event.set()
        
        # Cancel background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cleanup services
        try:
            # Cleanup job queue
            job_queue = get_job_queue()
            await job_queue.cleanup()
            logger.info("Job queue cleaned up")
            
            # Final cleanup of temporary files
            cleanup_service = get_cleanup_service()
            cleanup_service.force_cleanup()
            logger.info("Final cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Service shutdown complete")

# Create FastAPI application with lifespan management
app = FastAPI(
    title="Golf Video Anonymizer",
    description="Service for anonymizing faces in golf swing videos",
    version="1.0.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None,
    lifespan=lifespan
)

# Configure middleware based on environment
def setup_middleware(app: FastAPI):
    """Setup middleware for the application."""
    
    # Security middleware - configure based on environment
    allowed_hosts = ["localhost", "127.0.0.1"]
    if not config.DEBUG:
        # Add production hosts from environment
        production_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
        allowed_hosts.extend([host.strip() for host in production_hosts if host.strip()])
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=allowed_hosts
    )
    
    # CORS middleware - configure based on environment
    cors_origins = ["http://localhost:3000", "http://localhost:8000"]
    if not config.DEBUG:
        # Add production origins from environment
        production_origins = os.getenv("CORS_ORIGINS", "").split(",")
        cors_origins.extend([origin.strip() for origin in production_origins if origin.strip()])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

setup_middleware(app)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    return response

# Input sanitization middleware
@app.middleware("http")
async def sanitize_headers(request: Request, call_next):
    """Sanitize request headers."""
    sanitizer = get_input_sanitizer()
    
    # Sanitize headers
    sanitized_headers = sanitizer.sanitize_headers(dict(request.headers))
    
    # Update request headers (this is a simplified approach)
    # In production, you might want to reject requests with dangerous headers
    
    response = await call_next(request)
    return response

# Setup error handlers
setup_error_handlers(app)

# Include API routes
app.include_router(api_router)

# Mount static files for web interface
try:
    app.mount("/static", StaticFiles(directory="web"), name="static")
    logger.info("Static files mounted successfully")
except Exception as e:
    logger.warning(f"Failed to mount static files: {e}")

@app.get("/")
async def root():
    """Serve the web interface"""
    try:
        from fastapi.responses import FileResponse
        return FileResponse("web/index.html")
    except Exception as e:
        logger.error(f"Failed to serve web interface: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "Web interface is currently unavailable"
            }
        )

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with service status"""
    try:
        # Check core services
        health_status = {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "development" if config.DEBUG else "production",
            "services": {}
        }
        
        # Check job queue
        try:
            job_queue = get_job_queue()
            queue_stats = job_queue.get_queue_stats()
            health_status["services"]["job_queue"] = {
                "status": "healthy",
                "stats": queue_stats
            }
        except Exception as e:
            health_status["services"]["job_queue"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check file manager
        try:
            file_manager = get_file_manager()
            storage_info = file_manager.get_storage_info()
            health_status["services"]["file_manager"] = {
                "status": "healthy",
                "storage": storage_info
            }
        except Exception as e:
            health_status["services"]["file_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "message": str(e)
            }
        )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = asyncio.get_event_loop().time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    return app

if __name__ == "__main__":
    logger.info("Starting Golf Video Anonymizer service...")
    
    try:
        uvicorn.run(
            "main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=config.DEBUG,
            log_level="debug" if config.DEBUG else "info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)