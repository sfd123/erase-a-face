# Golf Video Anonymizer - Deployment Guide

This document provides comprehensive instructions for deploying and running the Golf Video Anonymizer service.

## Quick Start

### Development Environment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis** (required for job queue)
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:alpine
   
   # Or using Homebrew on macOS
   brew install redis
   brew services start redis
   ```

3. **Run Integration Tests**
   ```bash
   python test_integration.py
   ```

4. **Start the Service**
   ```bash
   # Development mode (auto-reload enabled)
   python main.py
   
   # Or using the startup script
   python start_server.py --reload
   ```

5. **Access the Service**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Production Environment

1. **Environment Configuration**
   ```bash
   # Copy and customize environment file
   cp .env.example .env
   
   # Edit .env with your production settings
   # IMPORTANT: Set SECRET_KEY for production!
   ```

2. **Validate Configuration**
   ```bash
   python start_server.py --validate-only
   ```

3. **Start Production Server**
   ```bash
   # Single worker
   ENVIRONMENT=production python start_server.py
   
   # Multiple workers (recommended for production)
   ENVIRONMENT=production python start_server.py --workers 4
   ```

## Architecture Overview

The service consists of several integrated components:

### Core Components

- **FastAPI Application** (`main.py`): Main web server with REST API
- **Job Queue** (`storage/job_queue.py`): Redis-based background job processing
- **File Manager** (`storage/file_manager.py`): Secure file handling and storage
- **Video Processor** (`processing/video_processor.py`): Core video processing logic
- **Batch Processor** (`processing/batch_processor.py`): Concurrent job processing
- **Cleanup Service** (`processing/cleanup_service.py`): Automatic file cleanup

### API Endpoints

- `POST /api/v1/upload` - Upload video for processing
- `GET /api/v1/status/{job_id}` - Get processing status
- `GET /api/v1/download/{job_id}` - Download processed video
- `GET /api/v1/health` - Service health check
- `GET /health` - Basic health check
- `GET /` - Web interface

### Background Services

The application automatically starts several background services:

1. **Job Queue Processing**: Continuously processes uploaded videos
2. **Cleanup Service**: Periodically removes old files and failed jobs
3. **Batch Processing**: Handles multiple jobs concurrently

## Configuration

### Environment Variables

The service supports configuration through environment variables:

#### Required for Production
- `SECRET_KEY`: Cryptographic key (minimum 32 characters)
- `ENVIRONMENT`: Set to "production"

#### Optional Configuration
- `API_HOST`: Host to bind to (default: 0.0.0.0)
- `API_PORT`: Port to bind to (default: 8000)
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_PASSWORD`: Redis password if required
- `MAX_FILE_SIZE`: Maximum upload size in bytes (default: 500MB)
- `MAX_CONCURRENT_JOBS`: Maximum concurrent processing jobs (default: 2)

### Configuration Files

The service uses a hierarchical configuration system:

1. **Default Configuration**: Built-in defaults in `config.py`
2. **Environment Variables**: Override defaults
3. **Environment-Specific Classes**: Development, Production, Testing

## Monitoring and Logging

### Health Checks

The service provides comprehensive health checks:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with service status
curl http://localhost:8000/api/v1/health
```

### Logging

Logs are configured based on environment:

- **Development**: Console output with DEBUG level
- **Production**: File logging with INFO level
- **Custom**: Set `LOG_LEVEL` and `LOG_FILE` environment variables

### Monitoring Endpoints

- `/api/v1/queue/stats` - Job queue statistics
- `/api/v1/admin/cleanup/stats` - Cleanup service statistics

## Security Features

### File Security
- Magic number validation for uploaded files
- File size limits and extension validation
- Secure temporary file handling
- Automatic cleanup of processed files

### API Security
- Rate limiting on all endpoints
- Input sanitization middleware
- CORS configuration
- Security headers (CSP, HSTS, etc.)
- Trusted host middleware

### Privacy Protection
- No permanent storage of original videos
- Secure deletion of temporary files
- No logging of video content
- Configurable file retention periods

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```
   Error: Redis connection failed
   Solution: Ensure Redis is running and accessible
   ```

2. **Permission Denied on File Operations**
   ```
   Error: Failed to create directory
   Solution: Check file system permissions for storage directories
   ```

3. **Import Errors**
   ```
   Error: ModuleNotFoundError
   Solution: Ensure all dependencies are installed (pip install -r requirements.txt)
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
DEBUG=true python start_server.py --log-level debug
```

### Integration Testing

Run the integration test suite to verify all components:

```bash
python test_integration.py
```

## Performance Tuning

### Concurrent Processing

Adjust concurrent job processing based on your hardware:

```bash
# Environment variable
export MAX_CONCURRENT_JOBS=4

# Or command line
python start_server.py --workers 4
```

### Memory Management

Monitor memory usage and adjust limits:

- `memory_limit_per_job_mb`: Memory limit per processing job
- `cpu_usage_threshold`: CPU threshold for throttling

### File Cleanup

Configure cleanup intervals based on storage capacity:

- `CLEANUP_INTERVAL_MINUTES`: How often to run cleanup
- `FILE_RETENTION_HOURS`: How long to keep processed files

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "start_server.py", "--workers", "4"]
```

### Systemd Service

```ini
[Unit]
Description=Golf Video Anonymizer
After=network.target redis.service

[Service]
Type=exec
User=www-data
WorkingDirectory=/opt/golf-video-anonymizer
Environment=ENVIRONMENT=production
ExecStart=/opt/golf-video-anonymizer/venv/bin/python start_server.py --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 500M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Recovery

### Data Backup

The service stores data in:
- `storage/uploads/` - Original uploaded files (temporary)
- `storage/processed/` - Processed video files (temporary)
- Redis database - Job queue and status information

### Recovery Procedures

1. **Service Recovery**: The service automatically handles interrupted jobs on startup
2. **File Recovery**: Temporary files are cleaned up automatically
3. **Queue Recovery**: Redis persistence handles job queue recovery

## Support

For issues and questions:

1. Check the logs for error messages
2. Run integration tests to verify setup
3. Review configuration settings
4. Check Redis connectivity and health