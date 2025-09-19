# Golf Video Anonymizer

A service that automatically detects and blurs faces in golf swing videos to protect privacy while preserving swing analysis capabilities.

## Project Structure

```
golf-video-anonymizer/
├── api/                 # REST API handlers
├── processing/          # Video processing and face detection
├── storage/            # File storage and job queue management
├── web/                # Web interface components
├── tests/              # Test suite
├── venv/               # Python virtual environment (created after setup)
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── setup.py           # Environment setup script
```

## Setup Instructions

1. **Create virtual environment and install dependencies:**
   ```bash
   python setup.py
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Run the application:**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## Development

- API documentation will be available at `http://localhost:8000/docs` when running
- Health check endpoint: `http://localhost:8000/health`

## Dependencies

- **FastAPI**: Web framework for building APIs
- **OpenCV**: Computer vision library for video processing and face detection
- **Redis**: Job queue and caching
- **Uvicorn**: ASGI server for running FastAPI applications

See `requirements.txt` for complete dependency list.