# Design Document

## Overview

The Golf Video Anonymizer is a video processing service that uses computer vision to detect and blur faces in golf swing videos. The system will be built as a web service with a REST API, utilizing OpenCV for video processing and face detection, with a simple web interface for file uploads and downloads.

## Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────┐
│   Web Interface │ (Upload/Download UI)
├─────────────────┤
│   REST API      │ (Express.js/FastAPI)
├─────────────────┤
│ Processing Core │ (Video processing logic)
├─────────────────┤
│ Computer Vision │ (OpenCV + face detection)
├─────────────────┤
│ File Storage    │ (Local/Cloud storage)
└─────────────────┘
```

### Technology Stack
- **Backend**: Python with FastAPI for REST API
- **Computer Vision**: OpenCV with Haar Cascades or DNN-based face detection
- **Video Processing**: FFmpeg integration via OpenCV
- **Frontend**: Simple HTML/JavaScript for file upload interface
- **Storage**: Local filesystem with option to extend to cloud storage
- **Queue**: Redis for job queue management (for scalability)

## Components and Interfaces

### 1. API Layer (`api/`)
- **VideoUploadHandler**: Handles file uploads, validation, and job creation
- **ProcessingStatusHandler**: Provides status updates on processing jobs
- **VideoDownloadHandler**: Serves processed video files

### 2. Processing Core (`processing/`)
- **VideoProcessor**: Main orchestrator for video processing pipeline
- **FaceDetector**: Handles face detection using OpenCV
- **FaceBlurrer**: Applies blur effects to detected face regions
- **VideoEncoder**: Manages video encoding and quality preservation

### 3. Storage Layer (`storage/`)
- **FileManager**: Handles file I/O operations
- **JobQueue**: Manages processing job queue and status tracking

### 4. Web Interface (`web/`)
- **UploadForm**: Simple drag-and-drop file upload interface
- **StatusDisplay**: Real-time processing status updates
- **DownloadLink**: Secure download links for processed videos

## Data Models

### ProcessingJob
```python
class ProcessingJob:
    job_id: str
    original_filename: str
    file_path: str
    status: JobStatus  # PENDING, PROCESSING, COMPLETED, FAILED
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    output_file_path: Optional[str]
    faces_detected: int
```

### VideoMetadata
```python
class VideoMetadata:
    duration: float
    fps: int
    resolution: Tuple[int, int]
    format: str
    file_size: int
```

### FaceDetection
```python
class FaceDetection:
    frame_number: int
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
```

## Processing Pipeline

### 1. Video Upload and Validation
- Accept video files via REST API
- Validate file format (MP4, MOV, AVI)
- Check file size limits (max 500MB)
- Generate unique job ID
- Store original file securely

### 2. Face Detection Pipeline
```python
def process_video(job_id: str) -> ProcessingResult:
    # Load video
    video = cv2.VideoCapture(input_path)
    
    # Initialize face detector (Haar Cascade or DNN)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    # Process frame by frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Detect faces
        faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        
        # Apply blur to detected faces
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face
        
        # Write processed frame
        output_video.write(frame)
```

### 3. Face Tracking Enhancement
- Use optical flow to track faces between frames
- Maintain consistent blur regions even with motion
- Handle temporary occlusions or detection failures

### 4. Quality Preservation
- Maintain original video resolution and frame rate
- Use high-quality encoding settings
- Preserve audio track if present
- Minimize compression artifacts

## Error Handling

### Input Validation Errors
- Invalid file format → Return 400 with supported formats
- File too large → Return 413 with size limit information
- Corrupted video → Return 422 with validation error

### Processing Errors
- Face detection failure → Log warning, continue processing
- Video encoding failure → Return 500 with retry option
- Storage errors → Return 503 with temporary unavailability message

### Recovery Mechanisms
- Automatic retry for transient failures
- Graceful degradation when face detection partially fails
- Cleanup of temporary files on error

## Testing Strategy

### Unit Tests
- Face detection accuracy with various lighting conditions
- Blur effect quality and consistency
- Video encoding quality preservation
- API endpoint validation and error handling

### Integration Tests
- End-to-end video processing pipeline
- File upload and download workflows
- Job queue and status tracking
- Error recovery scenarios

### Performance Tests
- Processing time for various video lengths
- Memory usage during processing
- Concurrent job handling
- Large file processing capabilities

### Test Data
- Sample golf swing videos with different:
  - Lighting conditions (bright, dim, backlit)
  - Camera angles (front, side, elevated)
  - Face orientations (frontal, profile, angled)
  - Video qualities (HD, 4K, mobile)
  - Multiple people in frame

## Security Considerations

### File Security
- Validate file types using magic numbers, not just extensions
- Scan uploaded files for malware
- Implement secure file storage with access controls
- Automatic cleanup of processed files after download

### API Security
- Rate limiting on upload endpoints
- File size restrictions
- Input sanitization
- CORS configuration for web interface

### Privacy Protection
- No permanent storage of original videos
- Secure deletion of temporary files
- No logging of video content or metadata
- Optional encryption of stored files