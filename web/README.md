# Golf Video Anonymizer - Web Interface

This directory contains the web interface for the Golf Video Anonymizer service, providing a user-friendly way to upload videos and download anonymized results.

## Files

### Core Interface
- `index.html` - Main web interface with drag-and-drop upload
- `styles.css` - CSS styles and responsive design
- `app.js` - JavaScript application logic and API integration

### Testing
- `test.html` - Interactive test page for manual testing
- `app.test.js` - Unit tests for JavaScript functionality
- `run-tests.js` - Test runner script

## Features

### File Upload
- **Drag & Drop**: Intuitive drag-and-drop interface
- **File Validation**: Client-side validation for file type and size
- **Progress Tracking**: Real-time upload and processing progress
- **Error Handling**: Clear error messages for validation failures

### Processing Status
- **Real-time Updates**: Automatic polling for job status updates
- **Progress Visualization**: Progress bar and status indicators
- **Job Information**: Display of job ID, filename, and processing details

### Results & Download
- **Processing Results**: Display of faces detected and processing summary
- **Secure Download**: Direct download of anonymized videos
- **File Management**: Automatic cleanup notifications

### User Experience
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Visual feedback during operations
- **Error Recovery**: Options to retry or start over
- **Accessibility**: Keyboard navigation and screen reader support

## API Integration

The web interface integrates with the following API endpoints:

- `POST /api/v1/upload` - Upload video files
- `GET /api/v1/status/{job_id}` - Check processing status
- `GET /api/v1/download/{job_id}` - Download processed videos
- `GET /api/v1/health` - Service health and configuration

## Configuration

The interface automatically loads configuration from the API health endpoint:
- Maximum file size limits
- Supported video formats
- Service status and version

## Testing

### Manual Testing
Open `test.html` in a browser to run interactive tests:
```bash
# Start the server
python main.py

# Open in browser
open http://localhost:8000/static/test.html
```

### Automated Testing
Run unit tests with Node.js:
```bash
# Run tests
node web/run-tests.js

# Or make executable and run
chmod +x web/run-tests.js
./web/run-tests.js
```

### Test Coverage
The test suite covers:
- File validation logic
- File size formatting
- Progress updates
- Section switching
- Error handling
- API response handling
- Drag and drop events
- Polling logic

## Browser Compatibility

The interface is compatible with modern browsers:
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Development

### Local Development
1. Start the FastAPI server: `python main.py`
2. Open the interface: `http://localhost:8000`
3. Use browser dev tools for debugging

### File Structure
```
web/
├── index.html          # Main interface
├── styles.css          # Styles and layout
├── app.js             # Application logic
├── test.html          # Manual test interface
├── app.test.js        # Unit tests
├── run-tests.js       # Test runner
└── README.md          # This file
```

### Adding Features
1. Update HTML structure in `index.html`
2. Add styles in `styles.css`
3. Implement logic in `app.js`
4. Add tests in `app.test.js`
5. Test manually with `test.html`

## Security Considerations

- Client-side validation is supplemented by server-side validation
- File uploads are sent directly to the API without client-side storage
- No sensitive data is stored in browser localStorage
- CORS is configured appropriately for the API

## Performance

- Efficient polling with configurable intervals
- Progress updates without blocking the UI
- Responsive design for various screen sizes
- Minimal external dependencies

## Troubleshooting

### Common Issues

**Upload fails with validation error:**
- Check file format (MP4, MOV, AVI only)
- Verify file size is under 500MB
- Ensure stable internet connection

**Status polling stops working:**
- Check browser console for errors
- Verify API server is running
- Check network connectivity

**Download doesn't start:**
- Ensure job is completed successfully
- Check browser popup/download settings
- Verify file hasn't been cleaned up

### Debug Mode
Open browser developer tools and check:
- Console for JavaScript errors
- Network tab for API requests
- Application tab for any stored data