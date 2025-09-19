# Golf Video Anonymizer - Test Suite

This directory contains a comprehensive test suite for the Golf Video Anonymizer application, covering unit tests, integration tests, and performance tests with high code coverage.

## Test Structure

### Test Categories

- **Unit Tests** (`test_*_unit.py`): Test individual components in isolation
- **Integration Tests** (`test_*_integration.py`): Test complete workflows and component interactions
- **Performance Tests** (`test_*_performance.py`): Test performance characteristics and scalability
- **API Tests** (`test_api_*.py`): Test REST API endpoints and web interface

### Test Files

| File | Purpose | Coverage |
|------|---------|----------|
| `test_comprehensive_unit.py` | Comprehensive unit tests for all core components | Models, Processing, Storage |
| `test_comprehensive_integration.py` | End-to-end workflow testing | Complete processing pipelines |
| `test_comprehensive_performance.py` | Performance and scalability testing | Memory usage, processing speed |
| `test_models.py` | Data model validation and serialization | ProcessingJob, VideoMetadata, FaceDetection |
| `test_face_detector.py` | Face detection functionality | FaceDetector, various conditions |
| `test_face_blurrer.py` | Face blurring algorithms | Different blur types and configurations |
| `test_video_processor.py` | Video processing pipeline | End-to-end video processing |
| `test_file_manager.py` | File operations and security | File I/O, validation, cleanup |
| `test_job_queue.py` | Job queue management | Redis operations, job lifecycle |
| `test_api_*.py` | API endpoint testing | HTTP requests, error handling |
| `test_web_interface.py` | Web interface functionality | Frontend interactions |
| `test_error_handling.py` | Error scenarios and recovery | Exception handling, edge cases |

## Test Configuration

### pytest.ini
The main pytest configuration file with:
- Test discovery patterns
- Coverage settings (80% minimum)
- Markers for test categorization
- Reporting options

### conftest.py
Shared test fixtures and configuration:
- Test data setup and teardown
- Mock objects for external dependencies
- Sample video generation
- Environment setup

## Running Tests

### Quick Start
```bash
# Run all tests with coverage
python run_tests.py --all

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --performance   # Performance tests only
python run_tests.py --fast          # Exclude slow tests
```

### Advanced Usage
```bash
# Run specific test file
python run_tests.py --file test_models.py

# Run tests matching keyword
python run_tests.py --keyword face

# Generate detailed coverage report
python run_tests.py --coverage --html

# Check test environment
python run_tests.py --check
```

### Direct pytest Usage
```bash
# Basic test run
pytest tests/

# With coverage
pytest --cov=. --cov-report=html tests/

# Specific markers
pytest -m unit tests/                    # Unit tests only
pytest -m "not slow" tests/              # Exclude slow tests
pytest -m "integration and not slow"     # Integration tests, not slow

# Parallel execution
pytest -n auto tests/                    # Auto-detect CPU cores
pytest -n 4 tests/                       # Use 4 processes
```

## Test Data

### Automatic Test Data
Test data is automatically generated during test execution:
- Sample images with face-like features
- Mock video frames with moving faces
- Temporary files for I/O testing

### Manual Test Data Setup
```bash
# Create comprehensive test videos
python tests/setup_test_data.py --all

# Create specific video types
python tests/setup_test_data.py --golf-swing
python tests/setup_test_data.py --multiple-faces
python tests/setup_test_data.py --no-faces
python tests/setup_test_data.py --challenging
python tests/setup_test_data.py --high-res
```

### Test Video Types
- **Golf Swing Video**: Realistic golf swing with detectable face
- **Multiple Faces**: Video with 2-3 people for multi-face detection
- **No Faces**: Landscape video without any faces
- **Challenging Conditions**: Various lighting conditions and noise
- **High Resolution**: Full HD video for performance testing

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for workflows
- `@pytest.mark.performance`: Performance and scalability tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.requires_redis`: Tests requiring Redis connection
- `@pytest.mark.requires_opencv`: Tests requiring OpenCV functionality

## Coverage Requirements

### Minimum Coverage Targets
- **Overall**: 80% minimum (enforced by pytest-cov)
- **Core Models**: 95%+ (ProcessingJob, VideoMetadata, FaceDetection)
- **Processing Components**: 85%+ (FaceDetector, FaceBlurrer, VideoProcessor)
- **API Endpoints**: 90%+ (All REST API routes)
- **Storage Layer**: 85%+ (FileManager, JobQueue)

### Coverage Exclusions
- Virtual environment files
- Test files themselves
- Setup and configuration files
- Third-party dependencies
- Debug and development utilities

## Mock Objects and Fixtures

### Key Fixtures
- `sample_image`: Test image with face-like features
- `sample_video_frames`: Sequence of video frames
- `sample_video_file`: Complete MP4 video file
- `mock_redis`: Mocked Redis connection
- `mock_opencv_cascades`: Mocked OpenCV cascade classifiers
- `temp_dir`: Temporary directory for file operations

### Mocking Strategy
- **External Dependencies**: Redis, OpenCV cascades
- **File System**: Temporary directories for isolation
- **Network Calls**: All external API calls mocked
- **Time-Dependent Operations**: Controlled timing for consistency

## Performance Testing

### Performance Metrics
- **Processing Speed**: Time per frame, total processing time
- **Memory Usage**: Peak memory, memory leaks
- **Scalability**: Performance with different video sizes
- **Concurrency**: Multiple job processing

### Performance Benchmarks
- Face detection: < 100ms per frame (640x480)
- Face blurring: < 50ms per face
- Video processing: < 2x real-time for HD video
- Memory usage: < 500MB for 4K video processing

## Continuous Integration

### GitHub Actions Integration
```yaml
# Example CI configuration
- name: Run Tests
  run: |
    python run_tests.py --all --coverage
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **OpenCV Not Found**
   ```bash
   # Install OpenCV
   pip install opencv-python opencv-contrib-python
   ```

2. **Redis Connection Failed**
   ```bash
   # Start Redis server
   redis-server
   # Or skip Redis tests
   pytest -m "not requires_redis"
   ```

3. **Test Data Missing**
   ```bash
   # Generate test data
   python tests/setup_test_data.py --all
   ```

4. **Coverage Too Low**
   ```bash
   # Generate detailed coverage report
   python run_tests.py --coverage --html
   # Open htmlcov/index.html to see uncovered lines
   ```

### Debug Mode
```bash
# Run tests with verbose output and no coverage
pytest -v -s --tb=long tests/

# Run single test with debugging
pytest -v -s tests/test_models.py::TestProcessingJob::test_create_new_job
```

## Contributing

### Adding New Tests
1. Follow naming convention: `test_*.py`
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Include docstrings explaining test purpose
4. Mock external dependencies
5. Ensure tests are deterministic and isolated

### Test Quality Guidelines
- Each test should test one specific behavior
- Use descriptive test names
- Include both positive and negative test cases
- Test edge cases and error conditions
- Maintain high code coverage
- Keep tests fast and reliable

### Performance Test Guidelines
- Use `@pytest.mark.performance` and `@pytest.mark.slow`
- Include baseline performance expectations
- Test with realistic data sizes
- Monitor memory usage
- Document performance characteristics

## Test Results and Reporting

### Coverage Reports
- **Terminal**: Summary coverage percentage
- **HTML**: Detailed line-by-line coverage (`htmlcov/index.html`)
- **XML**: Machine-readable format for CI (`coverage.xml`)

### Test Execution Reports
- **JUnit XML**: Compatible with CI systems
- **HTML Reports**: Detailed test results with timing
- **Performance Metrics**: Processing times and memory usage

### Continuous Monitoring
- Coverage trends over time
- Performance regression detection
- Test execution time monitoring
- Flaky test identification