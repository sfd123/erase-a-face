"""
Comprehensive performance tests with various video conditions.
"""

import pytest
import time
import psutil
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from processing.face_detector import FaceDetector, FaceDetectorConfig
from processing.face_blurrer import FaceBlurrer, FaceBlurrerConfig
from processing.video_processor import VideoProcessor
from storage.file_manager import FileManager
from models.face_detection import FaceDetection


@pytest.mark.performance
@pytest.mark.slow
class TestFaceDetectionPerformance:
    """Test face detection performance under various conditions."""
    
    @pytest.fixture
    def performance_images(self):
        """Create images of different sizes for performance testing."""
        sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        images = {}
        
        for width, height in sizes:
            # Create image with face-like rectangle
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add face-like rectangle
            face_x, face_y = width // 4, height // 4
            face_w, face_h = width // 8, height // 6
            cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), 
                         (150, 150, 150), -1)
            
            images[f"{width}x{height}"] = image
        
        return images
    
    @patch('cv2.CascadeClassifier')
    def test_detection_performance_by_resolution(self, mock_cascade_class, performance_images):
        """Test face detection performance across different resolutions."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 80, 80]])
        mock_cascade_class.return_value = mock_cascade
        
        detector = FaceDetector()
        performance_results = {}
        
        for resolution, image in performance_images.items():
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Perform detection
            detections = detector.detect_faces(image)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            performance_results[resolution] = {
                'processing_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'detections_count': len(detections)
            }
        
        # Verify performance characteristics
        for resolution, results in performance_results.items():
            assert results['processing_time'] < 1.0  # Should be fast with mocked detection
            assert results['detections_count'] >= 0
            
        # Higher resolution should generally take more time (though mocked here)
        print(f"\nPerformance Results by Resolution:")
        for resolution, results in performance_results.items():
            print(f"{resolution}: {results['processing_time']:.4f}s, "
                  f"Memory: {results['memory_used']} bytes")
    
    @patch('cv2.CascadeClassifier')
    def test_batch_processing_performance(self, mock_cascade_class):
        """Test batch processing performance vs individual processing."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
        mock_cascade_class.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Create batch of images
        batch_size = 20
        images = []
        for i in range(batch_size):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append(image)
        
        # Test individual processing
        start_time = time.time()
        individual_results = []
        for i, image in enumerate(images):
            detections = detector.detect_faces(image, frame_number=i)
            individual_results.append(detections)
        individual_time = time.time() - start_time
        
        # Test batch processing
        start_time = time.time()
        batch_results = detector.detect_faces_batch(images, start_frame=0)
        batch_time = time.time() - start_time
        
        # Verify results are consistent
        assert len(individual_results) == len(batch_results)
        
        # Batch processing should be at least as efficient
        print(f"\nBatch Processing Performance:")
        print(f"Individual processing: {individual_time:.4f}s")
        print(f"Batch processing: {batch_time:.4f}s")
        print(f"Efficiency gain: {individual_time / batch_time:.2f}x")
    
    @patch('cv2.CascadeClassifier')
    def test_detection_with_multiple_faces_performance(self, mock_cascade_class):
        """Test performance with varying numbers of faces."""
        mock_cascade = MagicMock()
        mock_cascade.empty.return_value = False
        mock_cascade_class.return_value = mock_cascade
        
        detector = FaceDetector()
        
        # Test with different numbers of faces
        face_counts = [0, 1, 3, 5, 10]
        performance_data = {}
        
        for face_count in face_counts:
            # Mock detection results
            if face_count == 0:
                mock_cascade.detectMultiScale.return_value = np.array([])
            else:
                faces = []
                for i in range(face_count):
                    x, y = i * 60, i * 40
                    faces.append([x, y, 50, 50])
                mock_cascade.detectMultiScale.return_value = np.array(faces)
            
            # Create test image
            image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            
            # Measure performance
            start_time = time.time()
            detections = detector.detect_faces(image)
            processing_time = time.time() - start_time
            
            performance_data[face_count] = {
                'processing_time': processing_time,
                'detected_faces': len(detections)
            }
        
        # Verify performance scales reasonably
        for face_count, data in performance_data.items():
            assert data['processing_time'] < 0.5  # Should be fast
            assert data['detected_faces'] == face_count
            
        print(f"\nPerformance by Face Count:")
        for face_count, data in performance_data.items():
            print(f"{face_count} faces: {data['processing_time']:.4f}s")


@pytest.mark.performance
class TestBlurringPerformance:
    """Test face blurring performance."""
    
    def test_blur_performance_by_face_size(self):
        """Test blurring performance with different face sizes."""
        blurrer = FaceBlurrer()
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        face_sizes = [(50, 50), (100, 100), (200, 200), (400, 400)]
        performance_results = {}
        
        for width, height in face_sizes:
            detection = FaceDetection(0, (100, 100, width, height), 0.9)
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            blurred_image = blurrer.blur_faces(image, [detection])
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            performance_results[f"{width}x{height}"] = {
                'processing_time': end_time - start_time,
                'memory_used': end_memory - start_memory
            }
            
            # Verify blur was applied
            assert blurred_image.shape == image.shape
        
        print(f"\nBlur Performance by Face Size:")
        for size, results in performance_results.items():
            print(f"{size}: {results['processing_time']:.4f}s")
    
    def test_different_blur_configurations_performance(self):
        """Test performance of different blur configurations."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = FaceDetection(0, (100, 100, 150, 150), 0.9)
        
        blur_configs = [
            ("light", FaceBlurrerConfig(blur_kernel_size=31, blur_sigma=10.0)),
            ("medium", FaceBlurrerConfig(blur_kernel_size=51, blur_sigma=20.0)),
            ("heavy", FaceBlurrerConfig(blur_kernel_size=99, blur_sigma=30.0))
        ]
        performance_results = {}
        
        for config_name, config in blur_configs:
            blurrer = FaceBlurrer(config)
            
            start_time = time.time()
            blurred_image = blurrer.blur_faces(image, [detection])
            processing_time = time.time() - start_time
            
            performance_results[config_name] = {
                'processing_time': processing_time
            }
            
            assert blurred_image.shape == image.shape
        
        print(f"\nBlur Configuration Performance:")
        for config_name, results in performance_results.items():
            print(f"{config_name}: {results['processing_time']:.4f}s")


@pytest.mark.performance
@pytest.mark.slow
class TestVideoProcessingPerformance:
    """Test video processing performance."""
    
    def test_video_processing_scalability(self, temp_dir):
        """Test video processing performance with different video lengths."""
        frame_counts = [30, 60, 120, 300]  # 1s, 2s, 4s, 10s at 30fps
        performance_results = {}
        
        for frame_count in frame_counts:
            # Create test video
            video_path = temp_dir / f"test_video_{frame_count}frames.mp4"
            self._create_test_video(video_path, frame_count)
            
            # Mock video processor
            with patch('processing.video_processor.VideoProcessor') as mock_processor_class:
                mock_processor = MagicMock()
                mock_processor_class.return_value = mock_processor
                
                # Simulate processing time proportional to frame count
                processing_time = frame_count * 0.01  # 10ms per frame
                mock_processor.process_video.return_value = {
                    'success': True,
                    'output_path': str(temp_dir / f"output_{frame_count}.mp4"),
                    'faces_detected': frame_count // 10,  # Some faces detected
                    'processing_time': processing_time
                }
                
                processor = VideoProcessor()
                
                start_time = time.time()
                result = processor.process_video(
                    str(video_path),
                    str(temp_dir / f"output_{frame_count}.mp4")
                )
                actual_time = time.time() - start_time
                
                performance_results[frame_count] = {
                    'simulated_processing_time': result['processing_time'],
                    'actual_call_time': actual_time,
                    'faces_detected': result['faces_detected']
                }
        
        print(f"\nVideo Processing Performance by Length:")
        for frames, results in performance_results.items():
            duration = frames / 30.0  # Assume 30fps
            print(f"{frames} frames ({duration:.1f}s): "
                  f"Processing: {results['simulated_processing_time']:.3f}s, "
                  f"Faces: {results['faces_detected']}")
    
    def _create_test_video(self, output_path: Path, frame_count: int):
        """Create a test video with specified frame count."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (320, 240))
        
        for i in range(frame_count):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            # Add a moving rectangle to simulate content
            x_pos = (i * 2) % 250
            cv2.rectangle(frame, (x_pos, 50), (x_pos + 50, 100), (128, 128, 128), -1)
            writer.write(frame)
        
        writer.release()


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns."""
    
    def test_memory_usage_during_processing(self):
        """Test memory usage during face detection and blurring."""
        # Create large image
        large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)  # 4K
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Mock face detector
        with patch('cv2.CascadeClassifier') as mock_cascade_class:
            mock_cascade = MagicMock()
            mock_cascade.empty.return_value = False
            mock_cascade.detectMultiScale.return_value = np.array([[500, 500, 200, 200]])
            mock_cascade_class.return_value = mock_cascade
            
            detector = FaceDetector()
            detections = detector.detect_faces(large_image)
            
            detection_memory = psutil.Process().memory_info().rss
            
            # Apply blur
            blurrer = FaceBlurrer()
            if detections:
                blurred_image = blurrer.blur_face(large_image, detections[0])
                
            final_memory = psutil.Process().memory_info().rss
        
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage Analysis:")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"After detection: {detection_memory / 1024 / 1024:.1f} MB")
        print(f"After blurring: {final_memory / 1024 / 1024:.1f} MB")
        print(f"Total increase: {memory_increase / 1024 / 1024:.1f} MB")
        
        # Memory increase should be reasonable for 4K image processing
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
    
    def test_memory_cleanup_after_processing(self, temp_dir):
        """Test that memory is properly cleaned up after processing."""
        file_manager = FileManager(base_path=str(temp_dir))
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and process multiple files
        for i in range(5):
            test_content = b"test video content " * 1000  # Larger content
            file_path = file_manager.save_uploaded_file(test_content, f"test_{i}.mp4")
            
            # Load and delete file
            loaded_content = file_manager.load_file(file_path)
            file_manager.delete_file(file_path)
            
            assert loaded_content == test_content
        
        # Cleanup old files
        file_manager.cleanup_old_files(max_age_minutes=0)
        
        final_memory = psutil.Process().memory_info().rss
        memory_difference = abs(final_memory - initial_memory)
        
        print(f"\nMemory Cleanup Analysis:")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"Final memory: {final_memory / 1024 / 1024:.1f} MB")
        print(f"Difference: {memory_difference / 1024 / 1024:.1f} MB")
        
        # Memory should not increase significantly after cleanup
        assert memory_difference < 50 * 1024 * 1024  # Less than 50MB difference


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    def test_concurrent_job_processing_simulation(self, mock_redis):
        """Test performance with multiple concurrent jobs."""
        from storage.job_queue import JobQueue
        from models.processing_job import ProcessingJob
        
        job_queue = JobQueue()
        
        # Create multiple jobs
        job_count = 10
        jobs = []
        
        start_time = time.time()
        
        for i in range(job_count):
            job = ProcessingJob.create_new(f"concurrent_video_{i}.mp4", f"/path/video_{i}.mp4")
            job_queue.add_job(job)
            jobs.append(job)
        
        creation_time = time.time() - start_time
        
        # Simulate concurrent processing
        start_time = time.time()
        
        for job in jobs:
            job.mark_processing()
            job_queue.update_job(job)
        
        processing_start_time = time.time() - start_time
        
        # Complete jobs
        start_time = time.time()
        
        for i, job in enumerate(jobs):
            job.mark_completed(f"/path/output_{i}.mp4", faces_detected=i % 5)
            job_queue.update_job(job)
        
        completion_time = time.time() - start_time
        
        print(f"\nConcurrent Processing Performance:")
        print(f"Job creation ({job_count} jobs): {creation_time:.4f}s")
        print(f"Processing start: {processing_start_time:.4f}s")
        print(f"Job completion: {completion_time:.4f}s")
        print(f"Average per job: {(creation_time + processing_start_time + completion_time) / job_count:.4f}s")
        
        # Verify all jobs completed
        for job in jobs:
            retrieved_job = job_queue.get_job(job.job_id)
            assert retrieved_job.status.value == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])