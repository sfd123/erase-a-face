"""
Integration test for performance optimizations.

This test verifies that the performance optimizations work correctly
in a real scenario without requiring complex video generation.
"""

import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from processing.video_processor import VideoProcessor, VideoProcessorConfig
from processing.batch_processor import BatchProcessor, BatchProcessorConfig
from processing.progress_service import progress_service, ProgressTracker
from models.processing_job import ProcessingJob


class TestPerformanceIntegration:
    """Integration tests for performance optimizations."""
    
    def test_video_processor_with_optimizations(self):
        """Test video processor with performance optimizations enabled."""
        # Create optimized configuration
        config = VideoProcessorConfig(
            enable_frame_skipping=True,
            frame_skip_ratio=2,
            max_resolution=(1280, 720),
            memory_limit_mb=512,
            max_frames_per_batch=10,
            preserve_audio=False
        )
        
        processor = VideoProcessor(config=config)
        
        # Verify configuration is applied
        assert processor.config.enable_frame_skipping is True
        assert processor.config.frame_skip_ratio == 2
        assert processor.config.max_resolution == (1280, 720)
        assert processor.config.memory_limit_mb == 512
        
        # Test memory usage monitoring
        memory_usage = processor._get_memory_usage()
        assert isinstance(memory_usage, int)
        assert memory_usage >= 0
        
        # Test resolution optimization
        optimized_width, optimized_height = processor._optimize_resolution(3840, 2160)
        assert optimized_width <= 1280
        assert optimized_height <= 720
        
        # Test frame skipping decision
        should_skip = processor._should_enable_frame_skipping(10000, 1920, 1080)
        assert isinstance(should_skip, bool)
        
        print("Video processor optimizations working correctly")
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization and configuration."""
        batch_config = BatchProcessorConfig(
            max_concurrent_jobs=2,
            max_worker_threads=4,
            memory_limit_per_job_mb=256
        )
        
        video_config = VideoProcessorConfig(
            max_frames_per_batch=15,
            preserve_audio=False
        )
        
        batch_processor = BatchProcessor(
            config=batch_config,
            video_processor_config=video_config
        )
        
        # Verify configuration
        assert batch_processor.config.max_concurrent_jobs == 2
        assert batch_processor.config.max_worker_threads == 4
        assert batch_processor.video_processor_config.max_frames_per_batch == 15
        
        # Test system resource monitoring
        resources = batch_processor.get_system_resources()
        assert isinstance(resources, dict)
        
        print("Batch processor initialization working correctly")
    
    def test_progress_service_functionality(self):
        """Test progress service functionality."""
        # Create a progress tracker
        tracker = progress_service.create_tracker("test_operation", total_frames=100)
        
        assert tracker is not None
        assert tracker.operation_id == "test_operation"
        assert tracker.metrics.total_frames == 100
        
        # Test progress updates
        tracker.start("Starting test operation")
        assert tracker.metrics.status.value == "initializing"
        
        tracker.update_progress(25.0, "Processing frames", current_frame=25)
        assert tracker.metrics.progress_percent == 25.0
        assert tracker.metrics.current_frame == 25
        
        tracker.update_system_metrics(memory_mb=128, cpu_percent=45.0)
        assert tracker.metrics.memory_usage_mb == 128
        assert tracker.metrics.cpu_usage_percent == 45.0
        
        # Test completion
        tracker.mark_completed("Test completed successfully")
        assert tracker.metrics.status.value == "completed"
        assert tracker.metrics.progress_percent == 100.0
        
        # Test retrieval
        retrieved_tracker = progress_service.get_tracker("test_operation")
        assert retrieved_tracker is not None
        assert retrieved_tracker.operation_id == "test_operation"
        
        # Test summary
        summary = tracker.get_summary()
        assert summary['operation_id'] == "test_operation"
        assert summary['status'] == "completed"
        assert summary['progress_percent'] == 100.0
        
        # Cleanup
        progress_service.remove_tracker("test_operation")
        
        print("Progress service functionality working correctly")
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_memory_management_integration(self, mock_writer, mock_capture):
        """Test memory management during processing."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30,    # FPS
            3: 1280,  # Width
            4: 720,   # Height
            7: 150    # Frame count
        }.get(prop, 0)
        mock_cap.read.side_effect = [(True, None)] * 10 + [(False, None)]
        mock_capture.return_value = mock_cap
        
        # Mock video writer
        mock_writer_instance = MagicMock()
        mock_writer_instance.isOpened.return_value = True
        mock_writer.return_value = mock_writer_instance
        
        # Create processor with memory limits
        config = VideoProcessorConfig(
            memory_limit_mb=256,
            temp_cleanup_interval=5,
            preserve_audio=False
        )
        
        processor = VideoProcessor(config=config)
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            job = ProcessingJob.create_new("test_memory.mp4", temp_path)
            
            # Mock the face detection and blurring to avoid complex setup
            with patch.object(processor, '_process_single_frame') as mock_process:
                mock_process.return_value = None  # Mock processed frame
                
                # Test memory management
                initial_memory = processor._get_memory_usage()
                processor._manage_memory()
                
                # Verify memory management doesn't crash
                assert isinstance(initial_memory, int)
                assert initial_memory >= 0
                
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("Memory management integration working correctly")
    
    def test_performance_stats_collection(self):
        """Test performance statistics collection."""
        config = VideoProcessorConfig(preserve_audio=False)
        processor = VideoProcessor(config=config)
        
        # Initialize performance stats
        processor.processing_stats['start_time'] = time.time()
        processor.processing_stats['frames_processed'] = 100
        processor.processing_stats['frames_skipped'] = 20
        processor.processing_stats['peak_memory_mb'] = 256
        processor.processing_stats['avg_frame_time'] = 0.033
        
        # Test stats retrieval
        stats = processor.get_processing_stats()
        
        assert 'frames_processed' in stats
        assert 'peak_memory_mb' in stats
        assert 'avg_frame_time' in stats
        
        # Test performance logging
        processor._log_performance_stats()  # Should not crash
        
        print("Performance statistics collection working correctly")
    
    def test_batch_progress_tracking(self):
        """Test progress tracking in batch processing."""
        from processing.batch_processor import BatchProgressTracker
        
        tracker = BatchProgressTracker()
        tracker.total_jobs = 3
        
        # Simulate job progress
        tracker.update_job_progress("job1", 0.5, "Processing job 1")
        tracker.update_job_progress("job2", 0.8, "Processing job 2")
        tracker.update_job_progress("job3", 0.2, "Processing job 3")
        
        # Test overall progress calculation
        progress = tracker.get_overall_progress()
        
        assert 'overall_progress' in progress
        assert 'completed_jobs' in progress
        assert 'total_jobs' in progress
        assert progress['total_jobs'] == 3
        
        # Mark jobs as completed
        tracker.mark_job_completed("job1", success=True)
        tracker.mark_job_completed("job2", success=True)
        tracker.mark_job_completed("job3", success=False)
        
        final_progress = tracker.get_overall_progress()
        assert final_progress['completed_jobs'] == 2
        assert final_progress['failed_jobs'] == 1
        
        print("Batch progress tracking working correctly")


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestPerformanceIntegration()
    
    print("Running performance integration tests...")
    
    test_suite.test_video_processor_with_optimizations()
    test_suite.test_batch_processor_initialization()
    test_suite.test_progress_service_functionality()
    test_suite.test_memory_management_integration()
    test_suite.test_performance_stats_collection()
    test_suite.test_batch_progress_tracking()
    
    print("\nAll performance integration tests passed!")