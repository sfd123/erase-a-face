"""
Performance tests for video processing with various video sizes and lengths.

This module contains comprehensive performance tests to validate the system's
ability to handle different video characteristics efficiently while maintaining
quality and managing system resources.
"""

import pytest
import time
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

from processing.video_processor import VideoProcessor, VideoProcessorConfig
from processing.batch_processor import BatchProcessor, BatchProcessorConfig
from models.processing_job import ProcessingJob
from storage.file_manager import FileManager


class VideoGenerator:
    """Helper class to generate test videos with different characteristics."""
    
    @staticmethod
    def create_test_video(width: int, height: int, fps: int, duration_seconds: int,
                         with_faces: bool = True, output_path: str = None) -> str:
        """
        Create a test video with specified characteristics.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            duration_seconds: Video duration in seconds
            with_faces: Whether to include face-like rectangles
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to the created video file
        """
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
        
        # Calculate total frames
        total_frames = fps * duration_seconds
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_num in range(total_frames):
                # Create a frame with gradient background
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add gradient background
                for y in range(height):
                    for x in range(width):
                        frame[y, x] = [
                            int(255 * x / width),
                            int(255 * y / height),
                            int(255 * (frame_num % 100) / 100)
                        ]
                
                # Add face-like rectangles if requested
                if with_faces:
                    # Add 1-3 moving face rectangles
                    num_faces = 1 + (frame_num // 30) % 3
                    for i in range(num_faces):
                        # Calculate moving position
                        x_offset = int(50 + 100 * np.sin(frame_num * 0.1 + i * 2))
                        y_offset = int(50 + 50 * np.cos(frame_num * 0.1 + i * 2))
                        
                        # Ensure face stays within bounds
                        face_size = 80
                        x = max(0, min(width - face_size, x_offset))
                        y = max(0, min(height - face_size, y_offset))
                        
                        # Draw face-like rectangle (skin color)
                        cv2.rectangle(frame, (x, y), (x + face_size, y + face_size), 
                                    (180, 150, 120), -1)
                        
                        # Add eyes
                        cv2.circle(frame, (x + 20, y + 25), 5, (0, 0, 0), -1)
                        cv2.circle(frame, (x + 60, y + 25), 5, (0, 0, 0), -1)
                        
                        # Add mouth
                        cv2.ellipse(frame, (x + 40, y + 55), (15, 8), 0, 0, 180, (0, 0, 0), 2)
                
                writer.write(frame)
                
        finally:
            writer.release()
        
        return output_path


class PerformanceTestSuite:
    """Test suite for performance validation."""
    
    def __init__(self):
        self.temp_files = []
        self.file_manager = FileManager()
    
    def cleanup(self):
        """Clean up temporary test files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
        self.temp_files.clear()
    
    def create_test_video(self, **kwargs) -> str:
        """Create a test video and track it for cleanup."""
        video_path = VideoGenerator.create_test_video(**kwargs)
        self.temp_files.append(video_path)
        return video_path


@pytest.fixture
def performance_suite():
    """Fixture providing performance test suite with cleanup."""
    suite = PerformanceTestSuite()
    yield suite
    suite.cleanup()


@pytest.fixture
def optimized_config():
    """Fixture providing optimized video processor configuration."""
    return VideoProcessorConfig(
        max_frames_per_batch=20,
        progress_callback_interval=5,
        enable_frame_skipping=True,
        frame_skip_ratio=2,
        max_resolution=(1280, 720),
        memory_limit_mb=512,
        chunk_size_mb=50,
        temp_cleanup_interval=25,
        preserve_audio=False  # Disable for performance tests
    )


class TestVideoProcessingPerformance:
    """Test video processing performance with different video characteristics."""
    
    def test_small_video_performance(self, performance_suite, optimized_config):
        """Test performance with small video files."""
        # Create small test video (480p, 5 seconds)
        video_path = performance_suite.create_test_video(
            width=640, height=480, fps=30, duration_seconds=5
        )
        
        processor = VideoProcessor(config=optimized_config)
        job = ProcessingJob.create_new("test_small.mp4", video_path)
        
        # Measure processing time
        start_time = time.time()
        output_path = processor.process_video(job)
        processing_time = time.time() - start_time
        
        # Verify results
        assert os.path.exists(output_path)
        assert processing_time < 30  # Should complete within 30 seconds
        
        # Check processing stats
        stats = processor.get_processing_stats()
        assert stats['frames_processed'] > 0
        assert stats['peak_memory_mb'] < 200  # Should use less than 200MB
        
        performance_suite.temp_files.append(output_path)
    
    def test_medium_video_performance(self, performance_suite, optimized_config):
        """Test performance with medium video files."""
        # Create medium test video (720p, 30 seconds)
        video_path = performance_suite.create_test_video(
            width=1280, height=720, fps=30, duration_seconds=30
        )
        
        processor = VideoProcessor(config=optimized_config)
        job = ProcessingJob.create_new("test_medium.mp4", video_path)
        
        # Track progress
        progress_updates = []
        def progress_callback(progress, message):
            progress_updates.append((progress, message, time.time()))
        
        # Measure processing time
        start_time = time.time()
        output_path = processor.process_video(job, progress_callback)
        processing_time = time.time() - start_time
        
        # Verify results
        assert os.path.exists(output_path)
        assert processing_time < 120  # Should complete within 2 minutes
        
        # Check progress reporting
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%
        
        # Check processing stats
        stats = processor.get_processing_stats()
        assert stats['frames_processed'] > 0
        assert stats['avg_frame_time'] > 0
        
        performance_suite.temp_files.append(output_path)
    
    def test_large_video_performance(self, performance_suite, optimized_config):
        """Test performance with large video files."""
        # Create large test video (1080p, 60 seconds)
        video_path = performance_suite.create_test_video(
            width=1920, height=1080, fps=30, duration_seconds=60
        )
        
        # Enable optimizations for large videos
        optimized_config.enable_frame_skipping = True
        optimized_config.max_resolution = (1280, 720)  # Downscale
        
        processor = VideoProcessor(config=optimized_config)
        job = ProcessingJob.create_new("test_large.mp4", video_path)
        
        # Measure processing time and memory
        start_time = time.time()
        peak_memory = 0
        
        def progress_callback(progress, message):
            nonlocal peak_memory
            current_memory = processor._get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        
        output_path = processor.process_video(job, progress_callback)
        processing_time = time.time() - start_time
        
        # Verify results
        assert os.path.exists(output_path)
        assert processing_time < 300  # Should complete within 5 minutes
        
        # Check memory usage
        assert peak_memory < optimized_config.memory_limit_mb * 1.5  # Allow some overhead
        
        # Check that frame skipping was used
        stats = processor.get_processing_stats()
        assert stats['frames_skipped'] > 0
        
        performance_suite.temp_files.append(output_path)
    
    def test_high_resolution_video_performance(self, performance_suite, optimized_config):
        """Test performance with high resolution videos."""
        # Create 4K test video (short duration)
        video_path = performance_suite.create_test_video(
            width=3840, height=2160, fps=24, duration_seconds=10
        )
        
        # Configure for high resolution processing
        optimized_config.max_resolution = (1920, 1080)  # Downscale 4K to 1080p
        optimized_config.max_frames_per_batch = 10  # Smaller batches for high res
        
        processor = VideoProcessor(config=optimized_config)
        job = ProcessingJob.create_new("test_4k.mp4", video_path)
        
        start_time = time.time()
        output_path = processor.process_video(job)
        processing_time = time.time() - start_time
        
        # Verify results
        assert os.path.exists(output_path)
        assert processing_time < 180  # Should complete within 3 minutes
        
        # Verify resolution was downscaled
        cap = cv2.VideoCapture(output_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        assert width <= 1920 and height <= 1080
        
        performance_suite.temp_files.append(output_path)
    
    def test_long_duration_video_performance(self, performance_suite, optimized_config):
        """Test performance with long duration videos."""
        # Create long test video (lower resolution, 5 minutes)
        video_path = performance_suite.create_test_video(
            width=854, height=480, fps=24, duration_seconds=300  # 5 minutes
        )
        
        # Configure for long video processing
        optimized_config.enable_frame_skipping = True
        optimized_config.frame_skip_ratio = 3  # Process every 3rd frame
        optimized_config.temp_cleanup_interval = 20
        
        processor = VideoProcessor(config=optimized_config)
        job = ProcessingJob.create_new("test_long.mp4", video_path)
        
        # Track memory usage over time
        memory_samples = []
        def progress_callback(progress, message):
            memory_samples.append(processor._get_memory_usage())
        
        start_time = time.time()
        output_path = processor.process_video(job, progress_callback)
        processing_time = time.time() - start_time
        
        # Verify results
        assert os.path.exists(output_path)
        assert processing_time < 600  # Should complete within 10 minutes
        
        # Check memory stability (shouldn't grow continuously)
        if len(memory_samples) > 10:
            early_avg = np.mean(memory_samples[:5])
            late_avg = np.mean(memory_samples[-5:])
            memory_growth = late_avg - early_avg
            assert memory_growth < 100  # Memory shouldn't grow by more than 100MB
        
        performance_suite.temp_files.append(output_path)


class TestBatchProcessingPerformance:
    """Test batch processing performance with multiple concurrent jobs."""
    
    def test_small_batch_performance(self, performance_suite):
        """Test performance with small batch of jobs."""
        # Create multiple small test videos
        jobs = []
        for i in range(3):
            video_path = performance_suite.create_test_video(
                width=640, height=480, fps=30, duration_seconds=5
            )
            job = ProcessingJob.create_new(f"batch_test_{i}.mp4", video_path)
            jobs.append(job)
        
        # Configure batch processor
        batch_config = BatchProcessorConfig(
            max_concurrent_jobs=2,
            max_worker_threads=3
        )
        
        processor_config = VideoProcessorConfig(
            max_frames_per_batch=15,
            preserve_audio=False
        )
        
        batch_processor = BatchProcessor(
            config=batch_config,
            video_processor_config=processor_config
        )
        
        # Process batch
        start_time = time.time()
        results = batch_processor.process_jobs_batch(jobs)
        batch_time = time.time() - start_time
        
        # Verify results
        assert results['batch_summary']['total_jobs'] == 3
        assert results['batch_summary']['successful_jobs'] == 3
        assert results['batch_summary']['failed_jobs'] == 0
        assert batch_time < 60  # Should complete within 1 minute
        
        # Verify all output files exist
        for job_id, result in results['job_results'].items():
            assert result['success']
            assert os.path.exists(result['output_path'])
            performance_suite.temp_files.append(result['output_path'])
    
    def test_concurrent_processing_efficiency(self, performance_suite):
        """Test that concurrent processing is more efficient than sequential."""
        # Create test videos
        jobs = []
        for i in range(4):
            video_path = performance_suite.create_test_video(
                width=854, height=480, fps=30, duration_seconds=10
            )
            job = ProcessingJob.create_new(f"concurrent_test_{i}.mp4", video_path)
            jobs.append(job)
        
        processor_config = VideoProcessorConfig(preserve_audio=False)
        
        # Test sequential processing (max_concurrent_jobs=1)
        sequential_config = BatchProcessorConfig(max_concurrent_jobs=1)
        sequential_processor = BatchProcessor(
            config=sequential_config,
            video_processor_config=processor_config
        )
        
        start_time = time.time()
        sequential_results = sequential_processor.process_jobs_batch(jobs[:2])
        sequential_time = time.time() - start_time
        
        # Test concurrent processing (max_concurrent_jobs=2)
        concurrent_config = BatchProcessorConfig(max_concurrent_jobs=2)
        concurrent_processor = BatchProcessor(
            config=concurrent_config,
            video_processor_config=processor_config
        )
        
        start_time = time.time()
        concurrent_results = concurrent_processor.process_jobs_batch(jobs[2:])
        concurrent_time = time.time() - start_time
        
        # Verify concurrent processing is more efficient
        # (allowing some overhead for thread management)
        efficiency_ratio = sequential_time / concurrent_time
        assert efficiency_ratio > 1.2  # At least 20% improvement
        
        # Clean up output files
        for results in [sequential_results, concurrent_results]:
            for result in results['job_results'].values():
                if result['success']:
                    performance_suite.temp_files.append(result['output_path'])
    
    def test_memory_management_in_batch(self, performance_suite):
        """Test memory management during batch processing."""
        # Create multiple medium-sized videos
        jobs = []
        for i in range(5):
            video_path = performance_suite.create_test_video(
                width=1280, height=720, fps=30, duration_seconds=15
            )
            job = ProcessingJob.create_new(f"memory_test_{i}.mp4", video_path)
            jobs.append(job)
        
        # Configure with memory limits
        batch_config = BatchProcessorConfig(
            max_concurrent_jobs=2,
            memory_limit_per_job_mb=300
        )
        
        processor_config = VideoProcessorConfig(
            memory_limit_mb=300,
            max_frames_per_batch=10,
            preserve_audio=False
        )
        
        batch_processor = BatchProcessor(
            config=batch_config,
            video_processor_config=processor_config
        )
        
        # Track system resources
        initial_resources = batch_processor.get_system_resources()
        
        # Process batch
        results = batch_processor.process_jobs_batch(jobs)
        
        # Check final resources
        final_resources = batch_processor.get_system_resources()
        
        # Verify all jobs completed successfully
        assert results['batch_summary']['successful_jobs'] == 5
        
        # Verify memory didn't grow excessively
        if initial_resources and final_resources:
            memory_growth = (final_resources.get('memory_usage_percent', 0) - 
                           initial_resources.get('memory_usage_percent', 0))
            assert memory_growth < 20  # Memory usage shouldn't increase by more than 20%
        
        # Clean up output files
        for result in results['job_results'].values():
            if result['success']:
                performance_suite.temp_files.append(result['output_path'])


class TestPerformanceOptimizations:
    """Test specific performance optimization features."""
    
    def test_frame_skipping_optimization(self, performance_suite):
        """Test frame skipping optimization for large videos."""
        # Create test video
        video_path = performance_suite.create_test_video(
            width=1920, height=1080, fps=30, duration_seconds=30
        )
        
        # Test without frame skipping
        config_no_skip = VideoProcessorConfig(
            enable_frame_skipping=False,
            preserve_audio=False
        )
        
        processor_no_skip = VideoProcessor(config=config_no_skip)
        job_no_skip = ProcessingJob.create_new("no_skip.mp4", video_path)
        
        start_time = time.time()
        output_no_skip = processor_no_skip.process_video(job_no_skip)
        time_no_skip = time.time() - start_time
        
        # Test with frame skipping
        config_skip = VideoProcessorConfig(
            enable_frame_skipping=True,
            frame_skip_ratio=2,
            preserve_audio=False
        )
        
        processor_skip = VideoProcessor(config=config_skip)
        job_skip = ProcessingJob.create_new("skip.mp4", video_path)
        
        start_time = time.time()
        output_skip = processor_skip.process_video(job_skip)
        time_skip = time.time() - start_time
        
        # Verify frame skipping improves performance
        assert time_skip < time_no_skip * 0.8  # At least 20% improvement
        
        # Verify frames were actually skipped
        stats_skip = processor_skip.get_processing_stats()
        assert stats_skip['frames_skipped'] > 0
        
        performance_suite.temp_files.extend([output_no_skip, output_skip])
    
    def test_resolution_downscaling_optimization(self, performance_suite):
        """Test resolution downscaling optimization."""
        # Create 4K test video
        video_path = performance_suite.create_test_video(
            width=3840, height=2160, fps=24, duration_seconds=10
        )
        
        # Test without downscaling
        config_full_res = VideoProcessorConfig(
            max_resolution=None,
            preserve_audio=False
        )
        
        processor_full = VideoProcessor(config=config_full_res)
        job_full = ProcessingJob.create_new("full_res.mp4", video_path)
        
        start_time = time.time()
        output_full = processor_full.process_video(job_full)
        time_full = time.time() - start_time
        
        # Test with downscaling
        config_downscale = VideoProcessorConfig(
            max_resolution=(1920, 1080),
            preserve_audio=False
        )
        
        processor_downscale = VideoProcessor(config=config_downscale)
        job_downscale = ProcessingJob.create_new("downscale.mp4", video_path)
        
        start_time = time.time()
        output_downscale = processor_downscale.process_video(job_downscale)
        time_downscale = time.time() - start_time
        
        # Verify downscaling improves performance
        assert time_downscale < time_full * 0.7  # At least 30% improvement
        
        # Verify output resolution
        cap = cv2.VideoCapture(output_downscale)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        assert width <= 1920 and height <= 1080
        
        performance_suite.temp_files.extend([output_full, output_downscale])
    
    def test_batch_processing_optimization(self, performance_suite):
        """Test batch processing vs sequential frame processing."""
        # Create test video
        video_path = performance_suite.create_test_video(
            width=1280, height=720, fps=30, duration_seconds=20
        )
        
        # Test sequential processing (batch size = 1)
        config_sequential = VideoProcessorConfig(
            max_frames_per_batch=1,
            preserve_audio=False
        )
        
        processor_sequential = VideoProcessor(config=config_sequential)
        job_sequential = ProcessingJob.create_new("sequential.mp4", video_path)
        
        start_time = time.time()
        output_sequential = processor_sequential.process_video(job_sequential)
        time_sequential = time.time() - start_time
        
        # Test batch processing (batch size = 20)
        config_batch = VideoProcessorConfig(
            max_frames_per_batch=20,
            preserve_audio=False
        )
        
        processor_batch = VideoProcessor(config=config_batch)
        job_batch = ProcessingJob.create_new("batch.mp4", video_path)
        
        start_time = time.time()
        output_batch = processor_batch.process_video(job_batch)
        time_batch = time.time() - start_time
        
        # Verify batch processing is more efficient
        # (Note: improvement may be modest for this test case)
        assert time_batch <= time_sequential * 1.1  # Should be at least as fast
        
        performance_suite.temp_files.extend([output_sequential, output_batch])


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks for different scenarios."""
    
    def test_performance_benchmark_suite(self, performance_suite):
        """Run comprehensive performance benchmark suite."""
        benchmark_results = {}
        
        # Define test scenarios
        scenarios = [
            ("small_480p_5s", {"width": 640, "height": 480, "fps": 30, "duration_seconds": 5}),
            ("medium_720p_30s", {"width": 1280, "height": 720, "fps": 30, "duration_seconds": 30}),
            ("large_1080p_60s", {"width": 1920, "height": 1080, "fps": 30, "duration_seconds": 60}),
            ("hd_720p_300s", {"width": 1280, "height": 720, "fps": 24, "duration_seconds": 300}),
        ]
        
        for scenario_name, video_params in scenarios:
            # Create test video
            video_path = performance_suite.create_test_video(**video_params)
            
            # Configure processor for scenario
            config = VideoProcessorConfig(
                enable_frame_skipping=video_params["duration_seconds"] > 60,
                max_resolution=(1920, 1080) if video_params["width"] > 1920 else None,
                preserve_audio=False
            )
            
            processor = VideoProcessor(config=config)
            job = ProcessingJob.create_new(f"{scenario_name}.mp4", video_path)
            
            # Run benchmark
            start_time = time.time()
            output_path = processor.process_video(job)
            processing_time = time.time() - start_time
            
            # Collect metrics
            stats = processor.get_processing_stats()
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            
            benchmark_results[scenario_name] = {
                'processing_time': processing_time,
                'input_file_size_mb': file_size_mb,
                'frames_processed': stats['frames_processed'],
                'frames_skipped': stats['frames_skipped'],
                'peak_memory_mb': stats['peak_memory_mb'],
                'avg_frame_time': stats['avg_frame_time'],
                'throughput_fps': stats['frames_processed'] / processing_time if processing_time > 0 else 0
            }
            
            performance_suite.temp_files.append(output_path)
        
        # Validate benchmark results
        for scenario, results in benchmark_results.items():
            assert results['processing_time'] > 0
            assert results['frames_processed'] > 0
            assert results['peak_memory_mb'] > 0
            assert results['throughput_fps'] > 0
            
            # Log results for analysis
            print(f"\n{scenario} Results:")
            print(f"  Processing Time: {results['processing_time']:.2f}s")
            print(f"  Throughput: {results['throughput_fps']:.2f} FPS")
            print(f"  Peak Memory: {results['peak_memory_mb']}MB")
            print(f"  Frames Processed: {results['frames_processed']}")
            print(f"  Frames Skipped: {results['frames_skipped']}")
        
        # Performance assertions
        small_scenario = benchmark_results["small_480p_5s"]
        assert small_scenario['processing_time'] < 30
        assert small_scenario['peak_memory_mb'] < 300
        
        medium_scenario = benchmark_results["medium_720p_30s"]
        assert medium_scenario['processing_time'] < 120
        assert medium_scenario['throughput_fps'] > 5
        
        large_scenario = benchmark_results["large_1080p_60s"]
        assert large_scenario['processing_time'] < 300
        assert large_scenario['frames_skipped'] > 0  # Should use frame skipping