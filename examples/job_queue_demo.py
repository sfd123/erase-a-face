#!/usr/bin/env python3
"""
Demo script showing JobQueue functionality.

This script demonstrates how to use the JobQueue class for managing
video processing jobs with Redis.
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.job_queue import JobQueue, JobQueueError
from models.processing_job import ProcessingJob, JobStatus


def main():
    """Demonstrate JobQueue functionality."""
    print("JobQueue Demo")
    print("=" * 50)
    
    try:
        # Initialize job queue
        print("1. Connecting to Redis...")
        queue = JobQueue("redis://localhost:6379/0")
        
        if not queue.health_check():
            print("❌ Redis connection failed!")
            return
        print("✅ Connected to Redis successfully")
        
        # Create a sample job
        print("\n2. Creating a sample job...")
        job = ProcessingJob.create_new("sample_golf_video.mp4", "/tmp/sample_golf_video.mp4")
        print(f"✅ Created job: {job.job_id}")
        
        # Enqueue the job
        print("\n3. Enqueuing job...")
        success = queue.enqueue_job(job)
        if success:
            print("✅ Job enqueued successfully")
        else:
            print("❌ Failed to enqueue job")
            return
        
        # Get queue statistics
        print("\n4. Queue statistics:")
        stats = queue.get_queue_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Dequeue the job (simulate worker picking up job)
        print("\n5. Dequeuing job (simulating worker)...")
        dequeued_job = queue.dequeue_job(timeout=5)
        if dequeued_job:
            print(f"✅ Dequeued job: {dequeued_job.job_id}")
            
            # Update job status to processing
            print("\n6. Updating job status to PROCESSING...")
            queue.update_job_status(dequeued_job.job_id, JobStatus.PROCESSING)
            
            # Simulate processing time
            print("   Simulating processing...")
            time.sleep(2)
            
            # Complete the job
            print("\n7. Completing job...")
            queue.update_job_status(
                dequeued_job.job_id,
                JobStatus.COMPLETED,
                output_file_path="/tmp/processed_video.mp4",
                faces_detected=3
            )
            print("✅ Job completed successfully")
            
            # Check final job status
            final_job = queue.get_job_status(dequeued_job.job_id)
            if final_job:
                print(f"\n8. Final job status:")
                print(f"   Status: {final_job.status.value}")
                print(f"   Faces detected: {final_job.faces_detected}")
                print(f"   Output file: {final_job.output_file_path}")
                print(f"   Processing duration: {final_job.processing_duration:.2f}s")
        else:
            print("❌ No job dequeued (timeout)")
        
        # Demonstrate retry mechanism
        print("\n9. Demonstrating retry mechanism...")
        failed_job = ProcessingJob.create_new("failed_video.mp4", "/tmp/failed_video.mp4")
        queue.enqueue_job(failed_job)
        
        # Dequeue and fail the job
        retry_job = queue.dequeue_job(timeout=1)
        if retry_job:
            queue.update_job_status(
                retry_job.job_id,
                JobStatus.FAILED,
                error_message="Simulated processing failure"
            )
            print(f"✅ Job {retry_job.job_id} marked as failed")
            
            # Retry the failed job
            retry_success = queue.retry_failed_job(retry_job.job_id)
            if retry_success:
                print("✅ Job queued for retry")
                
                # Get retry job
                retry_job_dequeued = queue.get_next_retry_job(timeout=1)
                if retry_job_dequeued:
                    print(f"✅ Retrieved retry job: {retry_job_dequeued.job_id}")
                    # Complete the retry
                    queue.update_job_status(
                        retry_job_dequeued.job_id,
                        JobStatus.COMPLETED,
                        output_file_path="/tmp/retry_success.mp4"
                    )
                    print("✅ Retry job completed successfully")
        
        # Final statistics
        print("\n10. Final queue statistics:")
        final_stats = queue.get_queue_stats()
        for key, value in final_stats.items():
            print(f"    {key}: {value}")
        
        print("\n🎉 Demo completed successfully!")
        
    except JobQueueError as e:
        print(f"❌ JobQueue error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()