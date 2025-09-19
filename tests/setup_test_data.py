#!/usr/bin/env python3
"""
Test data setup script for Golf Video Anonymizer.
Creates sample golf swing videos and test data for comprehensive testing.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys


def create_golf_swing_video(output_path: Path, duration_seconds: int = 3, fps: int = 30):
    """Create a realistic golf swing video with a person."""
    print(f"Creating golf swing video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create background (golf course green)
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Forest green
        
        # Add sky
        cv2.rectangle(frame, (0, 0), (width, height//3), (135, 206, 235), -1)  # Sky blue
        
        # Add ground/tee area
        cv2.rectangle(frame, (0, height*2//3), (width, height), (101, 67, 33), -1)  # Brown
        
        # Golf swing animation phases
        swing_progress = frame_num / total_frames
        
        # Person position (center of frame)
        person_x = width // 2
        person_y = height * 2 // 3
        
        # Head (face that should be detected)
        head_size = 25
        head_x = person_x - head_size // 2
        head_y = person_y - 120
        
        # Draw head/face
        cv2.circle(frame, (person_x, head_y + head_size//2), head_size, (220, 180, 140), -1)
        
        # Add facial features for better detection
        # Eyes
        eye_y = head_y + head_size//3
        cv2.circle(frame, (person_x - 8, eye_y), 2, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (person_x + 8, eye_y), 2, (0, 0, 0), -1)  # Right eye
        
        # Nose
        cv2.circle(frame, (person_x, eye_y + 8), 1, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(frame, (person_x, eye_y + 15), (4, 2), 0, 0, 180, (0, 0, 0), 1)
        
        # Body
        body_width = 40
        body_height = 80
        cv2.rectangle(frame, 
                     (person_x - body_width//2, person_y - 100), 
                     (person_x + body_width//2, person_y - 20), 
                     (100, 100, 200), -1)  # Blue shirt
        
        # Arms and golf club (animated based on swing progress)
        if swing_progress < 0.3:  # Backswing
            club_angle = -60 + (swing_progress * 120)  # -60 to +60 degrees
        elif swing_progress < 0.7:  # Downswing
            club_angle = 60 - ((swing_progress - 0.3) * 200)  # 60 to -140 degrees
        else:  # Follow through
            club_angle = -140 + ((swing_progress - 0.7) * 100)  # -140 to -40 degrees
        
        # Draw golf club
        club_length = 80
        club_end_x = person_x + int(club_length * np.cos(np.radians(club_angle)))
        club_end_y = person_y - 60 + int(club_length * np.sin(np.radians(club_angle)))
        
        cv2.line(frame, (person_x, person_y - 60), (club_end_x, club_end_y), (139, 69, 19), 3)
        
        # Arms
        arm_x = person_x + int(30 * np.cos(np.radians(club_angle + 20)))
        arm_y = person_y - 80 + int(30 * np.sin(np.radians(club_angle + 20)))
        cv2.line(frame, (person_x, person_y - 80), (arm_x, arm_y), (220, 180, 140), 5)
        
        # Legs
        cv2.line(frame, (person_x - 10, person_y - 20), (person_x - 15, person_y + 40), (0, 0, 139), 8)
        cv2.line(frame, (person_x + 10, person_y - 20), (person_x + 15, person_y + 40), (0, 0, 139), 8)
        
        # Add golf ball (small white circle)
        ball_x = person_x + 30
        ball_y = person_y
        if swing_progress > 0.5:  # Ball moves after impact
            ball_x += int((swing_progress - 0.5) * 200)
            ball_y -= int((swing_progress - 0.5) * 100)
        
        if ball_x < width and ball_y > 0:
            cv2.circle(frame, (ball_x, ball_y), 3, (255, 255, 255), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created golf swing video with {total_frames} frames")


def create_multiple_faces_video(output_path: Path, duration_seconds: int = 4, fps: int = 30):
    """Create a video with multiple people (faces) for testing."""
    print(f"Creating multiple faces video: {output_path}")
    
    width, height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Background
        frame = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
        
        # Person 1 - Golfer (main subject)
        person1_x = width // 3
        person1_y = height * 2 // 3
        
        # Face 1
        face1_size = 30
        cv2.circle(frame, (person1_x, person1_y - 100), face1_size, (220, 180, 140), -1)
        # Features
        cv2.circle(frame, (person1_x - 10, person1_y - 110), 2, (0, 0, 0), -1)
        cv2.circle(frame, (person1_x + 10, person1_y - 110), 2, (0, 0, 0), -1)
        cv2.circle(frame, (person1_x, person1_y - 95), 1, (0, 0, 0), -1)
        
        # Person 2 - Instructor/Observer (moving)
        person2_x = width * 2 // 3 + int(20 * np.sin(frame_num * 0.1))
        person2_y = height * 2 // 3
        
        # Face 2
        face2_size = 25
        cv2.circle(frame, (person2_x, person2_y - 90), face2_size, (200, 160, 120), -1)
        # Features
        cv2.circle(frame, (person2_x - 8, person2_y - 98), 2, (0, 0, 0), -1)
        cv2.circle(frame, (person2_x + 8, person2_y - 98), 2, (0, 0, 0), -1)
        cv2.circle(frame, (person2_x, person2_y - 85), 1, (0, 0, 0), -1)
        
        # Person 3 - Background person (smaller, further away)
        if frame_num > total_frames // 2:  # Appears halfway through
            person3_x = width // 6
            person3_y = height // 2
            
            # Face 3 (smaller)
            face3_size = 15
            cv2.circle(frame, (person3_x, person3_y), face3_size, (180, 140, 100), -1)
            # Features
            cv2.circle(frame, (person3_x - 5, person3_y - 3), 1, (0, 0, 0), -1)
            cv2.circle(frame, (person3_x + 5, person3_y - 3), 1, (0, 0, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created multiple faces video with {total_frames} frames")


def create_no_faces_video(output_path: Path, duration_seconds: int = 2, fps: int = 30):
    """Create a video with no faces for testing."""
    print(f"Creating no faces video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create landscape scene
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Green
        
        # Sky
        cv2.rectangle(frame, (0, 0), (width, height//2), (135, 206, 235), -1)
        
        # Add some geometric shapes (no faces)
        # Golf flag
        flag_x = width * 3 // 4
        cv2.line(frame, (flag_x, height//4), (flag_x, height*3//4), (139, 69, 19), 3)
        cv2.rectangle(frame, (flag_x, height//4), (flag_x + 30, height//4 + 20), (255, 0, 0), -1)
        
        # Golf cart (moving)
        cart_x = 50 + (frame_num * 2) % (width - 100)
        cart_y = height * 2 // 3
        cv2.rectangle(frame, (cart_x, cart_y), (cart_x + 60, cart_y + 30), (255, 255, 255), -1)
        cv2.circle(frame, (cart_x + 15, cart_y + 35), 8, (0, 0, 0), -1)  # Wheel
        cv2.circle(frame, (cart_x + 45, cart_y + 35), 8, (0, 0, 0), -1)  # Wheel
        
        writer.write(frame)
    
    writer.release()
    print(f"Created no faces video with {total_frames} frames")


def create_challenging_conditions_video(output_path: Path, duration_seconds: int = 5, fps: int = 30):
    """Create a video with challenging lighting and conditions."""
    print(f"Creating challenging conditions video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        progress = frame_num / total_frames
        
        # Varying lighting conditions
        if progress < 0.25:  # Bright/overexposed
            base_brightness = 200
        elif progress < 0.5:  # Normal
            base_brightness = 128
        elif progress < 0.75:  # Dim
            base_brightness = 60
        else:  # Backlit
            base_brightness = 180
        
        # Create frame with varying brightness
        frame = np.full((height, width, 3), base_brightness, dtype=np.uint8)
        
        # Person with face
        person_x = width // 2 + int(50 * np.sin(frame_num * 0.05))  # Slight movement
        person_y = height * 2 // 3
        
        # Adjust face brightness based on lighting condition
        if progress >= 0.75:  # Backlit - face should be darker
            face_brightness = max(40, base_brightness - 100)
        else:
            face_brightness = min(255, base_brightness + 20)
        
        # Face
        face_size = 28
        face_color = (face_brightness, face_brightness - 20, face_brightness - 40)
        cv2.circle(frame, (person_x, person_y - 100), face_size, face_color, -1)
        
        # Features (adjust contrast based on lighting)
        feature_color = (0, 0, 0) if face_brightness > 100 else (255, 255, 255)
        cv2.circle(frame, (person_x - 9, person_y - 108), 2, feature_color, -1)
        cv2.circle(frame, (person_x + 9, person_y - 108), 2, feature_color, -1)
        cv2.circle(frame, (person_x, person_y - 95), 1, feature_color, -1)
        
        # Add noise for challenging conditions
        if progress > 0.5:  # Add noise in latter half
            noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created challenging conditions video with {total_frames} frames")


def create_high_resolution_video(output_path: Path, duration_seconds: int = 2, fps: int = 30):
    """Create a high resolution video for performance testing."""
    print(f"Creating high resolution video: {output_path}")
    
    width, height = 1920, 1080  # Full HD
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for frame_num in range(total_frames):
        # Create high-res background
        frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
        
        # Large face for HD video
        person_x = width // 2
        person_y = height // 2
        
        face_size = 80  # Larger face for HD
        cv2.circle(frame, (person_x, person_y), face_size, (220, 180, 140), -1)
        
        # Detailed features
        cv2.circle(frame, (person_x - 25, person_y - 15), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (person_x + 25, person_y - 15), 5, (0, 0, 0), -1)  # Right eye
        cv2.circle(frame, (person_x, person_y + 10), 3, (0, 0, 0), -1)  # Nose
        cv2.ellipse(frame, (person_x, person_y + 30), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        writer.write(frame)
    
    writer.release()
    print(f"Created high resolution video with {total_frames} frames")


def main():
    """Main function to create all test videos."""
    parser = argparse.ArgumentParser(description="Create test data for Golf Video Anonymizer")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("tests/test_data"),
                       help="Output directory for test videos")
    parser.add_argument("--all", action="store_true", help="Create all test videos")
    parser.add_argument("--golf-swing", action="store_true", help="Create golf swing video")
    parser.add_argument("--multiple-faces", action="store_true", help="Create multiple faces video")
    parser.add_argument("--no-faces", action="store_true", help="Create no faces video")
    parser.add_argument("--challenging", action="store_true", help="Create challenging conditions video")
    parser.add_argument("--high-res", action="store_true", help="Create high resolution video")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating test videos in: {args.output_dir}")
    
    # Check if OpenCV is available
    try:
        cv2_version = cv2.__version__
        print(f"Using OpenCV version: {cv2_version}")
    except ImportError:
        print("Error: OpenCV not available. Please install opencv-python.")
        sys.exit(1)
    
    created_videos = []
    
    if args.all or args.golf_swing:
        video_path = args.output_dir / "sample_golf_swing.mp4"
        create_golf_swing_video(video_path, duration_seconds=3)
        created_videos.append(video_path)
    
    if args.all or args.multiple_faces:
        video_path = args.output_dir / "multiple_faces_video.mp4"
        create_multiple_faces_video(video_path, duration_seconds=4)
        created_videos.append(video_path)
    
    if args.all or args.no_faces:
        video_path = args.output_dir / "no_faces_video.mp4"
        create_no_faces_video(video_path, duration_seconds=2)
        created_videos.append(video_path)
    
    if args.all or args.challenging:
        video_path = args.output_dir / "challenging_conditions.mp4"
        create_challenging_conditions_video(video_path, duration_seconds=5)
        created_videos.append(video_path)
    
    if args.all or args.high_res:
        video_path = args.output_dir / "high_resolution_video.mp4"
        create_high_resolution_video(video_path, duration_seconds=2)
        created_videos.append(video_path)
    
    if not any([args.all, args.golf_swing, args.multiple_faces, args.no_faces, 
                args.challenging, args.high_res]):
        print("No video type specified. Use --all or specify individual types.")
        parser.print_help()
        return
    
    print(f"\nSuccessfully created {len(created_videos)} test videos:")
    for video_path in created_videos:
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {video_path.name} ({file_size:.1f} MB)")
    
    print(f"\nTest data setup complete! Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()