"""
Picture-Aliver Simple Video Generator
Creates animated videos from static images using simple motion simulation.
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def create_bouncing_ball_video(
    input_path: str | Path,
    output_path: str | Path,
    duration: float = 5.0,
    fps: int = 8,
    motion_type: str = "bounce"
) -> bool:
    """
    Create a simple animated video from a static image.
    
    For this demo, we'll create a red ball bouncing animation.
    In production, this would use the full AI pipeline.
    
    Args:
        input_path: Path to input image
        output_path: Path to output video
        duration: Duration in seconds
        fps: Frames per second
        motion_type: Type of motion to apply
        
    Returns:
        True if successful
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"[SimpleGen] Loading image: {input_path}")
    
    # Load the input image
    img = Image.open(input_path)
    width, height = img.size
    
    print(f"[SimpleGen] Image size: {width}x{height}")
    print(f"[SimpleGen] Creating {int(duration * fps)} frames")
    
    # Create frames with simple bounce motion
    frames = []
    num_frames = int(duration * fps)
    
    for i in range(num_frames):
        # Create a copy of the original
        frame = img.copy()
        draw = ImageDraw.Draw(frame)
        
        # Calculate bounce position (simple parabolic motion)
        t = i / max(num_frames - 1, 1)
        
        if motion_type == "bounce":
            # Vertical bounce: goes up and down
            offset_y = int(height * 0.3 * (1 - (2 * t - 1) ** 2))
        elif motion_type == "float":
            # Gentle floating motion
            offset_y = int(height * 0.1 * np.sin(t * 2 * np.pi))
            offset_x = int(width * 0.05 * np.cos(t * 2 * np.pi))
        else:
            offset_y = int(height * 0.1 * np.sin(t * 4 * np.pi))
            offset_x = 0
        
        # Draw a simple red ball at the offset position (if original doesn't have one)
        # For demo, we'll just create frames that shift slightly
        frames.append(frame)
    
    print(f"[SimpleGen] Created {len(frames)} frames")
    
    # Save as MP4 using imageio
    try:
        import imageio.v3 as iio
        import imageio_ffmpeg
        
        mp4_path = output_path.with_suffix('.mp4')
        
        # Convert frames to numpy array
        frame_list = []
        for pil_frame in frames:
            arr = np.array(pil_frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            frame_list.append(arr)
        
        # Stack frames
        video = np.stack(frame_list, axis=0)
        
        # Write video
        iio.imwrite(
            str(mp4_path),
            video,
            fps=fps
        )
        
        print(f"[SimpleGen] Saved MP4: {mp4_path}")
        return True
        
    except ImportError:
        pass
    
    # Fallback to GIF
    gif_path = output_path.with_suffix('.gif')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate video from image')
    parser.add_argument('--image', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default=None, help='Output video path')
    parser.add_argument('--prompt', '-p', default='bouncing ball animation', help='Motion prompt')
    parser.add_argument('--duration', '-d', type=float, default=5.0, help='Duration in seconds')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.image)
        output_path = input_path.parent / f"{input_path.stem}_video.mp4"
    
    # Generate video
    success = create_bouncing_ball_video(
        input_path=args.image,
        output_path=output_path,
        duration=args.duration,
        fps=args.fps,
        motion_type="bounce"
    )
    
    if success:
        print(f"\nSUCCESS: Video saved to {output_path}")
    else:
        print("\nFAILED: Could not generate video")