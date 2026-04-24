"""
Picture-Aliver CLI Entry Point

Usage:
    python main.py --image <path> --prompt "<text>"

Example:
    python main.py --image test_image.png --prompt "gentle wave animation"
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
except ImportError:
    torch = None


def load_config_yaml(config_path: str = "configs/default.yaml") -> dict | None:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        config_file = Path("src/picture_aliver/config.yaml")
    
    if not config_file.exists():
        print(f"[Warning] Config file not found: {config_path}")
        return None
    
    if yaml is None:
        print("[Warning] PyYAML not installed, using default config")
        return None
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Warning] Failed to load config: {e}")
        return None


def config_to_pipeline_kwargs(config_dict: dict | None) -> dict:
    """Convert YAML config dict to pipeline kwargs."""
    if config_dict is None:
        return {}
    
    pipeline_cfg = config_dict.get('pipeline', {})
    video_cfg = config_dict.get('video', {})
    gen_cfg = config_dict.get('generation', {})
    motion_cfg = config_dict.get('motion', {})
    output_cfg = config_dict.get('output', {})
    gpu_cfg = config_dict.get('gpu', {})
    
    device = gpu_cfg.get('device', None)
    if device == "auto":
        device = None
    
    return {
        'duration_seconds': video_cfg.get('duration_seconds', 3.0),
        'fps': video_cfg.get('fps', 8),
        'width': video_cfg.get('resolution', {}).get('width', 512),
        'height': video_cfg.get('resolution', {}).get('height', 512),
        'guidance_scale': gen_cfg.get('guidance_scale', 7.5),
        'num_inference_steps': gen_cfg.get('num_inference_steps', 25),
        'motion_strength': motion_cfg.get('strength', 0.8),
        'motion_mode': motion_cfg.get('mode', 'auto'),
        'motion_prompt': motion_cfg.get('prompt'),
        'quality': output_cfg.get('quality', 'medium'),
        'output_format': output_cfg.get('format', 'mp4'),
        'enable_stabilization': pipeline_cfg.get('enable_stabilization', True),
        'enable_interpolation': pipeline_cfg.get('enable_interpolation', False),
        'enable_quality_check': pipeline_cfg.get('enable_quality_check', True),
        'quality_max_retries': pipeline_cfg.get('quality_max_retries', 2),
        'device': device,
        'model_dir': None,
    }


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="Picture-Aliver",
        description="AI Image-to-Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image photo.jpg --prompt "cinematic animation"
  python main.py -i image.png -p "gentle motion"
  python main.py --image test.png --prompt "wind blowing" --duration 5
        """
    )
    
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Input image path"
    )
    
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Text prompt for video generation"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output video path (default: auto-generated)"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="Path to config.yaml file"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (overrides config)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("PICTURE-ALIVER - Image to Video Generation")
        print("=" * 60)
        print()
        
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[Error] Input image not found: {image_path}")
            return 1
        
        print(f"[Config] Loading: {args.config}")
        config_dict = load_config_yaml(args.config)
        
        print(f"[Config] Loading pipeline configuration...")
        kwargs = config_to_pipeline_kwargs(config_dict)
        
        if args.duration is not None:
            kwargs['duration_seconds'] = args.duration
        if args.fps is not None:
            kwargs['fps'] = args.fps
        if args.device is not None:
            kwargs['device'] = args.device
        
        print(f"[Config] Duration: {kwargs.get('duration_seconds', 3.0)}s @ {kwargs.get('fps', 8)}fps")
        print(f"[Config] Motion mode: {kwargs.get('motion_mode', 'auto')}")
        print()
        
        if torch is None:
            print("[Error] PyTorch is not installed")
            return 1
        
        device_name = kwargs.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[System] Using device: {device_name}")
        if torch.cuda.is_available():
            print(f"[System] GPU: {torch.cuda.get_device_name(0)}")
        print()
        
        print(f"[Stage 1/9] Loading image...")
        print(f"  Source: {image_path}")
        
        output_path = args.output
        if output_path is None:
            output_path = Path("outputs") / f"{image_path.stem}_video.mp4"
        
        print()
        print("[Stage 2/9] Running pipeline...")
        print()
        
        from src.picture_aliver.main import PipelineConfig, run_pipeline
        
        config = PipelineConfig(**kwargs)
        
        result = run_pipeline(
            image_path=image_path,
            prompt=args.prompt,
            config=config,
            output_path=output_path
        )
        
        if result.success:
            print()
            print("=" * 60)
            print("SUCCESS")
            print("=" * 60)
            print(f"Output video: {result.output_path}")
            print(f"Duration: {result.duration_seconds}s")
            print(f"Frames: {result.num_frames}")
            print(f"Processing time: {result.processing_time:.2f}s")
            if result.quality_score:
                print(f"Quality score: {result.quality_score:.2f}")
            print("=" * 60)
            return 0
        else:
            print()
            print("=" * 60)
            print("FAILED")
            print("=" * 60)
            for error in result.errors:
                print(f"  Error: {error}")
            print("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        print("\n[Interrupted] Processing cancelled by user")
        return 130
    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(f"Exception: {e}")
        if args.verbose:
            traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())