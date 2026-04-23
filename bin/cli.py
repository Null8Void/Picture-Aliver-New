#!/usr/bin/env python
"""Command-line interface for Image2Video AI system with unrestricted content support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.core.pipeline import Image2VideoPipeline, PipelineConfig
from src.core.device import get_torch_device
from src.core.model_registry import MODEL_REGISTRY, ContentRating, ModelCategory
from src.core.config_extension import GenerationMode, create_content_config
from src.utils.logger import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="Picture-Aliver",
        description="Convert images to videos using AI (Safe and Unrestricted modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe mode (default)
  %(prog)s -i photo.jpg -o video.mp4
  
  # Mature content mode
  %(prog)s -i photo.jpg -o video.mp4 --mode mature
  
  # Unrestricted/NSFW mode
  %(prog)s -i photo.jpg -o video.mp4 --mode nsfw --no-safety
  
  # Custom model selection
  %(prog)s -i photo.jpg -o video.mp4 --i2v-model "Open-SVD" --depth-model "MiDaS v3.1"
  
  # List available models
  %(prog)s --list-models
  %(prog)s --list-models --mode nsfw
        """
    )
    
    # Input/Output
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video path (default: input_stem_video.mp4)"
    )
    
    # Content Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["safe", "mature", "nsfw", "unrestricted"],
        default="safe",
        help="Content generation mode (default: safe)"
    )
    
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Disable all safety checks (equivalent to --mode nsfw)"
    )
    
    parser.add_argument(
        "--content-filter-strength",
        type=float,
        default=1.0,
        help="Content filter strength 0-1 (default: 1.0)"
    )
    
    # Prompt Settings
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation"
    )
    
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt"
    )
    
    # Video Settings
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of frames to generate (default: 24)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second (default: 8)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output resolution (default: 512)"
    )
    
    # Quality Settings
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="CFG guidance scale (default: 7.5)"
    )
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Number of denoising steps (default: 25)"
    )
    
    parser.add_argument(
        "--motion-strength",
        type=float,
        default=0.8,
        help="Motion strength 0-1 (default: 0.8)"
    )
    
    parser.add_argument(
        "--motion-mode",
        type=str,
        choices=["auto", "cinematic", "subtle", "environmental", "orbital", "zoom-in", "zoom-out", "pan-left", "pan-right"],
        default="cinematic",
        help="Motion style (default: cinematic)"
    )
    
    # Model Selection
    parser.add_argument(
        "--i2v-model",
        type=str,
        default=None,
        help="Image-to-video model override"
    )
    
    parser.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="Depth estimation model override"
    )
    
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default=None,
        help="Segmentation model override"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--list-i2v-models",
        action="store_true",
        help="List available I2V models"
    )
    
    parser.add_argument(
        "--show-vram",
        action="store_true",
        help="Show VRAM requirements for models"
    )
    
    # Technical
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
        help="Compute device"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["fp16", "int8", "none"],
        default="none",
        help="Model quantization (default: none)"
    )
    
    # Feature Flags
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth estimation"
    )
    
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Disable segmentation"
    )
    
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="Disable temporal consistency"
    )
    
    parser.add_argument(
        "--no-3d-effects",
        action="store_true",
        help="Disable depth-based effects"
    )
    
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Disable frame interpolation"
    )
    
    # Output Options
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (depth, segmentation, flow)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Picture-Aliver v1.0.0"
    )
    
    return parser


def list_models(rating_filter: Optional[str] = None, show_vram: bool = False):
    """List available models."""
    
    print("\n" + "=" * 70)
    print("PICTURE-ALIVER MODEL REGISTRY")
    print("=" * 70)
    
    ratings = ["safe", "mature", "nsfw"]
    if rating_filter:
        ratings = [rating_filter]
    
    for rating in ratings:
        content_rating = {
            "safe": ContentRating.SAFE,
            "mature": ContentRating.MATURE,
            "nsfw": ContentRating.NSFW,
        }.get(rating)
        
        if content_rating is None:
            continue
        
        models = MODEL_REGISTRY.get_by_category(ModelCategory.I2V, rating=content_rating)
        
        print(f"\n{'=' * 70}")
        print(f"  {rating.upper()} MODELS ({len(models)} image-to-video models)")
        print(f"{'=' * 70}")
        
        if not models:
            print("  No models available for this rating")
            continue
        
        for model in sorted(models, key=lambda m: m.vram_mb):
            print(f"\n  {model.name}")
            print(f"    ├─ Repo: {model.repo_id}")
            print(f"    ├─ VRAM: {model.vram_mb}MB")
            if model.resolution:
                print(f"    ├─ Resolution: {model.resolution[0]}x{model.resolution[1]}")
            if model.max_frames:
                print(f"    ├─ Max Frames: {model.max_frames}")
            if model.requires_base:
                print(f"    └─ Requires base model: Yes")
            else:
                print(f"    └─ Standalone: Yes")
            
            if show_vram and torch.cuda.is_available():
                vram_available = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                status = "✓ Can load" if model.vram_mb <= vram_available else "✗ Cannot load"
                print(f"      {status} ({vram_available:.0f}MB available)")
    
    print("\n" + "=" * 70)
    print("OTHER MODEL CATEGORIES")
    print("=" * 70)
    
    for category in [ModelCategory.DEPTH, ModelCategory.SEGMENTATION, ModelCategory.MOTION, ModelCategory.INTERPOLATION]:
        models = MODEL_REGISTRY.get_by_category(category)
        print(f"\n  {category.value.upper()}: {len(models)} models")
        for model in models[:5]:  # Show first 5
            print(f"    - {model.name} ({model.vram_mb}MB)")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")
    
    print("\n" + "=" * 70)
    print("\nUse --i2v-model, --depth-model, --segmentation-model to select models")
    print("Example: %(prog)s -i photo.jpg --i2v-model 'Open-SVD'")
    print("=" * 70 + "\n")


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    logger = setup_logger(
        name="Picture-Aliver",
        level=10 if parsed_args.verbose else 20
    )
    
    # Handle list commands
    if parsed_args.list_models:
        list_models(show_vram=parsed_args.show_vram)
        return 0
    
    if parsed_args.list_i2v_models:
        list_models(rating_filter="nsfw" if parsed_args.mode in ["mature", "nsfw", "unrestricted"] else None,
                   show_vram=True)
        return 0
    
    # Validate input
    input_path = Path(parsed_args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Determine output path
    if parsed_args.output:
        output_path = Path(parsed_args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_video.mp4"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine generation mode
    if parsed_args.no_safety:
        gen_mode = "nsfw"
        logger.warning("⚠️  Safety checks disabled - Unrestricted mode enabled")
    else:
        gen_mode = parsed_args.mode
    
    logger.info(f"Generation mode: {gen_mode.upper()}")
    
    # Create configuration
    content_config = create_content_config(
        mode=gen_mode,
        content_filter_strength=parsed_args.content_filter_strength
    )
    
    device = get_torch_device(parsed_args.device)
    
    config = PipelineConfig(
        enable_depth=not parsed_args.no_depth,
        enable_segmentation=not parsed_args.no_segmentation,
        enable_motion=True,
        enable_consistency=not parsed_args.no_consistency,
        use_3d_effects=not parsed_args.no_3d_effects,
        enable_interpolation=not parsed_args.no_interpolation,
        motion_mode=parsed_args.motion_mode,
        motion_strength=parsed_args.motion_strength,
        num_frames=parsed_args.num_frames,
        fps=parsed_args.fps,
        resolution=(parsed_args.resolution, parsed_args.resolution),
        guidance_scale=parsed_args.guidance_scale,
        num_inference_steps=parsed_args.num_inference_steps,
        verbose=True,
        content=content_config,
    )
    
    # Override models if specified
    if parsed_args.i2v_model:
        config.models.i2v_model = parsed_args.i2v_model
        logger.info(f"Using I2V model: {parsed_args.i2v_model}")
    if parsed_args.depth_model:
        config.models.depth_model = parsed_args.depth_model
        logger.info(f"Using depth model: {parsed_args.depth_model}")
    if parsed_args.segmentation_model:
        config.models.segmentation_model = parsed_args.segmentation_model
        logger.info(f"Using segmentation model: {parsed_args.segmentation_model}")
    
    # Create and run pipeline
    pipeline = Image2VideoPipeline(config=config, device=device)
    
    try:
        logger.info("=" * 50)
        logger.info("PICTURE-ALIVER - Image to Video Generation")
        logger.info("=" * 50)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Mode: {gen_mode.upper()}")
        logger.info(f"Frames: {parsed_args.num_frames}, FPS: {parsed_args.fps}")
        logger.info(f"I2V Model: {config.models.i2v_model}")
        logger.info("=" * 50)
        
        result = pipeline.process(
            image=str(input_path),
            prompt=parsed_args.prompt,
            negative_prompt=parsed_args.negative_prompt or "blurry, low quality, artifacts, static",
            num_frames=parsed_args.num_frames,
            fps=parsed_args.fps,
            guidance_scale=parsed_args.guidance_scale,
            num_inference_steps=parsed_args.num_inference_steps,
            motion_strength=parsed_args.motion_strength,
            output_path=str(output_path),
            seed=parsed_args.seed
        )
        
        logger.info("=" * 50)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Video saved to: {output_path}")
        logger.info(f"Duration: {result.duration:.2f}s")
        logger.info(f"Processing time: {result.processing_time:.2f}s")
        
        if parsed_args.save_intermediate:
            intermediate_dir = output_path.parent / f"{output_path.stem}_intermediate"
            pipeline.save_intermediate(result, intermediate_dir)
            logger.info(f"Intermediate results saved to: {intermediate_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        pipeline.clear_cache()


if __name__ == "__main__":
    sys.exit(main())