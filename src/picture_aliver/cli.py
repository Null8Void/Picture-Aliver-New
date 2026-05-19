"""
Pic Aliver - CLI Entry Point

Usage:
    picaliver                          # Interactive model selector
    picaliver --list-models            # List all models
    picaliver --model "DreamShaper XL" # Interactive experience for a model
    picaliver --model "DreamShaper XL" --mode txt2img --prompt "a cat"
    picaliver --model "Polaris" --mode img2video --input start.png --prompt "dance"

Each model is its own experience. Modes available depend on model type:
  - TEXT2IMAGE models: txt2img, img2img
  - VIDEO_MOTION models: txt2img, img2img, img2video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .experiences import (
    ModelExperienceLauncher,
    run_one_shot,
    GenerationResult,
)


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="picaliver",
        description="Pic Aliver - Model Experience Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes by model type:
  TEXT2IMAGE models:  txt2img, img2img
  VIDEO_MOTION models: txt2img, img2img, img2video

Examples:
  picaliver                         # Interactive mode
  picaliver --list-models           # List all models
  picaliver -m "DreamShaper XL"     # Enter a model experience
  picaliver -m "DreamShaper XL" --mode txt2img -p "a cat"
  picaliver -m "Polaris" --mode img2video -i start.png -p "dance"
        """,
    )
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name to launch")
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List all available models")
    parser.add_argument("--version", "-v", action="store_true",
                        help="Show version")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["txt2img", "img2img", "img2video"],
                        help="Generation mode (requires --model)")
    parser.add_argument("--prompt", "-p", type=str, default="",
                        help="Text prompt for generation")
    parser.add_argument("--negative", "-n", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input image path (for img2img or img2video)")
    parser.add_argument("--end-input", type=str, default=None,
                        help="End frame image path (for img2video animation)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--steps", type=int, default=25,
                        help="Denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--strength", type=float, default=0.75,
                        help="Denoising strength (img2img)")
    parser.add_argument("--frames", type=int, default=24,
                        help="Number of video frames")
    parser.add_argument("--fps", type=int, default=8,
                        help="Video FPS")
    
    args = parser.parse_args(argv)
    
    if args.version:
        try:
            from . import __version__
            print(f"Pic Aliver v{__version__}")
        except ImportError:
            print("Pic Aliver v1.0.0")
        return 0
    
    if args.list_models:
        launcher = ModelExperienceLauncher()
        launcher.list_models()
        return 0
    
    if args.model and args.mode:
        kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
        }
        
        if args.mode in ("img2img", "img2video"):
            kwargs["input_image"] = args.input
        
        if args.mode == "img2video":
            kwargs["num_frames"] = args.frames
            kwargs["fps"] = args.fps
            if args.end_input:
                kwargs["end_image"] = args.end_input
        
        if args.mode == "img2img":
            kwargs["strength"] = args.strength
        
        result = run_one_shot(
            model=args.model,
            mode=args.mode,
            output=args.output,
            **kwargs,
        )
        
        if result.success:
            print(f"✓ {result.output_path}")
            return 0
        else:
            print(f"✗ {result.error}", file=sys.stderr)
            return 1
    
    if args.model:
        launcher = ModelExperienceLauncher()
        launcher.launch(args.model, mode=args.mode)
        return 0
    
    launcher = ModelExperienceLauncher()
    launcher.interactive_select()
    return 0


if __name__ == "__main__":
    sys.exit(main())
