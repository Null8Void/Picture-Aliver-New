"""
Picture-Aliver: AI-Powered Image to Video Animation System

A production-grade system for converting single images into coherent animated videos.
Supports human, furry, landscape, and object content with dynamic adaptation.
Supports both image-to-video and text-to-image/video generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import torch

from image_loader import ImageLoader
from depth_estimator import DepthEstimator
from segmentation import SegmentationModule
from motion_generator import MotionGenerator
from video_generator import VideoGenerator
from stabilizer import VideoStabilizer
from exporter import VideoExporter
from text_to_image import TextToImageGenerator, TextToVideoGenerator
from quality_control import QualityController, QualityReport
from gpu_optimization import GPUOptimizer, VRAMTier


class PictureAliver:
    """
    Main system orchestrator for image-to-video conversion.
    
    This class coordinates all subsystems to transform a static image
    into a smooth, coherent animated video with proper motion and
    temporal consistency.
    
    Attributes:
        device: PyTorch device for computation (CPU/CUDA)
        enable_cuda: Whether CUDA acceleration is enabled
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        enable_cuda: bool = True,
        model_dir: Optional[Path] = None
    ):
        self.device = self._setup_device(device, enable_cuda)
        self.dtype = torch.float32
        self.model_dir = model_dir or Path("./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_loader: Optional[ImageLoader] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.segmentation: Optional[SegmentationModule] = None
        self.motion_generator: Optional[MotionGenerator] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.stabilizer: Optional[VideoStabilizer] = None
        self.exporter: Optional[VideoExporter] = None
        self.text_to_image: Optional[TextToImageGenerator] = None
        self.text_to_video: Optional[TextToVideoGenerator] = None
        
        self.quality_controller: Optional[QualityController] = None
        self.gpu_optimizer: Optional[GPUOptimizer] = None
        
        self._initialized = False
        self._quality_auto_correct = True
        self._quality_max_retries = 2
    
    def _setup_device(
        self,
        device: Optional[torch.device],
        enable_cuda: bool
    ) -> torch.device:
        """Setup the compute device."""
        if device is not None:
            return device
        
        if enable_cuda and torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            print(f"[PictureAliver] Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"[PictureAliver] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return cuda_device
        
        print("[PictureAliver] Using CPU")
        return torch.device("cpu")
    
    def initialize(self) -> None:
        """Initialize all subsystems."""
        if self._initialized:
            return
        
        print("[PictureAliver] Initializing subsystems...")
        
        self.image_loader = ImageLoader(device=self.device)
        
        self.depth_estimator = DepthEstimator(
            device=self.device,
            model_type="zoedepth",
            model_dir=self.model_dir
        )
        
        self.segmentation = SegmentationModule(
            device=self.device,
            model_dir=self.model_dir
        )
        
        self.motion_generator = MotionGenerator(
            device=self.device,
            depth_estimator=self.depth_estimator
        )
        
        self.video_generator = VideoGenerator(
            device=self.device,
            depth_estimator=self.depth_estimator
        )
        
        self.stabilizer = VideoStabilizer(device=self.device)
        
        self.exporter = VideoExporter(device=self.device)
        
        self.text_to_image = TextToImageGenerator(device=self.device)
        self.text_to_video = TextToVideoGenerator(device=self.device)
        
        self.gpu_optimizer = GPUOptimizer(device=self.device)
        self.quality_controller = QualityController(
            device=self.device,
            max_retries=self._quality_max_retries,
            auto_correct=self._quality_auto_correct
        )
        
        self._initialized = True
        print("[PictureAliver] All subsystems initialized")
        
        benchmark = self.gpu_optimizer.get_benchmark()
        print(f"[PictureAliver] GPU Tier: {self.gpu_optimizer.config.tier.value}")
        print(f"[PictureAliver] Max Resolution: {benchmark.max_resolution}")
        print(f"[PictureAliver] FP16 Enabled: {self.gpu_optimizer.config.use_fp16}")
    
    def process(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        prompt: str = "",
        negative_prompt: str = "",
        duration_seconds: float = 3.0,
        fps: int = 8,
        motion_strength: float = 0.8,
        motion_mode: str = "auto",
        motion_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None,
        content_type_hint: Optional[str] = None,
        apply_stabilization: bool = True,
        output_format: str = "mp4",
        quality: str = "medium",
        enable_interpolation: bool = False
    ) -> dict:
        """
        Process an image to generate an animated video.
        
        Args:
            image_path: Path to input image
            output_path: Path for output video
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            duration_seconds: Video duration (5-120 seconds)
            fps: Frames per second
            motion_strength: Strength of motion (0-1)
            motion_mode: Motion mode (auto, cinematic, zoom, pan, subtle, furry)
            motion_prompt: Natural language motion description
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            content_type_hint: Hint for content type
            apply_stabilization: Whether to apply stabilization
            output_format: Output format (mp4, webm, gif)
            quality: Quality preset (low, medium, high, ultra)
            enable_interpolation: Enable frame interpolation
            
        Returns:
            Dictionary with processing metadata
        """
        duration_seconds = max(5.0, min(120.0, duration_seconds))
        num_frames = int(duration_seconds * fps)
        num_frames = max(num_frames, 40)
        
        if not self._initialized:
            self.initialize()
        
        import time
        start_time = time.time()
        
        image_path = Path(image_path)
        output_path = Path(output_path)
        
        print(f"[PictureAliver] Loading image: {image_path}")
        image_tensor = self.image_loader.load(image_path)
        
        print("[PictureAliver] Estimating depth...")
        depth_map = self.depth_estimator.estimate(image_tensor)
        
        print("[PictureAliver] Performing segmentation...")
        segmentation = self.segmentation.segment(image_tensor)
        
        content_type = content_type_hint or self.segmentation.detect_content_type(
            image_tensor, segmentation
        )
        print(f"[PictureAliver] Detected content type: {content_type}")
        print(f"[PictureAliver] Generating {num_frames} frames ({duration_seconds}s @ {fps}fps)...")
        
        print("[PictureAliver] Generating motion vectors...")
        motion_field = self.motion_generator.generate(
            image_tensor,
            depth_map,
            segmentation,
            mode=motion_mode,
            strength=motion_strength,
            num_frames=num_frames,
            motion_prompt=motion_prompt
        )
        
        print(f"[PictureAliver] Generating video frames...")
        video_frames = self.video_generator.generate(
            image_tensor=image_tensor,
            depth_map=depth_map,
            motion_field=motion_field,
            segmentation=segmentation,
            prompt=prompt or self._get_default_prompt(content_type),
            negative_prompt=negative_prompt or self._get_default_negative_prompt(content_type),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            content_type=content_type
        )
        
        if apply_stabilization:
            print("[PictureAliver] Stabilizing video...")
            video_frames = self.stabilizer.stabilize(
                video_frames,
                motion_field=motion_field
            )
        
        if video_frames:
            video_frames = video_frames.pad_to_duration(duration_seconds, fps)
        
        if self.quality_controller:
            print("[PictureAliver] Running quality control...")
            frames_tensor = video_frames.to_tensor()
            report, corrections = self.quality_controller.assess(frames_tensor)
            
            print(f"[PictureAliver] Quality Score: {report.overall_score:.2f}")
            
            if report.needs_correction:
                print(f"[PictureAliver] Issues detected: {[i.value for i in report.issues]}")
                
                if corrections:
                    if corrections.get("guidance_scale"):
                        guidance_scale *= corrections.get("guidance_scale", 1.0)
                        print(f"[PictureAliver] Adjusted guidance_scale to {guidance_scale:.1f}")
                    
                    if corrections.get("motion_strength"):
                        motion_strength = corrections.get("motion_strength")
                        print(f"[PictureAliver] Adjusted motion_strength to {motion_strength}")
                    
                    if corrections.get("new_seed"):
                        import random
                        seed = random.randint(0, 2**32 - 1)
                        print(f"[PictureAliver] Using new seed: {seed}")
        
        print(f"[PictureAliver] Exporting to {output_path}...")
        
        from exporter import ExportOptions, VideoSpec, QualityPreset, VideoFormat
        
        quality_map = {"low": QualityPreset.LOW, "medium": QualityPreset.MEDIUM, 
                      "high": QualityPreset.HIGH, "ultra": QualityPreset.ULTRA}
        
        options = ExportOptions(
            video_spec=VideoSpec(
                duration_seconds=duration_seconds,
                fps=fps,
                format=VideoFormat(output_format),
                quality=quality_map.get(quality, QualityPreset.MEDIUM)
            ),
            enable_interpolation=enable_interpolation
        )
        
        self.exporter.export(video_frames, output_path, options)
        
        elapsed = time.time() - start_time
        metadata = {
            "input_image": str(image_path),
            "output_video": str(output_path),
            "duration_seconds": duration_seconds,
            "fps": fps,
            "num_frames": num_frames,
            "content_type": content_type,
            "motion_mode": motion_mode,
            "processing_time": elapsed,
            "device": str(self.device)
        }
        
        print(f"[PictureAliver] Complete in {elapsed:.2f}s")
        return metadata
    
    def generate_image(
        self,
        prompt: str,
        output_path: Optional[Union[str, Path]] = None,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Text description
            output_path: Optional path to save image
            negative_prompt: What to avoid
            width: Output width
            height: Output height
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            
        Returns:
            Generated image tensor
        """
        if not self._initialized:
            self.initialize()
        
        if self.text_to_image is None:
            self.text_to_image = TextToImageGenerator(device=self.device)
        
        if not self.text_to_image._initialized:
            self.text_to_image.initialize()
        
        image = self.text_to_image.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or self._get_default_negative_prompt("scene"),
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if output_path:
            self._save_image(image, output_path)
        
        return image
    
    def generate_video_from_text(
        self,
        prompt: str,
        output_path: Union[str, Path],
        motion_prompt: Optional[str] = None,
        negative_prompt: str = "",
        duration_seconds: float = 3.0,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        quality: str = "medium",
        output_format: str = "mp4"
    ) -> dict:
        """
        Generate video directly from text prompt.
        
        Args:
            prompt: Text description
            output_path: Path for output video
            motion_prompt: Motion description
            negative_prompt: What to avoid
            duration_seconds: Video duration (5-120 seconds)
            fps: Frames per second
            width: Frame width
            height: Frame height
            guidance_scale: CFG scale
            num_inference_steps: Denoising steps
            quality: Quality preset
            output_format: Output format
            
        Returns:
            Processing metadata
        """
        duration_seconds = max(5.0, min(120.0, duration_seconds))
        num_frames = int(duration_seconds * fps)
        num_frames = max(num_frames, 40)
        
        if not self._initialized:
            self.initialize()
        
        import time
        start_time = time.time()
        
        if self.text_to_video is None:
            self.text_to_video = TextToVideoGenerator(device=self.device)
        
        if not self.text_to_video._initialized:
            self.text_to_video.initialize()
        
        print(f"[PictureAliver] Generating video from text: '{prompt}'")
        print(f"[PictureAliver] Duration: {duration_seconds}s @ {fps}fps ({num_frames} frames)")
        
        video_frames = self.text_to_video.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or self._get_default_negative_prompt("scene"),
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            motion_prompt=motion_prompt
        )
        
        if video_frames:
            video_frames = video_frames.pad_to_duration(duration_seconds, fps)
        
        if self.stabilizer:
            video_frames = self.stabilizer.stabilize(video_frames)
        
        from exporter import ExportOptions, VideoSpec, QualityPreset, VideoFormat
        
        quality_map = {"low": QualityPreset.LOW, "medium": QualityPreset.MEDIUM, 
                      "high": QualityPreset.HIGH, "ultra": QualityPreset.ULTRA}
        
        options = ExportOptions(
            video_spec=VideoSpec(
                duration_seconds=duration_seconds,
                fps=fps,
                format=VideoFormat(output_format),
                quality=quality_map.get(quality, QualityPreset.MEDIUM)
            ),
            enable_interpolation=duration_seconds > 30
        )
        
        self.exporter.export(video_frames, output_path, options)
        
        elapsed = time.time() - start_time
        
        return {
            "prompt": prompt,
            "motion_prompt": motion_prompt,
            "output_video": str(output_path),
            "duration_seconds": duration_seconds,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": elapsed,
            "device": str(self.device)
        }
    
    def _get_default_prompt(self, content_type: str) -> str:
        """Get default prompt based on content type."""
        prompts = {
            "human": "smooth animation, natural motion, high quality, fluid movement",
            "furry": "animated furry character, smooth fur animation, natural motion, high quality",
            "animal": "animated animal, natural motion, high quality, fluid movement",
            "landscape": "cinematic landscape, natural wind movement, clouds drifting, high quality",
            "object": "smooth animation, subtle motion, high quality",
            "scene": "cinematic scene, smooth animation, high quality"
        }
        return prompts.get(content_type, "smooth animation, high quality")
    
    def _get_default_negative_prompt(self, content_type: str) -> str:
        """Get default negative prompt based on content type."""
        base = "blurry, low quality, artifacts, static, jittery, distorted"
        if content_type == "furry":
            return f"{base}, broken anatomy, melting fur, distorted face"
        return base
    
    def _save_image(self, image: torch.Tensor, output_path: Union[str, Path]) -> None:
        """Save image tensor to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(image_np).save(output_path)
        print(f"[PictureAliver] Saved image: {output_path}")
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def __repr__(self) -> str:
        return f"PictureAliver(device={self.device})"


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Picture-Aliver: Convert images to animated videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Image to Video):
  python main.py --input photo.jpg --output video.mp4
  python main.py -i photo.jpg -o video.mp4 --motion-prompt "cinematic camera pan"
  python main.py -i furry.png -o animation.mp4 --motion-prompt "gentle tail wag with breathing"

Examples (Text to Image):
  python main.py --text-prompt "a cute wolf in a forest" --output-image image.png

Examples (Text to Video):
  python main.py --text-prompt "a cat stretching" --output video.mp4 --text-to-video
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Generation mode")
    
    img2vid_parser = subparsers.add_parser("img2vid", help="Image to video")
    img2vid_parser.add_argument("-i", "--input", required=True, help="Input image path")
    img2vid_parser.add_argument("-o", "--output", required=True, help="Output video path")
    img2vid_parser.add_argument("--prompt", default="", help="Generation prompt")
    img2vid_parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    img2vid_parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (5-120)")
    img2vid_parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    img2vid_parser.add_argument("--motion-strength", type=float, default=0.8)
    img2vid_parser.add_argument("--motion-mode", default="auto",
                       choices=["auto", "cinematic", "zoom", "pan", "subtle", "furry"])
    img2vid_parser.add_argument("--motion-prompt", help="Motion description")
    img2vid_parser.add_argument("--no-stabilization", action="store_true")
    img2vid_parser.add_argument("--format", default="mp4", choices=["mp4", "webm", "gif"])
    img2vid_parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "ultra"])
    img2vid_parser.add_argument("--interpolate", action="store_true", help="Enable frame interpolation")
    
    txt2img_parser = subparsers.add_parser("txt2img", help="Text to image")
    txt2img_parser.add_argument("--prompt", required=True, help="Text description")
    txt2img_parser.add_argument("-o", "--output", required=True, help="Output image path")
    txt2img_parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    txt2img_parser.add_argument("--width", type=int, default=512)
    txt2img_parser.add_argument("--height", type=int, default=512)
    txt2img_parser.add_argument("--steps", type=int, default=30)
    txt2img_parser.add_argument("--scale", type=float, default=7.5)
    txt2img_parser.add_argument("--seed", type=int)
    
    txt2vid_parser = subparsers.add_parser("txt2vid", help="Text to video")
    txt2vid_parser.add_argument("--prompt", required=True, help="Text description")
    txt2vid_parser.add_argument("-o", "--output", required=True, help="Output video path")
    txt2vid_parser.add_argument("--motion-prompt", help="Motion description")
    txt2vid_parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (5-120)")
    txt2vid_parser.add_argument("--fps", type=int, default=8)
    txt2vid_parser.add_argument("--width", type=int, default=512)
    txt2vid_parser.add_argument("--height", type=int, default=512)
    txt2vid_parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "ultra"])
    
    parser.add_argument("-i", "--input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument("--text-prompt", help="Text prompt (alias for --prompt)")
    parser.add_argument("--output-image", help="Output image path (for txt2img)")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (5-120)")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--motion-strength", type=float, default=0.8)
    parser.add_argument("--motion-mode", default="auto",
                       choices=["auto", "cinematic", "zoom", "pan", "subtle", "furry"])
    parser.add_argument("--motion-prompt", help="Motion description")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--content-type",
                       choices=["human", "furry", "animal", "landscape", "object", "scene"])
    parser.add_argument("--no-stabilization", action="store_true")
    parser.add_argument("--format", default="mp4", choices=["mp4", "webm", "gif"])
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "ultra"])
    parser.add_argument("--interpolate", action="store_true", help="Enable frame interpolation")
    parser.add_argument("--text-to-video", action="store_true", help="Generate video from text")
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality control")
    parser.add_argument("--gpu-tier", choices=["2gb", "4gb", "8gb", "12gb", "24gb"], help="Override GPU tier")
    parser.add_argument("--benchmark", action="store_true", help="Show GPU benchmark table")
    parser.add_argument("--model-dir", type=Path, default=Path("./models"))
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--no-cuda", action="store_true")
    
    args = parser.parse_args()
    
    if args.benchmark:
        from gpu_optimization import print_benchmark_table
        print_benchmark_table()
        return 0
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    mode = args.mode
    
    enable_cuda = not args.no_cuda
    device = None
    if args.device:
        device = torch.device(args.device)
    
    system = PictureAliver(
        device=device,
        enable_cuda=enable_cuda,
        model_dir=args.model_dir
    )
    
    try:
        duration = max(5.0, min(120.0, args.duration))
        
        if mode == "img2vid":
            metadata = system.process(
                image_path=args.input,
                output_path=args.output,
                prompt=args.prompt or args.text_prompt or "",
                negative_prompt=args.negative_prompt,
                duration_seconds=duration,
                fps=args.fps,
                motion_strength=args.motion_strength,
                motion_mode=args.motion_mode,
                motion_prompt=args.motion_prompt,
                content_type_hint=args.content_type,
                apply_stabilization=not args.no_stabilization,
                output_format=args.format,
                quality=args.quality,
                enable_interpolation=args.interpolate
            )
        elif mode == "txt2img":
            system.generate_image(
                prompt=args.prompt,
                output_path=args.output,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.scale,
                seed=args.seed
            )
            metadata = {"output_image": args.output}
        elif mode == "txt2vid":
            metadata = system.generate_video_from_text(
                prompt=args.prompt,
                output_path=args.output,
                motion_prompt=args.motion_prompt,
                duration_seconds=duration,
                fps=args.fps,
                width=args.width,
                height=args.height,
                quality=args.quality
            )
        else:
            prompt = args.prompt or args.text_prompt or ""
            output = args.output or args.output_image or ""
            
            if args.input and output and not args.text_to_video:
                metadata = system.process(
                    image_path=args.input,
                    output_path=output,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    num_frames=args.frames,
                    fps=args.fps,
                    motion_strength=args.motion_strength,
                    motion_mode=args.motion_mode,
                    motion_prompt=args.motion_prompt,
                    content_type_hint=args.content_type,
                    apply_stabilization=not args.no_stabilization,
                    output_format=args.format
                )
            elif prompt and output and args.text_to_video:
                metadata = system.generate_video_from_text(
                    prompt=prompt,
                    output_path=output,
                    motion_prompt=args.motion_prompt,
                    num_frames=args.frames,
                    fps=args.fps,
                    width=args.width,
                    height=args.height
                )
            elif prompt and output:
                system.generate_image(
                    prompt=prompt,
                    output_path=output,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.scale,
                    seed=args.seed
                )
                metadata = {"output_image": output}
            else:
                print("Error: Insufficient arguments")
                parser.print_help()
                return 1
        
        print("\n=== Processing Complete ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[PictureAliver] Interrupted by user")
        return 130
    except Exception as e:
        print(f"[PictureAliver] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        system.clear_cache()


if __name__ == "__main__":
    sys.exit(main())