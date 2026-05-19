"""
Model Experience System

Each model is its own interactive experience with modes determined by model type:
- IMAGE models (TEXT2IMAGE): txt2img, img2img
- VIDEO models (VIDEO_MOTION): txt2img, img2img, img2video

Extensible: add new experience types by subclassing ModelExperience.
"""

from __future__ import annotations

import os
import sys
import gc
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import torch
import numpy as np

# Ensure core package is importable (works for both pip-installed and python -m)
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.model_registry import (
    MODEL_REGISTRY, ModelInfo, ModelCategory, ContentRating,
)
from core.model_manager import ModelManager

OUTPUT_DIR = Path("./outputs")


@dataclass
class GenerationResult:
    """Result of a generation."""
    success: bool
    output_path: Optional[Path] = None
    mode: str = ""
    model_name: str = ""
    prompt: str = ""
    seed: Optional[int] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def timestamp() -> str:
    """Get timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(text: str, max_len: int = 50) -> str:
    """Convert text to a safe filename fragment."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)
    return safe.strip()[:max_len].strip()


class ModelExperience(ABC):
    """
    Base class for a model experience.
    
    Each model experience provides generation modes appropriate to the model type.
    """
    
    def __init__(
        self,
        model_info: ModelInfo,
        manager: ModelManager,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.model_info = model_info
        self.manager = manager
        self.output_dir = output_dir / sanitize_filename(model_info.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        return self.model_info.name
    
    @abstractmethod
    def available_modes(self) -> List[Dict[str, Any]]:
        """Return list of available modes with metadata."""
        ...
    
    @abstractmethod
    def run_mode(self, mode: str, **kwargs) -> GenerationResult:
        """Run a specific generation mode."""
        ...
    
    def interactive_loop(self) -> None:
        """Run interactive experience loop for this model."""
        while True:
            self._show_header()
            modes = self.available_modes()
            
            print("\nAvailable Modes:")
            for i, mode in enumerate(modes, 1):
                print(f"  [{i}] {mode['label']}")
            print(f"  [B] Back to model selection")
            print(f"  [Q] Quit")
            
            choice = input("\nChoose: ").strip().upper()
            
            if choice == "Q":
                print("\nGoodbye!")
                sys.exit(0)
            elif choice == "B":
                return
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(modes):
                        mode = modes[idx]
                        self._run_interactive_mode(mode)
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Invalid choice")
    
    def _show_header(self) -> None:
        """Display model experience header."""
        info = self.model_info
        print("\n" + "=" * 60)
        print(f"  {info.name}")
        print(f"  {'─' * 58}")
        print(f"  Pipeline: {info.pipeline_type or info.category.value}")
        print(f"  VRAM: {info.vram_mb}MB")
        print(f"  Resolution: {info.resolution or 'N/A'}")
        print("=" * 60)
    
    def _run_interactive_mode(self, mode: Dict[str, Any]) -> None:
        """Run an interactive session for a specific mode."""
        mode_id = mode["id"]
        print(f"\n── {mode['label']} ──")
        
        try:
            if mode_id == "txt2img":
                result = self._interactive_txt2img()
            elif mode_id == "img2img":
                result = self._interactive_img2img()
            elif mode_id == "img2video":
                result = self._interactive_img2video()
            else:
                print(f"Unknown mode: {mode_id}")
                return
            
            if result and result.success and result.output_path:
                print(f"\n  ✓ Saved: {result.output_path}")
            elif result and result.error:
                print(f"\n  ✗ Error: {result.error}")
                
        except KeyboardInterrupt:
            print("\n  Cancelled")
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def _interactive_txt2img(self) -> Optional[GenerationResult]:
        """Interactive text-to-image generation."""
        prompt = input("Prompt: ").strip()
        if not prompt:
            print("  Prompt required")
            return None
        
        neg = input("Negative prompt (optional): ").strip()
        seed_str = input("Seed (optional, Enter=random): ").strip()
        seed = int(seed_str) if seed_str else None
        
        width = self._prompt_int("Width", self.model_info.resolution[0] if self.model_info.resolution else 1024)
        height = self._prompt_int("Height", self.model_info.resolution[1] if self.model_info.resolution else 1024)
        steps = self._prompt_int("Steps", 25)
        guidance = self._prompt_float("CFG Scale", 7.5)
        
        print(f"\n  Generating...")
        start = time.time()
        
        result = self.run_mode(
            "txt2img",
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )
        
        result.processing_time = time.time() - start
        if result.success:
            print(f"  Done in {result.processing_time:.1f}s")
        return result
    
    def _interactive_img2img(self) -> Optional[GenerationResult]:
        """Interactive image-to-image generation."""
        path_str = input("Input image path: ").strip()
        if not path_str:
            print("  Path required")
            return None
        input_path = Path(path_str)
        if not input_path.exists():
            print(f"  File not found: {input_path}")
            return None
        
        prompt = input("Prompt: ").strip()
        if not prompt:
            print("  Prompt required")
            return None
        
        neg = input("Negative prompt (optional): ").strip()
        strength = self._prompt_float("Denoising strength (0-1)", 0.75)
        seed_str = input("Seed (optional, Enter=random): ").strip()
        seed = int(seed_str) if seed_str else None
        steps = self._prompt_int("Steps", 25)
        
        print(f"\n  Generating...")
        start = time.time()
        
        result = self.run_mode(
            "img2img",
            prompt=prompt,
            negative_prompt=neg,
            input_image=input_path,
            strength=strength,
            steps=steps,
            seed=seed,
        )
        
        result.processing_time = time.time() - start
        if result.success:
            print(f"  Done in {result.processing_time:.1f}s")
        return result
    
    def _interactive_img2video(self) -> Optional[GenerationResult]:
        """Interactive image-to-video generation."""
        path_str = input("Start frame image path: ").strip()
        if not path_str:
            print("  Path required")
            return None
        start_path = Path(path_str)
        if not start_path.exists():
            print(f"  File not found: {start_path}")
            return None
        
        end_path_str = input("End frame image path (optional, Enter=loop): ").strip()
        end_path = Path(end_path_str) if end_path_str else None
        if end_path and not end_path.exists():
            print(f"  File not found: {end_path}")
            return None
        
        prompt = input("Motion description: ").strip()
        num_frames = self._prompt_int("Number of frames", 24)
        fps = self._prompt_int("FPS", 8)
        
        print(f"\n  Generating {num_frames} frames at {fps}fps...")
        start = time.time()
        
        result = self.run_mode(
            "img2video",
            input_image=start_path,
            end_image=end_path,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
        )
        
        result.processing_time = time.time() - start
        if result.success:
            print(f"  Done in {result.processing_time:.1f}s")
        return result
    
    def _prompt_int(self, label: str, default: int) -> int:
        """Prompt for an integer with default."""
        val = input(f"{label} [{default}]: ").strip()
        return int(val) if val else default
    
    def _prompt_float(self, label: str, default: float) -> float:
        """Prompt for a float with default."""
        val = input(f"{label} [{default}]: ").strip()
        return float(val) if val else default


# =============================================================================
# IMAGE MODEL EXPERIENCE (TEXT2IMAGE)
# =============================================================================

class ImageModelExperience(ModelExperience):
    """
    Experience for TEXT2IMAGE models.
    Modes: txt2img, img2img
    """
    
    def available_modes(self) -> List[Dict[str, Any]]:
        return [
            {"id": "txt2img", "label": "Text-to-Image", "description": "Generate from text prompt"},
            {"id": "img2img", "label": "Image-to-Image", "description": "Transform an existing image"},
        ]
    
    def run_mode(self, mode: str, **kwargs) -> GenerationResult:
        if mode == "txt2img":
            return self._do_txt2img(**kwargs)
        elif mode == "img2img":
            return self._do_img2img(**kwargs)
        else:
            return GenerationResult(False, error=f"Unknown mode: {mode}")
    
    def _do_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        try:
            progress_cb = kwargs.pop('_progress_callback', None)
            self.manager.load_image_model(self.model_info.name)
            image = self.manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                progress_callback=progress_cb,
            )
            
            mode_dir = self.output_dir / "txt2img"
            mode_dir.mkdir(exist_ok=True)
            ts = timestamp()
            slug = sanitize_filename(prompt)
            fname = f"{ts}_{slug}.png"
            out_path = mode_dir / fname
            
            from PIL import Image as PILImage
            image = image.clamp(0, 1)
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in (1, 3, 4):
                img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            elif image.dim() == 3:
                img_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            else:
                img_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            PILImage.fromarray(img_np).save(out_path)
            
            return GenerationResult(
                success=True,
                output_path=out_path,
                mode="txt2img",
                model_name=self.model_info.name,
                prompt=prompt,
                seed=seed,
                metadata={"width": width, "height": height, "steps": steps},
            )
        except Exception as e:
            return GenerationResult(False, error=str(e))
    
    def _do_img2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_image: Optional[Union[str, Path]] = None,
        strength: float = 0.75,
        steps: int = 25,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        try:
            from PIL import Image as PILImage
            progress_cb = kwargs.pop('_progress_callback', None)
            
            if progress_cb:
                progress_cb(0, steps, "Loading image...")
            
            input_path = Path(input_image)
            pil_image = PILImage.open(input_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(self.manager.device)
            
            self.manager.load_image_model(self.model_info.name)
            
            if seed is not None:
                torch.manual_seed(seed)
            
            pipeline = self.manager._image_pipeline
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=pil_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=kwargs.get("guidance_scale", 7.5),
            )
            if progress_cb:
                def _cb(step, _t, _l):
                    progress_cb(step + 1, steps, f"Step {step+1}/{steps}")
                    return False
                pipe_kwargs["callback"] = _cb
                pipe_kwargs["callback_steps"] = 1
            
            if hasattr(pipeline, "img2img"):
                result = pipeline.img2img(**pipe_kwargs)
            else:
                pipe_kwargs["output_type"] = "pil"
                result = pipeline(**pipe_kwargs)
            
            if hasattr(result, "images"):
                out_img = result.images[0]
            else:
                out_img = result
            
            mode_dir = self.output_dir / "img2img"
            mode_dir.mkdir(exist_ok=True)
            ts = timestamp()
            slug = sanitize_filename(prompt)
            fname = f"{ts}_{slug}.png"
            out_path = mode_dir / fname
            
            if isinstance(out_img, PILImage.Image):
                out_img.save(out_path)
            elif isinstance(out_img, torch.Tensor):
                img_np = (out_img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(out_path)
            elif isinstance(out_img, np.ndarray):
                PILImage.fromarray(out_img).save(out_path)
            
            return GenerationResult(
                success=True,
                output_path=out_path,
                mode="img2img",
                model_name=self.model_info.name,
                prompt=prompt,
                seed=seed,
                metadata={"strength": strength},
            )
        except Exception as e:
            return GenerationResult(False, error=str(e))


# =============================================================================
# VIDEO MODEL EXPERIENCE (VIDEO_MOTION)
# =============================================================================

class VideoModelExperience(ModelExperience):
    """
    Experience for VIDEO_MOTION models (Polaris, Lynx One).
    Modes: txt2img, img2img, img2video
    """
    
    def available_modes(self) -> List[Dict[str, Any]]:
        return [
            {"id": "txt2img", "label": "Text-to-Image", "description": "Generate an image from text"},
            {"id": "img2img", "label": "Image-to-Image", "description": "Transform an existing image"},
            {"id": "img2video", "label": "Image-to-Video", "description": "Animate from image(s) to video"},
        ]
    
    def run_mode(self, mode: str, **kwargs) -> GenerationResult:
        if mode in ("txt2img", "img2img"):
            image_exp = ImageModelExperience(self.model_info, self.manager, self.output_dir.parent)
            return image_exp.run_mode(mode, **kwargs)
        elif mode == "img2video":
            return self._do_img2video(**kwargs)
        else:
            return GenerationResult(False, error=f"Unknown mode: {mode}")
    
    def _do_img2video(
        self,
        input_image: Optional[Union[str, Path]] = None,
        end_image: Optional[Union[str, Path]] = None,
        prompt: str = "",
        num_frames: int = 24,
        fps: int = 8,
        **kwargs,
    ) -> GenerationResult:
        try:
            from PIL import Image as PILImage
            progress_cb = kwargs.pop('_progress_callback', None)
            
            input_path = Path(input_image)
            pil_image = PILImage.open(input_path).convert("RGB")
            start_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            start_tensor = start_tensor.to(self.manager.device)
            
            end_tensor = None
            if end_image:
                end_path = Path(end_image)
                pil_end = PILImage.open(end_path).convert("RGB")
                end_tensor = torch.from_numpy(np.array(pil_end)).permute(2, 0, 1).float() / 255.0
                end_tensor = end_tensor.to(self.manager.device)
            
            self.manager.load_motion_model(self.model_info.name)
            
            frames = self.manager.generate_video(
                start_frame=start_tensor,
                end_frame=end_tensor,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                motion_strength=kwargs.get("motion_strength", 0.8),
                progress_callback=progress_cb,
            )
            
            mode_dir = self.output_dir / "img2video"
            mode_dir.mkdir(exist_ok=True)
            ts = timestamp()
            slug = sanitize_filename(prompt) if prompt else "video"
            video_path = mode_dir / f"{ts}_{slug}.mp4"
            frames_dir = mode_dir / f"{ts}_{slug}_frames"
            frames_dir.mkdir(exist_ok=True)
            
            try:
                import cv2
                frame_list = []
                for i, frame in enumerate(frames):
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu()
                        if frame.dim() == 3:
                            frame = frame.permute(1, 2, 0).numpy()
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        if frame.shape[-1] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_list.append(frame)
                
                if frame_list:
                    h, w = frame_list[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                    for frame in frame_list:
                        writer.write(frame)
                    writer.release()
                
                for i, frame in enumerate(frames):
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu()
                        if frame.dim() == 3:
                            frame = frame.permute(1, 2, 0).numpy()
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    PILImage.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
                
            except ImportError:
                for i, frame in enumerate(frames):
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu()
                        if frame.dim() == 3:
                            frame = frame.permute(1, 2, 0).numpy()
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    PILImage.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
                video_path = frames_dir
            
            return GenerationResult(
                success=True,
                output_path=video_path,
                mode="img2video",
                model_name=self.model_info.name,
                prompt=prompt,
                metadata={"num_frames": num_frames, "fps": fps},
            )
        except Exception as e:
            return GenerationResult(False, error=str(e))


# =============================================================================
# EXPERIENCE LAUNCHER
# =============================================================================

# =============================================================================
# VIDEO DIFFUSION EXPERIENCE (Wan2.2 I2V / T2V)
# =============================================================================

class VideoDiffusionExperience(ModelExperience):
    """
    Experience for native video diffusion models (Wan2.2 I2V / T2V).
    Modes: txt2video, img2video
    """

    def available_modes(self) -> List[Dict[str, Any]]:
        return [
            {"id": "txt2video", "label": "Text-to-Video", "description": "Generate video from text"},
            {"id": "img2video", "label": "Image-to-Video", "description": "Animate from image to video"},
        ]

    def run_mode(self, mode: str, **kwargs) -> GenerationResult:
        if mode == "img2video":
            return self._do_img2video(**kwargs)
        elif mode == "txt2video":
            return self._do_txt2video(**kwargs)
        else:
            return GenerationResult(False, error=f"Unknown mode: {mode}")

    def _do_img2video(
        self,
        prompt: str = "",
        input_image: Optional[Union[str, Path]] = None,
        num_frames: int = 49,
        fps: int = 16,
        **kwargs,
    ) -> GenerationResult:
        try:
            from PIL import Image as PILImage
            progress_cb = kwargs.pop('_progress_callback', None)

            self.manager.load_image_model(self.model_info.name)
            pipeline = self.manager._image_pipeline

            pipe_kwargs = {"prompt": prompt}
            if "negative_prompt" in kwargs:
                pipe_kwargs["negative_prompt"] = kwargs["negative_prompt"]

            if input_image:
                input_path = Path(input_image)
                pipe_kwargs["image"] = PILImage.open(input_path).convert("RGB")

            steps = kwargs.get("steps", 25)
            pipe_kwargs["num_inference_steps"] = steps
            pipe_kwargs["guidance_scale"] = kwargs.get("guidance_scale", 7.5)

            if progress_cb:
                def _cb(step, _t, _l):
                    progress_cb(step + 1, steps, f"Step {step+1}/{steps}")
                    return False
                pipe_kwargs["callback"] = _cb
                pipe_kwargs["callback_steps"] = 1

            result = pipeline(**pipe_kwargs, num_frames=num_frames)

            mode_dir = self.output_dir / "img2video"
            mode_dir.mkdir(exist_ok=True)
            ts = timestamp()
            slug = sanitize_filename(prompt) if prompt else "video"
            frames_dir = mode_dir / f"{ts}_{slug}_frames"
            frames_dir.mkdir(exist_ok=True)

            if hasattr(result, "frames"):
                frames_list = result.frames[0] if result.frames else []
            elif isinstance(result, dict):
                frames_list = result.get("frames", [])
            elif isinstance(result, torch.Tensor):
                if result.dim() == 5:
                    frames_list = [result[0][i] for i in range(result.shape[1])]
                elif result.dim() == 4:
                    frames_list = [result[i] for i in range(result.shape[0])]
                else:
                    frames_list = []
            else:
                frames_list = []
            if not frames_list:
                return GenerationResult(False, error="Pipeline returned no frames")
            video_path = None

            for i, frame in enumerate(frames_list):
                if isinstance(frame, PILImage.Image):
                    frame.save(frames_dir / f"frame_{i:04d}.png")
                else:
                    img_np = (frame.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    PILImage.fromarray(img_np).save(frames_dir / f"frame_{i:04d}.png")

            try:
                import cv2
                frame_paths = sorted(frames_dir.glob("frame_*.png"))
                if frame_paths:
                    first = cv2.imread(str(frame_paths[0]))
                    h, w = first.shape[:2]
                    video_path = mode_dir / f"{ts}_{slug}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                    for fp in frame_paths:
                        writer.write(cv2.imread(str(fp)))
                    writer.release()
            except ImportError:
                video_path = frames_dir

            return GenerationResult(
                success=True,
                output_path=video_path or frames_dir,
                mode="img2video",
                model_name=self.model_info.name,
                prompt=prompt,
                metadata={"num_frames": num_frames, "fps": fps},
            )
        except Exception as e:
            return GenerationResult(False, error=str(e))

    def _do_txt2video(
        self,
        prompt: str = "",
        num_frames: int = 49,
        fps: int = 16,
        **kwargs,
    ) -> GenerationResult:
        try:
            progress_cb = kwargs.pop('_progress_callback', None)
            self.manager.load_image_model(self.model_info.name)
            pipeline = self.manager._image_pipeline

            pipe_kwargs = {"prompt": prompt}
            if "negative_prompt" in kwargs:
                pipe_kwargs["negative_prompt"] = kwargs["negative_prompt"]

            steps = kwargs.get("steps", 25)
            pipe_kwargs["num_inference_steps"] = steps
            pipe_kwargs["guidance_scale"] = kwargs.get("guidance_scale", 7.5)

            if progress_cb:
                def _cb(step, _t, _l):
                    progress_cb(step + 1, steps, f"Step {step+1}/{steps}")
                    return False
                pipe_kwargs["callback"] = _cb
                pipe_kwargs["callback_steps"] = 1

            result = pipeline(**pipe_kwargs, num_frames=num_frames)

            mode_dir = self.output_dir / "txt2video"
            mode_dir.mkdir(exist_ok=True)
            ts = timestamp()
            slug = sanitize_filename(prompt) if prompt else "video"
            frames_dir = mode_dir / f"{ts}_{slug}_frames"
            frames_dir.mkdir(exist_ok=True)

            if hasattr(result, "frames"):
                frames_list = result.frames[0] if result.frames else []
            elif isinstance(result, dict):
                frames_list = result.get("frames", [])
            elif isinstance(result, torch.Tensor):
                if result.dim() == 5:
                    frames_list = [result[0][i] for i in range(result.shape[1])]
                elif result.dim() == 4:
                    frames_list = [result[i] for i in range(result.shape[0])]
                else:
                    frames_list = []
            else:
                frames_list = []
            if not frames_list:
                return GenerationResult(False, error="Pipeline returned no frames")

            for i, frame in enumerate(frames_list):
                if isinstance(frame, PILImage.Image):
                    frame.save(frames_dir / f"frame_{i:04d}.png")
                else:
                    img_np = (frame.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    PILImage.fromarray(img_np).save(frames_dir / f"frame_{i:04d}.png")

            video_path = None
            try:
                import cv2
                frame_paths = sorted(frames_dir.glob("frame_*.png"))
                if frame_paths:
                    first = cv2.imread(str(frame_paths[0]))
                    h, w = first.shape[:2]
                    video_path = mode_dir / f"{ts}_{slug}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                    for fp in frame_paths:
                        writer.write(cv2.imread(str(fp)))
                    writer.release()
            except ImportError:
                video_path = frames_dir

            return GenerationResult(
                success=True,
                output_path=video_path or frames_dir,
                mode="txt2video",
                model_name=self.model_info.name,
                prompt=prompt,
                metadata={"num_frames": num_frames, "fps": fps},
            )
        except Exception as e:
            return GenerationResult(False, error=str(e))


EXPERIENCE_MAP: Dict[ModelCategory, type] = {
    ModelCategory.TEXT2IMAGE: ImageModelExperience,
    ModelCategory.VIDEO_MOTION: VideoModelExperience,
    ModelCategory.I2V: VideoDiffusionExperience,
    ModelCategory.TEXT2VIDEO: VideoDiffusionExperience,
}


class ModelExperienceLauncher:
    """
    Launcher that discovers models and creates appropriate experiences.
    """
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.manager = ModelManager()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_models(self) -> None:
        """Print all available models grouped by category."""
        self.manager.print_model_summary()
    
    def get_experience(self, model_name: str) -> Optional[ModelExperience]:
        """Get the appropriate experience for a model."""
        info = (
            self.manager.get_image_model_info(model_name)
            or self.manager.get_motion_model_info(model_name)
        )
        if info is None:
            available = self.manager.available_image_models + self.manager.available_motion_models
            print(f"Unknown model: '{model_name}'")
            print(f"Available: {', '.join(available)}")
            return None
        
        exp_class = EXPERIENCE_MAP.get(info.category)
        if exp_class is None:
            print(f"No experience defined for category: {info.category}")
            return None
        
        return exp_class(info, self.manager, self.output_dir)
    
    def launch(self, model_name: str, mode: Optional[str] = None) -> None:
        """Launch a model experience, optionally in a specific mode."""
        exp = self.get_experience(model_name)
        if exp is None:
            return
        
        if mode:
            modes = {m["id"]: m for m in exp.available_modes()}
            if mode not in modes:
                print(f"Mode '{mode}' not available for {model_name}")
                print(f"Available: {', '.join(modes.keys())}")
                return
            exp._run_interactive_mode(modes[mode])
        else:
            exp.interactive_loop()
    
    def interactive_select(self) -> None:
        """Interactive model selection menu."""
        while True:
            self._show_selection_menu()
            
            images = self.manager.available_image_models
            motions = self.manager.available_motion_models
            all_models = images + motions
            total = len(all_models)
            
            choice = input("\nSelect model number (or Q to quit): ").strip().upper()
            
            if choice == "Q":
                print("Goodbye!")
                return
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < total:
                    model_name = all_models[idx]
                    self.launch(model_name)
                else:
                    print(f"Enter 1-{total}")
            except ValueError:
                print(f"Enter a number 1-{total} or Q")
    
    def _show_selection_menu(self) -> None:
        """Display the model selection menu."""
        images = self.manager.available_image_models
        motions = self.manager.available_motion_models
        
        print("\n" + "=" * 60)
        print("  PIC ALIVER - Model Selector")
        print("=" * 60)
        
        idx = 1
        print("\n  Image Models:")
        categories = self.manager.list_models_by_style()
        for style, model_list in categories.items():
            if style == "Motion":
                continue
            print(f"\n    [{style}]")
            for name in model_list:
                if name in images:
                    info = self.manager.get_image_model_info(name)
                    tag = f" ({info.pipeline_type})" if info else ""
                    print(f"    {idx:2d}. {name}{tag}")
                    idx += 1
        
        print("\n  Video/Motion Models:")
        print(f"\n    [Motion]")
        for name in motions:
            info = self.manager.get_motion_model_info(name)
            tag = f" ({info.pipeline_type})" if info else ""
            print(f"    {idx:2d}. {name}{tag}")
            idx += 1
        
        print(f"\n  Total: {idx - 1} models")


def launch_interactive(output_dir: str = "./outputs") -> None:
    """Main entry point for interactive experience."""
    launcher = ModelExperienceLauncher(Path(output_dir))
    launcher.interactive_select()


def run_one_shot(
    model: str,
    mode: str = "txt2img",
    prompt: str = "",
    input_image: Optional[str] = None,
    output: Optional[str] = None,
    **kwargs,
) -> GenerationResult:
    """One-shot generation from a specific model and mode."""
    launcher = ModelExperienceLauncher()
    exp = launcher.get_experience(model)
    if exp is None:
        return GenerationResult(False, error=f"Model not found: {model}")
    
    gen_kwargs = {"prompt": prompt}
    if input_image:
        gen_kwargs["input_image"] = input_image
    gen_kwargs.update(kwargs)
    
    result = exp.run_mode(mode, **gen_kwargs)
    
    if output and result.success and result.output_path:
        import shutil
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.output_path, out)
        result.output_path = out
    
    return result
