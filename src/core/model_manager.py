"""
Pic Aliver Model Manager

Central manager for all AI models.
Handles model lifecycle, switching, and orchestration for:
- TEXT2IMAGE models (Solarmix XL, DreamShaper XL, BluePencil XL, etc.)
- VIDEO_MOTION models (Polaris, Lynx One)
- LoRA adapters

Filter areas are left empty for later implementation.
"""

from __future__ import annotations

import os
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple

import torch
import torch.nn as nn

from .model_registry import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelCategory,
    ContentRating,
    get_registry,
)
from .model_loader import ModelLoader


class ModelManager:
    """
    Central manager for all models.
    
    Provides:
    - Model discovery and selection
    - Hot-swapping between image models
    - Motion pipeline selection (Polaris vs Lynx One)
    - Memory management across model switches
    - Unified generation interface
    
    Usage:
        manager = ModelManager()
        manager.load_image_model("Solarmix XL")
        image = manager.generate_image(prompt="...")
        
        manager.load_motion_model("Lynx One")
        frames = manager.generate_motion(start_frame, end_frame)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        default_rating: ContentRating = ContentRating.MATURE,
    ):
        if cache_dir is None:
            if "HF_HUB_CACHE" in os.environ:
                cache_dir = Path(os.environ["HF_HUB_CACHE"])
            elif "HUGGINGFACE_HUB_CACHE" in os.environ:
                cache_dir = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
            elif "HF_HOME" in os.environ:
                cache_dir = Path(os.environ["HF_HOME"]) / "hub"
            else:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_rating = default_rating
        
        self.loader = ModelLoader(
            cache_dir=self.cache_dir,
            device=self.device,
            default_rating=default_rating,
        )
        
        # Currently active models
        self._active_image_model: Optional[ModelInfo] = None
        self._active_motion_model: Optional[ModelInfo] = None
        self._active_lora: Optional[ModelInfo] = None
        
        # Loaded pipeline references
        self._image_pipeline: Optional[nn.Module] = None
        self._motion_pipeline: Optional[nn.Module] = None
        
        # Track available models by category
        self._image_models: Dict[str, ModelInfo] = {}
        self._motion_models: Dict[str, ModelInfo] = {}
        
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover all available Pic Aliver models from registry."""
        registry = get_registry()
        
        for name, info in registry.models.items():
            if info.category in (ModelCategory.TEXT2IMAGE, ModelCategory.I2V, ModelCategory.TEXT2VIDEO):
                self._image_models[name] = info
            elif info.category == ModelCategory.VIDEO_MOTION:
                self._motion_models[name] = info
    
    @property
    def available_image_models(self) -> List[str]:
        """Get list of available image model names."""
        return list(self._image_models.keys())
    
    @property
    def available_motion_models(self) -> List[str]:
        """Get list of available motion model names."""
        return list(self._motion_models.keys())
    
    @property
    def active_model_name(self) -> Optional[str]:
        """Get currently active image model name."""
        return self._active_image_model.name if self._active_image_model else None
    
    @property
    def active_motion_name(self) -> Optional[str]:
        """Get currently active motion model name."""
        return self._active_motion_model.name if self._active_motion_model else None
    
    def get_image_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get ModelInfo for a named image model."""
        return self._image_models.get(name)
    
    def get_motion_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get ModelInfo for a named motion model."""
        return self._motion_models.get(name)
    
    def load_image_model(
        self,
        model_name: str,
        variant: Optional[str] = None,
        force_reload: bool = False,
    ) -> nn.Module:
        """
        Load and activate a Pic Aliver image model.
        
        Automatically unloads previous model to save VRAM.
        
        Args:
            model_name: Name of the model (e.g., "Solarmix XL")
            variant: Model variant ("fp16", "int8")
            force_reload: Force reload even if cached
            
        Returns:
            Loaded pipeline module
        """
        model_info = self._image_models.get(model_name)
        if model_info is None:
            available = ", ".join(self.available_image_models)
            raise ValueError(
                f"Unknown image model '{model_name}'. Available: {available}"
            )
        
        if not force_reload and self._active_image_model and self._active_image_model.name == model_name:
            return self._image_pipeline
        
        if self._image_pipeline is not None and model_name != self._active_image_model.name:
            self._unload_image_model()
        
        print(f"[PicAliver] Loading image model: {model_name}")
        print(f"  Repository: {model_info.repo_id}")
        print(f"  Pipeline: {model_info.pipeline_type}")
        print(f"  VRAM: {model_info.vram_mb}MB")
        
        self._image_pipeline = self.loader.load_model(
            model_info,
            force_reload=force_reload,
            variant=variant,
        )
        
        self._active_image_model = model_info
        print(f"[PicAliver] Model active: {model_name}")
        
        return self._image_pipeline
    
    def load_motion_model(
        self,
        model_name: str,
        force_reload: bool = False,
    ) -> nn.Module:
        """
        Load and activate a Pic Aliver motion/animation model.
        
        Args:
            model_name: "Polaris" or "Lynx One"
            force_reload: Force reload even if cached
            
        Returns:
            Loaded motion pipeline
        """
        model_info = self._motion_models.get(model_name)
        if model_info is None:
            available = ", ".join(self.available_motion_models)
            raise ValueError(
                f"Unknown motion model '{model_name}'. Available: {available}"
            )
        
        if not force_reload and self._active_motion_model and self._active_motion_model.name == model_name:
            return self._motion_pipeline
        
        if self._motion_pipeline is not None:
            self._unload_motion_model()
        
        print(f"[PicAliver] Loading motion model: {model_name}")
        print(f"  Pipeline: {model_info.pipeline_type}")
        print(f"  Max frames: {model_info.max_frames} @ {model_info.fps}fps")
        
        self._motion_pipeline = self.loader.load_model(
            model_info,
            force_reload=force_reload,
        )
        
        self._active_motion_model = model_info
        print(f"[PicAliver] Motion model active: {model_name}")
        
        return self._motion_pipeline
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate an image using the active (or specified) Pic Aliver model.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Output width
            height: Output height
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            model_name: Optional model override
            progress_callback: Callable(current, total, stage)
            
        Returns:
            Image tensor (C, H, W)
        """
        if model_name:
            self.load_image_model(model_name)
        
        if self._image_pipeline is None:
            raise RuntimeError(
                "No image model loaded. Call load_image_model() first."
            )
        
        if seed is not None:
            torch.manual_seed(seed)
        
        model_info = self._active_image_model
        pipeline_type = model_info.pipeline_type if model_info else "sdxl"
        
        try:
            if hasattr(self._image_pipeline, "to"):
                self._image_pipeline = self._image_pipeline.to(self.device)
            
            pipe_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            if progress_callback:
                def _cb(step, _t, _l):
                    progress_callback(step + 1, num_inference_steps, f"Step {step+1}/{num_inference_steps}")
                    return False
                pipe_kwargs["callback"] = _cb
                pipe_kwargs["callback_steps"] = 1
            
            result = self._image_pipeline(**pipe_kwargs)
            
            if hasattr(result, "images"):
                image = result.images[0]
            elif isinstance(result, torch.Tensor):
                image = result
            else:
                image = result
            
            import numpy as np
            from PIL import Image
            
            if isinstance(image, Image.Image):
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            elif isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image)
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.float() / 255.0
            else:
                image_tensor = image
            
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.dim() == 3 and image_tensor.shape[0] in (1, 3, 4):
                pass
            elif image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"[PicAliver] Generation error: {e}")
            raise
    
    def generate_video(
        self,
        start_frame: torch.Tensor,
        end_frame: Optional[torch.Tensor] = None,
        prompt: str = "",
        num_frames: int = 24,
        fps: int = 8,
        motion_strength: float = 0.8,
        model_name: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Generate video frames using active motion model.
        
        Uses Lynx One for start/end animation or Polaris for looping.
        
        Args:
            start_frame: Starting frame (C, H, W)
            end_frame: Optional ending frame (Lynx One only)
            prompt: Motion description
            num_frames: Number of frames
            fps: Frames per second
            motion_strength: Motion intensity
            model_name: "Polaris" or "Lynx One"
            
        Returns:
            List of frame tensors
        """
        if model_name:
            self.load_motion_model(model_name)
        
        if self._motion_pipeline is None:
            raise RuntimeError(
                "No motion model loaded. Call load_motion_model() first."
            )
        
        pipeline_type = self._active_motion_model.pipeline_type if self._active_motion_model else "polaris"
        
        if pipeline_type == "lynx":
            frames = self._motion_pipeline.generate(
                start_frame=start_frame,
                end_frame=end_frame,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                motion_strength=motion_strength,
                **kwargs,
            )
        else:
            frames = self._motion_pipeline.generate_loop(
                image=start_frame,
                num_frames=num_frames,
                fps=fps,
                motion_strength=motion_strength,
                **kwargs,
            )
        
        return frames
    
    def _unload_image_model(self) -> None:
        """Unload current image model to free VRAM."""
        if self._image_pipeline is not None:
            model_name = self._active_image_model.name if self._active_image_model else "unknown"
            print(f"[PicAliver] Unloading image model: {model_name}")
            del self._image_pipeline
            self._image_pipeline = None
            self._active_image_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _unload_motion_model(self) -> None:
        """Unload current motion model to free VRAM."""
        if self._motion_pipeline is not None:
            model_name = self._active_motion_model.name if self._active_motion_model else "unknown"
            print(f"[PicAliver] Unloading motion model: {model_name}")
            del self._motion_pipeline
            self._motion_pipeline = None
            self._active_motion_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all(self) -> None:
        """Unload all models and clear VRAM."""
        self._unload_image_model()
        self._unload_motion_model()
        self.loader.unload_all()
    
    def list_models_by_style(self) -> Dict[str, List[str]]:
        """List available models grouped by style category."""
        categories = {
            "Premium": ["Solarmix XL", "Supernova XL", "Solarmix XL Premium"],
            "Experimental": ["Sirius", "Alpha Centauri Flux"],
            "Photoreal": ["DreamShaper XL", "Lite DreamShaper XL", "DreamShaper XL Turbo"],
            "Anime": ["BluePencil XL", "Lite BluePencil XL", "BluePencil XL LCM", "Anything", "Kotosmix"],
            "Furry": ["Compass Mix XL", "Lite Compass Mix XL", "Indigomix V", 
                      "YiffyMix", "YiffyMix XL", "YiffyMix v5", "YiffyMix v6",
                      "Fluffyrock", "Fluffyrock-Unbound"],
            "Flux": ["Alpha Centauri Flux", "Frosting Lane Flux"],
            "Motion": ["Polaris", "Lynx One"],
        }
        return categories
    
    def print_model_summary(self) -> None:
        """Print a formatted summary of all Pic Aliver models."""
        print("\n" + "=" * 60)
        print("PIC ALIVER - MODEL SUMMARY")
        print("=" * 60)
        
        by_style = self.list_models_by_style()
        for style, models in by_style.items():
            print(f"\n  [{style}]")
            for name in models:
                info = self._image_models.get(name) or self._motion_models.get(name)
                if info:
                    status = " [ACTIVE]" if (self._active_image_model and self._active_image_model.name == name) or \
                                              (self._active_motion_model and self._active_motion_model.name == name) else ""
                    print(f"    - {name} ({info.pipeline_type}, {info.vram_mb}MB){status}")
        
        print(f"\n  Total: {len(self._image_models)} image + {len(self._motion_models)} motion models")
        print("=" * 60)


def create_model_manager(
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> ModelManager:
    """Factory function to create a ModelManager."""
    torch_device = torch.device(device) if device else None
    return ModelManager(
        device=torch_device,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )
