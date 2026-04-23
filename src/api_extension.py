"""Extended API with unrestricted content support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch

from .core.pipeline import Image2VideoPipeline, PipelineResult
from .core.config_extension import (
    PipelineConfig,
    ContentConfig,
    ModelConfig,
    GenerationMode,
    ContentRating,
    ConfigBuilder,
    create_content_config,
    create_model_config,
)
from .core.model_registry import MODEL_REGISTRY, ModelInfo, ContentRating
from .core.model_loader import ModelLoader
from .core.device import get_torch_device, DeviceManager


class PictureAliver:
    """High-level API for image-to-video generation.
    
    Supports three content modes:
    - SAFE: General models only, safety checks enabled
    - MATURE: Some unrestricted models, moderate content
    - NSFW/UNRESTRICTED: Full model access, no restrictions
    
    Example usage:
        >>> from picture_aliver import PictureAliver
        >>> 
        >>> # Safe mode (default)
        >>> app = PictureAliver(mode="safe")
        >>> result = app.convert("photo.jpg", "video.mp4")
        >>> 
        >>> # Unrestricted mode
        >>> app = PictureAliver(mode="nsfw")
        >>> result = app.convert("photo.jpg", "video.mp4")
    """
    
    def __init__(
        self,
        mode: str = "safe",
        device: Optional[str] = None,
        vram_mb: Optional[int] = None,
        quality: str = "high",
        auto_select_models: bool = True,
        **kwargs
    ):
        """Initialize PictureAliver.
        
        Args:
            mode: Content mode ('safe', 'mature', 'nsfw', 'unrestricted')
            device: Compute device ('cuda', 'cpu', 'mps')
            vram_mb: Available VRAM in MB (auto-detected if None)
            quality: Quality preference ('high', 'medium', 'fast')
            auto_select_models: Automatically select best models
            **kwargs: Additional configuration options
        """
        self.mode = mode.lower()
        self._setup_mode()
        
        self.device = get_torch_device(device) if device else None
        
        if vram_mb is None:
            vram_mb = self._detect_vram_mb()
        self.vram_mb = vram_mb
        
        self.quality = quality
        
        self.model_loader = ModelLoader(
            default_rating=self.content_rating,
            device=self.device,
        )
        
        self.config = self._build_config(auto_select_models, **kwargs)
        
        self.pipeline = Image2VideoPipeline(
            config=self.config,
            device=self.device
        )
        
        self._initialized = False
    
    def _setup_mode(self):
        """Setup content mode and rating."""
        mode_map = {
            "safe": (GenerationMode.SAFE, ContentRating.SAFE),
            "general": (GenerationMode.SAFE, ContentRating.SAFE),
            "mature": (GenerationMode.MATURE, ContentRating.MATURE),
            "nsfw": (GenerationMode.UNRESTRICTED, ContentRating.NSFW),
            "unrestricted": (GenerationMode.UNRESTRICTED, ContentRating.NSFW),
        }
        
        gen_mode, content_rating = mode_map.get(
            self.mode, 
            (GenerationMode.SAFE, ContentRating.SAFE)
        )
        
        self.generation_mode = gen_mode
        self.content_rating = content_rating
    
    def _detect_vram_mb(self) -> int:
        """Detect available VRAM."""
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
        return 8000  # Assume 8GB for CPU
    
    def _build_config(self, auto_select: bool, **kwargs) -> PipelineConfig:
        """Build pipeline configuration."""
        content_config = create_content_config(mode=self.mode)
        
        model_config = None
        if auto_select:
            model_config = create_model_config(
                rating=self.content_rating,
                vram_mb=self.vram_mb,
                quality_preference=self.quality
            )
        else:
            model_config = ModelConfig()
        
        config = PipelineConfig(
            content=content_config,
            models=model_config,
            **kwargs
        )
        
        return config
    
    def initialize(self) -> None:
        """Initialize the pipeline."""
        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True
    
    def convert(
        self,
        image: Union[str, Path],
        output_path: Union[str, Path],
        prompt: str = "",
        negative_prompt: str = "",
        **kwargs
    ) -> PipelineResult:
        """Convert an image to video.
        
        Args:
            image: Input image path
            output_path: Output video path
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            **kwargs: Additional generation parameters
            
        Returns:
            PipelineResult with generated video
        """
        if not self._initialized:
            self.initialize()
        
        return self.pipeline.process(
            image=str(image),
            output_path=str(output_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs
        )
    
    def convert_with_motion(
        self,
        image: Union[str, Path],
        output_path: Union[str, Path],
        motion_style: str = "cinematic",
        **kwargs
    ) -> PipelineResult:
        """Convert with specific motion style.
        
        Args:
            image: Input image path
            output_path: Output video path
            motion_style: Motion style ('cinematic', 'subtle', 'environmental', etc.)
            **kwargs: Additional parameters
        """
        motion_modes = {
            "cinematic": "cinematic",
            "subtle": "subtle", 
            "environmental": "environmental",
            "orbital": "orbital",
            "zoom-in": "zoom-in",
            "zoom-out": "zoom-out",
            "pan-left": "pan-left",
            "pan-right": "pan-right",
        }
        
        style = motion_modes.get(motion_style, "cinematic")
        kwargs["motion_mode"] = style
        
        return self.convert(image, output_path, **kwargs)
    
    def set_mode(self, mode: str) -> "PictureAliver":
        """Change content mode (returns new instance).
        
        Args:
            mode: New mode ('safe', 'mature', 'nsfw')
            
        Returns:
            New PictureAliver instance with updated mode
        """
        return PictureAliver(
            mode=mode,
            vram_mb=self.vram_mb,
            quality=self.quality,
        )
    
    def get_available_models(self) -> Dict[str, list]:
        """Get available models for current content rating.
        
        Returns:
            Dictionary of models by category
        """
        return {
            "i2v": MODEL_REGISTRY.get_by_category(
                __import__('src.core.model_registry', fromlist=['ModelCategory']).ModelCategory.I2V,
                rating=self.content_rating
            ),
            "depth": MODEL_REGISTRY.get_by_category(
                __import__('src.core.model_registry', fromlist=['ModelCategory']).ModelCategory.DEPTH,
                rating=self.content_rating
            ),
            "segmentation": MODEL_REGISTRY.get_by_category(
                __import__('src.core.model_registry', fromlist=['ModelCategory']).ModelCategory.SEGMENTATION,
                rating=self.content_rating
            ),
        }
    
    def set_model(self, category: str, model_name: str) -> "PictureAliver":
        """Set a specific model for a category.
        
        Args:
            category: Model category ('i2v', 'depth', 'segmentation')
            model_name: Name of the model from registry
            
        Returns:
            Self for chaining
        """
        model_map = {
            "i2v": "i2v_model",
            "depth": "depth_model",
            "segmentation": "segmentation_model",
            "motion": "motion_model",
            "interpolation": "interpolation_model",
        }
        
        attr = model_map.get(category.lower())
        if attr:
            setattr(self.config.models, attr, model_name)
        
        return self
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        self.pipeline.clear_cache()
        self.model_loader.unload_all()
    
    def get_vram_info(self) -> Dict[str, int]:
        """Get VRAM information."""
        return self.model_loader.get_vram_info()
    
    def __repr__(self) -> str:
        return (
            f"PictureAliver(mode={self.mode}, rating={self.content_rating.value}, "
            f"vram={self.vram_mb}MB, quality={self.quality})"
        )


class PictureAliverBuilder:
    """Builder for creating PictureAliver with custom configuration.
    
    Example:
        >>> app = PictureAliverBuilder() \\
        ...     .mode("nsfw") \\
        ...     .vram(12000) \\
        ...     .frames(24) \\
        ...     .resolution(768) \\
        ...     .motion("cinematic") \\
        ...     .build()
    """
    
    def __init__(self):
        self._mode = "safe"
        self._vram = None
        self._quality = "high"
        self._frames = 24
        self._fps = 8
        self._resolution = 512
        self._motion = "cinematic"
        self._i2v_model = None
        self._depth_model = None
        self._seg_model = None
    
    def mode(self, mode: str) -> "PictureAliverBuilder":
        """Set content mode."""
        self._mode = mode
        return self
    
    def vram(self, mb: int) -> "PictureAliverBuilder":
        """Set available VRAM."""
        self._vram = mb
        return self
    
    def quality(self, quality: str) -> "PictureAliverBuilder":
        """Set quality preference."""
        self._quality = quality
        return self
    
    def frames(self, num: int) -> "PictureAliverBuilder":
        """Set number of frames."""
        self._frames = num
        return self
    
    def fps(self, value: int) -> "PictureAliverBuilder":
        """Set frames per second."""
        self._fps = value
        return self
    
    def resolution(self, size: int) -> "PictureAliverBuilder":
        """Set resolution (square)."""
        self._resolution = size
        return self
    
    def motion(self, style: str) -> "PictureAliverBuilder":
        """Set motion style."""
        self._motion = style
        return self
    
    def i2v_model(self, name: str) -> "PictureAliverBuilder":
        """Set I2V model."""
        self._i2v_model = name
        return self
    
    def depth_model(self, name: str) -> "PictureAliverBuilder":
        """Set depth model."""
        self._depth_model = name
        return self
    
    def seg_model(self, name: str) -> "PictureAliverBuilder":
        """Set segmentation model."""
        self._seg_model = name
        return self
    
    def build(self) -> PictureAliver:
        """Build PictureAliver instance."""
        app = PictureAliver(
            mode=self._mode,
            vram_mb=self._vram,
            quality=self._quality,
            num_frames=self._frames,
            fps=self._fps,
            resolution=(self._resolution, self._resolution),
            motion_mode=self._motion,
        )
        
        if self._i2v_model:
            app.set_model("i2v", self._i2v_model)
        if self._depth_model:
            app.set_model("depth", self._depth_model)
        if self._seg_model:
            app.set_model("segmentation", self._seg_model)
        
        return app


def create_app(
    mode: str = "safe",
    **kwargs
) -> PictureAliver:
    """Factory function to create PictureAliver instance.
    
    Args:
        mode: Content mode ('safe', 'mature', 'nsfw')
        **kwargs: Additional parameters
        
    Returns:
        Configured PictureAliver instance
    """
    return PictureAliver(mode=mode, **kwargs)


def create_safe_app(**kwargs) -> PictureAliver:
    """Create safe mode app."""
    return PictureAliver(mode="safe", **kwargs)


def create_mature_app(**kwargs) -> PictureAliver:
    """Create mature content app."""
    return PictureAliver(mode="mature", **kwargs)


def create_unrestricted_app(**kwargs) -> PictureAliver:
    """Create unrestricted/NSFW app."""
    return PictureAliver(mode="nsfw", **kwargs)


def convert_image(
    image: Union[str, Path],
    output_path: Union[str, Path],
    mode: str = "safe",
    **kwargs
) -> PipelineResult:
    """Convenience function to convert a single image.
    
    Example:
        >>> result = convert_image("photo.jpg", "video.mp4", mode="nsfw")
    """
    app = PictureAliver(mode=mode)
    return app.convert(image, output_path, **kwargs)


__all__ = [
    "PictureAliver",
    "PictureAliverBuilder",
    "create_app",
    "create_safe_app",
    "create_mature_app",
    "create_unrestricted_app",
    "convert_image",
    "PipelineConfig",
    "ContentConfig",
    "ModelConfig",
    "ContentRating",
    "GenerationMode",
    "MODEL_REGISTRY",
]