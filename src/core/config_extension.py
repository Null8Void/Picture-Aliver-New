"""Extended pipeline configuration with content rating support."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from ..core.model_registry import ContentRating, ModelCategory


class GenerationMode(Enum):
    """Generation mode based on content rating."""
    SAFE = "safe"           # General/safe content only
    MATURE = "mature"        # Mature content with some restrictions
    UNRESTRICTED = "nsfw"    # No content restrictions


class MotionStyle(Enum):
    """Available motion styles."""
    CINEMATIC = "cinematic"
    SUBTLE = "subtle"
    ENVIRONMENTAL = "environmental"
    ORBITAL = "orbital"
    ZOOM_IN = "zoom-in"
    ZOOM_OUT = "zoom-out"
    PAN_LEFT = "pan-left"
    PAN_RIGHT = "pan-right"
    CUSTOM = "custom"


@dataclass
class ContentConfig:
    """Configuration for content rating and model selection.
    
    This controls which models are available and how content is generated.
    """
    mode: GenerationMode = GenerationMode.SAFE
    
    allow_nsfw_models: bool = False
    allow_mature_models: bool = True
    
    content_filter_strength: float = 1.0
    
    prompt_safety_check: bool = True
    output_safety_check: bool = False
    
    warn_on_mature: bool = True
    require_consent: bool = False
    
    @property
    def content_rating(self) -> ContentRating:
        """Get current content rating based on mode."""
        if self.mode == GenerationMode.UNRESTRICTED:
            return ContentRating.NSFW
        elif self.mode == GenerationMode.MATURE:
            return ContentRating.MATURE
        return ContentRating.SAFE
    
    @property
    def is_unrestricted(self) -> bool:
        """Check if unrestricted mode is enabled."""
        return self.mode == GenerationMode.UNRESTRICTED
    
    @property
    def is_safe(self) -> bool:
        """Check if safe mode is enabled."""
        return self.mode == GenerationMode.SAFE


@dataclass
class ModelConfig:
    """Configuration for model selection."""
    
    i2v_model: str = "SVD (Stable Video Diffusion)"
    depth_model: str = "ZoeDepth"
    segmentation_model: str = "SAM-ViT-Large"
    motion_model: str = "AnimateDiff-Motion-Adapter-SDXL"
    interpolation_model: str = "RIFE-v4"
    
    auto_select_best: bool = True
    
    prefer_quality_over_speed: bool = True
    
    max_vram_mb: int = 12000
    
    quantization: Optional[str] = None
    
    def get_model_for_category(self, category: ModelCategory) -> str:
        """Get model name for a category."""
        mapping = {
            ModelCategory.I2V: self.i2v_model,
            ModelCategory.DEPTH: self.depth_model,
            ModelCategory.SEGMENTATION: self.segmentation_model,
            ModelCategory.MOTION: self.motion_model,
            ModelCategory.INTERPOLATION: self.interpolation_model,
        }
        return mapping.get(category, "")


@dataclass
class PipelineConfig:
    """Extended pipeline configuration."""
    
    content: ContentConfig = field(default_factory=ContentConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    
    enable_depth: bool = True
    enable_segmentation: bool = True
    enable_motion: bool = True
    enable_consistency: bool = True
    use_3d_effects: bool = True
    enable_interpolation: bool = True
    
    motion_mode: MotionStyle = MotionStyle.CINEMATIC
    motion_strength: float = 0.8
    
    num_frames: int = 24
    fps: int = 8
    resolution: tuple = (512, 512)
    
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    
    output_format: str = "mp4"
    verbose: bool = True
    
    seed: Optional[int] = None
    
    def get_resolution_for_quality(self) -> tuple:
        """Get target resolution based on quality level."""
        if self.resolution[0] >= 768:
            return self.resolution
        return self.resolution
    
    def requires_high_vram(self) -> bool:
        """Check if config requires high VRAM."""
        return any([
            self.resolution[0] >= 768 or self.resolution[1] >= 768,
            self.num_frames >= 32,
            self.enable_interpolation,
        ])
    
    def get_estimated_vram_mb(self) -> int:
        """Estimate VRAM requirements."""
        base = 2000  # Depth + segmentation
        
        resolution = self.resolution
        if resolution[0] >= 1024:
            base += 4000
        elif resolution[0] >= 768:
            base += 2000
        else:
            base += 1000
        
        if self.num_frames >= 24:
            base += 2000
        elif self.num_frames >= 16:
            base += 1000
        
        if self.enable_interpolation:
            base += 1000
        
        return base


def create_content_config(
    mode: str = "safe",
    **kwargs
) -> ContentConfig:
    """Create content configuration from mode string."""
    mode_map = {
        "safe": GenerationMode.SAFE,
        "general": GenerationMode.SAFE,
        "mature": GenerationMode.MATURE,
        "nsfw": GenerationMode.UNRESTRICTED,
        "unrestricted": GenerationMode.UNRESTRICTED,
    }
    
    gen_mode = mode_map.get(mode.lower(), GenerationMode.SAFE)
    
    config = ContentConfig(mode=gen_mode)
    
    if gen_mode == GenerationMode.UNRESTRICTED:
        config.allow_nsfw_models = True
        config.allow_mature_models = True
        config.prompt_safety_check = False
        config.output_safety_check = False
        config.warn_on_mature = False
    elif gen_mode == GenerationMode.MATURE:
        config.allow_nsfw_models = False
        config.allow_mature_models = True
        config.warn_on_mature = True
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_model_config(
    rating: ContentRating,
    vram_mb: int,
    quality_preference: str = "high"
) -> ModelConfig:
    """Create model configuration based on content rating and hardware."""
    
    config = ModelConfig()
    config.max_vram_mb = vram_mb
    config.prefer_quality_over_speed = quality_preference == "high"
    
    if rating == ContentRating.SAFE:
        config.i2v_model = "SVD (Stable Video Diffusion)"
        config.depth_model = "ZoeDepth"
        config.segmentation_model = "SAM-ViT-Large"
        config.motion_model = "AnimateDiff-Motion-Adapter-SDXL"
        config.interpolation_model = "RIFE-v4"
    
    elif rating == ContentRating.MATURE:
        config.i2v_model = "ZeroScope-v2"
        config.depth_model = "MiDaS v3.1"
        config.segmentation_model = "SAM-ViT-Base"
        config.motion_model = "AnimateDiff-Motion-Adapter"
        config.interpolation_model = "RIFE-v4"
    
    elif rating == ContentRating.NSFW:
        if vram_mb >= 12000:
            config.i2v_model = "Open-SVD"
            config.motion_model = "AnimateDiff-Motion-Adapter"
        elif vram_mb >= 8000:
            config.i2v_model = "OpenGIF-Unrestricted"
        else:
            config.i2v_model = "ZeroScope-v2"
        
        config.depth_model = "MiDaS v3.1"
        config.segmentation_model = "SAM-ViT-Base"
        config.interpolation_model = "RIFE-v4"
    
    if vram_mb < 6000:
        config.segmentation_model = "MobileSAM"
        config.interpolation_model = ""
    
    return config


class ConfigBuilder:
    """Builder for creating pipeline configurations."""
    
    def __init__(self):
        self._content = ContentConfig()
        self._models = ModelConfig()
        self._pipeline = {}
    
    def set_mode(self, mode: str) -> "ConfigBuilder":
        """Set generation mode."""
        self._content = create_content_config(mode)
        return self
    
    def set_vram(self, vram_mb: int) -> "ConfigBuilder":
        """Set available VRAM."""
        self._models.max_vram_mb = vram_mb
        return self
    
    def set_quality(self, quality: str) -> "ConfigBuilder":
        """Set quality preference."""
        self._models.prefer_quality_over_speed = quality == "high"
        return self
    
    def set_frames(self, num_frames: int) -> "ConfigBuilder":
        """Set number of frames."""
        self._pipeline["num_frames"] = num_frames
        return self
    
    def set_resolution(self, resolution: tuple) -> "ConfigBuilder":
        """Set output resolution."""
        self._pipeline["resolution"] = resolution
        return self
    
    def enable_depth(self, enable: bool = True) -> "ConfigBuilder":
        """Enable or disable depth estimation."""
        self._pipeline["enable_depth"] = enable
        return self
    
    def enable_segmentation(self, enable: bool = True) -> "ConfigBuilder":
        """Enable or disable segmentation."""
        self._pipeline["enable_segmentation"] = enable
        return self
    
    def enable_interpolation(self, enable: bool = True) -> "ConfigBuilder":
        """Enable or disable frame interpolation."""
        self._pipeline["enable_interpolation"] = enable
        return self
    
    def set_motion_style(self, style: str) -> "ConfigBuilder":
        """Set motion style."""
        style_map = {
            "cinematic": MotionStyle.CINEMATIC,
            "subtle": MotionStyle.SUBTLE,
            "environmental": MotionStyle.ENVIRONMENTAL,
            "orbital": MotionStyle.ORBITAL,
            "zoom-in": MotionStyle.ZOOM_IN,
            "zoom-out": MotionStyle.ZOOM_OUT,
            "pan-left": MotionStyle.PAN_LEFT,
            "pan-right": MotionStyle.PAN_RIGHT,
        }
        self._pipeline["motion_mode"] = style_map.get(style, MotionStyle.CINEMATIC)
        return self
    
    def build(self) -> PipelineConfig:
        """Build final configuration."""
        config = PipelineConfig(
            content=self._content,
            models=self._models,
            **self._pipeline
        )
        
        rating = self._content.content_rating
        if self._models.auto_select_best:
            models = create_model_config(
                rating=rating,
                vram_mb=self._models.max_vram_mb,
                quality_preference="high" if self._models.prefer_quality_over_speed else "fast"
            )
            config.models = models
        
        return config
    
    def build_safe(self) -> PipelineConfig:
        """Build safe configuration."""
        return self.set_mode("safe").build()
    
    def build_mature(self) -> PipelineConfig:
        """Build mature configuration."""
        return self.set_mode("mature").build()
    
    def build_unrestricted(self) -> PipelineConfig:
        """Build unrestricted configuration."""
        return self.set_mode("nsfw").build()