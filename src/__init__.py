"""Picture-Aliver: Image to Video AI System

Production-ready AI system for converting images to animated videos.
Supports Safe, Mature, and Unrestricted (NSFW) content generation modes.
"""

__version__ = "1.0.0"
__author__ = "AI Engineering Team"

from .core.pipeline import Image2VideoPipeline
from .core.config import Config
from .core.device import DeviceManager
from .core.model_registry import (
    MODEL_REGISTRY,
    ContentRating,
    ModelCategory,
    ModelInfo,
    get_registry,
    get_nsfw_models,
    get_safe_models,
)
from .core.model_loader import ModelLoader
from .core.config_extension import (
    PipelineConfig,
    ContentConfig,
    ModelConfig,
    GenerationMode,
    ConfigBuilder,
    create_content_config,
    create_model_config,
)

from .image2video import Image2Video

__all__ = [
    # Core
    "Image2VideoPipeline",
    "Image2Video",
    "Config",
    "DeviceManager",
    
    # Model Management
    "MODEL_REGISTRY",
    "ModelLoader",
    "ContentRating",
    "ModelCategory",
    "ModelInfo",
    "get_registry",
    "get_nsfw_models",
    "get_safe_models",
    
    # Configuration
    "PipelineConfig",
    "ContentConfig",
    "ModelConfig",
    "GenerationMode",
    "ConfigBuilder",
    "create_content_config",
    "create_model_config",
]