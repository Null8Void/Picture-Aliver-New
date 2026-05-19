"""Core module for Image2Video AI system"""

from .config import Config
from .device import DeviceManager
from .pipeline import Image2VideoPipeline
from .model_registry import (
    MODEL_REGISTRY, ModelRegistry, ModelInfo,
    ModelCategory, ContentRating,
    get_registry, get_nsfw_models, get_safe_models,
)
from .model_loader import ModelLoader
from .model_manager import ModelManager, create_model_manager as create_model_manager
from .polaris_pipeline import PolarisPipeline, PolarisLoopOptimizer
from .lynx_pipeline import LynxOnePipeline, LynxFrameBlender
from .debug import (
    DebugManager, GenerationTrace, ModelLoadRecord,
    TimingEvent, VRAMSnapshot, trace_generation, debug as debug_instance,
)

__all__ = [
    "Config", "DeviceManager", "Image2VideoPipeline",
    "MODEL_REGISTRY", "ModelRegistry", "ModelInfo",
    "ModelCategory", "ContentRating",
    "get_registry", "get_nsfw_models", "get_safe_models",
    "ModelLoader",
    "ModelManager", "create_model_manager",
    "PolarisPipeline", "PolarisLoopOptimizer",
    "LynxOnePipeline", "LynxFrameBlender",
    "DebugManager", "GenerationTrace", "ModelLoadRecord",
    "TimingEvent", "VRAMSnapshot", "trace_generation", "debug_instance",
]