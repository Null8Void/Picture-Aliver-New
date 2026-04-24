"""
Picture-Aliver: AI-Powered Image to Video Animation System
"""

__version__ = "1.0.0"

from .image_loader import ImageLoader
from .depth_estimator import DepthEstimator, DepthResult
from .segmentation import SegmentationModule, SegmentationResult, ContentType
from .motion_generator import FurryMotionGenerator
from .motion_prompt import (
    MotionPromptParser, MotionPromptMapper, MotionParameters,
    MotionCategory, MotionIntensity
)
from .video_generator import VideoGenerator, VideoFrames, GenerationConfig
from .text_to_image import TextToImageGenerator, TextToVideoGenerator, T2IConfig
from .stabilizer import VideoStabilizer, StabilizationConfig
from .exporter import (
    VideoExporter, VideoFormat, VideoSpec, ExportOptions,
    QualityPreset, Codec, export_video
)
from .quality_control import (
    QualityController, QualityReport, QualityDetector, QualityIssue,
    assess_video_quality
)
from .gpu_optimization import (
    GPUOptimizer, VRAMTier, GPUConfig, ModelOffloader,
    optimize_model_for_device, print_benchmark_table
)
from .main import Pipeline, PipelineConfig, PipelineResult, run_pipeline

__all__ = [
    "ImageLoader",
    "DepthEstimator",
    "DepthResult",
    "SegmentationModule",
    "SegmentationResult",
    "ContentType",
    "FurryMotionGenerator",
    "MotionPromptParser",
    "MotionPromptMapper",
    "MotionParameters",
    "MotionCategory",
    "MotionIntensity",
    "VideoGenerator",
    "VideoFrames",
    "GenerationConfig",
    "TextToImageGenerator",
    "TextToVideoGenerator",
    "T2IConfig",
    "VideoStabilizer",
    "StabilizationConfig",
    "VideoExporter",
    "VideoFormat",
    "VideoSpec",
    "ExportOptions",
    "QualityPreset",
    "Codec",
    "export_video",
    "QualityController",
    "QualityReport",
    "QualityDetector",
    "QualityIssue",
    "assess_video_quality",
    "GPUOptimizer",
    "VRAMTier",
    "GPUConfig",
    "ModelOffloader",
    "optimize_model_for_device",
    "print_benchmark_table",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]