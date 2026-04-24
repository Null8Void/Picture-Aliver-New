"""
Picture-Aliver: AI-Powered Image to Video Animation System

A production-grade system for converting single images into coherent animated videos.
"""

from __future__ import annotations

from .image_loader import ImageLoader
from .depth_estimator import DepthEstimator, DepthResult
from .segmentation import SegmentationModule, SegmentationResult, ContentType
from .motion_generator import (
    MotionGenerator, MotionField, MotionMode, CameraTrajectory, FurryMotionGenerator
)
from .motion_prompt import (
    MotionPromptParser, MotionPromptMapper, MotionParameters, MotionCategory, MotionIntensity
)
from .video_generator import VideoGenerator, VideoFrames, GenerationConfig
from .text_to_image import TextToImageGenerator, TextToVideoGenerator, T2IConfig
from .stabilizer import VideoStabilizer, StabilizationConfig
from .exporter import (
    VideoExporter, ExportConfig, VideoFormat, VideoSpec, ExportOptions, 
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
from .main import PictureAliver

__version__ = "1.0.0"

__all__ = [
    "ImageLoader",
    "DepthEstimator",
    "DepthResult",
    "SegmentationModule",
    "SegmentationResult",
    "ContentType",
    "MotionGenerator",
    "MotionField",
    "MotionMode",
    "CameraTrajectory",
    "FurryMotionGenerator",
    "MotionPromptParser",
    "MotionPromptMapper",
    "MotionParameters",
    "MotionCategory",
    "MotionIntensity",
    "VideoGenerator",
    "VideoFrames",
    "GenerationConfig",
    "VideoStabilizer",
    "StabilizationConfig",
    "VideoExporter",
    "ExportConfig",
    "VideoFormat",
    "export_video",
    "PictureAliver",
]

__doc__ = """
Picture-Aliver: AI Image to Video Animation System
===================================================

Features:
- Depth estimation using MiDaS/ZoeDepth
- Semantic segmentation with content type detection
- Multi-mode motion generation (cinematic, zoom, pan, furry-specific)
- Natural language motion prompts ("gentle tail wag", "dramatic zoom")
- Diffusion-based video generation
- Temporal stabilization
- Multiple export formats (MP4, WebM, GIF)

Usage:
    from picture_aliver import PictureAliver
    
    system = PictureAliver()
    system.initialize()
    
    # Using motion mode
    metadata = system.process(
        image_path="input.jpg",
        output_path="output.mp4",
        motion_mode="cinematic",
        num_frames=24
    )
    
    # Using natural language motion prompt
    metadata = system.process(
        image_path="furry.png",
        output_path="animation.mp4",
        motion_prompt="gentle tail wag with breathing"
    )

Requirements:
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- FFmpeg (for video export)
"""