"""Main pipeline orchestration for Image2Video AI system."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from PIL import Image

from ..core.config import Config
from ..core.device import DeviceManager, get_torch_device

from ..modules.depth import DepthEstimator, DepthMap
from ..modules.segmentation import Segmentor, SegmentationMask
from ..modules.motion import (
    FlowEstimator, FlowField, MotionMagnitude,
    CameraMotionGenerator, CameraTrajectory,
    ObjectMotionGenerator, ObjectMotion,
    EnvironmentalMotionGenerator, EnvironmentalEffect,
    DepthParallaxGenerator, ParallaxConfig,
    PhysicsMotionGenerator, PhysicsConfig,
    FurryMotionGenerator, FurryMotionType
)
from ..modules.generation import (
    VideoGenerator,
    TemporalConsistencyManager,
    VideoFrames,
    GenerationConfig,
    MotionGuidance,
    DepthConditioner, DepthConditionConfig,
    ControlNetGuidance, ControlNetConfig,
    LatentConsistencyManager, LatentConsistencyConfig,
    OpticalFlowStabilizer, StabilizationConfig,
    FrameInterpolator, InterpolationConfig,
    ArtifactReducer, ArtifactReductionConfig,
    ContentAnalyzer, ContentType, DynamicPipelineAdapter,
    FurryModelRegistry, FurryModelConfig, get_furry_model
)


@dataclass
class PipelineConfig:
    """Configuration for the main pipeline.
    
    Attributes:
        enable_depth: Enable depth estimation
        enable_segmentation: Enable segmentation
        enable_motion: Enable motion estimation
        enable_consistency: Enable temporal consistency
        use_3d_effects: Use depth-based effects
        motion_mode: Motion mode (auto, camera, flow, keypoints)
        output_format: Output video format
        verbose: Verbose logging
        enable_artifact_reduction: Enable artifact reduction
        enable_furry_support: Enable furry content support
        enable_nsfw: Enable unrestricted content generation
        content_filter: Content filtering level (none, basic, strict)
    """
    enable_depth: bool = True
    enable_segmentation: bool = True
    enable_motion: bool = True
    enable_consistency: bool = True
    use_3d_effects: bool = True
    motion_mode: str = "auto"
    output_format: str = "mp4"
    verbose: bool = True
    enable_artifact_reduction: bool = True
    enable_furry_support: bool = True
    enable_nsfw: bool = False
    content_filter: str = "none"
    
    def __post_init__(self):
        valid_modes = ["auto", "camera", "flow", "keyframes", "zoom", "pan"]
        if self.motion_mode not in valid_modes:
            self.motion_mode = "auto"
        valid_filters = ["none", "basic", "strict"]
        if self.content_filter not in valid_filters:
            self.content_filter = "none"


@dataclass
class PipelineResult:
    """Result from the pipeline execution.
    
    Attributes:
        video_frames: Generated video frames
        depth_map: Estimated depth map
        segmentation: Segmentation result
        flow_field: Estimated flow field
        metadata: Additional metadata
        processing_time: Total processing time
        content_analysis: Content analysis result
        motion_generators: Active motion generators
    """
    video_frames: VideoFrames
    depth_map: Optional[DepthMap] = None
    segmentation: Optional[SegmentationMask] = None
    flow_field: Optional[FlowField] = None
    metadata: Dict = field(default_factory=dict)
    processing_time: float = 0.0
    content_analysis: Optional[Any] = None
    motion_generators: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_frames(self) -> int:
        return len(self.video_frames)
    
    @property
    def fps(self) -> int:
        return self.metadata.get("fps", 8)
    
    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


class Image2VideoPipeline:
    """Main pipeline orchestrating the entire image-to-video conversion.
    
    This pipeline:
    1. Analyzes the input image (depth, segmentation, semantics)
    2. Estimates motion patterns (optical flow, camera motion)
    3. Generates video frames using diffusion models
    4. Ensures temporal consistency throughout
    5. Applies post-processing and effects
    6. Outputs a coherent video
    7. Supports dynamic content adaptation (human, furry, landscape, etc.)
    8. Reduces artifacts through architectural solutions
    9. Supports unrestricted content generation for mature content
    
    Args:
        config: Pipeline configuration
        device: Compute device
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or PipelineConfig()
        self.device_manager = DeviceManager()
        self.device = device or get_torch_device()
        
        self.depth_estimator: Optional[DepthEstimator] = None
        self.segmentor: Optional[Segmentor] = None
        self.flow_estimator: Optional[FlowEstimator] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.temporal_manager: Optional[TemporalConsistencyManager] = None
        
        self.camera_motion: Optional[CameraMotionGenerator] = None
        self.object_motion: Optional[ObjectMotionGenerator] = None
        self.env_motion: Optional[EnvironmentalMotionGenerator] = None
        self.parallax: Optional[DepthParallaxGenerator] = None
        self.physics_motion: Optional[PhysicsMotionGenerator] = None
        self.furry_motion: Optional[FurryMotionGenerator] = None
        
        self.depth_conditioner: Optional[DepthConditioner] = None
        self.controlnet_guidance: Optional[ControlNetGuidance] = None
        self.latent_consistency: Optional[LatentConsistencyManager] = None
        self.flow_stabilizer: Optional[OpticalFlowStabilizer] = None
        self.frame_interpolator: Optional[FrameInterpolator] = None
        self.artifact_reducer: Optional[ArtifactReducer] = None
        
        self.content_analyzer: Optional[ContentAnalyzer] = None
        self.pipeline_adapter: Optional[DynamicPipelineAdapter] = None
        self.furry_models: Optional[FurryModelRegistry] = None
        
        self._initialized = False
        self._log_history: List[str] = []
        self._content_type: Optional[ContentType] = None
    
    def initialize(
        self,
        init_depth: bool = True,
        init_segmentation: bool = True,
        init_motion: bool = True,
        init_generation: bool = True
    ) -> None:
        """Initialize all pipeline components.
        
        Args:
            init_depth: Initialize depth estimator
            init_segmentation: Initialize segmentor
            init_motion: Initialize flow estimator
            init_generation: Initialize video generator
        """
        if self._initialized:
            return
        
        if self.config.verbose:
            print("Initializing Image2Video Pipeline...")
        
        if init_depth and self.config.enable_depth:
            self._log("Initializing depth estimator...")
            self.depth_estimator = DepthEstimator(device=self.device)
            self.depth_estimator.initialize()
        
        if init_segmentation and self.config.enable_segmentation:
            self._log("Initializing segmentor...")
            self.segmentor = Segmentor(device=self.device)
            self.segmentor.initialize()
        
        if init_motion and self.config.enable_motion:
            self._log("Initializing motion estimator...")
            self.flow_estimator = FlowEstimator(device=self.device)
            self.flow_estimator.initialize()
            
            self._log("Initializing camera motion generator...")
            self.camera_motion = CameraMotionGenerator(device=self.device)
            
            self._log("Initializing object motion generator...")
            self.object_motion = ObjectMotionGenerator(device=self.device)
            
            self._log("Initializing environmental motion generator...")
            self.env_motion = EnvironmentalMotionGenerator(device=self.device)
            
            self._log("Initializing depth parallax generator...")
            self.parallax = DepthParallaxGenerator(device=self.device)
            
            self._log("Initializing physics motion generator...")
            self.physics_motion = PhysicsMotionGenerator(device=self.device)
            
            if self.config.enable_furry_support:
                self._log("Initializing furry motion generator...")
                self.furry_motion = FurryMotionGenerator(device=self.device)
        
        if init_generation:
            self._log("Initializing video generator...")
            gen_config = GenerationConfig(
                num_frames=24,
                fps=8,
                motion_mode=self.config.motion_mode
            )
            self.video_generator = VideoGenerator(
                config=gen_config,
                device=self.device
            )
            
            if self.depth_estimator:
                self.video_generator.set_depth_estimator(self.depth_estimator)
            if self.segmentor:
                self.video_generator.set_segmentor(self.segmentor)
            
            if self.config.enable_artifact_reduction:
                self._log("Initializing artifact reduction system...")
                self.depth_conditioner = DepthConditioner(
                    device=self.device,
                    model_type="zoedepth"
                )
                self.controlnet_guidance = ControlNetGuidance(device=self.device)
                self.latent_consistency = LatentConsistencyManager(
                    num_frames=24,
                    device=self.device
                )
                self.flow_stabilizer = OpticalFlowStabilizer(device=self.device)
                self.frame_interpolator = FrameInterpolator(
                    device=self.device,
                    model_type="rife"
                )
                self.artifact_reducer = ArtifactReducer(
                    device=self.device,
                    enable_depth_conditioning=True,
                    enable_controlnet=True,
                    enable_latent_consistency=True,
                    enable_flow_stabilization=True,
                    enable_frame_interpolation=True
                )
        
        if self.config.enable_consistency:
            self.temporal_manager = TemporalConsistencyManager(
                num_frames=24,
                device=self.device
            )
        
        if self.config.enable_furry_support:
            self._log("Initializing content analyzer...")
            self.content_analyzer = ContentAnalyzer(
                device=self.device,
                nsfw_enabled=self.config.enable_nsfw
            )
            self.pipeline_adapter = DynamicPipelineAdapter(
                device=self.device,
                enable_nsfw=self.config.enable_nsfw
            )
            self.furry_models = FurryModelRegistry()
        
        self._initialized = True
        self._log("Pipeline initialization complete.")
    
    def process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: int = 24,
        fps: int = 8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        motion_strength: float = 0.8,
        output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        return_intermediate: bool = False,
        motion_preset: Optional[str] = None,
        content_type_hint: Optional[str] = None
    ) -> PipelineResult:
        """Process an image to generate a video.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            num_frames: Number of frames to generate
            fps: Frames per second
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of denoising steps
            motion_strength: Strength of motion (0-1)
            output_path: Path to save output video
            seed: Random seed for reproducibility
            return_intermediate: Return intermediate results
            motion_preset: Motion preset (cinematic, zoom, pan, subtle, furry_tail, etc.)
            content_type_hint: Hint for content type (human, furry, landscape, etc.)
            
        Returns:
            PipelineResult with generated video and metadata
        """
        start_time = time.time()
        
        if not self._initialized:
            self.initialize()
        
        image = self._load_image(image)
        
        self._log(f"Processing image of shape {image.shape}...")
        
        depth_map = None
        segmentation = None
        flow_field = None
        motion_guidance = None
        content_analysis = None
        motion_generators = {}
        
        if self.content_analyzer is not None:
            self._log("Analyzing content...")
            content_analysis = self.content_analyzer.analyze(image)
            self._content_type = content_analysis.content_type
            self._log(f"Detected content type: {content_analysis.content_type.value}")
            
            if content_type_hint:
                try:
                    hint_type = ContentType(content_type_hint.lower())
                    content_analysis.content_type = hint_type
                    self._content_type = hint_type
                except ValueError:
                    pass
        
        if self.config.enable_depth and self.depth_estimator is not None:
            self._log("Estimating depth...")
            depth_map = self.depth_estimator.estimate(image, return_normal=False)
        
        if self.config.enable_segmentation and self.segmentor is not None:
            self._log("Performing segmentation...")
            segmentation = self.segmentor.segment_with_prompts(
                image,
                prompt_type="automatic",
                points_per_side=32
            )
        
        if self.config.enable_motion and self.camera_motion is not None:
            self._log("Generating motion...")
            flow_field = self._generate_motion(
                image, num_frames, motion_preset, content_analysis, depth_map
            )
            
            if self.camera_motion:
                motion_generators["camera"] = self.camera_motion
            if self.object_motion:
                motion_generators["object"] = self.object_motion
            if self.env_motion:
                motion_generators["environment"] = self.env_motion
            if self.parallax:
                motion_generators["parallax"] = self.parallax
            if self.physics_motion:
                motion_generators["physics"] = self.physics_motion
            if self.furry_motion and self._content_type in [ContentType.FURRY, ContentType.ANIMAL]:
                motion_generators["furry"] = self.furry_motion
        
        if self.depth_estimator is not None:
            motion_guidance = MotionGuidance(
                flow_field=flow_field,
                depth_map=depth_map,
                segmentation=segmentation
            )
        
        self._log(f"Generating {num_frames} frames...")
        
        selected_model = None
        if self.furry_models and self._content_type in [ContentType.FURRY, ContentType.ANIMAL]:
            furry_config = self.furry_models.get_model_config(
                self._content_type.value,
                nsfw=self.config.enable_nsfw
            )
            selected_model = furry_config.get("model_name", "dreamshaper")
            self._log(f"Using furry model: {selected_model}")
        
        gen_config = GenerationConfig(
            num_frames=num_frames,
            fps=fps,
            prompt=prompt or self._generate_prompt(image, content_analysis),
            negative_prompt=negative_prompt or self._get_default_negative_prompt(content_analysis),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            motion_strength=motion_strength
        )
        
        if self.video_generator is not None:
            self.video_generator.config = gen_config
            video_frames = self.video_generator.generate(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                motion_guidance=motion_guidance,
                seed=seed
            )
        else:
            video_frames = self._generate_simple_video(image, num_frames)
        
        if self.artifact_reducer is not None and self.config.enable_artifact_reduction:
            self._log("Applying artifact reduction...")
            video_tensor = video_frames.to_video()
            
            if depth_map is not None and self.depth_conditioner is not None:
                if isinstance(depth_map, DepthMap):
                    depth = depth_map.depth
                else:
                    depth = depth_map
                video_tensor = self.artifact_reducer.reduce(
                    video_tensor,
                    depth=depth,
                    segmentation=segmentation
                )
            else:
                video_tensor = self.artifact_reducer.reduce(
                    video_tensor
                )
            
            video_frames = VideoFrames()
            for frame in video_tensor:
                video_frames.append(frame)
        
        if self.temporal_manager is not None:
            self._log("Applying temporal consistency...")
            video_tensor = video_frames.to_video()
            
            video_tensor = self.temporal_manager.temporal_smooth(video_tensor)
            video_tensor = self.temporal_manager.reduce_flickering(video_tensor)
            
            video_frames = VideoFrames()
            for frame in video_tensor:
                video_frames.append(frame)
        
        if self.config.use_3d_effects and depth_map is not None:
            self._log("Applying depth effects...")
            if isinstance(depth_map, DepthMap):
                depth = depth_map.depth
            else:
                depth = depth_map
            video_frames = self.video_generator.apply_depth_effects(
                video_frames, depth
            )
        
        processing_time = time.time() - start_time
        self._log(f"Processing complete in {processing_time:.2f}s")
        
        if output_path is not None:
            self._log(f"Saving video to {output_path}...")
            self.save_video(video_frames, output_path, fps=fps)
        
        metadata = {
            "fps": fps,
            "num_frames": num_frames,
            "duration": num_frames / fps,
            "processing_time": processing_time,
            "motion_mode": self.config.motion_mode,
            "content_type": self._content_type.value if self._content_type else "unknown",
            "selected_model": selected_model,
        }
        
        result = PipelineResult(
            video_frames=video_frames,
            depth_map=depth_map,
            segmentation=segmentation,
            flow_field=flow_field,
            metadata=metadata,
            processing_time=processing_time,
            content_analysis=content_analysis,
            motion_generators=motion_generators
        )
        
        return result
    
    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """Load and preprocess image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        h, w = image.shape[:2]
        
        target_size = 512
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return tensor.to(self.device)
    
    def _generate_prompt(
        self,
        image: torch.Tensor,
        content_analysis: Optional[Any] = None
    ) -> str:
        """Generate a simple prompt based on image content."""
        if content_analysis and content_analysis.content_type == ContentType.FURRY:
            return "animated furry character, smooth motion, high quality, fluid animation"
        elif content_analysis and content_analysis.content_type == ContentType.ANIMAL:
            return "animated animal, natural motion, high quality"
        elif content_analysis and content_analysis.content_type == ContentType.LANDSCAPE:
            return "cinematic landscape, natural motion, wind movement, high quality"
        return "natural motion, smooth animation, high quality"
    
    def _get_default_negative_prompt(
        self,
        content_analysis: Optional[Any] = None
    ) -> str:
        """Get default negative prompt based on content type."""
        base_negative = "blurry, low quality, artifacts, static, jittery motion"
        if content_analysis and content_analysis.content_type == ContentType.FURRY:
            return f"{base_negative}, distorted face, broken anatomy, melting fur"
        return base_negative
    
    def _generate_motion(
        self,
        image: torch.Tensor,
        num_frames: int,
        motion_preset: Optional[str],
        content_analysis: Optional[Any],
        depth_map: Optional[Any] = None
    ) -> FlowField:
        """Generate motion based on preset and content type."""
        h, w = image.shape[-2:]
        
        if motion_preset == "cinematic" and self.camera_motion:
            trajectory = CameraTrajectory(
                camera_type="orbital",
                start_angle=0.0,
                end_angle=360.0,
                radius_start=1.0,
                radius_end=0.95
            )
            flows = self.camera_motion.generate_trajectory(image, trajectory, num_frames)
            return FlowField(flow=flows[-1])
        
        elif motion_preset == "zoom" and self.camera_motion:
            trajectory = CameraTrajectory(
                camera_type="dolly",
                zoom_range=(1.0, 1.3),
                radius_start=1.0,
                radius_end=0.7
            )
            flows = self.camera_motion.generate_trajectory(image, trajectory, num_frames)
            return FlowField(flow=flows[-1])
        
        elif motion_preset == "pan" and self.camera_motion:
            trajectory = CameraTrajectory(
                camera_type="pan",
                start_angle=0.0,
                end_angle=180.0,
                radius_start=1.0,
                radius_end=1.0
            )
            flows = self.camera_motion.generate_trajectory(image, trajectory, num_frames)
            return FlowField(flow=flows[-1])
        
        elif motion_preset == "subtle":
            x_coords, y_coords = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'
            )
            flow = np.zeros((h, w, 2), dtype=np.float32)
            wave_strength = 1.0
            flow[..., 0] = wave_strength * np.sin(y_coords / h * 4 * np.pi)
            flow[..., 1] = wave_strength * np.cos(x_coords / w * 4 * np.pi)
            return FlowField(flow=flow)
        
        elif motion_preset == "furry_tail" and self.furry_motion:
            flows = self.furry_motion.generate_tail_motion(
                image,
                num_frames=num_frames,
                intensity=0.8
            )
            return FlowField(flow=flows[-1])
        
        elif motion_preset == "furry_ears" and self.furry_motion:
            flows = self.furry_motion.generate_ear_motion(
                image,
                num_frames=num_frames,
                intensity=0.7
            )
            return FlowField(flow=flows[-1])
        
        elif content_analysis and content_analysis.content_type in [ContentType.FURRY, ContentType.ANIMAL]:
            if self.furry_motion:
                flows = self.furry_motion.generate(
                    image,
                    num_frames=num_frames,
                    motion_types=None,
                    intensity=0.7
                )
                return FlowField(flow=flows[-1])
        
        elif content_analysis and content_analysis.content_type == ContentType.LANDSCAPE:
            if self.env_motion:
                flows = self.env_motion.generate(
                    image,
                    num_frames=num_frames,
                    effects=None,
                    strength=0.5
                )
                return FlowField(flow=flows[-1])
        
        if self.parallax and depth_map is not None:
            if isinstance(depth_map, DepthMap):
                depth = depth_map.depth
            else:
                depth = depth_map
            flows = self.parallax.generate(
                image,
                depth,
                num_frames=num_frames
            )
            return FlowField(flow=flows[-1])
        
        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            indexing='xy'
        )
        flow = np.zeros((h, w, 2), dtype=np.float32)
        wave_strength = 2.0
        flow[..., 0] = wave_strength * np.sin(y_coords / h * 4 * np.pi)
        flow[..., 1] = wave_strength * np.cos(x_coords / w * 4 * np.pi)
        return FlowField(flow=flow)
    
    def _estimate_flow(
        self,
        image: torch.Tensor
    ) -> Optional[FlowField]:
        """Estimate optical flow from image."""
        if self.flow_estimator is None:
            return None
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return self.flow_estimator._estimate_farnback(
            image.squeeze(0).cpu().numpy(),
            image.squeeze(0).cpu().numpy()
        )
    
    def _generate_camera_flow(
        self,
        image: torch.Tensor,
        num_frames: int
    ) -> FlowField:
        """Generate camera motion flow."""
        h, w = image.shape[-2:]
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 2 * np.pi
            
            pan_strength = 0.05 * w
            tilt_strength = 0.03 * h
            zoom_strength = 0.02
            
            dx = pan_strength * np.sin(phase)
            dy = tilt_strength * np.sin(phase * 0.7)
            dz = zoom_strength * np.sin(phase * 0.5)
            
            flow = np.zeros((h, w, 2), dtype=np.float32)
            
            x_coords, y_coords = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing='xy'
            )
            
            center_x = w / 2
            center_y = h / 2
            
            flow[..., 0] = ((x_coords - center_x) / center_x * dx)
            flow[..., 1] = ((y_coords - center_y) / center_y * dy)
            
            flows.append(flow)
        
        return FlowField(flow=flows[-1])
    
    def _generate_default_flow(
        self,
        image: torch.Tensor,
        num_frames: int
    ) -> FlowField:
        """Generate default motion flow."""
        h, w = image.shape[-2:]
        
        x_coords, y_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            indexing='xy'
        )
        
        flow = np.zeros((h, w, 2), dtype=np.float32)
        
        wave_strength = 2.0
        flow[..., 0] = wave_strength * np.sin(y_coords / h * 4 * np.pi)
        flow[..., 1] = wave_strength * np.cos(x_coords / w * 4 * np.pi)
        
        return FlowField(flow=flow)
    
    def _generate_simple_video(
        self,
        image: torch.Tensor,
        num_frames: int
    ) -> VideoFrames:
        """Generate simple video without diffusion model."""
        frames = VideoFrames()
        h, w = image.shape[-2:]
        
        for i in range(num_frames):
            t = i / num_frames
            
            zoom = 1.0 + 0.05 * np.sin(t * 2 * np.pi)
            dx = 0.03 * w * np.sin(t * 2 * np.pi)
            dy = 0.02 * h * np.sin(t * 2 * np.pi * 0.7)
            
            frame = image.clone()
            
            if zoom != 1.0:
                scaled = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    scale_factor=zoom,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
                
                new_h, new_w = scaled.shape[-2:]
                start_y = max(0, (new_h - h) // 2 - int(dy))
                start_x = max(0, (new_w - w) // 2 - int(dx))
                
                frame = scaled[
                    :,
                    start_y:start_y + h,
                    start_x:start_x + w
                ]
                
                if frame.shape[-2:] != (h, w):
                    frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
            
            frames.append(frame)
        
        return frames
    
    def save_video(
        self,
        frames: VideoFrames,
        output_path: Union[str, Path],
        fps: int = 8,
        codec: str = "libx264",
        quality: int = 23
    ) -> None:
        """Save video frames to file.
        
        Args:
            frames: Video frames to save
            output_path: Output file path
            fps: Frames per second
            codec: Video codec
            quality: Quality setting (lower is better)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import cv2
            
            frame_list = frames.to_list()
            
            if not frame_list:
                return
            
            h, w = frame_list[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (w, h)
            )
            
            for frame in frame_list:
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            
            writer.release()
            
        except ImportError:
            import subprocess
            import tempfile
            
            temp_dir = Path(tempfile.mkdtemp())
            
            for i, frame in enumerate(frames.to_list()):
                frame_path = temp_dir / f"frame_{i:04d}.png"
                Image.fromarray(frame).save(frame_path)
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%04d.png"),
                "-c:v", codec,
                "-crf", str(quality),
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                self._log("FFmpeg not available. Saving frames individually...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(frames.to_list()):
                    frame_path = output_path.parent / f"{output_path.stem}_frame_{i:04d}.png"
                    Image.fromarray(frame).save(frame_path)
    
    def save_intermediate(
        self,
        result: PipelineResult,
        output_dir: Union[str, Path]
    ) -> None:
        """Save intermediate results for debugging.
        
        Args:
            result: Pipeline result
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if result.depth_map is not None:
            depth = result.depth_map
            if isinstance(depth, DepthMap):
                depth_np = depth.normalized
                if isinstance(depth_np, torch.Tensor):
                    depth_np = depth_np.cpu().numpy()
            else:
                depth_np = depth
            
            depth_img = (depth_np * 255).astype(np.uint8)
            if depth_img.ndim == 2:
                Image.fromarray(depth_img).save(output_dir / "depth.png")
            else:
                Image.fromarray(depth_img).save(output_dir / "depth.png")
        
        if result.segmentation is not None:
            seg_vis = result.segmentation.visualize()
            Image.fromarray(seg_vis).save(output_dir / "segmentation.png")
        
        if result.flow_field is not None:
            flow_vis = result.flow_field.visualize()
            Image.fromarray(flow_vis).save(output_dir / "flow.png")
    
    def _log(self, message: str) -> None:
        """Log a message."""
        self._log_history.append(message)
        if self.config.verbose:
            print(f"[Image2Video] {message}")
    
    def get_log_history(self) -> List[str]:
        """Get processing log history."""
        return self._log_history.copy()
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return (
            f"Image2VideoPipeline("
            f"device={self.device}, "
            f"depth={self.depth_estimator is not None}, "
            f"segmentation={self.segmentor is not None}, "
            f"motion={self.flow_estimator is not None}, "
            f"generation={self.video_generator is not None}, "
            f"furry={self.furry_motion is not None}, "
            f"artifact_reducer={self.artifact_reducer is not None}, "
            f"content_type={self._content_type})"
        )


def create_pipeline(
    device: Optional[str] = None,
    enable_depth: bool = True,
    enable_segmentation: bool = True,
    enable_motion: bool = True,
    enable_consistency: bool = True,
    enable_artifact_reduction: bool = True,
    enable_furry_support: bool = True,
    enable_nsfw: bool = False,
    content_filter: str = "none"
) -> Image2VideoPipeline:
    """Factory function to create a configured pipeline.
    
    Args:
        device: Target device string
        enable_depth: Enable depth estimation
        enable_segmentation: Enable segmentation
        enable_motion: Enable motion estimation
        enable_consistency: Enable temporal consistency
        enable_artifact_reduction: Enable artifact reduction
        enable_furry_support: Enable furry content support
        enable_nsfw: Enable unrestricted content generation
        content_filter: Content filtering level (none, basic, strict)
        
    Returns:
        Configured Image2VideoPipeline
    """
    config = PipelineConfig(
        enable_depth=enable_depth,
        enable_segmentation=enable_segmentation,
        enable_motion=enable_motion,
        enable_consistency=enable_consistency,
        enable_artifact_reduction=enable_artifact_reduction,
        enable_furry_support=enable_furry_support,
        enable_nsfw=enable_nsfw,
        content_filter=content_filter
    )
    
    torch_device = get_torch_device(device) if device else None
    
    pipeline = Image2VideoPipeline(config=config, device=torch_device)
    
    return pipeline