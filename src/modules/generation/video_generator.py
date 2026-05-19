"""Video generation using diffusion models with motion transfer."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .types import VideoFrames, GenerationConfig, MotionGuidance, SceneContext
from .temporal_consistency import TemporalConsistencyManager, MotionPropagator


class VideoGenerator:
    """Video generation using diffusion models.
    
    Supports multiple backends:
    - AnimateDiff: Stable Diffusion with motion modules
    - VideoLDM: Latent video diffusion
    - SVD: Stable Video Diffusion
    - I2VGen: Image-to-Video generation
    
    Args:
        config: Generation configuration
        device: Target compute device
    """
    
    SUPPORTED_MODELS = ["animatediff", "videoldm", "svd", "i2vgen", "custom"]
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or GenerationConfig()
        self.device = device or torch.device("cpu")
        self._cache_dir = self._resolve_cache_dir()
        
        self.pipeline = None
        self.motion_adapter = None
        self.temporal_manager = TemporalConsistencyManager(
            num_frames=self.config.num_frames,
            device=self.device
        )
        self.motion_propagator = MotionPropagator(device=self.device)
        
        self.model_type = "animatediff"
        self._initialized = False
        
        self.depth_estimator = None
        self.segmentor = None
    
    @staticmethod
    def _resolve_cache_dir() -> str:
        if "HF_HUB_CACHE" in os.environ:
            return os.environ["HF_HUB_CACHE"]
        if "HUGGINGFACE_HUB_CACHE" in os.environ:
            return os.environ["HUGGINGFACE_HUB_CACHE"]
        if "HF_HOME" in os.environ:
            return str(Path(os.environ["HF_HOME"]) / "hub")
        return "models/generation"

    @property
    def _cache(self) -> str:
        return self._cache_dir

    def initialize(self) -> None:
        """Initialize the video generation model."""
        if self._initialized:
            return
        
        try:
            self._init_animatediff()
        except Exception as e:
            print(f"AnimateDiff initialization failed: {e}")
            try:
                self._init_videoldm()
            except Exception as e2:
                print(f"VideoLDM initialization failed: {e2}")
                self._init_custom()
        
        self._initialized = True
    
    def _init_animatediff(self) -> None:
        """Initialize AnimateDiff pipeline."""
        try:
            from diffusers import (
                AnimateDiffPipeline,
                MotionAdapter,
                DDIMScheduler,
                AutoencoderKL
            )
            from huggingface_hub import hf_hub_download
            
            adapter_path = "guoyww/animatediff-motion-adapter-sdxl-beta"
            
            try:
                self.motion_adapter = MotionAdapter.from_pretrained(
                    adapter_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    cache_dir=self._cache
                )
            except Exception:
                self.motion_adapter = None
            
            if self.motion_adapter is not None:
                self.pipeline = AnimateDiffPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    motion_adapter=self.motion_adapter,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    cache_dir=self._cache
                )
            else:
                self.pipeline = AnimateDiffPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    cache_dir=self._cache
                )
            
            if self.device.type == "cuda":
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.enable_attention_slicing()
            
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.model_type = "animatediff"
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AnimateDiff: {e}")
    
    def _init_videoldm(self) -> None:
        """Initialize VideoLDM pipeline."""
        try:
            from diffusers import DiffusionPipeline
            
            self.pipeline = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir=self._cache
            )
            
            self.pipeline = self.pipeline.to(self.device)
            self.model_type = "videoldm"
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VideoLDM: {e}")
    
    def _init_custom(self) -> None:
        """Initialize custom fallback generator."""
        self.model_type = "custom"
        
        class SimpleVideoGenerator(nn.Module):
            def __init__(self, num_frames=24):
                super().__init__()
                self.num_frames = num_frames
                
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                self.temporal = nn.LSTM(256, 256, batch_first=True)
                
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 7, padding=3),
                    nn.Sigmoid(),
                )
            
            def forward(self, x):
                B, T, C, H, W = x.shape
                features = self.encoder(x.view(B * T, C, H, W))
                _, (hidden, _) = self.temporal(features.view(B, T, -1))
                features = hidden[-1].unsqueeze(1).expand(-1, T, -1)
                features = features.reshape(B * T, 256, H // 4, W // 4)
                output = self.decoder(features)
                return output.view(B, T, C, H, W)
        
        self.pipeline = SimpleVideoGenerator(num_frames=self.config.num_frames)
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.eval()
    
    def set_depth_estimator(self, estimator) -> None:
        """Set depth estimator for 3D-aware generation."""
        self.depth_estimator = estimator
    
    def set_segmentor(self, segmentor) -> None:
        """Set segmentor for object-aware generation."""
        self.segmentor = segmentor
    
    def generate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        num_frames: Optional[int] = None,
        motion_guidance: Optional[MotionGuidance] = None,
        seed: Optional[int] = None
    ) -> VideoFrames:
        """Generate video from image.
        
        Args:
            image: Source image
            prompt: Text prompt for generation
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            num_frames: Number of frames to generate
            motion_guidance: Optional motion guidance
            seed: Random seed
            
        Returns:
            VideoFrames object with generated frames
        """
        if not self._initialized:
            self.initialize()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if image.max() > 1:
            image = image / 255.0
        
        image = image.to(self.device)
        
        prompt = prompt or self.config.prompt
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        num_frames = num_frames or self.config.num_frames
        
        if seed is not None:
            torch.manual_seed(seed)
        elif self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        
        if self.model_type == "animatediff":
            return self._generate_animatediff(
                image, prompt, guidance_scale, num_inference_steps, num_frames
            )
        elif self.model_type == "videoldm":
            return self._generate_videoldm(
                image, prompt, guidance_scale, num_inference_steps, num_frames
            )
        else:
            return self._generate_custom(
                image, num_frames, motion_guidance
            )
    
    def _generate_animatediff(
        self,
        image: torch.Tensor,
        prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        num_frames: int
    ) -> VideoFrames:
        """Generate video using AnimateDiff."""
        from diffusers import AnimateDiffPipeline, DDIMScheduler
        
        h, w = image.shape[-2:]
        target_h, target_w = self.config.resolution
        
        if h != target_h or w != target_w:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        
        init_image = self._tensor_to_pil(image)
        
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=target_h,
            width=target_w,
            generator=torch.Generator(device=self.device),
        )
        
        frames = VideoFrames()
        for frame in output.frames[0]:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)
        
        return frames
    
    def _generate_videoldm(
        self,
        image: torch.Tensor,
        prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        num_frames: int
    ) -> VideoFrames:
        """Generate video using VideoLDM."""
        h, w = image.shape[-2:]
        target_h, target_w = self.config.resolution
        
        if h != target_h or w != target_w:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=target_h,
            width=target_w,
            generator=torch.Generator(device=self.device),
        )
        
        frames = VideoFrames()
        for frame in output.frames[0]:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)
        
        return frames
    
    def _generate_custom(
        self,
        image: torch.Tensor,
        num_frames: int,
        motion_guidance: Optional[MotionGuidance] = None
    ) -> VideoFrames:
        """Generate video using custom model."""
        frames = VideoFrames()
        h, w = image.shape[-2:]
        
        if motion_guidance is not None and motion_guidance.flow_field is not None:
            flow = motion_guidance.flow_field
            if isinstance(flow, torch.Tensor):
                flow_np = flow.cpu().numpy()
            else:
                flow_np = flow.flow if hasattr(flow, 'flow') else flow
            
            frame_sequence = self.motion_propagator.propagate_from_keyframes(
                image.unsqueeze(0),
                torch.tensor([0, num_frames - 1]),
                num_frames
            )
            
            for i in range(num_frames):
                alpha = i / (num_frames - 1) if num_frames > 1 else 0
                eased = alpha * alpha * (3 - 2 * alpha)
                
                frame = image * (1 - eased) + frame_sequence[i] * eased
                
                frame = frame + torch.randn_like(frame) * 0.05
                frame = torch.clamp(frame, 0, 1)
                
                frames.append(frame)
        else:
            camera_angles = self._generate_camera_motion(num_frames)
            
            for i in range(num_frames):
                offset_x = int(camera_angles[i, 0] * w * 0.1)
                offset_y = int(camera_angles[i, 1] * h * 0.1)
                zoom = 1.0 + camera_angles[i, 2] * 0.1
                
                if zoom != 1.0:
                    scaled_h, scaled_w = int(h * zoom), int(w * zoom)
                    frame_scaled = F.interpolate(
                        image.unsqueeze(0),
                        size=(scaled_h, scaled_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    
                    start_y = (scaled_h - h) // 2 - offset_y
                    start_x = (scaled_w - w) // 2 - offset_x
                    
                    frame = frame_scaled[
                        :,
                        max(0, start_y):max(0, start_y) + h,
                        max(0, start_x):max(0, start_x) + w
                    ]
                    
                    if frame.shape[-2:] != (h, w):
                        frame = F.interpolate(
                            frame.unsqueeze(0),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0)
                else:
                    frame = image
                
                wave_offset = math.sin(i / num_frames * 2 * math.pi) * 0.02
                frame = frame + wave_offset
                frame = torch.clamp(frame, 0, 1)
                
                frames.append(frame)
        
        return frames
    
    def _generate_camera_motion(self, num_frames: int) -> torch.Tensor:
        """Generate smooth camera motion parameters."""
        t = torch.linspace(0, 2 * math.pi, num_frames)
        
        pan = 0.3 * torch.sin(t)
        tilt = 0.2 * torch.sin(t * 0.7 + 1)
        zoom = 0.15 * torch.sin(t * 0.5 + 2)
        
        camera = torch.stack([pan, tilt, zoom], dim=1)
        
        return camera
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
    
    def apply_motion_transfer(
        self,
        frames: VideoFrames,
        motion_guidance: MotionGuidance
    ) -> VideoFrames:
        """Apply motion transfer to generated frames.
        
        Args:
            frames: Generated frames
            motion_guidance: Motion guidance with flow fields
            
        Returns:
            Motion-enhanced frames
        """
        if motion_guidance.flow_field is None:
            return frames
        
        video_tensor = frames.to_video()
        
        if motion_guidance.camera_motion is not None:
            camera = motion_guidance.camera_motion
            zoom = camera.get("scale", 1.0)
            pan = camera.get("pan", 0.0)
            tilt = camera.get("tilt", 0.0)
            
            for i, frame in enumerate(video_tensor):
                t = i / len(video_tensor)
                
                current_zoom = 1.0 + zoom * math.sin(t * 2 * math.pi) * 0.1
                current_pan = pan * math.sin(t * 2 * math.pi) * 0.1
                current_tilt = tilt * math.sin(t * 2 * math.pi) * 0.1
                
                h, w = frame.shape[-2:]
                
                if current_zoom != 1.0:
                    scaled = F.interpolate(
                        frame.unsqueeze(0),
                        scale_factor=current_zoom,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    
                    new_h, new_w = scaled.shape[-2:]
                    start_y = (new_h - h) // 2 + int(current_tilt * h)
                    start_x = (new_w - w) // 2 + int(current_pan * w)
                    
                    frame = scaled[
                        :,
                        max(0, start_y):max(0, start_y) + h,
                        max(0, start_x):max(0, start_x) + w
                    ]
                    
                    if frame.shape[-2:] != (h, w):
                        frame = F.interpolate(
                            frame.unsqueeze(0),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0)
                
                video_tensor[i] = frame
        
        smoothed = self.temporal_manager.temporal_smooth(video_tensor)
        filtered = self.temporal_manager.reduce_flickering(smoothed)
        
        result = VideoFrames()
        for frame in filtered:
            result.append(frame)
        
        return result
    
    def apply_depth_effects(
        self,
        frames: VideoFrames,
        depth_map: torch.Tensor
    ) -> VideoFrames:
        """Apply depth-based effects to frames.
        
        Args:
            frames: Video frames
            depth_map: Depth map [H, W]
            
        Returns:
            Frames with depth effects
        """
        video_tensor = frames.to_video()
        T, C, H, W = video_tensor.shape
        
        depth = depth_map
        if isinstance(depth, torch.Tensor) and depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        
        result_frames = []
        
        for t in range(T):
            frame = video_tensor[t]
            
            parallax = self._compute_parallax(depth.squeeze(), t / T)
            
            for c in range(C):
                frame_c = frame[c].unsqueeze(0).unsqueeze(0)
                warped = F.grid_sample(
                    frame_c,
                    parallax,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True
                )
                frame = frame.clone()
                frame[c] = warped.squeeze(0).squeeze(0)
            
            result_frames.append(frame)
        
        result = VideoFrames()
        for frame in result_frames:
            result.append(frame)
        
        return result
    
    def _compute_parallax(
        self,
        depth: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Compute parallax grid for depth effect."""
        H, W = depth.shape
        
        y_coords = torch.linspace(-1, 1, H, device=depth.device)
        x_coords = torch.linspace(-1, 1, W, device=depth.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='x')
        
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        strength = 0.1 * t
        
        grid_x = grid_x + depth_norm * strength * torch.sign(grid_x)
        grid_y = grid_y + depth_norm * strength * torch.sign(grid_y)
        
        grid_x = grid_x.clamp(-1, 1)
        grid_y = grid_y.clamp(-1, 1)
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        return grid
    
    def make_loopable(
        self,
        frames: VideoFrames,
        blend_frames: int = 4
    ) -> VideoFrames:
        """Make video loopable by blending end to start.
        
        Args:
            frames: Video frames
            blend_frames: Number of frames to blend
            
        Returns:
            Loopable video frames
        """
        video_tensor = frames.to_video()
        
        T = video_tensor.shape[0]
        
        looped = self.temporal_manager.enforce_loop_consistency(
            video_tensor,
            loop_frames=blend_frames,
            strength=0.6
        )
        
        result = VideoFrames()
        for frame in looped:
            result.append(frame)
        
        return result
    
    def __repr__(self) -> str:
        return f"VideoGenerator(model={self.model_type}, device={self.device})"


class MotionAwareInterpolator:
    """Motion-aware frame interpolation for smooth video."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.flow_estimator = None
    
    def set_flow_estimator(self, estimator) -> None:
        """Set flow estimator for interpolation."""
        self.flow_estimator = estimator
    
    def interpolate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        num_intermediate: int = 2
    ) -> List[torch.Tensor]:
        """Interpolate frames with motion compensation.
        
        Args:
            frame1: First frame [C, H, W]
            frame2: Second frame [C, H, W]
            flow: Optical flow [2, H, W]
            num_intermediate: Number of intermediate frames
            
        Returns:
            List of interpolated frames
        """
        if flow is not None and self.flow_estimator is not None:
            return self._motion_compensated_interpolation(
                frame1, frame2, flow, num_intermediate
            )
        else:
            return self._linear_interpolation(frame1, frame2, num_intermediate)
    
    def _linear_interpolation(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_intermediate: int
    ) -> List[torch.Tensor]:
        """Simple linear interpolation."""
        frames = []
        
        for i in range(num_intermediate + 2):
            alpha = i / (num_intermediate + 1)
            frame = (1 - alpha) * frame1 + alpha * frame2
            frames.append(frame)
        
        return frames[1:-1]
    
    def _motion_compensated_interpolation(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flow: torch.Tensor,
        num_intermediate: int
    ) -> List[torch.Tensor]:
        """Motion-compensated interpolation."""
        frames = []
        
        for i in range(num_intermediate + 2):
            t = i / (num_intermediate + 1)
            
            flow_t = flow * t
            
            if frame1.dim() == 3:
                frame1 = frame1.unsqueeze(0)
            if frame2.dim() == 3:
                frame2 = frame2.unsqueeze(0)
            
            grid = self._flow_to_grid(flow_t, frame1.shape[-2:])
            
            warped1 = F.grid_sample(
                frame1,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            warped2 = F.grid_sample(
                frame2,
                self._invert_grid(grid),
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            frame = (1 - t) * warped1 + t * warped2
            
            frames.append(frame.squeeze(0) if frame.dim() == 4 else frame)
        
        return frames[1:-1]
    
    def _flow_to_grid(
        self,
        flow: torch.Tensor,
        size: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert flow to sampling grid."""
        H, W = size
        
        y_coords = torch.linspace(-1, 1, H, device=flow.device)
        x_coords = torch.linspace(-1, 1, W, device=flow.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='x')
        
        grid_x = grid_x + flow[0] / W * 2
        grid_y = grid_y + flow[1] / H * 2
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        return grid
    
    def _invert_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Invert sampling grid."""
        return torch.cat([-grid[..., 0:1], -grid[..., 1:2]], dim=-1)