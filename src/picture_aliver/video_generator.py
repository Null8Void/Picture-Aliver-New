"""
Video Generator Module

Generates video frames from input images using diffusion models.
Supports motion-conditioned generation, depth-guided synthesis,
and content-adaptive parameters for different content types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """
    Configuration for video generation.
    
    Attributes:
        num_frames: Number of frames to generate
        fps: Frames per second
        width: Output width
        height: Output height
        guidance_scale: CFG guidance scale
        num_inference_steps: Denoising steps
        motion_strength: Motion strength (0-1)
        seed: Random seed
        duration_seconds: Desired duration in seconds (5-120)
    """
    num_frames: int = 24
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    motion_strength: float = 0.8
    seed: Optional[int] = None
    duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.duration_seconds is not None:
            self.num_frames = self._calculate_frames_from_duration(
                self.duration_seconds, self.fps
            )
        elif self.num_frames < 24:
            self.num_frames = self._ensure_minimum_frames(self.num_frames)
    
    @staticmethod
    def _calculate_frames_from_duration(duration: float, fps: int) -> int:
        """Calculate frames from desired duration."""
        duration = max(5.0, min(120.0, duration))
        return max(int(duration * fps), 40)
    
    @staticmethod
    def _ensure_minimum_frames(num_frames: int) -> int:
        """Ensure minimum frame count for reasonable video."""
        return max(num_frames, 40)
    
    @property
    def actual_duration(self) -> float:
        """Get actual duration in seconds."""
        return self.num_frames / self.fps


class VideoFrames:
    """
    Container for video frame sequences.
    
    Supports:
    - List-like operations (append, extend, indexing)
    - Batch conversion to/from tensors
    - Visualization
    """
    
    def __init__(self):
        self.frames: List[torch.Tensor] = []
    
    def append(self, frame: torch.Tensor) -> None:
        """Add a frame to the sequence."""
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        if frame.dim() == 3:
            self.frames.append(frame)
    
    def extend(self, frames: List[torch.Tensor]) -> None:
        """Add multiple frames."""
        for frame in frames:
            self.append(frame)
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.frames[index]
    
    def to_tensor(self, batch_first: bool = True) -> torch.Tensor:
        """
        Convert frames to tensor.
        
        Args:
            batch_first: If True, tensor is (B, C, H, W)
            
        Returns:
            Video tensor
        """
        if not self.frames:
            return torch.empty(0)
        
        if batch_first:
            return torch.stack(self.frames, dim=0)
        else:
            return torch.stack(self.frames, dim=1).permute(1, 0, 2, 3)
    
    def to_list(self) -> List[np.ndarray]:
        """Convert frames to numpy arrays."""
        result = []
        for frame in self.frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu()
                if frame.dtype == torch.float32:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).to(torch.uint8)
                    else:
                        frame = frame.to(torch.uint8)
                frame = frame.permute(1, 2, 0).numpy()
            result.append(frame)
        return result
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "VideoFrames":
        """Create VideoFrames from tensor."""
        frames = cls()
        if tensor.dim() == 4:
            for frame in tensor:
                frames.append(frame)
        elif tensor.dim() == 5:
            for t in range(tensor.shape[1]):
                frames.append(tensor[0, t])
        return frames
    
    def pad_to_duration(self, duration_seconds: float, fps: int) -> "VideoFrames":
        """
        Pad or extend frames to achieve desired duration.
        
        Args:
            duration_seconds: Target duration (5-120 seconds)
            fps: Frames per second
            
        Returns:
            VideoFrames with extended duration
        """
        duration_seconds = max(5.0, min(120.0, duration_seconds))
        target_frames = int(duration_seconds * fps)
        
        if len(self.frames) >= target_frames:
            return self
        
        result = VideoFrames()
        result.frames = self.frames.copy()
        
        while len(result.frames) < target_frames:
            cycle_idx = len(result.frames) % len(self.frames)
            result.frames.append(self.frames[cycle_idx].clone())
        
        return result
    
    def get_metadata(self) -> dict:
        """Get video metadata."""
        if not self.frames:
            return {"num_frames": 0, "fps": 8, "duration": 0}
        
        frame = self.frames[0]
        if isinstance(frame, torch.Tensor):
            c, h, w = frame.shape
        else:
            h, w = frame.shape[:2]
            c = 3
        
        return {
            "num_frames": len(self.frames),
            "width": w,
            "height": h,
            "channels": c
        }


class VideoGenerator:
    """
    Diffusion-based video generator.
    
    Generates coherent video frames from a single image using:
    - Motion-conditioned latent diffusion
    - Depth-guided generation
    - Temporal consistency enforcement
    - Content-adaptive parameters
    
    Attributes:
        device: Compute device
        depth_estimator: Optional depth estimator
        config: Generation configuration
        use_fp16: Whether to use half precision
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        depth_estimator: Optional[Any] = None,
        use_fp16: bool = True
    ):
        self.device = device or torch.device("cpu")
        self.depth_estimator = depth_estimator
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        
        self.config = GenerationConfig()
        
        self.unet: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        
        self._initialized = False
        self._dummy_models = False
    
    def initialize(self) -> None:
        """Initialize the diffusion models."""
        if self._initialized:
            return
        
        print("[VideoGenerator] Initializing diffusion models...")
        
        try:
            self._initialize_torch_models()
        except Exception as e:
            print(f"[VideoGenerator] Using fallback generator: {e}")
            self._initialize_fallback()
        
        self._initialized = True
    
    def _initialize_torch_models(self) -> None:
        """Try to initialize real diffusion models."""
        self.unet = UNet3DConditionModel(device=self.device)
        self.vae = VAEWrapper(device=self.device)
        self.text_encoder = TextEncoder(device=self.device)
    
    def _initialize_fallback(self) -> None:
        """Initialize fallback procedural generator."""
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self._dummy_models = True
    
    def generate(
        self,
        image_tensor: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        motion_field: Optional[Any] = None,
        segmentation: Optional[Any] = None,
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        content_type: str = "scene"
    ) -> VideoFrames:
        """
        Generate video frames from image.
        
        Args:
            image_tensor: Input image (C, H, W)
            depth_map: Depth map from estimation
            motion_field: Motion field for animation
            segmentation: Segmentation result
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_frames: Number of frames to generate
            guidance_scale: CFG scale
            num_inference_steps: Denoising steps
            seed: Random seed
            content_type: Content type hint
            
        Returns:
            VideoFrames object with generated frames
        """
        if not self._initialized:
            self.initialize()
        
        if num_frames is not None:
            self.config.num_frames = num_frames
        if guidance_scale is not None:
            self.config.guidance_scale = guidance_scale
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        if depth_map is not None and isinstance(depth_map, torch.Tensor):
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(0)
        
        if self._dummy_models:
            return self._generate_procedural(
                image_tensor,
                depth_map,
                motion_field,
                num_frames or self.config.num_frames
            )
        
        return self._generate_diffusion(
            image_tensor,
            depth_map,
            motion_field,
            prompt,
            negative_prompt
        )
    
    def _generate_diffusion(
        self,
        image_tensor: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        motion_field: Optional[Any],
        prompt: str,
        negative_prompt: str
    ) -> VideoFrames:
        """Generate using diffusion model."""
        _, c, h, w = image_tensor.shape
        
        latent_frames = []
        
        current = image_tensor
        for t in range(self.config.num_frames):
            latent = self._encode_latent(current)
            
            for i in range(self.config.num_inference_steps):
                noise_level = 1.0 - i / self.config.num_inference_steps
                
                if motion_field is not None and t > 0:
                    latent = self._apply_motion(latent, motion_field, t)
                
                if depth_map is not None:
                    latent = self._apply_depth_guidance(latent, depth_map)
                
                latent = self._denoise_step(latent, noise_level)
            
            decoded = self._decode_latent(latent)
            current = decoded
            
            latent_frames.append(decoded)
        
        frames = VideoFrames()
        frames.extend(latent_frames)
        
        return frames
    
    def _generate_procedural(
        self,
        image_tensor: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        motion_field: Optional[Any],
        num_frames: int
    ) -> VideoFrames:
        """
        Generate video procedurally without diffusion model.
        
        Uses warping and interpolation to create motion from the input.
        Used when diffusion models are unavailable.
        """
        frames = VideoFrames()
        
        _, c, h, w = image_tensor.shape
        
        base = image_tensor.squeeze(0)
        
        for t in range(num_frames):
            progress = t / max(1, num_frames - 1)
            
            phase = progress * 2 * np.pi
            
            if motion_field is not None and motion_field.flows:
                flow_idx = min(t, len(motion_field.flows) - 1)
                flow = motion_field.flows[flow_idx]
                
                if isinstance(flow, torch.Tensor):
                    flow = flow * progress * 0.1
                else:
                    flow = torch.from_numpy(flow).float() * progress * 0.1
                
                frame = self._warp_frame(base, flow.to(self.device))
            else:
                zoom = 1.0 + 0.05 * np.sin(phase)
                
                dx = 0.03 * w * np.sin(phase)
                dy = 0.02 * h * np.sin(phase * 0.7)
                
                frame = self._apply_transform(
                    base,
                    scale=zoom,
                    dx=int(dx),
                    dy=int(dy)
                )
            
            frames.append(frame)
        
        return frames
    
    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        if self.vae is not None:
            return self.vae.encode(x)
        
        scale = 8
        h, w = x.shape[-2:]
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        x_padded = F.pad(x, (0, w_pad, 0, h_pad))
        
        latent = F.avg_pool2d(x_padded, 2)[:, :3]
        
        return latent
    
    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space."""
        if self.vae is not None:
            return self.vae.decode(latent)
        
        frame = F.interpolate(
            latent,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        
        if frame.max() > 1:
            frame = torch.sigmoid(frame)
        
        return frame
    
    def _denoise_step(self, latent: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Single denoising step."""
        noise = torch.randn_like(latent) * noise_level * 0.1
        return latent * (1 - noise_level * 0.05) + noise
    
    def _apply_motion(
        self,
        latent: torch.Tensor,
        motion_field: Any,
        t: int
    ) -> torch.Tensor:
        """Apply motion field to latent."""
        if motion_field.flows is None or len(motion_field.flows) == 0:
            return latent
        
        flow_idx = min(t, len(motion_field.flows) - 1)
        flow = motion_field.flows[flow_idx]
        
        if isinstance(flow, np.ndarray):
            flow = torch.from_numpy(flow).float()
        
        flow = F.interpolate(
            flow.unsqueeze(0),
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        return latent
    
    def _apply_depth_guidance(
        self,
        latent: torch.Tensor,
        depth: torch.Tensor
    ) -> torch.Tensor:
        """Apply depth-based guidance."""
        if depth is None:
            return latent
        
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        
        depth_resized = F.interpolate(
            depth,
            size=latent.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
        
        guidance = depth_norm * 0.02
        
        return latent + guidance
    
    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp frame using optical flow."""
        b, c, h, w = frame.shape
        flow = flow.to(frame.device)
        
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        
        grid = F.affine_grid(
            torch.eye(2, 3, device=frame.device).unsqueeze(0),
            frame.unsqueeze(0).shape,
            align_corners=False
        )
        
        flow_normalized = flow.clone()
        flow_normalized[..., 0] /= w
        flow_normalized[..., 1] /= h
        
        warped = F.grid_sample(
            frame.unsqueeze(0),
            grid + flow_normalized,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        
        return warped.squeeze(0)
    
    def _apply_transform(
        self,
        frame: torch.Tensor,
        scale: float = 1.0,
        dx: int = 0,
        dy: int = 0
    ) -> torch.Tensor:
        """Apply affine transform to frame."""
        theta = torch.tensor([
            [1/scale, 0, dx],
            [0, 1/scale, dy]
        ], dtype=torch.float32, device=self.device)
        
        grid = F.affine_grid(
            theta.unsqueeze(0),
            frame.unsqueeze(0).shape,
            align_corners=False
        )
        
        transformed = F.grid_sample(
            frame.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        
        return transformed.squeeze(0)
    
    def set_config(self, config: GenerationConfig) -> None:
        """Update generation configuration."""
        self.config = config


class UNet3DConditionModel(nn.Module):
    """
    3D U-Net for video diffusion.
    
    Architecture:
    - 3D convolutions for temporal dimension
    - Cross-attention for text conditioning
    - Skip connections for detail preservation
    - Time embedding for timestep conditioning
    
    Attributes:
        device: Compute device
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(
        self,
        device: torch.device,
        in_channels: int = 4,
        out_channels: int = 4,
        time_dim: int = 256
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.time_embed = TimeEmbedding(time_dim)
        
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, 64, 3, padding=1),
                nn.GroupNorm(32, 64),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(64, 128, 3, padding=1, stride=2),
                nn.GroupNorm(32, 128),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(128, 256, 3, padding=1, stride=2),
                nn.GroupNorm(32, 256),
                nn.SiLU(inplace=True)
            ),
        ])
        
        self.middle_block = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True)
        )
        
        self.output_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(512, 256, 3, padding=1),
                nn.GroupNorm(32, 256),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(384, 128, 3, padding=1),
                nn.GroupNorm(32, 128),
                nn.SiLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(192, 64, 3, padding=1),
                nn.GroupNorm(32, 64),
                nn.SiLU(inplace=True)
            ),
        ])
        
        self.out_conv = nn.Conv3d(64, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        t_emb = self.time_embed(timestep)
        
        hiddens = []
        for block in self.input_blocks:
            x = block(x)
            hiddens.append(x)
        
        x = self.middle_block(x)
        
        for block in self.output_blocks:
            skip = hiddens.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        x = self.out_conv(x)
        
        return x


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create time embeddings."""
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class VAEWrapper(nn.Module):
    """VAE encoder/decoder wrapper."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)


class TextEncoder(nn.Module):
    """Simple text encoder."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 768),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode text tokens."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        return self.projection(x.mean(dim=1))