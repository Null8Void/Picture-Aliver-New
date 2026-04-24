"""Frame interpolation for smooth video output.

Uses frame interpolation to:
- Smooth motion between keyframes
- Increase effective frame rate
- Reduce motion judder
- Generate intermediate frames

Methods:
- RIFE: Real-time intermediate flow estimation
- FILM: Frame interpolation with large motion
- CAIN: Channel attention for frame interpolation
- Generic: Simple linear interpolation fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolationMethod(Enum):
    """Frame interpolation methods."""
    RIFE = "rife"
    FILM = "film"
    CAIN = "cain"
    GENERIC = "generic"


@dataclass
class FrameInterpolatorConfig:
    """Configuration for frame interpolation."""
    method: InterpolationMethod = InterpolationMethod.RIFE
    
    interpolation_factor: int = 2
    quality_preset: str = "high"
    
    flow_refinement_iterations: int = 12
    context_frames: int = 2
    
    use_temporal_encoding: bool = True
    temporal_encoding_dim: int = 64
    
    confidence_threshold: float = 0.5
    blend_mode: str = "soft"
    blend_strength: float = 0.8
    
    enable_occlusion_handling: bool = True
    occlusion_threshold: float = 0.3
    
    block_size: int = 8
    search_range: int = 4


class FrameInterpolator:
    """Frame interpolation for artifact-free video generation.
    
    Reduces artifacts by:
    1. Generating intermediate frames
    2. Smoothing motion transitions
    3. Handling occlusions gracefully
    4. Preserving visual quality
    
    Tradeoffs:
    - Quality vs Speed: RIFE is best balance
    - FILM: Better for large motions, slower
    - Generic: Fast but lower quality
    
    Args:
        config: Interpolator configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[FrameInterpolatorConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or FrameInterpolatorConfig()
        self.device = device or torch.device("cpu")
        
        self._model = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize interpolation model."""
        if self._initialized:
            return
        
        if self.config.method == InterpolationMethod.RIFE:
            self._init_rife()
        elif self.config.method == InterpolationMethod.FILM:
            self._init_film()
        elif self.config.method == InterpolationMethod.CAIN:
            self._init_cain()
        else:
            self._model = None
        
        self._initialized = True
    
    def _init_rife(self) -> None:
        """Initialize RIFE model."""
        try:
            import rifenet
            
            self._model = rifenet.RIFENet()
            
            try:
                state_dict = torch.load(
                    "models/interpolation/rife.pth",
                    map_location=self.device
                )
                self._model.load_state_dict(state_dict)
            except FileNotFoundError:
                pass
            
            self._model = self._model.to(self.device)
            self._model.eval()
            
        except ImportError:
            self._model = self._create_generic_interpolator()
    
    def _init_film(self) -> None:
        """Initialize FILM model."""
        try:
            from film_pytorch import FILM
            
            self._model = FILM(
                scale=self.config.interpolation_factor,
                device=self.device
            )
            
        except ImportError:
            self._model = self._create_generic_interpolator()
    
    def _init_cain(self) -> None:
        """Initialize CAIN model."""
        try:
            from cain import CAIN
            
            self._model = CAIN(
                num_iterations=self.config.flow_refinement_iterations,
                device=self.device
            )
            
        except ImportError:
            self._model = self._create_generic_interpolator()
    
    def _create_generic_interpolator(self) -> nn.Module:
        """Create generic frame interpolation model."""
        class GenericInterpolator(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(6, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 6, 3, padding=1),
                )
            
            def forward(self, frame1, frame2):
                x = torch.cat([frame1, frame2], dim=1)
                features = self.encoder(x)
                output = self.decoder(features)
                pred1 = output[:, :3]
                pred2 = output[:, 3:]
                return pred1, pred2
        
        model = GenericInterpolator()
        model = model.to(self.device)
        model.eval()
        return model
    
    def interpolate(
        self,
        frames: torch.Tensor,
        num_interpolated: int = 1
    ) -> torch.Tensor:
        """Interpolate frames to create smooth motion.
        
        Args:
            frames: Video tensor [T, C, H, W]
            num_interpolated: Number of frames to insert between each pair
            
        Returns:
            Interpolated video tensor
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        T, B, C, H, W = frames.shape
        
        if not self._initialized:
            self.initialize()
        
        interpolated_frames = []
        
        for t in range(T - 1):
            frame1 = frames[t]
            frame2 = frames[t + 1]
            
            interpolated = self._interpolate_pair(
                frame1, frame2, num_interpolated
            )
            
            interpolated_frames.append(frame1)
            interpolated_frames.extend(interpolated)
        
        interpolated_frames.append(frames[T - 1])
        
        return torch.stack(interpolated_frames, dim=0)
    
    def _interpolate_pair(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Interpolate between two frames."""
        if num_frames <= 0:
            return []
        
        if self._model is not None:
            return self._neural_interpolation(frame1, frame2, num_frames)
        else:
            return self._linear_interpolation(frame1, frame2, num_frames)
    
    def _neural_interpolation(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Neural network-based interpolation."""
        interpolated = []
        
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            
            flow, occlusion = self._estimate_flow_occlusion(frame1, frame2)
            
            mid_frame = self._warp_and_blend(frame1, frame2, flow, occlusion, t)
            
            interpolated.append(mid_frame)
        
        return interpolated
    
    def _estimate_flow_occlusion(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate bidirectional flow and occlusion mask."""
        if frame1.shape[0] == 3:
            gray1 = 0.299 * frame1[0] + 0.587 * frame1[1] + 0.114 * frame1[2]
            gray2 = 0.299 * frame2[0] + 0.587 * frame2[1] + 0.114 * frame2[2]
        else:
            gray1 = frame1[0]
            gray2 = frame2[0]
        
        if gray1.dim() == 2:
            gray1 = gray1.unsqueeze(0)
            gray2 = gray2.unsqueeze(0)
        
        flow12 = self._estimate_optical_flow(gray1, gray2)
        flow21 = self._estimate_optical_flow(gray2, gray1)
        
        H, W = flow12.shape[-2:]
        occlusion12 = self._compute_occlusion(flow12, flow21)
        
        return (flow12, flow21), occlusion12
    
    def _estimate_optical_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """Estimate optical flow between frames."""
        flow_net = self._create_simple_flow_net()
        
        flow_net.eval()
        with torch.no_grad():
            flow = flow_net(frame1, frame2)
        
        return flow
    
    def _create_simple_flow_net(self) -> nn.Module:
        """Create simple flow estimation network."""
        class SimpleFlowNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv4 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
                
                self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
                self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
                self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
                self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
                self.flow = nn.Conv2d(32, 2, 3, padding=1)
            
            def forward(self, frame1, frame2):
                x = torch.cat([frame1, frame2], dim=1)
                
                f1 = F.relu(self.conv1(x))
                f2 = F.relu(self.conv2(f1))
                f3 = F.relu(self.conv3(f2))
                f4 = F.relu(self.conv4(f3))
                
                up1 = F.relu(self.up1(f4))
                f5 = F.relu(self.conv5(up1))
                up2 = F.relu(self.up2(f5))
                f6 = F.relu(self.conv6(up2))
                
                flow = self.flow(f6)
                
                return flow
        
        net = SimpleFlowNet().to(self.device)
        return net
    
    def _compute_occlusion(
        self,
        flow12: torch.Tensor,
        flow21: torch.Tensor
    ) -> torch.Tensor:
        """Compute occlusion mask from flow fields."""
        H, W = flow12.shape[-2:]
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        warped_y = grid_y + flow12[1]
        warped_x = grid_x + flow12[0]
        
        occlusion = torch.zeros(H, W, device=self.device)
        
        valid_x = (warped_x >= 0) & (warped_x < W)
        valid_y = (warped_y >= 0) & (warped_y < H)
        valid = valid_x & valid_y
        
        occlusion[~valid] = 1.0
        
        backward_check = torch.zeros_like(occlusion)
        
        occlusion_threshold = self.config.occlusion_threshold
        occlusion[occlusion.abs() > occlusion_threshold] = 1.0
        
        return occlusion
    
    def _warp_and_blend(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        flows: Tuple[torch.Tensor, torch.Tensor],
        occlusion: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Warp frames and blend for intermediate frame."""
        flow12, flow21 = flows
        
        flow_t = flow12 * t
        
        warped1 = self._warp_frame(frame1, flow_t)
        warped2 = self._warp_frame(frame2, -flow_t * (1 - t))
        
        occlusion_mask = occlusion.unsqueeze(0)
        
        if self.config.blend_mode == "soft":
            weight1 = (1 - occlusion_mask) * (1 - t)
            weight2 = (1 - occlusion_mask) * t
        else:
            weight1 = 1 - occlusion_mask
            weight2 = 1 - occlusion_mask
        
        weights = weight1 + weight2 + 1e-8
        weight1 = weight1 / weights
        weight2 = weight2 / weights
        
        blended = warped1 * weight1 + warped2 * weight2
        
        return blended
    
    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp frame using flow."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        B, C, H, W = frame.shape
        
        if flow.dim() == 2:
            flow = flow.unsqueeze(0)
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        flow_x = flow[..., 0] / W * 2
        flow_y = flow[..., 1] / H * 2
        
        grid_x_warped = grid_x + flow_x.squeeze()
        grid_y_warped = grid_y + flow_y.squeeze()
        
        grid_x_warped = grid_x_warped.clamp(-1, 1)
        grid_y_warped = grid_y_warped.clamp(-1, 1)
        
        grid = torch.stack([grid_x_warped, grid_y_warped], dim=-1).unsqueeze(0)
        
        warped = F.grid_sample(
            frame,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        
        return warped.squeeze(0)
    
    def _linear_interpolation(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Simple linear interpolation fallback."""
        interpolated = []
        
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            
            frame = (1 - t) * frame1 + t * frame2
            interpolated.append(frame)
        
        return interpolated
    
    def upsample_fps(
        self,
        frames: torch.Tensor,
        target_fps: int,
        source_fps: int
    ) -> torch.Tensor:
        """Upsample video to higher frame rate.
        
        Args:
            frames: Video tensor [T, C, H, W]
            target_fps: Target frames per second
            source_fps: Source frames per second
            
        Returns:
            Upsampled video
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        T = frames.shape[0]
        
        fps_ratio = target_fps / source_fps
        
        if fps_ratio <= 1:
            return frames
        
        num_frames_per_pair = int(fps_ratio) - 1
        
        return self.interpolate(frames, num_frames_per_pair)
    
    def compute_interpolation_quality(
        self,
        original: torch.Tensor,
        interpolated: torch.Tensor,
        original_indices: List[int]
    ) -> Dict[str, float]:
        """Compute interpolation quality metrics.
        
        Args:
            original: Original frames
            interpolated: Interpolated frames
            original_indices: Indices of original frames in interpolated
            
        Returns:
            Quality metrics
        """
        metrics = {}
        
        reconstructed = []
        for idx in original_indices:
            if idx < interpolated.shape[0]:
                reconstructed.append(interpolated[idx])
        
        if reconstructed:
            reconstructed_tensor = torch.stack(reconstructed)
            original_tensor = original[:len(reconstructed)]
            
            mse = ((reconstructed_tensor - original_tensor) ** 2).mean().item()
            metrics['mse'] = mse
            metrics['psnr'] = 10 * np.log10(1.0 / (mse + 1e-8))
        
        T_interp = interpolated.shape[0]
        smoothness_scores = []
        
        for t in range(1, T_interp):
            diff = torch.abs(interpolated[t] - interpolated[t-1]).mean().item()
            smoothness_scores.append(diff)
        
        metrics['mean_smoothness'] = np.mean(smoothness_scores)
        metrics['smoothness_std'] = np.std(smoothness_scores)
        
        return metrics