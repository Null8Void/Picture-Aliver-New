"""
Video Stabilizer Module

Stabilizes generated video frames using optical flow and motion fields.
Applies temporal smoothing, flicker reduction, and motion compensation
to produce smooth, professional-looking videos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StabilizationConfig:
    """
    Configuration for video stabilization.
    
    Attributes:
        temporal_window: Window size for temporal smoothing
        flow_weight: Weight for optical flow guidance
        preserve_motion: How much original motion to preserve
        flicker_threshold: Threshold for flicker detection
        enable_warping: Enable frame warping
    """
    temporal_window: int = 3
    flow_weight: float = 0.5
    preserve_motion: float = 0.7
    flicker_threshold: float = 0.1
    enable_warping: bool = True


class VideoStabilizer:
    """
    Video stabilization and enhancement system.
    
    Applies:
    - Temporal smoothing to reduce jitter
    - Flicker reduction for consistent brightness
    - Motion compensation using optical flow
    - Frame interpolation for smoothness
    - Color consistency across frames
    
    Attributes:
        device: Compute device
        config: Stabilization configuration
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[StabilizationConfig] = None
    ):
        self.device = device or torch.device("cpu")
        self.config = config or StabilizationConfig()
        
        self.flow_estimator = FlowBasedStabilizer(device=self.device)
        self.temporal_smoother = TemporalSmoother(device=self.device)
        self.flicker_reducer = FlickerReducer(device=self.device)
        self.color_stabilizer = ColorStabilizer(device=self.device)
        
        self._initialized = True
    
    def stabilize(
        self,
        video_frames: Any,
        motion_field: Optional[Any] = None,
        reference_frame: Optional[torch.Tensor] = None
    ) -> Any:
        """
        Stabilize video frames.
        
        Args:
            video_frames: Input frames (VideoFrames or tensor)
            motion_field: Optional motion field for compensation
            reference_frame: Optional reference frame for alignment
            
        Returns:
            Stabilized video frames
        """
        if isinstance(video_frames, torch.Tensor):
            is_tensor = True
        elif hasattr(video_frames, 'to_tensor'):
            is_tensor = False
            frames_tensor = video_frames.to_tensor()
        else:
            raise ValueError("video_frames must be VideoFrames or tensor")
        
        if is_tensor:
            frames_tensor = video_frames
        
        num_frames = frames_tensor.shape[0] if frames_tensor.dim() == 4 else frames_tensor.shape[1]
        
        if num_frames == 0:
            return video_frames
        
        frames_tensor = self._ensure_format(frames_tensor)
        
        if self.config.enable_warping and motion_field is not None:
            frames_tensor = self.flow_estimator.stabilize(frames_tensor, motion_field)
        
        frames_tensor = self.temporal_smoother.smooth(
            frames_tensor,
            window_size=self.config.temporal_window
        )
        
        frames_tensor = self.flicker_reducer.reduce(
            frames_tensor,
            threshold=self.config.flicker_threshold
        )
        
        frames_tensor = self.color_stabilizer.stabilize(frames_tensor)
        
        if hasattr(video_frames, 'from_tensor'):
            return video_frames.from_tensor(frames_tensor)
        
        return frames_tensor
    
    def _ensure_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is in (B, C, H, W) format."""
        if tensor.dim() == 5:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor


class FlowBasedStabilizer(nn.Module):
    """
    Optical flow-based video stabilization.
    
    Uses motion vectors to smooth trajectory and reduce sudden movements.
    Preserves intentional motion while reducing unwanted jitter.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def stabilize(
        self,
        frames: torch.Tensor,
        motion_field: Any
    ) -> torch.Tensor:
        """
        Stabilize using flow information.
        
        Args:
            frames: Video frames (B, C, H, W)
            motion_field: Motion field with flow vectors
            
        Returns:
            Stabilized frames
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        b, t, c, h, w = frames.shape
        
        smoothed_frames = []
        smoothed_trajectory = self._smooth_trajectory(motion_field, t)
        
        for i in range(t):
            current = frames[:, i]
            
            if i > 0 and motion_field is not None:
                flow_idx = min(i, len(motion_field.flows) - 1)
                flow = motion_field.flows[flow_idx]
                
                if isinstance(flow, np.ndarray):
                    flow = torch.from_numpy(flow).float()
                
                flow = flow.to(self.device)
                
                if flow.dim() == 2:
                    flow = flow.unsqueeze(0)
                
                stabilized = self._warp_with_smoothed_flow(
                    current,
                    flow,
                    smoothed_trajectory[i]
                )
            else:
                stabilized = current
            
            smoothed_frames.append(stabilized)
        
        result = torch.stack(smoothed_frames, dim=1)
        
        if squeeze_output:
            result = result.squeeze(1)
        
        return result
    
    def _smooth_trajectory(
        self,
        motion_field: Any,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Compute smoothed camera trajectory."""
        trajectory = []
        
        for i in range(num_frames):
            if motion_field is not None and motion_field.flows:
                flow_idx = min(i, len(motion_field.flows) - 1)
                flow = motion_field.flows[flow_idx]
                
                if isinstance(flow, np.ndarray):
                    flow = torch.from_numpy(flow).float()
                
                mean_flow = flow.mean(dim=(0, 1))
            else:
                mean_flow = torch.zeros(2, device=self.device)
            
            trajectory.append(mean_flow)
        
        smoothed = []
        smoothed.append(trajectory[0])
        
        for i in range(1, len(trajectory)):
            alpha = 0.3
            smoothed_frame = alpha * trajectory[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    def _warp_with_smoothed_flow(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
        smoothed_flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp frame using flow difference."""
        flow_diff = flow - smoothed_flow
        
        if flow_diff.dim() == 3:
            flow_diff = flow_diff.unsqueeze(0)
        
        grid = F.affine_grid(
            torch.eye(2, 3, device=self.device).unsqueeze(0),
            frame.shape,
            align_corners=False
        )
        
        flow_normalized = flow_diff.clone()
        flow_normalized[..., 0] = flow_normalized[..., 0] / frame.shape[-1]
        flow_normalized[..., 1] = flow_normalized[..., 1] / frame.shape[-2]
        
        warped = F.grid_sample(
            frame,
            grid + flow_normalized,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        
        return warped


class TemporalSmoother(nn.Module):
    """
    Temporal smoothing for frame sequences.
    
    Uses Gaussian and bilateral filtering in time dimension
    to smooth out frame-to-frame variations while preserving
    sharp transitions.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def smooth(
        self,
        frames: torch.Tensor,
        window_size: int = 3
    ) -> torch.Tensor:
        """
        Apply temporal smoothing.
        
        Args:
            frames: Video frames (B, T, C, H, W) or (T, C, H, W)
            window_size: Size of smoothing window
            
        Returns:
            Smoothed frames
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        
        b, t, c, h, w = frames.shape
        
        kernel_size = min(window_size, t)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        
        sigma = kernel_size / 6
        
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, 1, 1).to(self.device)
        
        padded = F.pad(frames, (0, 0, 0, 0, 0, 0, kernel_size//2, kernel_size//2), mode="replicate")
        
        smoothed = F.conv3d(padded, kernel.expand(c, 1, kernel_size, 1, 1))
        
        if squeeze_batch:
            smoothed = smoothed.squeeze(0)
        
        return smoothed
    
    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        x = torch.arange(size, dtype=torch.float32, device=self.device) - size // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel


class FlickerReducer(nn.Module):
    """
    Flicker reduction for video sequences.
    
    Detects and reduces frame-to-frame brightness variations
    while preserving intentional light changes.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def reduce(
        self,
        frames: torch.Tensor,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Reduce flickering in video.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            threshold: Flicker detection threshold
            
        Returns:
            Deflickered frames
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        
        b, t, c, h, w = frames.shape
        
        frame_means = frames.mean(dim=(2, 3, 4))
        
        mean_diff = torch.abs(frame_means[:, 1:] - frame_means[:, :-1])
        mean_diff = torch.cat([torch.zeros_like(mean_diff[:, :1]), mean_diff], dim=1)
        
        flicker_mask = mean_diff > threshold
        
        luminance_avg = frame_means.mean(dim=1, keepdim=True).mean(dim=0)
        
        corrected = frames.clone()
        
        for i in range(t):
            if flicker_mask[:, i].any():
                adjustment = luminance_avg - frame_means[:, i]
                adjustment = adjustment.view(b, 1, 1, 1, 1)
                corrected[:, i] = frames[:, i] + adjustment * 0.5
        
        if squeeze_batch:
            corrected = corrected.squeeze(0)
        
        return corrected


class ColorStabilizer(nn.Module):
    """
    Color consistency stabilization.
    
    Ensures colors remain consistent across frames by applying
    histogram matching and gamma correction.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def stabilize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply color stabilization.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            
        Returns:
            Color-stabilized frames
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        
        b, t, c, h, w = frames.shape
        
        if frames.max() <= 1.0:
            frames_normalized = frames
        else:
            frames_normalized = frames / 255.0
        
        reference = frames_normalized.mean(dim=1, keepdim=True)
        
        stabilized = []
        for i in range(t):
            frame = frames_normalized[:, i]
            
            corrected = self._match_histogram(frame, reference[:, 0])
            
            stabilized.append(corrected)
        
        result = torch.stack(stabilized, dim=1)
        
        if squeeze_batch:
            result = result.squeeze(0)
        
        return result
    
    def _match_histogram(
        self,
        source: torch.Tensor,
        reference: torch.Tensor
    ) -> torch.Tensor:
        """Match histogram of source to reference."""
        source_adjusted = torch.zeros_like(source)
        
        for ch in range(source.shape[1] if source.dim() > 2 else 3):
            if source.dim() > 2:
                src_ch = source[:, ch]
                ref_ch = reference[:, ch]
            else:
                src_ch = source[ch]
                ref_ch = reference[ch]
            
            src_mean = src_ch.mean()
            ref_mean = ref_ch.mean()
            src_std = src_ch.std() + 1e-8
            ref_std = ref_ch.std() + 1e-8
            
            adjusted = (src_ch - src_mean) / src_std * ref_std + ref_mean
            adjusted = adjusted.clamp(0, 1)
            
            if source.dim() > 2:
                source_adjusted[:, ch] = adjusted
            else:
                source_adjusted[ch] = adjusted
        
        return source_adjusted


class FrameInterpolator(nn.Module):
    """
    Frame interpolation for smoother video.
    
    Generates intermediate frames to increase frame rate
    and smooth motion transitions.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def interpolate(
        self,
        frames: torch.Tensor,
        factor: int = 2
    ) -> torch.Tensor:
        """
        Interpolate frames.
        
        Args:
            frames: Video frames (B, T, C, H, W)
            factor: Interpolation factor
            
        Returns:
            Interpolated frames (B, T*factor, C, H, W)
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True
        
        b, t, c, h, w = frames.shape
        
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        output_frames = []
        
        for i in range(t - 1):
            output_frames.append(frames[:, i])
            
            for j in range(1, factor):
                alpha = j / factor
                
                interp = (1 - alpha) * frames[:, i] + alpha * frames[:, i + 1]
                output_frames.append(interp)
        
        output_frames.append(frames[:, -1])
        
        result = torch.stack(output_frames, dim=1)
        
        if squeeze_batch:
            result = result.squeeze(0)
        
        return result