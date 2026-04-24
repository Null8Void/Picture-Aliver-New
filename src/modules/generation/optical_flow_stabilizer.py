"""Optical flow-based stabilization for artifact reduction.

Uses optical flow to:
- Stabilize camera shake
- Reduce frame-to-frame warping
- Preserve natural motion
- Detect and fix motion anomalies

Methods:
- RAFT: High accuracy optical flow
- GMFlow: Transformer-based flow
- Farneback: CPU-friendly fallback
- Motion path smoothing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StabilizationMethod(Enum):
    """Stabilization methods."""
    MOTION_SMOOTHING = "motion_smoothing"
    TRAJECTORY_SMOOTHING = "trajectory_smoothing"
    FLOW_COMPENSATION = "flow_compensation"
    FEATURE_MATCHING = "feature_matching"


@dataclass
class StabilizationConfig:
    """Configuration for optical flow stabilization."""
    method: StabilizationMethod = StabilizationMethod.TRAJECTORY_SMOOTHING
    
    smoothing_window: int = 5
    smoothing_strength: float = 0.8
    
    motion_threshold: float = 2.0
    deviation_threshold: float = 5.0
    
    preserve_intended_motion: bool = True
    intended_motion_scale: float = 0.5
    
    use_flow_magnitude_weighting: bool = True
    flow_confidence_weighting: float = 0.3
    
    temporal_consistency: float = 0.7
    edge_aware_smoothing: bool = True
    
    max_correction_per_frame: float = 10.0


class OpticalFlowStabilizer:
    """Optical flow-based video stabilization.
    
    Reduces artifacts through:
    1. Motion path smoothing: Removes camera shake
    2. Flow compensation: Corrects erroneous flow
    3. Trajectory smoothing: Ensures smooth motion
    4. Anomaly detection: Fixes motion outliers
    
    Tradeoffs:
    - Quality vs Speed: RAFT is most accurate but slower
    - Farneback: Fast, lower accuracy
    - Flow compensation: Best for correcting errors
    
    Args:
        config: Stabilization configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[StabilizationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or StabilizationConfig()
        self.device = device or torch.device("cpu")
        
        self._flow_estimator = None
        self._trajectories: List[torch.Tensor] = []
        self._corrections: List[torch.Tensor] = []
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize flow estimator."""
        if self._initialized:
            return
        
        from ..motion.flow_estimator import FlowEstimator
        
        self._flow_estimator = FlowEstimator(device=self.device)
        self._flow_estimator.initialize()
        
        self._initialized = True
    
    def stabilize(
        self,
        frames: torch.Tensor,
        flows: Optional[List[torch.Tensor]] = None,
        return_flows: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Stabilize video frames using optical flow.
        
        Args:
            frames: Video tensor [T, C, H, W]
            flows: Optional pre-computed flow fields
            return_flows: Whether to return flow fields
            
        Returns:
            Stabilized frames, optionally with flow fields
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        T, B, C, H, W = frames.shape
        
        if not self._initialized:
            self.initialize()
        
        if flows is None:
            flows = self._compute_flows(frames)
        
        if self.config.method == StabilizationMethod.TRAJECTORY_SMOOTHING:
            stabilized, corrected_flows = self._trajectory_stabilize(frames, flows)
        elif self.config.method == StabilizationMethod.MOTION_SMOOTHING:
            stabilized, corrected_flows = self._motion_smooth_stabilize(frames, flows)
        elif self.config.method == StabilizationMethod.FLOW_COMPENSATION:
            stabilized, corrected_flows = self._flow_compensate(frames, flows)
        else:
            stabilized, corrected_flows = self._feature_match_stabilize(frames, flows)
        
        self._trajectories.append(corrected_flows)
        
        if return_flows:
            return stabilized, corrected_flows
        
        return stabilized
    
    def _compute_flows(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """Compute optical flow between consecutive frames."""
        T = frames.shape[0]
        
        flows = []
        
        for t in range(T - 1):
            frame1 = frames[t]
            frame2 = frames[t + 1]
            
            if frame1.shape[0] == 3:
                gray1 = 0.299 * frame1[0] + 0.587 * frame1[1] + 0.114 * frame1[2]
                gray2 = 0.299 * frame2[0] + 0.587 * frame2[1] + 0.114 * frame2[2]
            else:
                gray1 = frame1[0]
                gray2 = frame2[0]
            
            flow_field = self._flow_estimator.estimate(gray1, gray2)
            
            if hasattr(flow_field, 'flow'):
                flow = flow_field.flow
            else:
                flow = flow_field
            
            flows.append(flow)
        
        flows.append(torch.zeros_like(flows[-1]) if flows else torch.zeros(2, frames.shape[-2], frames.shape[-1]))
        
        return flows
    
    def _trajectory_stabilize(
        self,
        frames: torch.Tensor,
        flows: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Trajectory-based stabilization."""
        T = frames.shape[0]
        
        trajectory = self._extract_trajectory(flows)
        
        smoothed_trajectory = self._smooth_trajectory(trajectory)
        
        corrections = []
        
        for t in range(T):
            correction = smoothed_trajectory[t] - trajectory[t]
            
            correction_mag = torch.sqrt(correction[0]**2 + correction[1]**2)
            max_correction = self.config.max_correction_per_frame
            correction = correction * torch.clamp(
                correction_mag / max_correction,
                max=1.0
            ).clamp(min=1e-6)
            
            corrections.append(correction)
        
        self._corrections = corrections
        
        stabilized = self._apply_corrections(frames, corrections)
        
        return stabilized, corrections
    
    def _motion_smooth_stabilize(
        self,
        frames: torch.Tensor,
        flows: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Motion smoothing stabilization."""
        T = frames.shape[0]
        
        corrected_flows = []
        
        for t in range(len(flows)):
            flow = flows[t]
            
            magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
            
            anomaly_mask = magnitude > self.config.motion_threshold
            
            corrected_flow = flow.clone()
            for t_other in range(len(flows)):
                if t_other == t:
                    continue
                other_mag = torch.sqrt(flows[t_other][0]**2 + flows[t_other][1]**2)
                other_anomaly = other_mag > self.config.motion_threshold
                
                if not other_anomaly and t_other > 0:
                    weight = 0.5
                    corrected_flow = (
                        (1 - weight) * corrected_flow + 
                        weight * flows[t_other] * (~anomaly_mask).float()
                    )
                    break
            
            corrected_flows.append(corrected_flow)
        
        trajectory = self._extract_trajectory(corrected_flows)
        smoothed = self._smooth_trajectory(trajectory)
        
        corrections = []
        for t in range(T):
            if t < len(smoothed) and t < len(trajectory):
                correction = smoothed[t] - trajectory[t]
                corrections.append(correction)
            else:
                corrections.append(torch.zeros(2, frames.shape[-2], frames.shape[-1], device=self.device))
        
        stabilized = self._apply_corrections(frames, corrections)
        
        return stabilized, corrected_flows
    
    def _flow_compensate(
        self,
        frames: torch.Tensor,
        flows: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Flow compensation stabilization."""
        T = frames.shape[0]
        
        compensated = frames.clone()
        
        compensated_flows = []
        
        for t in range(1, T):
            flow = flows[t - 1]
            
            magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
            
            if magnitude.mean() < self.config.motion_threshold:
                compensated[t] = frames[t]
            else:
                warped = self._warp_frame(frames[t], -flow)
                compensated[t] = warped
            
            compensated_flows.append(flow)
        
        compensated_flows.append(torch.zeros_like(flows[-1]) if flows else torch.zeros(2, frames.shape[-2], frames.shape[-1]))
        
        return compensated, compensated_flows
    
    def _feature_match_stabilize(
        self,
        frames: torch.Tensor,
        flows: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Feature matching based stabilization."""
        T = frames.shape[0]
        
        stabilized = frames.clone()
        
        reference = frames[0]
        if reference.shape[0] == 3:
            ref_gray = 0.299 * reference[0] + 0.587 * reference[1] + 0.114 * reference[2]
        else:
            ref_gray = reference[0]
        
        stabilized_flows = []
        
        for t in range(1, T):
            current = frames[t]
            if current.shape[0] == 3:
                curr_gray = 0.299 * current[0] + 0.587 * current[1] + 0.114 * current[2]
            else:
                curr_gray = current[0]
            
            flow_to_ref = self._flow_estimator.estimate(curr_gray, ref_gray)
            
            if hasattr(flow_to_ref, 'flow'):
                flow = flow_to_ref.flow
            else:
                flow = flow_to_ref
            
            flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2)
            
            if flow_mag.mean() < self.config.deviation_threshold:
                stabilized[t] = self._warp_frame(frames[t], -flow)
            
            stabilized_flows.append(flow)
        
        stabilized_flows.append(torch.zeros_like(stabilized_flows[-1]) if stabilized_flows else torch.zeros(2, frames.shape[-2], frames.shape[-1]))
        
        return stabilized, stabilized_flows
    
    def _extract_trajectory(
        self,
        flows: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Extract motion trajectory from flows."""
        T = len(flows) + 1
        
        trajectory = [torch.zeros(2, flows[0].shape[-2], flows[0].shape[-1], device=self.device)]
        
        cumulative = torch.zeros(2, device=self.device)
        
        for i, flow in enumerate(flows):
            flow_mean = torch.mean(flow, dim=[1, 2])
            cumulative = cumulative + flow_mean
            
            traj_tensor = torch.zeros_like(flow)
            traj_tensor[0] = cumulative[0]
            traj_tensor[1] = cumulative[1]
            
            traj_tensor = traj_tensor / (i + 1) * (i + 1)
            
            trajectory.append(traj_tensor)
        
        return trajectory
    
    def _smooth_trajectory(
        self,
        trajectory: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply smoothing to motion trajectory."""
        if len(trajectory) < self.config.smoothing_window:
            return trajectory
        
        smoothed = []
        window = self.config.smoothing_window
        alpha = self.config.smoothing_strength
        
        for t in range(len(trajectory)):
            start = max(0, t - window // 2)
            end = min(len(trajectory), t + window // 2 + 1)
            
            neighbors = trajectory[start:end]
            mean_traj = torch.stack(neighbors).mean(dim=0)
            
            smooth_t = alpha * trajectory[t] + (1 - alpha) * mean_traj
            smoothed.append(smooth_t)
        
        return smoothed
    
    def _apply_corrections(
        self,
        frames: torch.Tensor,
        corrections: List[torch.Tensor]
    ) -> torch.Tensor:
        """Apply stabilization corrections to frames."""
        T = frames.shape[0]
        
        stabilized = frames.clone()
        
        for t in range(T):
            if t < len(corrections):
                correction = corrections[t]
                
                stabilized[t] = self._warp_frame(frames[t], -correction)
        
        return stabilized
    
    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp frame using flow field."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        B, C, H, W = frame.shape
        
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        flow_x = flow[0] if flow.dim() == 3 else flow[..., 0]
        flow_y = flow[1] if flow.dim() == 3 else flow[..., 1]
        
        grid_x = grid_x + flow_x / W * 2
        grid_y = grid_y + flow_y / H * 2
        
        grid_x = grid_x.clamp(-1, 1)
        grid_y = grid_y.clamp(-1, 1)
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        warped = F.grid_sample(
            frame,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        
        return warped.squeeze(0)
    
    def detect_motion_anomalies(
        self,
        flows: List[torch.Tensor]
    ) -> List[bool]:
        """Detect motion anomalies in flow sequence.
        
        Args:
            flows: List of flow fields
            
        Returns:
            List of boolean anomaly flags
        """
        anomalies = []
        
        magnitudes = []
        for flow in flows:
            mag = torch.sqrt(flow[0]**2 + flow[1]**2).mean().item()
            magnitudes.append(mag)
        
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        for t, mag in enumerate(magnitudes):
            is_anomaly = abs(mag - mean_mag) > 2 * std_mag
            is_anomaly = is_anomaly or mag > self.config.motion_threshold * 2
            anomalies.append(is_anomaly)
        
        return anomalies
    
    def compute_stabilization_metrics(
        self,
        original: torch.Tensor,
        stabilized: torch.Tensor
    ) -> Dict[str, float]:
        """Compute stabilization quality metrics.
        
        Args:
            original: Original frames [T, C, H, W]
            stabilized: Stabilized frames
            
        Returns:
            Dictionary of metrics
        """
        T = original.shape[0]
        
        metrics = {}
        
        frame_diffs = []
        for t in range(T):
            diff = torch.abs(original[t] - stabilized[t]).mean().item()
            frame_diffs.append(diff)
        
        metrics['mean_diff'] = np.mean(frame_diffs)
        metrics['max_diff'] = np.max(frame_diffs)
        metrics['stability_score'] = 1.0 - np.mean(frame_diffs)
        
        inter_frame_orig = []
        inter_frame_stab = []
        
        for t in range(1, T):
            orig_diff = torch.abs(original[t] - original[t-1]).mean().item()
            stab_diff = torch.abs(stabilized[t] - stabilized[t-1]).mean().item()
            inter_frame_orig.append(orig_diff)
            inter_frame_stab.append(stab_diff)
        
        metrics['original_smoothness'] = np.std(inter_frame_orig)
        metrics['stabilized_smoothness'] = np.std(inter_frame_stab)
        metrics['smoothness_improvement'] = (
            metrics['original_smoothness'] - metrics['stabilized_smoothness']
        )
        
        return metrics