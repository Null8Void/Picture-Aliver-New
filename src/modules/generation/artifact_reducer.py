"""Main artifact reducer integrating all reduction methods.

Combines:
- Depth conditioning
- ControlNet guidance
- Latent consistency
- Optical flow stabilization
- Frame interpolation

For comprehensive artifact reduction without filtering/censorship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ArtifactConfig:
    """Configuration for artifact reduction."""
    enable_depth_conditioning: bool = True
    enable_controlnet: bool = True
    enable_latent_consistency: bool = True
    enable_flow_stabilization: bool = True
    enable_interpolation: bool = True
    
    depth_strength: float = 0.8
    controlnet_strength: float = 0.8
    latent_strength: float = 0.7
    stabilization_strength: float = 0.8
    interpolation_factor: int = 2
    
    processing_order: List[str] = field(default_factory=lambda: [
        "depth_conditioning",
        "controlnet_guidance",
        "latent_consistency",
        "flow_stabilization",
        "interpolation"
    ])
    
    verbose: bool = False
    cache_results: bool = True


class ArtifactReducer:
    """Comprehensive artifact reduction system.
    
    Reduces visual artifacts through architectural solutions:
    
    1. DEPTH CONDITIONING
       - Reduces: Flickering, warping, face distortion
       - Method: MiDaS/ZoeDepth guided latent warping
       - Tradeoff: Quality vs Speed (ensemble slower but better)
    
    2. CONTROLNET GUIDANCE
       - Reduces: Structural inconsistency, face distortion
       - Method: Depth/edge/pose conditioning
       - Tradeoff: Precision vs Generality
    
    3. LATENT CONSISTENCY
       - Reduces: Flickering, temporal inconsistency
       - Method: Cross-frame attention, trajectory smoothing
       - Tradeoff: Coherence vs Natural variation
    
    4. FLOW STABILIZATION
       - Reduces: Camera shake, motion warping
       - Method: Optical flow path smoothing
       - Tradeoff: Stability vs Intended motion
    
    5. FRAME INTERPOLATION
       - Reduces: Motion judder, frame gaps
       - Method: RIFE/FILM intermediate frame generation
       - Tradeoff: Smoothness vs Detail preservation
    
    Args:
        config: Artifact reduction configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[ArtifactConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or ArtifactConfig()
        self.device = device or torch.device("cpu")
        
        self._depth_conditioner = None
        self._controlnet_guidance = None
        self._latent_consistency = None
        self._flow_stabilizer = None
        self._frame_interpolator = None
        
        self._initialized = False
        self._cache: Dict[str, torch.Tensor] = {}
    
    def initialize(self) -> None:
        """Initialize all reduction components."""
        if self._initialized:
            return
        
        if self.config.enable_depth_conditioning:
            self._init_depth_conditioner()
        
        if self.config.enable_controlnet:
            self._init_controlnet()
        
        if self.config.enable_latent_consistency:
            self._init_latent_consistency()
        
        if self.config.enable_flow_stabilization:
            self._init_flow_stabilizer()
        
        if self.config.enable_interpolation:
            self._init_interpolator()
        
        self._initialized = True
    
    def _init_depth_conditioner(self) -> None:
        """Initialize depth conditioner."""
        from .depth_conditioning import DepthConditioner, DepthConditioningConfig
        
        config = DepthConditioningConfig(
            strength=self.config.depth_strength,
            temporal_smoothing=0.7,
            preserve_edges=True
        )
        
        self._depth_conditioner = DepthConditioner(config, self.device)
        self._depth_conditioner.initialize()
    
    def _init_controlnet(self) -> None:
        """Initialize ControlNet guidance."""
        from .controlnet_guidance import ControlNetGuidance, ControlNetConfig, ControlNetType
        
        config = ControlNetConfig(
            controlnet_type=ControlNetType.DEPTH,
            strength=self.config.controlnet_strength
        )
        
        self._controlnet_guidance = ControlNetGuidance(config, self.device)
        self._controlnet_guidance.initialize()
    
    def _init_latent_consistency(self) -> None:
        """Initialize latent consistency manager."""
        from .latent_consistency import LatentConsistencyManager, LatentConsistencyConfig
        
        config = LatentConsistencyConfig(
            enable_cross_attention=True,
            cross_attention_strength=self.config.latent_strength,
            temporal_smoothing=0.7
        )
        
        self._latent_consistency = LatentConsistencyManager(config, self.device)
        self._latent_consistency.initialize()
    
    def _init_flow_stabilizer(self) -> None:
        """Initialize optical flow stabilizer."""
        from .optical_flow_stabilizer import OpticalFlowStabilizer, StabilizationConfig
        
        config = StabilizationConfig(
            smoothing_strength=self.config.stabilization_strength,
            preserve_intended_motion=True
        )
        
        self._flow_stabilizer = OpticalFlowStabilizer(config, self.device)
        self._flow_stabilizer.initialize()
    
    def _init_interpolator(self) -> None:
        """Initialize frame interpolator."""
        from .frame_interpolator import FrameInterpolator, FrameInterpolatorConfig
        
        config = FrameInterpolatorConfig(
            interpolation_factor=self.config.interpolation_factor
        )
        
        self._frame_interpolator = FrameInterpolator(config, self.device)
        self._frame_interpolator.initialize()
    
    def reduce_artifacts(
        self,
        frames: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        reference_frame: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply comprehensive artifact reduction.
        
        Args:
            frames: Video frames [T, C, H, W]
            depth_map: Optional depth map for guidance
            segmentation: Optional segmentation mask
            reference_frame: Optional reference for consistency
            
        Returns:
            Artifact-reduced frames
        """
        if not self._initialized:
            self.initialize()
        
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        result = frames.clone()
        
        for step in self.config.processing_order:
            if self.config.verbose:
                print(f"Applying: {step}")
            
            if step == "depth_conditioning" and self._depth_conditioner is not None:
                result = self._apply_depth_conditioning(result, depth_map)
            
            elif step == "controlnet_guidance" and self._controlnet_guidance is not None:
                result = self._apply_controlnet_guidance(result, depth_map, segmentation)
            
            elif step == "latent_consistency" and self._latent_consistency is not None:
                result = self._apply_latent_consistency(result, depth_map, segmentation)
            
            elif step == "flow_stabilization" and self._flow_stabilizer is not None:
                result = self._apply_flow_stabilization(result)
            
            elif step == "interpolation" and self._frame_interpolator is not None:
                result = self._apply_interpolation(result)
        
        return result
    
    def _apply_depth_conditioning(
        self,
        frames: torch.Tensor,
        depth_map: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply depth-based artifact reduction."""
        if depth_map is None:
            first_frame = frames[0]
            
            if self._depth_conditioner is not None:
                depth_map = self._depth_conditioner.estimate_depth(first_frame)
        
        if depth_map is None:
            return frames
        
        T = frames.shape[0]
        
        depth_sequence = []
        for t in range(T):
            d = depth_map.clone()
            depth_sequence.append(d)
        
        smooth_depth = self._depth_conditioner.temporal_depth_smooth(depth_sequence)
        
        for t in range(T):
            timestep_ratio = t / max(T - 1, 1)
            
            guided = self._depth_conditioner.apply_depth_guidance(
                frames[t].unsqueeze(0),
                smooth_depth[t].unsqueeze(0),
                timestep_ratio
            )
            
            frames[t] = guided.squeeze(0)
        
        return frames
    
    def _apply_controlnet_guidance(
        self,
        frames: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        segmentation: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply ControlNet-based guidance."""
        T = frames.shape[0]
        
        if frames.shape[0] == 3:
            first_frame = frames[:3]
        else:
            first_frame = frames[0:1]
        
        control = self._controlnet_guidance.preprocess(
            first_frame,
            self._controlnet_guidance.config.controlnet_type
        )
        
        controls = self._controlnet_guidance.create_control_sequence(
            first_frame,
            T,
            self._controlnet_guidance.config.controlnet_type
        )
        
        for t in range(T):
            step_ratio = t / T
            
            guided = self._controlnet_guidance.apply_guidance(
                frames[t].unsqueeze(0),
                controls[t],
                int(step_ratio * 100),
                100
            )
            
            frames[t] = guided.squeeze(0)
        
        return frames
    
    def _apply_latent_consistency(
        self,
        frames: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        segmentation: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply latent consistency enforcement."""
        if self._latent_consistency is None:
            return frames
        
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        consistent = self._latent_consistency.enforce_consistency(
            frames,
            timestep=0,
            depth_map=depth_map,
            segmentation=segmentation
        )
        
        return consistent.squeeze(0) if consistent.dim() == 5 else consistent
    
    def _apply_flow_stabilization(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """Apply optical flow stabilization."""
        if self._flow_stabilizer is None:
            return frames
        
        stabilized = self._flow_stabilizer.stabilize(frames)
        
        return stabilized.squeeze(0) if stabilized.dim() == 5 else stabilized
    
    def _apply_interpolation(
        self,
        frames: torch.Tensor
    ) -> torch.Tensor:
        """Apply frame interpolation."""
        if self._frame_interpolator is None:
            return frames
        
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        if self.config.interpolation_factor > 1:
            interpolated = self._frame_interpolator.interpolate(
                frames,
                self.config.interpolation_factor - 1
            )
            return interpolated.squeeze(0) if interpolated.dim() == 5 else interpolated
        
        return frames.squeeze(0) if frames.dim() == 5 else frames
    
    def estimate_depth(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """Estimate depth for conditioning."""
        if not self._initialized:
            self.initialize()
        
        if self._depth_conditioner is not None:
            return self._depth_conditioner.estimate_depth(image)
        
        return torch.zeros(1, 1, image.shape[-2], image.shape[-1], device=self.device)
    
    def compute_quality_metrics(
        self,
        original: torch.Tensor,
        reduced: torch.Tensor
    ) -> Dict[str, float]:
        """Compute artifact reduction quality metrics.
        
        Args:
            original: Original frames
            reduced: Artifact-reduced frames
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        T = min(original.shape[0], reduced.shape[0])
        
        flicker_scores = []
        for t in range(1, T):
            orig_diff = torch.abs(original[t] - original[t-1]).mean()
            red_diff = torch.abs(reduced[t] - reduced[t-1]).mean()
            
            flicker_scores.append((orig_diff - red_diff).item())
        
        metrics['flicker_reduction'] = np.mean(flicker_scores) if flicker_scores else 0
        
        warping_scores = []
        for t in range(T):
            orig_grad = self._compute_gradient_magnitude(original[t])
            red_grad = self._compute_gradient_magnitude(reduced[t])
            
            warping_scores.append((orig_grad - red_grad).item())
        
        metrics['warping_reduction'] = np.mean(warping_scores) if warping_scores else 0
        
        stability_scores = []
        for t in range(1, T):
            stability = torch.abs(reduced[t] - reduced[t-1]).std().item()
            stability_scores.append(stability)
        
        metrics['stability'] = np.mean(stability_scores) if stability_scores else 0
        
        return metrics
    
    def _compute_gradient_magnitude(
        self,
        frame: torch.Tensor
    ) -> float:
        """Compute gradient magnitude for warping analysis."""
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]
        
        grad_x = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        grad_y = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2).mean().item()
        
        return magnitude
    
    def clear_cache(self) -> None:
        """Clear cached results."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return (
            f"ArtifactReducer("
            f"depth={self._depth_conditioner is not None}, "
            f"controlnet={self._controlnet_guidance is not None}, "
            f"latent={self._latent_consistency is not None}, "
            f"flow={self._flow_stabilizer is not None}, "
            f"interp={self._frame_interpolator is not None}"
            f")"
        )