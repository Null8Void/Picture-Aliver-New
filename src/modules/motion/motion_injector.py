"""Motion injection system for diffusion model conditioning.

Handles:
- Motion field conditioning
- Depth-based parallax
- Flow field encoding
- ControlNet integration
- Temporal attention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningType(Enum):
    """Types of motion conditioning."""
    CONTROLNET = "controlnet"
    REFERENCE = "reference"
    TEMPORAL_ATTENTION = "temporal_attention"
    MOTION_LORA = "motion_lora"
    FLOW_INJECTION = "flow_injection"
    LATENT_WARP = "latent_warp"


class MotionInjectionStrategy(Enum):
    """Strategies for injecting motion."""
    ADDITIVE = "additive"
    ATTENTION = "attention"
    WARPING = "warping"
    HYPERNETWORK = "hypernetwork"
    ADAPTER = "adapter"


@dataclass
class MotionConditioning:
    """Motion conditioning data for diffusion models."""
    conditioning_type: ConditioningType
    control_image: Optional[torch.Tensor] = None
    latent_flow: Optional[torch.Tensor] = None
    temporal_weights: Optional[torch.Tensor] = None
    motion_strength: float = 1.0
    
    depth_map: Optional[torch.Tensor] = None
    segmentation: Optional[torch.Tensor] = None
    
    camera_params: Optional[Dict] = None
    object_motions: Optional[List] = None
    
    extra_channels: Optional[torch.Tensor] = None


class MotionInjector:
    """Injects motion conditioning into video diffusion models.
    
    Supports multiple strategies:
    - ControlNet: Uses flow/depth as control signal
    - Reference: Uses initial frame as reference
    - Temporal Attention: Modulates cross-frame attention
    - Motion LoRA: Lightweight adapter for motion
    - Flow Injection: Direct flow field injection
    - Latent Warping: Warps latent representations
    
    Args:
        strategy: Injection strategy
        device: Target compute device
    """
    
    def __init__(
        self,
        strategy: MotionInjectionStrategy = MotionInjectionStrategy.CONTROLNET,
        device: Optional[torch.device] = None
    ):
        self.strategy = strategy
        self.device = device or torch.device("cpu")
        
        self._initialized = False
        self._controlnet = None
        self._motion_adapter = None
        
        self._conditioning_cache: Dict[str, torch.Tensor] = {}
    
    def initialize(self) -> None:
        """Initialize injection components."""
        if self._initialized:
            return
        
        if self.strategy == MotionInjectionStrategy.CONTROLNET:
            self._init_controlnet()
        elif self.strategy == MotionInjectionStrategy.MOTION_LORA:
            self._init_motion_lora()
        elif self.strategy == MotionInjectionStrategy.ADAPTER:
            self._init_adapter()
        
        self._initialized = True
    
    def _init_controlnet(self) -> None:
        """Initialize ControlNet for motion control."""
        pass
    
    def _init_motion_lora(self) -> None:
        """Initialize Motion LoRA adapter."""
        pass
    
    def _init_adapter(self) -> None:
        """Initialize motion adapter."""
        pass
    
    def prepare_conditioning(
        self,
        initial_image: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        camera_motion: Optional[torch.Tensor] = None,
        object_flows: Optional[List[torch.Tensor]] = None,
        env_flows: Optional[torch.Tensor] = None,
        num_frames: int = 16
    ) -> MotionConditioning:
        """Prepare motion conditioning from various inputs.
        
        Args:
            initial_image: First frame [C, H, W]
            depth_map: Depth map [H, W]
            segmentation: Segmentation mask
            camera_motion: Camera motion parameters
            object_flows: List of object motion fields
            env_flows: Environmental motion field
            num_frames: Number of frames
            
        Returns:
            MotionConditioning object
        """
        control_image = self._create_control_image(
            initial_image, depth_map, segmentation, num_frames
        )
        
        latent_flow = self._combine_flows(
            camera_motion, object_flows, env_flows, num_frames
        )
        
        temporal_weights = self._compute_temporal_weights(
            latent_flow, num_frames
        )
        
        return MotionConditioning(
            conditioning_type=ConditioningType.CONTROLNET,
            control_image=control_image,
            latent_flow=latent_flow,
            temporal_weights=temporal_weights,
            motion_strength=1.0,
            depth_map=depth_map,
            segmentation=segmentation,
            camera_params=camera_motion,
            object_motions=object_flows
        )
    
    def _create_control_image(
        self,
        image: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        segmentation: Optional[torch.Tensor],
        num_frames: int
    ) -> torch.Tensor:
        """Create control image for conditioning.
        
        Args:
            image: Base image [C, H, W]
            depth_map: Optional depth map
            segmentation: Optional segmentation
            num_frames: Number of frames to create
            
        Returns:
            Control image tensor [num_frames, 3+1, H, W]
        """
        C, H, W = image.shape
        
        if depth_map is not None:
            if depth_map.dim() == 2:
                depth_map = depth_map.unsqueeze(0)
            if depth_map.shape[0] != 1:
                depth_map = depth_map[:1]
            depth_map = F.interpolate(
                depth_map.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_channels = depth_normalized.repeat(num_frames, 1, 1)
        else:
            depth_channels = torch.zeros(num_frames, H, W, device=self.device)
        
        if segmentation is not None:
            if isinstance(segmentation, torch.Tensor):
                seg = segmentation
            else:
                seg = torch.from_numpy(segmentation)
            seg = seg.to(self.device).float()
            
            if seg.dim() == 2:
                seg = seg.unsqueeze(0)
            
            seg = F.interpolate(
                seg.unsqueeze(0),
                size=(H, W),
                mode="nearest"
            ).squeeze(0)
            
            seg_channels = seg.repeat(num_frames, 1, 1)
        else:
            seg_channels = torch.zeros(num_frames, H, W, device=self.device)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image_expanded = image.expand(num_frames, -1, -1, -1)
        
        control = torch.cat([image_expanded, depth_channels.unsqueeze(1)], dim=1)
        control = torch.cat([control, seg_channels.unsqueeze(1)], dim=1)
        
        return control
    
    def _combine_flows(
        self,
        camera_motion: Optional[torch.Tensor],
        object_flows: Optional[List[torch.Tensor]],
        env_flows: Optional[torch.Tensor],
        num_frames: int
    ) -> torch.Tensor:
        """Combine motion flows into latent space.
        
        Args:
            camera_motion: Camera motion [T, 2, H, W]
            object_flows: List of object flows
            env_flows: Environmental flow [2, H, W]
            num_frames: Number of frames
            
        Returns:
            Combined latent flow [T, 2, H, W]
        """
        if camera_motion is not None:
            combined = camera_motion
        else:
            combined = torch.zeros(num_frames, 2, 512, 512, device=self.device)
        
        if object_flows is not None:
            for obj_flow in object_flows:
                if obj_flow is not None:
                    if obj_flow.dim() == 3:
                        obj_flow = obj_flow.unsqueeze(0)
                    if obj_flow.shape[0] < num_frames:
                        obj_flow = obj_flow.repeat(
                            (num_frames + obj_flow.shape[0] - 1) // obj_flow.shape[0], 1, 1, 1
                        )[:num_frames]
                    combined = combined + obj_flow
        
        if env_flows is not None:
            env_expanded = env_flows.unsqueeze(0).expand(num_frames, -1, -1, -1)
            combined = combined + env_expanded
        
        return combined
    
    def _compute_temporal_weights(
        self,
        flow_field: torch.Tensor,
        num_frames: int
    ) -> torch.Tensor:
        """Compute temporal attention weights from flow.
        
        Args:
            flow_field: Flow field [T, 2, H, W]
            num_frames: Number of frames
            
        Returns:
            Temporal weights [T, T]
        """
        T = num_frames
        H, W = flow_field.shape[2], flow_field.shape[3]
        
        flow_magnitude = torch.sqrt(flow_field[:, 0:1] ** 2 + flow_field[:, 1:2] ** 2)
        flow_magnitude = flow_magnitude.mean(dim=[2, 3])
        
        magnitude_normalized = (flow_magnitude - flow_magnitude.min()) / (
            flow_magnitude.max() - flow_magnitude.min() + 1e-8
        )
        
        weights = torch.zeros(T, T, device=self.device)
        for i in range(T):
            for j in range(T):
                temporal_dist = abs(i - j)
                motion_similarity = 1.0 - abs(
                    magnitude_normalized[i] - magnitude_normalized[j]
                )
                weights[i, j] = torch.exp(-temporal_dist * 0.5) * motion_similarity
        
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        return weights
    
    def inject_into_diffusion(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int,
        model: nn.Module
    ) -> torch.Tensor:
        """Inject motion conditioning into diffusion model.
        
        Args:
            latents: Current latent state [B, C, H, W]
            conditioning: Motion conditioning data
            timestep: Current diffusion timestep
            model: Diffusion model
            
        Returns:
            Modified latents
        """
        if not self._initialized:
            self.initialize()
        
        if self.strategy == MotionInjectionStrategy.CONTROLNET:
            return self._inject_controlnet(latents, conditioning, timestep, model)
        elif self.strategy == MotionInjectionStrategy.FLOW_INJECTION:
            return self._inject_flow(latents, conditioning, timestep)
        elif self.strategy == MotionInjectionStrategy.WARPING:
            return self._inject_warping(latents, conditioning, timestep)
        elif self.strategy == MotionInjectionStrategy.ATTENTION:
            return self._inject_attention(latents, conditioning, timestep, model)
        elif self.strategy == MotionInjectionStrategy.ADAPTER:
            return self._inject_adapter(latents, conditioning, timestep, model)
        else:
            return latents
    
    def _inject_controlnet(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int,
        model: nn.Module
    ) -> torch.Tensor:
        """Inject via ControlNet-style conditioning."""
        if conditioning.control_image is None:
            return latents
        
        control = conditioning.control_image
        if control.dim() == 4:
            control = control[0:1]
        
        flow_strength = conditioning.motion_strength
        temporal_decay = 1.0 - timestep / 1000.0
        
        combined_strength = flow_strength * temporal_decay
        
        if conditioning.latent_flow is not None:
            latent_H, latent_W = latents.shape[2], latents.shape[3]
            flow = conditioning.latent_flow[0:1]
            flow_resized = F.interpolate(
                flow,
                size=(latent_H, latent_W),
                mode="bilinear",
                align_corners=False
            )
            flow_strength_map = torch.sqrt(
                flow_resized[0:1, 0:1] ** 2 + flow_resized[0:1, 1:2] ** 2
            )
            flow_strength_map = flow_strength_map / (flow_strength_map.max() + 1e-8)
            
            flow_contribution = flow_resized * flow_strength_map * combined_strength
            
            latents = latents + flow_contribution
        
        return latents
    
    def _inject_flow(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int
    ) -> torch.Tensor:
        """Direct flow injection into latents."""
        if conditioning.latent_flow is None:
            return latents
        
        latent_H, latent_W = latents.shape[2], latents.shape[3]
        flow = F.interpolate(
            conditioning.latent_flow[0:1],
            size=(latent_H, latent_W),
            mode="bilinear",
            align_corners=False
        )
        
        strength = conditioning.motion_strength * (1.0 - timestep / 1000.0)
        
        latents = latents + flow * strength
        
        return latents
    
    def _inject_warping(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int
    ) -> torch.Tensor:
        """Warping-based motion injection."""
        if conditioning.latent_flow is None:
            return latents
        
        latent_H, latent_W = latents.shape[2], latents.shape[3]
        flow = conditioning.latent_flow[0:1]
        
        flow_resized = F.interpolate(
            flow,
            size=(latent_H, latent_W),
            mode="bilinear",
            align_corners=False
        )
        
        y_coords = torch.linspace(-1, 1, latent_H, device=self.device)
        x_coords = torch.linspace(-1, 1, latent_W, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        flow_scale = 0.1
        grid_x = grid_x + flow_resized[0, 0] / latent_W * 2 * flow_scale
        grid_y = grid_y + flow_resized[0, 1] / latent_H * 2 * flow_scale
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        strength = conditioning.motion_strength * (1.0 - timestep / 1000.0)
        
        warped = F.grid_sample(
            latents,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        
        latents = (1 - strength) * latents + strength * warped
        
        return latents
    
    def _inject_attention(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int,
        model: nn.Module
    ) -> torch.Tensor:
        """Attention-based motion injection."""
        if conditioning.temporal_weights is None:
            return latents
        
        T = conditioning.temporal_weights.shape[0]
        
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        
        B, C, H, W = latents.shape
        
        latent_seq = latents.view(B, C, H * W).transpose(1, 2)
        
        attention_scores = conditioning.temporal_weights.unsqueeze(0)
        
        attended = torch.bmm(attention_scores, latent_seq.transpose(1, 2))
        attended = attended.transpose(1, 2).view(B, C, H, W)
        
        strength = conditioning.motion_strength * (1.0 - timestep / 1000.0)
        latents = (1 - strength) * latents + strength * attended
        
        return latents
    
    def _inject_adapter(
        self,
        latents: torch.Tensor,
        conditioning: MotionConditioning,
        timestep: int,
        model: nn.Module
    ) -> torch.Tensor:
        """Adapter-based motion injection."""
        if self._motion_adapter is None:
            self._init_adapter()
        
        if self._motion_adapter is None:
            return latents
        
        adapter_input = conditioning.latent_flow
        if adapter_input is not None:
            adapter_output = self._motion_adapter(adapter_input)
            
            strength = conditioning.motion_strength * (1.0 - timestep / 1000.0)
            
            latents = latents + adapter_output * strength
        
        return latents
    
    def create_motion_embedding(
        self,
        flow_field: torch.Tensor,
        embedding_dim: int = 128
    ) -> torch.Tensor:
        """Create motion embedding from flow field.
        
        Args:
            flow_field: Flow field [2, H, W]
            embedding_dim: Embedding dimension
            
        Returns:
            Motion embedding [embedding_dim]
        """
        if flow_field.dim() == 3:
            flow_field = flow_field.unsqueeze(0)
        
        B, C, H, W = flow_field.shape
        
        magnitude = torch.sqrt(flow_field[:, 0:1] ** 2 + flow_field[:, 1:2] ** 2)
        angle = torch.atan2(flow_field[:, 1:2], flow_field[:, 0:1])
        
        encoded = torch.cat([magnitude, angle], dim=1)
        
        pooled = F.adaptive_avg_pool2d(encoded, (1, 1)).squeeze(-1).squeeze(-1)
        
        embedding = self._learn_embedding(pooled, embedding_dim)
        
        return embedding
    
    def _learn_embedding(
        self,
        features: torch.Tensor,
        output_dim: int
    ) -> torch.Tensor:
        """Learn embedding from features."""
        input_dim = features.shape[1]
        
        weight = torch.randn(input_dim, output_dim, device=self.device) * 0.02
        bias = torch.zeros(output_dim, device=self.device)
        
        projected = torch.matmul(features, weight) + bias
        
        return projected
    
    def modulate_latents(
        self,
        latents: torch.Tensor,
        motion_embedding: torch.Tensor,
        scale: float = 1.0
    ) -> torch.Tensor:
        """Modulate latents with motion embedding.
        
        Args:
            latents: Latent tensor [B, C, H, W]
            motion_embedding: Motion embedding [D]
            scale: Modulation scale
            
        Returns:
            Modulated latents
        """
        if motion_embedding.dim() == 1:
            motion_embedding = motion_embedding.unsqueeze(0)
        
        B, C, H, W = latents.shape
        D = motion_embedding.shape[1]
        
        scale_factor = motion_embedding[:, :C].view(B, C, 1, 1)
        shift_factor = motion_embedding[:, C:2*C].view(B, C, 1, 1) if D > C else torch.zeros_like(scale_factor)
        
        modulated = (1 + scale * scale_factor) * latents + scale * shift_factor
        
        return modulated
    
    def encode_to_latent(
        self,
        image: torch.Tensor,
        vae: nn.Module
    ) -> torch.Tensor:
        """Encode image to latent space.
        
        Args:
            image: Image tensor [B, C, H, W]
            vae: VAE model
            
        Returns:
            Latent tensor
        """
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
        
        return latents
    
    def decode_from_latent(
        self,
        latents: torch.Tensor,
        vae: nn.Module
    ) -> torch.Tensor:
        """Decode latents to image space.
        
        Args:
            latents: Latent tensor
            vae: VAE model
            
        Returns:
            Image tensor
        """
        with torch.no_grad():
            decoded = vae.decode(latents).sample
        
        return decoded