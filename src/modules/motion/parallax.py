"""Depth-based parallax generation for 3D motion effects.

Provides:
- Depth map analysis
- Parallax flow generation
- Layer separation
- 3D-aware motion warping
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ParallaxConfig:
    """Configuration for depth parallax generation."""
    parallax_strength: float = 0.1
    depth_threshold_near: float = 0.2
    depth_threshold_far: float = 0.8
    layer_count: int = 5
    blend_mode: str = "weighted"
    preserve_edges: bool = True
    temporal_smoothing: float = 0.8
    
    camera_motion: Optional[Tuple[float, float, float]] = None


class DepthLayer:
    """Represents a depth layer in parallax system."""
    
    def __init__(
        self,
        depth_range: Tuple[float, float],
        motion_scale: float,
        mask: Optional[torch.Tensor] = None
    ):
        self.depth_range = depth_range
        self.motion_scale = motion_scale
        self.mask = mask
        self.flow_field: Optional[torch.Tensor] = None
        
        self.min_depth, self.max_depth = depth_range
    
    @property
    def is_valid(self) -> bool:
        """Check if layer is valid."""
        if self.mask is None:
            return True
        return self.mask.any()
    
    def contains_depth(self, depth_value: float) -> bool:
        """Check if depth value falls in this layer."""
        return self.min_depth <= depth_value <= self.max_depth


class DepthParallaxGenerator:
    """Generates parallax motion from depth maps.
    
    Creates 3D motion effects by:
    1. Separating scene into depth layers
    2. Assigning motion scales based on depth
    3. Generating parallax flow fields
    4. Warping frames with depth-aware transforms
    
    Args:
        config: Parallax configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[ParallaxConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or ParallaxConfig()
        self.device = device or torch.device("cpu")
        
        self._layers: List[DepthLayer] = []
        self._depth_cache: Optional[torch.Tensor] = None
    
    def generate_layers(
        self,
        depth_map: torch.Tensor,
        num_layers: Optional[int] = None
    ) -> List[DepthLayer]:
        """Generate depth layers from depth map.
        
        Args:
            depth_map: Depth map [H, W] or [1, H, W]
            num_layers: Override number of layers
            
        Returns:
            List of DepthLayer objects
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        depth_map = depth_map.to(self.device)
        
        if depth_map.max() > 1:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        H, W = depth_map.shape[-2:]
        
        num_layers = num_layers or self.config.layer_count
        
        self._layers = []
        
        for i in range(num_layers):
            min_ratio = i / num_layers
            max_ratio = (i + 1) / num_layers
            
            min_val = depth_map.min() + min_ratio * (depth_map.max() - depth_map.min())
            max_val = depth_map.min() + max_ratio * (depth_map.max() - depth_map.min())
            
            motion_scale = 1.0 - (i / num_layers)
            motion_scale = 0.2 + 0.8 * motion_scale
            
            layer_mask = (depth_map >= min_val) & (depth_map < max_val)
            
            layer = DepthLayer(
                depth_range=(min_val.item(), max_val.item()),
                motion_scale=motion_scale,
                mask=layer_mask.float()
            )
            
            self._layers.append(layer)
        
        self._depth_cache = depth_map
        
        return self._layers
    
    def generate_parallax_flow(
        self,
        depth_map: torch.Tensor,
        camera_motion: Optional[Tuple[float, float, float]] = None,
        num_frames: int = 24
    ) -> torch.Tensor:
        """Generate parallax flow field.
        
        Args:
            depth_map: Depth map [H, W]
            camera_motion: (pan, tilt, zoom) camera parameters
            num_frames: Number of frames
            
        Returns:
            Flow field [num_frames, 2, H, W]
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        depth_map = depth_map.to(self.device)
        
        if camera_motion is None:
            camera_motion = self.config.camera_motion or (0.0, 0.0, 0.0)
        
        pan, tilt, zoom = camera_motion
        
        H, W = depth_map.shape[-2:]
        
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        flows = []
        for frame_idx in range(num_frames):
            t = frame_idx / num_frames
            
            eased_t = self._ease_in_out(t)
            
            parallax_scale = self.config.parallax_strength * eased_t
            
            center_x = W / 2
            center_y = H / 2
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            relative_x = (grid_x - center_x) / center_x
            relative_y = (grid_y - center_y) / center_y
            
            depth_weight = 1.0 - depth_norm.squeeze()
            depth_weight = depth_weight ** 2
            
            pan_flow = relative_x * pan * depth_weight * parallax_scale * W / 2
            tilt_flow = relative_y * tilt * depth_weight * parallax_scale * H / 2
            
            zoom_scale = 1.0 + zoom * eased_t * depth_weight
            zoom_flow_x = (grid_x - center_x) * (zoom_scale - 1)
            zoom_flow_y = (grid_y - center_y) * (zoom_scale - 1)
            
            flow_x = pan_flow + zoom_flow_x
            flow_y = tilt_flow + zoom_flow_y
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def generate_layered_flow(
        self,
        depth_map: torch.Tensor,
        camera_motion: Tuple[float, float, float],
        num_frames: int = 24
    ) -> List[torch.Tensor]:
        """Generate separate flow for each depth layer.
        
        Args:
            depth_map: Depth map
            camera_motion: Camera motion parameters
            num_frames: Number of frames
            
        Returns:
            List of flow tensors, one per layer
        """
        if not self._layers:
            self.generate_layers(depth_map)
        
        pan, tilt, zoom = camera_motion
        H, W = depth_map.shape[-2:]
        
        layer_flows = []
        
        for layer in self._layers:
            if layer.mask is None or not layer.is_valid:
                layer_flow = torch.zeros(2, H, W, device=self.device)
                layer_flows.append(layer_flow)
                continue
            
            flows = []
            for frame_idx in range(num_frames):
                t = frame_idx / num_frames
                eased_t = self._ease_in_out(t)
                
                parallax_scale = self.config.parallax_strength * eased_t * layer.motion_scale
                
                center_x = W / 2
                center_y = H / 2
                
                y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
                x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
                grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
                
                relative_x = (grid_x - center_x) / center_x
                relative_y = (grid_y - center_y) / center_y
                
                flow_x = relative_x * pan * parallax_scale * W / 2
                flow_y = relative_y * tilt * parallax_scale * H / 2
                
                flow = torch.stack([flow_x, flow_y], dim=0)
                flows.append(flow)
            
            layer.flow_field = torch.stack(flows, dim=0)
            layer_flows.append(layer.flow_field)
        
        return layer_flows
    
    def warp_with_parallax(
        self,
        frames: torch.Tensor,
        depth_map: torch.Tensor,
        camera_motion: Optional[Tuple[float, float, float]] = None
    ) -> torch.Tensor:
        """Apply parallax warping to video frames.
        
        Args:
            frames: Video tensor [T, C, H, W]
            depth_map: Depth map [H, W]
            camera_motion: Camera motion parameters
            
        Returns:
            Warped video tensor
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        T, B, C, H, W = frames.shape
        
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0)
        
        depth_map = F.interpolate(
            depth_map,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        if camera_motion is None:
            camera_motion = (0.1, 0.0, 0.0)
        
        pan, tilt, zoom = camera_motion
        
        warped_frames = []
        for t in range(T):
            frame = frames[t]
            
            eased_t = self._ease_in_out(t / T)
            
            center_x = W / 2
            center_y = H / 2
            
            y_coords = torch.linspace(-1, 1, H, device=self.device)
            x_coords = torch.linspace(-1, 1, W, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            depth_weight = 1.0 - depth_norm.squeeze()
            depth_weight = depth_weight ** 2
            
            flow_scale = self.config.parallax_strength * eased_t
            
            grid_x_warp = grid_x + pan * depth_weight * flow_scale
            grid_y_warp = grid_y + tilt * depth_weight * flow_scale
            
            zoom_scale = 1.0 + zoom * depth_weight * flow_scale
            grid_x_warp = grid_x_warp * zoom_scale
            grid_y_warp = grid_y_warp * zoom_scale
            
            grid_x_warp = grid_x_warp.clamp(-1, 1)
            grid_y_warp = grid_y_warp.clamp(-1, 1)
            
            grid = torch.stack([grid_x_warp, grid_y_warp], dim=-1).unsqueeze(0)
            
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            
            warped = F.grid_sample(
                frame,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )
            
            if self.config.preserve_edges:
                warped = self._apply_edge_preservation(warped, depth_norm)
            
            warped_frames.append(warped.squeeze(0))
        
        return torch.stack(warped_frames, dim=0)
    
    def _apply_edge_preservation(
        self,
        frame: torch.Tensor,
        depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Preserve edges during parallax warping."""
        depth_threshold = self.config.depth_threshold_far
        
        edge_mask = (depth_map > depth_threshold).float()
        
        edge_mask = F.max_pool2d(
            edge_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1
        ).squeeze()
        
        edge_mask = F.interpolate(
            edge_mask.unsqueeze(0).unsqueeze(0),
            size=frame.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze()
        
        return frame * (1 - edge_mask) + frame * edge_mask
    
    def generate_depth_aware_flow(
        self,
        depth_map: torch.Tensor,
        motion_field: torch.Tensor,
        num_frames: int = 24
    ) -> torch.Tensor:
        """Generate depth-modulated motion field.
        
        Args:
            depth_map: Depth map [H, W]
            motion_field: Base motion field [2, H, W]
            num_frames: Number of frames
            
        Returns:
            Depth-modulated flow [T, 2, H, W]
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        H, W = depth_map.shape[-2:]
        
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        depth_weight = 1.0 - depth_norm.squeeze()
        depth_weight = depth_weight ** 2
        
        modulated_flows = []
        for t in range(num_frames):
            eased_t = self._ease_in_out(t / num_frames)
            
            modulated_x = motion_field[0] * depth_weight * (1 + eased_t * 0.5)
            modulated_y = motion_field[1] * depth_weight * (1 + eased_t * 0.5)
            
            modulated = torch.stack([modulated_x, modulated_y], dim=0)
            modulated_flows.append(modulated)
        
        return torch.stack(modulated_flows, dim=0)
    
    def create_depth_pyramid(
        self,
        depth_map: torch.Tensor,
        levels: int = 4
    ) -> List[torch.Tensor]:
        """Create depth pyramid for multi-scale parallax.
        
        Args:
            depth_map: Depth map [H, W]
            levels: Number of pyramid levels
            
        Returns:
            List of depth maps at different scales
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        
        pyramid = [depth_map]
        
        current = depth_map
        for _ in range(levels - 1):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyramid.append(current)
        
        return pyramid
    
    def project_to_3d(
        self,
        depth_map: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map [H, W]
            intrinsics: Camera intrinsics [3, 3]
            
        Returns:
            Point cloud [H*W, 3]
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        B, _, H, W = depth_map.shape
        
        if intrinsics is None:
            fx = fy = W / 2
            cx, cy = W / 2, H / 2
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=self.device, dtype=torch.float32)
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        u = grid_x.reshape(-1)
        v = grid_y.reshape(-1)
        z = depth_map.squeeze().reshape(-1)
        
        x = z * (u - intrinsics[0, 2]) / intrinsics[0, 0]
        y = z * (v - intrinsics[1, 2]) / intrinsics[1, 1]
        
        points = torch.stack([x, y, z], dim=-1)
        
        return points
    
    def _ease_in_out(self, t: float) -> float:
        """Ease-in-out interpolation."""
        return t * t * (3 - 2 * t)
    
    def get_layers(self) -> List[DepthLayer]:
        """Get current depth layers."""
        return self._layers.copy()
    
    def reset(self) -> None:
        """Reset generator state."""
        self._layers.clear()
        self._depth_cache = None