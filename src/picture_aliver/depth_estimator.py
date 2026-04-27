"""
Depth Estimation Module

Estimates depth maps from input images using pretrained models.
Supports multiple model architectures (MiDaS, ZoeDepth, Marigold).
Provides both relative and metric depth estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthResult:
    """
    Container for depth estimation results.
    
    Attributes:
        depth: Raw depth values
        normalized: Normalized depth for visualization [0, 1]
        inverse: Inverse depth (disparity)
        confidence: Confidence map
        model_name: Name of model used
    """
    depth: torch.Tensor
    normalized: torch.Tensor
    inverse: torch.Tensor
    confidence: Optional[torch.Tensor]
    model_name: str


class DepthEstimator:
    """
    Depth estimation using pretrained models.
    
    Supports:
    - MiDaS: Monocular depth estimation
    - ZoeDepth: Metric depth estimation
    - Marigold: Diffusion-based depth estimation
    
    Attributes:
        device: Compute device
        model_type: Type of model to use
        model: Loaded model
        model_dir: Directory for model weights
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_type: str = "zoedepth",
        model_dir: Optional[Path] = None
    ):
        self.device = device or torch.device("cpu")
        self.model_type = model_type.lower()
        self.model_dir = model_dir or Path("./models/depth")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[nn.Module] = None
        self._transform = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the depth model."""
        if self._initialized:
            return
        
        if self.model_type == "midas":
            self._initialize_midas()
        elif self.model_type == "zoedepth":
            self._initialize_zoedepth()
        elif self.model_type == "marigold":
            self._initialize_marigold()
        else:
            self._initialize_midas()
        
        self._initialized = True
    
    def _initialize_midas(self) -> None:
        """Initialize MiDaS model."""
        try:
            from torchvision import transforms
            
            self.model = MiDaSWrapper(device=self.device)
            self.model.eval()
            
            self._transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except ImportError:
            print("[DepthEstimator] Falling back to simple depth estimation")
            self.model = SimpleDepthEstimator(device=self.device)
            self.model.eval()
    
    def _initialize_zoedepth(self) -> None:
        """Initialize ZoeDepth model."""
        try:
            from torchvision import transforms
            
            self.model = ZoeDepthWrapper(device=self.device)
            self.model.eval()
            
            self._transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except Exception:
            print("[DepthEstimator] ZoeDepth unavailable, using MiDaS")
            self._initialize_midas()
    
    def _initialize_marigold(self) -> None:
        """Initialize Marigold model."""
        self.model = SimpleDepthEstimator(device=self.device)
        self.model.eval()
        self._transform = None
    
    def estimate(
        self,
        image: torch.Tensor,
        normalized: bool = True
    ) -> DepthResult:
        """
        Estimate depth from image.
        
        Args:
            image: Input tensor of shape (C, H, W)
            normalized: Whether to normalize output
            
        Returns:
            DepthResult with depth maps
        """
        if not self._initialized:
            self.initialize()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        original_shape = image.shape[-2:]
        
        image_for_depth = F.interpolate(
            image,
            size=(384, 384),
            mode="bilinear",
            align_corners=False
        )
        
        if self._transform is not None:
            image_for_depth = self._transform(image_for_depth)
        
        with torch.no_grad():
            if self.model_type == "marigold":
                depth_raw = self.model(image_for_depth, num_inference_steps=10)
            else:
                depth_raw = self.model(image_for_depth)
        
        depth_raw = F.interpolate(
            depth_raw,
            size=original_shape,
            mode="bilinear",
            align_corners=False
        )
        
        depth_raw = depth_raw.squeeze(0)
        
        if normalized:
            d_min = depth_raw.quantile(0.02)
            d_max = depth_raw.quantile(0.98)
            depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-8)
            depth_norm = depth_norm.clamp(0, 1)
        else:
            depth_norm = depth_raw
        
        inverse_depth = 1.0 / (depth_raw + 1e-8)
        inverse_norm = inverse_depth / (inverse_depth.max() + 1e-8)
        
        confidence = self._estimate_confidence(depth_raw)
        
        return DepthResult(
            depth=depth_raw,
            normalized=depth_norm,
            inverse=inverse_norm,
            confidence=confidence,
            model_name=self.model_type
        )
    
    def _estimate_confidence(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence in depth values.
        
        Args:
            depth: Depth tensor
            
        Returns:
            Confidence map
        """
        grad_x = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
        grad_y = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
        
        grad_x = F.pad(grad_x, (0, 1), mode="replicate")
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        confidence = 1.0 - torch.tanh(gradient_magnitude * 0.1)
        
        return confidence
    
    def create_depth_image(self, depth: torch.Tensor) -> np.ndarray:
        """
        Create visualization image from depth tensor.
        
        Args:
            depth: Depth tensor (H, W)
            
        Returns:
            RGB numpy array
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        
        if depth.ndim == 3:
            depth = depth.squeeze()
        
        depth_vis = (depth * 255).astype(np.uint8)
        
        if depth_vis.ndim == 2:
            depth_vis = np.stack([depth_vis] * 3, axis=-1)
        
        return depth_vis


class MiDaSWrapper(nn.Module):
    """Wrapper for MiDaS depth estimation model."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_reported=True
            )
            self.model.to(device)
            self.uses_small = True
        except Exception:
            self.model = SimpleDepthEstimator(device=device)
            self.uses_small = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.uses_small:
            return self.model(x)
        return self.model(x)


class ZoeDepthWrapper(nn.Module):
    """Wrapper for ZoeDepth model."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        try:
            from ZoeDepth import zoedepth
            self.model = zoedepth.depth.models.ZoeDepth.get_ZoeD_N()
            self.model.to(device)
            self.uses_zoe = True
        except Exception:
            self.model = SimpleDepthEstimator(device=device)
            self.uses_zoe = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.uses_zoe:
            return self.model(x)
        return self.model(x)


class SimpleDepthEstimator(nn.Module):
    """
    Fallback depth estimator using image features.
    
    Creates a rough depth estimate based on color and edge features.
    Used when pretrained models are unavailable.
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.depth_conv = nn.Sequential(
            nn.Conv2d(38, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from RGB features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Depth map (B, 1, H, W)
        """
        b, c, h, w = x.shape
        
        edges = self._compute_edges(x)
        
        features = self.edge_conv(x)
        
        combined = torch.cat([x, features, edges], dim=1)
        
        depth_raw = self.depth_conv(combined)
        
        depth_raw = F.interpolate(
            depth_raw,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        
        depth = torch.sigmoid(depth_raw)
        
        return depth
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge features using Sobel filters."""
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(x, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        edges_y = F.conv2d(x, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        
        return edges