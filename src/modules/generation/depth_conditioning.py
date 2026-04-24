"""Depth conditioning for artifact reduction.

Uses depth maps to guide video generation, reducing:
- Flickering (depth consistency between frames)
- Warping (geometric coherence)
- Face distortion (depth preservation)
- Structural inconsistency (scene depth ordering)

Methods:
- MiDaS: Generic monocular depth estimation
- ZoeDepth: Metric depth for indoor/outdoor
- Marigold: Diffusion-based depth regression
- Depth-Anything: Foundation model depth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthMethod(Enum):
    """Depth estimation methods."""
    MIDAS = "midas"
    ZOEDEPTH = "zoedepth"
    MARIGOLD = "marigold"
    DEPTH_ANYTHING = "depth_anything"
    ENSEMBLE = "ensemble"


@dataclass
class DepthConditioningConfig:
    """Configuration for depth conditioning."""
    method: DepthMethod = DepthMethod.ZOEDEPTH
    strength: float = 0.8
    temporal_smoothing: float = 0.7
    preserve_edges: bool = True
    edge_threshold: float = 0.1
    
    confidence_threshold: float = 0.5
    use_refined_depth: bool = True
    refine_iterations: int = 3
    
    scale_factor: float = 1.0
    offset: float = 0.0
    normalize: bool = True
    
    blend_mode: str = "weighted"
    depth_weight_far: float = 0.5
    depth_weight_near: float = 1.0


class DepthConditioner:
    """Depth-based conditioning for artifact reduction.
    
    Reduces artifacts through:
    1. Depth-guided latent warping
    2. Per-layer motion modulation
    3. Temporal depth consistency
    4. Edge-preserved blending
    
    Tradeoffs:
    - Quality vs Speed: Ensemble is most accurate but slowest
    - ZoeDepth: Fast, good metric accuracy
    - MiDaS: General purpose, moderate speed
    - Marigold: Highest quality, slowest
    
    Args:
        config: Depth conditioning configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[DepthConditioningConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or DepthConditioningConfig()
        self.device = device or torch.device("cpu")
        
        self._models: Dict[DepthMethod, nn.Module] = {}
        self._initialized = False
        self._depth_cache: Optional[torch.Tensor] = None
    
    def initialize(self) -> None:
        """Initialize depth estimation models."""
        if self._initialized:
            return
        
        if self.config.method == DepthMethod.ZOEDEPTH:
            self._init_zoedepth()
        elif self.config.method == DepthMethod.MIDAS:
            self._init_midas()
        elif self.config.method == DepthMethod.MARIGOLD:
            self._init_marigold()
        elif self.config.method == DepthMethod.DEPTH_ANYTHING:
            self._init_depth_anything()
        elif self.config.method == DepthMethod.ENSEMBLE:
            self._init_ensemble()
        
        self._initialized = True
    
    def _init_zoedepth(self) -> None:
        """Initialize ZoeDepth."""
        try:
            from zoedepth.models import ZoeDepth
            model = ZoeDepth.from_pretrained("Zoedepth/zoedepth-nyu")
            model = model.to(self.device)
            model.eval()
            self._models[DepthMethod.ZOEDEPTH] = model
        except ImportError:
            self._models[DepthMethod.ZOEDEPTH] = self._create_fallback_depth()
    
    def _init_midas(self) -> None:
        """Initialize MiDaS."""
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            model = model.to(self.device)
            model.eval()
            self._models[DepthMethod.MIDAS] = model
            self._models["midas_processor"] = processor
        except ImportError:
            self._models[DepthMethod.MIDAS] = self._create_fallback_depth()
    
    def _init_marigold(self) -> None:
        """Initialize Marigold."""
        try:
            from diffusers import DiffusionPipeline
            model = DiffusionPipeline.from_pretrained("prs-eth/marigold-depth")
            model = model.to(self.device)
            model.eval()
            self._models[DepthMethod.MARIGOLD] = model
        except ImportError:
            self._models[DepthMethod.MARIGOLD] = self._create_fallback_depth()
    
    def _init_depth_anything(self) -> None:
        """Initialize Depth-Anything."""
        try:
            from transformers import AutoModel, AutoImageProcessor
            model = AutoModel.from_pretrained("LiheYoung/depth_anything_vitl14")
            processor = AutoImageProcessor.from_pretrained("LiheYoung/depth_anything_vitl14")
            model = model.to(self.device)
            model.eval()
            self._models[DepthMethod.DEPTH_ANYTHING] = model
            self._models["depth_anything_processor"] = processor
        except ImportError:
            self._models[DepthMethod.DEPTH_ANYTHING] = self._create_fallback_depth()
    
    def _init_ensemble(self) -> None:
        """Initialize ensemble of all methods."""
        self._init_zoedepth()
        self._init_midas()
        self._init_depth_anything()
    
    def _create_fallback_depth(self) -> nn.Module:
        """Create simple depth estimation fallback."""
        class SimpleDepth(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(32, 1, 1),
                )
            
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        model = SimpleDepth()
        model = model.to(self.device)
        model.eval()
        return model
    
    def estimate_depth(
        self,
        image: torch.Tensor,
        method: Optional[DepthMethod] = None
    ) -> torch.Tensor:
        """Estimate depth from image.
        
        Args:
            image: Image tensor [C, H, W] or [B, C, H, W]
            method: Override estimation method
            
        Returns:
            Depth map [H, W] or [B, H, W]
        """
        if not self._initialized:
            self.initialize()
        
        method = method or self.config.method
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        original_size = image.shape[-2:]
        
        if method == DepthMethod.ZOEDEPTH:
            depth = self._estimate_zoedepth(image)
        elif method == DepthMethod.MIDAS:
            depth = self._estimate_midas(image)
        elif method == DepthMethod.MARIGOLD:
            depth = self._estimate_marigold(image)
        elif method == DepthMethod.DEPTH_ANYTHING:
            depth = self._estimate_depth_anything(image)
        elif method == DepthMethod.ENSEMBLE:
            depth = self._estimate_ensemble(image)
        else:
            depth = self._models.get(method, self._create_fallback_depth())(image)
        
        depth = F.interpolate(
            depth,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
        
        if self.config.normalize:
            depth_min = depth.view(depth.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
            depth_max = depth.view(depth.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        self._depth_cache = depth
        
        return depth
    
    def _estimate_zoedepth(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using ZoeDepth."""
        model = self._models.get(DepthMethod.ZOEDEPTH)
        if model is None:
            return self._create_fallback_depth()(image)
        
        with torch.no_grad():
            try:
                depth = model.infer(image)
            except Exception:
                depth = model(image)
        
        if depth.dim() == 3:
            depth = depth.squeeze(1)
        
        return depth
    
    def _estimate_midas(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using MiDaS."""
        model = self._models.get(DepthMethod.MIDAS)
        processor = self._models.get("midas_processor")
        
        if model is None:
            return self._create_fallback_depth()(image)
        
        with torch.no_grad():
            if processor is not None:
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                depth = outputs.predicted_depth
            else:
                depth = model(image)
        
        if depth.dim() == 3:
            depth = depth.squeeze(1)
        
        return depth
    
    def _estimate_marigold(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using Marigold."""
        model = self._models.get(DepthMethod.MARIGOLD)
        if model is None:
            return self._create_fallback_depth()(image)
        
        with torch.no_grad():
            rgb = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb = (np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
            depth = model.predict_legacy(rgb, denormalize=False)
        
        depth = torch.from_numpy(depth).float().to(self.device)
        
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        
        return depth
    
    def _estimate_depth_anything(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using Depth-Anything."""
        model = self._models.get(DepthMethod.DEPTH_ANYTHING)
        processor = self._models.get("depth_anything_processor")
        
        if model is None:
            return self._create_fallback_depth()(image)
        
        with torch.no_grad():
            if processor is not None:
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                depth = outputs.predicted_depth
            else:
                depth = model(image)
        
        if depth.dim() == 3:
            depth = depth.squeeze(1)
        
        return depth
    
    def _estimate_ensemble(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth using ensemble of methods."""
        depths = []
        
        if DepthMethod.ZOEDEPTH in self._models:
            d = self._estimate_zoedepth(image)
            depths.append(d)
        
        if DepthMethod.MIDAS in self._models:
            d = self._estimate_midas(image)
            depths.append(d)
        
        if DepthMethod.DEPTH_ANYTHING in self._models:
            d = self._estimate_depth_anything(image)
            depths.append(d)
        
        if not depths:
            return self._create_fallback_depth()(image)
        
        if len(depths) == 1:
            return depths[0]
        
        depths_tensor = torch.stack([d for d in depths if d.shape == depths[0].shape])
        
        median_depth = torch.median(depths_tensor, dim=0)[0]
        
        return median_depth
    
    def apply_depth_guidance(
        self,
        latents: torch.Tensor,
        depth_map: torch.Tensor,
        timestep: float = 0.5
    ) -> torch.Tensor:
        """Apply depth guidance to latents.
        
        Args:
            latents: Latent tensor [B, C, H, W]
            depth_map: Depth map [B, 1, H, W]
            timestep: Diffusion timestep (0-1)
            
        Returns:
            Depth-guided latents
        """
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        
        depth_norm = depth_map
        if depth_norm.max() > 1:
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
        
        strength = self.config.strength * (1.0 - timestep)
        
        depth_weight = 1.0 + (1.0 - depth_norm) * (self.config.depth_weight_near - 1)
        
        guided = latents * depth_weight
        
        edge_mask = self._detect_depth_edges(depth_norm)
        
        if self.config.preserve_edges:
            guided = guided * (1 - edge_mask * 0.3) + latents * edge_mask * 0.3
        
        return guided
    
    def _detect_depth_edges(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Detect edges in depth map for preservation."""
        grad_x = F.conv2d(
            depth_map,
            torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        grad_y = F.conv2d(
            depth_map,
            torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_mask = (edge_magnitude > self.config.edge_threshold).float()
        
        return edge_mask
    
    def temporal_depth_smooth(
        self,
        depth_sequence: List[torch.Tensor],
        smoothing_strength: Optional[float] = None
    ) -> List[torch.Tensor]:
        """Apply temporal smoothing to depth sequence.
        
        Reduces flickering by enforcing consistency across frames.
        
        Args:
            depth_sequence: List of depth maps
            smoothing_strength: Override smoothing strength
            
        Returns:
            Smoothed depth sequence
        """
        smoothing = smoothing_strength or self.config.temporal_smoothing
        
        if not depth_sequence:
            return []
        
        smoothed = [depth_sequence[0]]
        
        for i in range(1, len(depth_sequence)):
            prev = smoothed[i - 1]
            curr = depth_sequence[i]
            
            if prev.shape != curr.shape:
                curr = F.interpolate(
                    curr.unsqueeze(0),
                    size=prev.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
            
            new_depth = smoothing * prev + (1 - smoothing) * curr
            
            smoothed.append(new_depth)
        
        return smoothed
    
    def create_depth_pyramid(
        self,
        depth_map: torch.Tensor,
        levels: int = 4
    ) -> List[torch.Tensor]:
        """Create multi-scale depth pyramid for hierarchical guidance.
        
        Args:
            depth_map: Depth map [B, 1, H, W]
            levels: Number of pyramid levels
            
        Returns:
            List of depth maps at different scales
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0)
        
        pyramid = [depth_map]
        current = depth_map
        
        for _ in range(levels - 1):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyramid.append(current)
        
        return pyramid
    
    def warp_with_depth(
        self,
        frame: torch.Tensor,
        depth_map: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Depth-aware frame warping.
        
        Applies motion with depth modulation to reduce warping artifacts.
        
        Args:
            frame: Frame tensor [C, H, W]
            depth_map: Depth map [H, W]
            flow: Motion flow [2, H, W]
            
        Returns:
            Warped frame
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
        
        depth_norm = depth_map / (depth_map.max() + 1e-8)
        
        depth_weight = 1.0 + (1.0 - depth_norm) * 0.5
        
        modulated_flow = flow * depth_weight
        
        B, C, H, W = frame.shape
        
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        grid_x = grid_x + modulated_flow[0, 0] / W * 2
        grid_y = grid_y + modulated_flow[0, 1] / H * 2
        
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
    
    def get_depth_layers(
        self,
        depth_map: torch.Tensor,
        num_layers: int = 5
    ) -> List[Tuple[float, float, torch.Tensor]]:
        """Separate depth map into layers for per-layer motion.
        
        Args:
            depth_map: Depth map [H, W]
            num_layers: Number of depth layers
            
        Returns:
            List of (min_depth, max_depth, layer_mask) tuples
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        layers = []
        
        for i in range(num_layers):
            min_val = depth_min + (depth_max - depth_min) * i / num_layers
            max_val = depth_min + (depth_max - depth_min) * (i + 1) / num_layers
            
            layer_mask = (depth_map >= min_val) & (depth_map < max_val)
            
            layers.append((min_val.item(), max_val.item(), layer_mask.float()))
        
        return layers


class DepthConsistencyLoss(nn.Module):
    """Loss function for depth consistency across frames.
    
    Reduces flickering by enforcing:
    - Temporal depth consistency
    - Depth ordering preservation
    - Edge alignment
    
    Args:
        weight: Loss weight
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        depth_sequence: torch.Tensor,
        reference_depth: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute depth consistency loss.
        
        Args:
            depth_sequence: Depth sequence [T, 1, H, W]
            reference_depth: Reference depth for alignment [1, H, W]
            
        Returns:
            Loss value
        """
        if depth_sequence.dim() == 3:
            depth_sequence = depth_sequence.unsqueeze(1)
        
        T = depth_sequence.shape[0]
        
        temporal_variance = torch.var(depth_sequence, dim=0)
        
        loss = temporal_variance.mean()
        
        if reference_depth is not None:
            if reference_depth.dim() == 2:
                reference_depth = reference_depth.unsqueeze(0).unsqueeze(0)
            
            consistency_loss = torch.abs(depth_sequence - reference_depth).mean()
            loss = loss + consistency_loss
        
        return self.weight * loss
    
    def compute_gradient_loss(
        self,
        depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient-based consistency loss.
        
        Args:
            depth_map: Depth map [1, 1, H, W]
            
        Returns:
            Gradient loss
        """
        grad_x = depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1]
        grad_y = depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :]
        
        grad_loss = grad_x.abs().mean() + grad_y.abs().mean()
        
        return self.weight * grad_loss