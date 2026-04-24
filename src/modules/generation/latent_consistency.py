"""Latent consistency management for temporal coherence.

Ensures latent space consistency across video frames, reducing:
- Flickering (consistent latent trajectory)
- Face distortion (identity preservation)
- Structural inconsistency (spatial coherence)
- Warping (geometric stability)

Methods:
- Cross-frame attention
- Latent trajectory smoothing
- Identity preservation loss
- Structural consistency enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyMetric(Enum):
    """Metrics for measuring latent consistency."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    STRUCTURAL_SIMILARITY = "ssim"
    PERCEPTUAL = "perceptual"
    TEMPORAL_VARIATION = "temporal_variation"


@dataclass
class LatentConsistencyConfig:
    """Configuration for latent consistency."""
    enable_cross_attention: bool = True
    cross_attention_strength: float = 0.5
    
    temporal_smoothing: float = 0.7
    smoothing_window: int = 3
    
    identity_weight: float = 0.3
    structural_weight: float = 0.5
    
    use_perceptual_loss: bool = True
    perceptual_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    latent_stride: int = 8
    consistency_check_interval: int = 4
    
    trajectory_smoothing: float = 0.8
    max_trajectory_deviation: float = 0.1


class LatentConsistencyManager:
    """Manages latent consistency for artifact-free video generation.
    
    Reduces artifacts through:
    1. Cross-frame attention: Ensures temporal coherence
    2. Latent smoothing: Reduces flickering
    3. Identity preservation: Maintains face consistency
    4. Structural consistency: Keeps geometry stable
    
    Tradeoffs:
    - Quality vs Speed: More iterations = better but slower
    - Attention-based: High quality, moderate speed
    - Simple smoothing: Fast, lower quality
    
    Args:
        config: Consistency configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[LatentConsistencyConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or LatentConsistencyConfig()
        self.device = device or torch.device("cpu")
        
        self._latent_history: List[torch.Tensor] = []
        self._reference_latent: Optional[torch.Tensor] = None
        self._trajectory: List[torch.Tensor] = []
        
        self._perceptual_extractor: Optional[nn.Module] = None
    
    def initialize(self) -> None:
        """Initialize consistency components."""
        if self.config.use_perceptual_loss:
            self._init_perceptual_extractor()
    
    def _init_perceptual_extractor(self) -> None:
        """Initialize perceptual feature extractor."""
        try:
            import torchvision.models as models
            
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            vgg = vgg.features.to(self.device)
            vgg.eval()
            
            for param in vgg.parameters():
                param.requires_grad = False
            
            self._perceptual_extractor = vgg
            
        except ImportError:
            self._perceptual_extractor = self._create_simple_extractor()
    
    def _create_simple_extractor(self) -> nn.Module:
        """Create simple feature extractor fallback."""
        class SimpleExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            
            def forward(self, x):
                f1 = F.relu(self.conv1(x))
                f2 = F.relu(self.conv2(f1))
                f3 = F.relu(self.conv3(f2))
                return [f1, f2, f3]
        
        return SimpleExtractor().to(self.device)
    
    def set_reference(self, latent: torch.Tensor) -> None:
        """Set reference latent for identity preservation.
        
        Args:
            latent: Reference latent tensor
        """
        self._reference_latent = latent.detach().clone()
    
    def enforce_consistency(
        self,
        latents: torch.Tensor,
        timestep: int = 0,
        depth_map: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Enforce latent consistency across frames.
        
        Args:
            latents: Latent tensor [T, C, H, W] or [B, C, H, W]
            timestep: Current diffusion timestep
            depth_map: Optional depth map for structural guidance
            segmentation: Optional segmentation for region guidance
            
        Returns:
            Consistency-enforced latents
        """
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        
        T, B, C, H, W = latents.shape
        
        if self._reference_latent is None and T > 0:
            self._reference_latent = latents[0].detach()
        
        consistent = latents.clone()
        
        if self.config.enable_cross_attention:
            consistent = self._apply_cross_frame_attention(consistent)
        
        consistent = self._apply_temporal_smoothing(consistent)
        
        if self._reference_latent is not None:
            consistent = self._enforce_identity(consistent, depth_map, segmentation)
        
        if depth_map is not None:
            consistent = self._enforce_structural_consistency(
                consistent, depth_map
            )
        
        consistent = self._smooth_trajectory(consistent)
        
        self._update_history(consistent)
        
        return consistent
    
    def _apply_cross_frame_attention(
        self,
        latents: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-frame attention for temporal coherence.
        
        Reduces flickering by ensuring each frame attends to neighboring frames.
        
        Args:
            latents: Latent tensor [T, B, C, H, W]
            
        Returns:
            Attention-enhanced latents
        """
        T, B, C, H, W = latents.shape
        
        if T < 2:
            return latents
        
        result = latents.clone()
        
        for t in range(T):
            neighbors = []
            weights = []
            
            for dt in range(-1, 2):
                if dt == 0:
                    continue
                nt = t + dt
                if 0 <= nt < T:
                    neighbors.append(latents[nt])
                    
                    temporal_weight = 1.0 / (abs(dt) + 1)
                    weights.append(temporal_weight)
            
            if not neighbors:
                continue
            
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            
            attended = torch.zeros_like(latents[t])
            for neighbor, weight in zip(neighbors, weights):
                attended = attended + neighbor * weight
            
            strength = self.config.cross_attention_strength
            result[t] = (1 - strength) * latents[t] + strength * attended
        
        return result
    
    def _apply_temporal_smoothing(
        self,
        latents: torch.Tensor
    ) -> torch.Tensor:
        """Apply temporal smoothing to latents.
        
        Reduces high-frequency variations that cause flickering.
        
        Args:
            latents: Latent tensor [T, B, C, H, W]
            
        Returns:
            Smoothed latents
        """
        T = latents.shape[0]
        
        if T < 2:
            return latents
        
        window = self.config.smoothing_window
        alpha = self.config.temporal_smoothing
        
        result = latents.clone()
        
        for t in range(T):
            neighbors = []
            for dt in range(-window, window + 1):
                nt = t + dt
                if 0 <= nt < T:
                    neighbors.append(latents[nt])
            
            if len(neighbors) > 1:
                mean_latent = torch.stack(neighbors).mean(dim=0)
                
                result[t] = (1 - alpha) * latents[t] + alpha * mean_latent
        
        return result
    
    def _enforce_identity(
        self,
        latents: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        segmentation: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Enforce identity preservation using reference latent.
        
        Reduces face distortion by maintaining subject identity.
        
        Args:
            latents: Latent tensor [T, B, C, H, W]
            depth_map: Optional depth map
            segmentation: Optional segmentation mask
            
        Returns:
            Identity-preserved latents
        """
        if self._reference_latent is None:
            return latents
        
        ref = self._reference_latent
        
        if ref.dim() == 4:
            ref = ref.unsqueeze(0)
        
        T = latents.shape[0]
        
        result = latents.clone()
        
        for t in range(T):
            if depth_map is not None:
                depth_loss = self._compute_depth_consistency(
                    latents[t], ref, depth_map
                )
                weight = self.config.identity_weight * (1 - depth_loss)
            else:
                weight = self.config.identity_weight
            
            ref_expanded = ref.expand(T, -1, -1, -1, -1)
            
            result[t] = (1 - weight) * latents[t] + weight * ref_expanded[t]
        
        return result
    
    def _compute_depth_consistency(
        self,
        latent: torch.Tensor,
        reference: torch.Tensor,
        depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Compute depth-aware consistency weight."""
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        depth_norm = depth_map / (depth_map.max() + 1e-8)
        
        near_mask = (depth_norm < 0.3).float()
        
        diff = torch.abs(latent - reference).mean()
        
        return diff
    
    def _enforce_structural_consistency(
        self,
        latents: torch.Tensor,
        depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Enforce structural consistency using depth map.
        
        Reduces warping by maintaining geometric coherence.
        
        Args:
            latents: Latent tensor [T, B, C, H, W]
            depth_map: Depth map
            
        Returns:
            Structurally-consistent latents
        """
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        
        T = latents.shape[0]
        
        depth_norm = depth_map / (depth_map.max() + 1e-8)
        
        structural_weight = self.config.structural_weight
        
        result = latents.clone()
        
        for t in range(T):
            t_ratio = t / max(T - 1, 1)
            temporal_weight = structural_weight * (1 - abs(t_ratio - 0.5) * 2)
            
            for dt in range(-1, 2):
                nt = t + dt
                if 0 <= nt < T and dt != 0:
                    depth_diff = torch.abs(depth_norm - depth_norm).mean()
                    depth_weight = 1.0 / (1 + depth_diff)
                    
                    mix_weight = temporal_weight * depth_weight * abs(dt) / 2
                    result[t] = result[t] * (1 - mix_weight) + latents[nt] * mix_weight
        
        return result
    
    def _smooth_trajectory(
        self,
        latents: torch.Tensor
    ) -> torch.Tensor:
        """Smooth the latent trajectory over time.
        
        Ensures smooth camera/object motion.
        
        Args:
            latents: Latent tensor [T, B, C, H, W]
            
        Returns:
            Trajectory-smoothed latents
        """
        T = latents.shape[0]
        
        if T < 3:
            return latents
        
        self._trajectory.append(latents.detach())
        
        if len(self._trajectory) > 10:
            self._trajectory.pop(0)
        
        result = latents.clone()
        
        alpha = self.config.trajectory_smoothing
        
        for t in range(T):
            deviations = []
            for dt in range(-1, 2):
                nt = t + dt
                if 0 <= nt < T:
                    dev = torch.abs(latents[t] - latents[nt]).mean()
                    deviations.append(dev)
            
            avg_dev = sum(deviations) / len(deviations) if deviations else 0
            
            if avg_dev > self.config.max_trajectory_deviation:
                smooth_mean = latents[max(0, t-1):min(T, t+2)].mean(dim=0)
                result[t] = (1 - alpha) * latents[t] + alpha * smooth_mean
        
        return result
    
    def compute_consistency_metric(
        self,
        latents: torch.Tensor,
        metric: ConsistencyMetric = ConsistencyMetric.EUCLIDEAN
    ) -> Dict[str, float]:
        """Compute consistency metrics for evaluation.
        
        Args:
            latents: Latent tensor [T, C, H, W]
            metric: Type of metric to compute
            
        Returns:
            Dictionary of metric values
        """
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
        
        T = latents.shape[0]
        
        metrics = {}
        
        if metric in [ConsistencyMetric.EUCLIDEAN, ConsistencyMetric.COSINE]:
            inter_frame_dists = []
            
            for t in range(1, T):
                if metric == ConsistencyMetric.EUCLIDEAN:
                    dist = torch.abs(latents[t] - latents[t-1]).mean()
                else:
                    cos_sim = F.cosine_similarity(
                        latents[t].flatten(),
                        latents[t-1].flatten(),
                        dim=0
                    )
                    dist = 1 - cos_sim
                
                inter_frame_dists.append(dist.item())
            
            metrics['mean_inter_frame'] = np.mean(inter_frame_dists)
            metrics['std_inter_frame'] = np.std(inter_frame_dists)
            metrics['max_inter_frame'] = np.max(inter_frame_dists)
        
        if metric == ConsistencyMetric.TEMPORAL_VARIATION:
            temporal_vars = []
            
            for c in range(latents.shape[1]):
                for h in range(latents.shape[2]):
                    for w in range(latents.shape[3]):
                        var = torch.var(latents[:, c, h, w]).item()
                        temporal_vars.append(var)
            
            metrics['temporal_variation'] = np.mean(temporal_vars)
            metrics['temporal_variation_std'] = np.std(temporal_vars)
        
        if metric == ConsistencyMetric.PERCEPTUAL and self._perceptual_extractor is not None:
            with torch.no_grad():
                features = self._perceptual_extractor(latents[0:1])
                
                if isinstance(features, list):
                    metrics['perceptual_complexity'] = sum(f.mean().item() for f in features)
                else:
                    metrics['perceptual_complexity'] = features.mean().item()
        
        return metrics
    
    def _update_history(self, latents: torch.Tensor) -> None:
        """Update latent history for temporal analysis."""
        for latent in latents:
            self._latent_history.append(latent.detach())
            
            if len(self._latent_history) > 100:
                self._latent_history.pop(0)
    
    def get_history(self) -> List[torch.Tensor]:
        """Get latent history."""
        return self._latent_history.copy()
    
    def reset(self) -> None:
        """Reset consistency manager state."""
        self._latent_history.clear()
        self._trajectory.clear()
        self._reference_latent = None


class ConsistencyLoss(nn.Module):
    """Loss function for latent consistency training.
    
    Args:
        weight: Loss weight
        metric: Consistency metric to use
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        metric: ConsistencyMetric = ConsistencyMetric.TEMPORAL_VARIATION
    ):
        super().__init__()
        self.weight = weight
        self.metric = metric
    
    def forward(
        self,
        latents: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute consistency loss.
        
        Args:
            latents: Latent tensor [T, C, H, W]
            reference: Optional reference for identity loss
            
        Returns:
            Loss value
        """
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
        
        T = latents.shape[0]
        
        if T < 2:
            return torch.tensor(0.0, device=latents.device)
        
        if self.metric == ConsistencyMetric.TEMPORAL_VARIATION:
            variance = torch.var(latents, dim=0)
            loss = variance.mean()
        
        elif self.metric == ConsistencyMetric.EUCLIDEAN:
            diff_sum = torch.tensor(0.0, device=latents.device)
            count = 0
            
            for t in range(1, T):
                diff = torch.abs(latents[t] - latents[t-1])
                diff_sum = diff_sum + diff.mean()
                count = count + 1
            
            loss = diff_sum / count if count > 0 else torch.tensor(0.0, device=latents.device)
        
        elif self.metric == ConsistencyMetric.COSINE:
            cos_diffs = []
            
            for t in range(1, T):
                cos_sim = F.cosine_similarity(
                    latents[t].flatten().unsqueeze(0),
                    latents[t-1].flatten().unsqueeze(0),
                    dim=1
                )
                cos_diffs.append(1 - cos_sim)
            
            loss = torch.stack(cos_diffs).mean()
        
        else:
            loss = torch.var(latents, dim=0).mean()
        
        if reference is not None:
            identity_loss = torch.abs(latents - reference).mean()
            loss = loss + identity_loss * 0.5
        
        return self.weight * loss