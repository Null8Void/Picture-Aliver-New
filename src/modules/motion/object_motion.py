"""Object motion generation for hair, cloth, and small movements.

Generates realistic motion for:
- Hair movement (flowing, wind effects)
- Cloth simulation (fabric physics)
- Small object movements
- Particle effects
- Surface ripples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ObjectMotionType(Enum):
    """Types of object motion."""
    HAIR = "hair"
    CLOTH = "cloth"
    FOLIAGE = "foliage"
    WATER = "water"
    SMOKE = "smoke"
    DUST = "dust"
    RIPPLE = "ripple"
    CLOTHING = "clothing"
    FABRIC = "fabric"
    GENERAL = "general"


class MotionDirection(Enum):
    """Global motion directions."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    DIAGONAL = "diagonal"
    RANDOM = "random"
    WIND = "wind"


@dataclass
class MotionRegion:
    """Region in image with specific motion properties."""
    mask: Union[np.ndarray, torch.Tensor]
    center: Tuple[float, float]
    motion_type: ObjectMotionType
    strength: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    priority: int = 0
    
    def __post_init__(self):
        if isinstance(self.mask, torch.Tensor):
            self.mask = self.mask.bool()
    
    @property
    def is_valid(self) -> bool:
        """Check if region is valid (non-empty)."""
        if isinstance(self.mask, np.ndarray):
            return self.mask.any()
        elif isinstance(self.mask, torch.Tensor):
            return self.mask.any().item()
        return False


@dataclass
class ObjectMotion:
    """Motion data for a specific object region."""
    region_id: int
    motion_type: ObjectMotionType
    flow_field: torch.Tensor
    strength_map: torch.Tensor
    phase: float = 0.0
    frequency: float = 1.0
    
    @property
    def has_motion(self) -> bool:
        """Check if region has non-zero motion."""
        return self.flow_field.abs().sum() > 1e-6
    
    def get_flow_at_time(self, t: float) -> torch.Tensor:
        """Get flow field at time t."""
        modulated = self.flow_field * self.strength_map
        phase_shift = 2 * np.pi * self.frequency * t + self.phase
        return modulated * torch.cos(torch.tensor(phase_shift))


@dataclass
class ObjectMotionConfig:
    """Configuration for object motion generation."""
    motion_type: ObjectMotionType = ObjectMotionType.HAIR
    direction: MotionDirection = MotionDirection.RIGHT
    strength: float = 0.5
    frequency: float = 0.5
    randomness: float = 0.2
    temporal_coherence: float = 0.9
    edge_blend: float = 0.3
    
    hair_config: Optional[Dict] = None
    cloth_config: Optional[Dict] = None
    foliage_config: Optional[Dict] = None
    water_config: Optional[Dict] = None
    
    def __post_init__(self):
        if self.hair_config is None:
            self.hair_config = {
                "strand_length": 50,
                "stiffness": 0.3,
                "damping": 0.1,
                "gravity": 0.05
            }
        if self.cloth_config is None:
            self.cloth_config = {
                "spring_constant": 0.5,
                "damping": 0.2,
                "gravity": 0.1
            }
        if self.foliage_config is None:
            self.foliage_config = {
                "leaf_size": 10,
                "rustle_intensity": 0.3
            }


class ObjectMotionGenerator:
    """Generates realistic object motion for video synthesis.
    
    Creates motion fields for:
    - Hair strands (flowing with physics)
    - Cloth/fabric (subtle movement)
    - Foliage (leaves, trees)
    - Water surfaces (ripples, waves)
    - Particle effects (smoke, dust)
    
    Args:
        config: Object motion configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[ObjectMotionConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or ObjectMotionConfig()
        self.device = device or torch.device("cpu")
        
        self._motion_cache: Dict[int, ObjectMotion] = {}
        self._flow_history: List[torch.Tensor] = []
    
    def generate(
        self,
        regions: List[MotionRegion],
        num_frames: int,
        depth_map: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None
    ) -> List[ObjectMotion]:
        """Generate motion for object regions.
        
        Args:
            regions: List of motion regions
            num_frames: Number of frames
            depth_map: Optional depth map for depth-aware motion
            segmentation: Optional segmentation for region identification
            
        Returns:
            List of ObjectMotion objects
        """
        motions = []
        
        for i, region in enumerate(regions):
            if not region.is_valid:
                continue
            
            motion = self._generate_region_motion(
                region, num_frames, depth_map
            )
            motions.append(motion)
        
        if self.config.temporal_coherence > 0:
            motions = self._apply_temporal_coherence(motions)
        
        return motions
    
    def _generate_region_motion(
        self,
        region: MotionRegion,
        num_frames: int,
        depth_map: Optional[torch.Tensor] = None
    ) -> ObjectMotion:
        """Generate motion for single region."""
        mask = region.mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        mask = mask.to(self.device)
        
        H, W = mask.shape
        flow_field = torch.zeros(2, H, W, device=self.device)
        strength_map = torch.zeros(H, W, device=self.device)
        
        if region.motion_type == ObjectMotionType.HAIR:
            flow_field, strength_map = self._generate_hair_motion(
                mask, H, W, region
            )
        elif region.motion_type == ObjectMotionType.CLOTH:
            flow_field, strength_map = self._generate_cloth_motion(
                mask, H, W, region
            )
        elif region.motion_type == ObjectMotionType.FOLIAGE:
            flow_field, strength_map = self._generate_foliage_motion(
                mask, H, W, region
            )
        elif region.motion_type == ObjectMotionType.WATER:
            flow_field, strength_map = self._generate_water_motion(
                mask, H, W, region
            )
        elif region.motion_type == ObjectMotionType.RIPPLE:
            flow_field, strength_map = self._generate_ripple_motion(
                mask, H, W, region, depth_map
            )
        else:
            flow_field, strength_map = self._generate_general_motion(
                mask, H, W, region
            )
        
        if depth_map is not None:
            flow_field = self._apply_depth_modulation(flow_field, depth_map, mask)
        
        flow_field = flow_field * region.strength
        strength_map = strength_map * region.strength
        
        return ObjectMotion(
            region_id=region.center,
            motion_type=region.motion_type,
            flow_field=flow_field,
            strength_map=strength_map,
            phase=region.phase,
            frequency=region.frequency
        )
    
    def _generate_hair_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate realistic hair-like motion."""
        hair_config = self.config.hair_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        direction = self._get_direction_vector()
        
        strand_length = hair_config.get("strand_length", 50)
        stiffness = hair_config.get("stiffness", 0.3)
        damping = hair_config.get("damping", 0.1)
        
        distance_from_anchor = (grid_x - region.center[0])**2 + (grid_y - region.center[1])**2
        distance_from_anchor = torch.sqrt(distance_from_anchor + 1e-6)
        
        strand_influence = torch.exp(-distance_from_anchor / (strand_length * 0.5))
        
        height_influence = (H - grid_y) / H
        mask_influence = mask.float()
        
        base_flow = direction.view(2, 1, 1) * strand_length * 0.01
        
        flow_magnitude = strand_influence * height_influence * mask_influence
        
        flow_magnitude = flow_magnitude * self.config.strength
        
        flow_x = base_flow[0] * flow_magnitude
        flow_y = base_flow[1] * flow_magnitude
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        strength_map = flow_magnitude * stiffness
        
        noise = torch.randn_like(flow_field) * 0.1 * damping
        flow_field = flow_field + noise * strength_map.unsqueeze(0)
        
        return flow_field, strength_map
    
    def _generate_cloth_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cloth-like fabric motion."""
        cloth_config = self.config.cloth_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        direction = self._get_direction_vector()
        
        spring_constant = cloth_config.get("spring_constant", 0.5)
        damping = cloth_config.get("damping", 0.2)
        gravity = cloth_config.get("gravity", 0.1)
        
        edge_distance = torch.minimum(
            grid_x, torch.minimum(W - grid_x, torch.minimum(grid_y, H - grid_y))
        )
        edge_influence = torch.exp(-edge_distance / 20)
        
        top_influence = grid_y / H
        
        cloth_flow = direction.view(2, 1, 1) * self.config.strength * 0.5
        gravity_flow = torch.tensor([0.0, gravity * top_influence * mask.float()])
        
        flow_x = (cloth_flow[0] + gravity_flow[0]) * mask.float()
        flow_y = (cloth_flow[1] + gravity_flow[1]) * mask.float()
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        strength_map = mask.float() * (1 - edge_influence * 0.5) * spring_constant
        
        noise = torch.randn_like(flow_field) * 0.05 * damping * mask.float()
        flow_field = flow_field + noise
        
        return flow_field, strength_map
    
    def _generate_foliage_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate natural foliage/leaf motion."""
        foliage_config = self.config.foliage_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        direction = self._get_direction_vector()
        
        leaf_size = foliage_config.get("leaf_size", 10)
        rustle_intensity = foliage_config.get("rustle_intensity", 0.3)
        
        spatial_phase = (
            grid_x / leaf_size * 2 * np.pi +
            grid_y / leaf_size * np.pi
        )
        
        base_flow = direction.view(2, 1, 1) * self.config.strength * 0.3
        
        rustle = rustle_intensity * self.config.strength
        
        flow_x = base_flow[0] + rustle * torch.sin(spatial_phase + region.phase)
        flow_y = base_flow[1] + rustle * 0.5 * torch.cos(spatial_phase * 1.3 + region.phase)
        
        flow_field = torch.stack([flow_x, flow_y], dim=0) * mask.float()
        
        strength_map = mask.float() * (0.3 + 0.7 * torch.rand(H, W, device=self.device))
        
        return flow_field, strength_map
    
    def _generate_water_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate water surface motion (waves, ripples)."""
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        wave_freq = 0.05 + self.config.frequency * 0.1
        wave_amp = 0.3 * self.config.strength
        
        center_x, center_y = region.center
        
        distance = torch.sqrt(
            (grid_x - center_x)**2 + (grid_y - center_y)**2
        )
        
        wave_phase = distance * wave_freq - region.phase * 2 * np.pi
        
        radial_flow = torch.stack([
            (grid_x - center_x) / (distance + 1),
            (grid_y - center_y) / (distance + 1)
        ], dim=0)
        
        wave_intensity = torch.exp(-distance / (W * 0.3))
        
        flow_x = radial_flow[0] * wave_amp * wave_intensity * torch.sin(wave_phase)
        flow_y = radial_flow[1] * wave_amp * wave_intensity * torch.sin(wave_phase)
        
        flow_field = torch.stack([flow_x, flow_y], dim=0) * mask.float()
        
        strength_map = mask.float() * wave_intensity * 0.8
        
        return flow_field, strength_map
    
    def _generate_ripple_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion,
        depth_map: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate circular ripple motion."""
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        center_x, center_y = region.center
        
        distance = torch.sqrt(
            (grid_x - center_x)**2 + (grid_y - center_y)**2
        )
        
        ripple_freq = 0.1 * self.config.frequency
        ripple_amp = 0.2 * self.config.strength
        
        wave = ripple_amp * torch.sin(distance * ripple_freq - region.phase * 2 * np.pi)
        
        radial = torch.stack([
            (grid_x - center_x) / (distance + 1),
            (grid_y - center_y) / (distance + 1)
        ], dim=0)
        
        flow_x = radial[0] * wave * mask.float()
        flow_y = radial[1] * wave * mask.float()
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        strength_map = mask.float() * torch.exp(-distance / (W * 0.2))
        
        return flow_field, strength_map
    
    def _generate_general_motion(
        self,
        mask: torch.Tensor,
        H: int,
        W: int,
        region: MotionRegion
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate general object motion."""
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        direction = self._get_direction_vector()
        
        base_flow = direction.view(2, 1, 1) * self.config.strength * 0.2
        
        noise_scale = 0.05 * self.config.randomness
        noise = torch.randn(2, H, W, device=self.device) * noise_scale
        
        flow_field = (base_flow + noise) * mask.float()
        
        strength_map = mask.float() * self.config.strength
        
        return flow_field, strength_map
    
    def _get_direction_vector(self) -> torch.Tensor:
        """Get direction vector based on config."""
        if self.config.direction == MotionDirection.LEFT:
            return torch.tensor([-1.0, 0.0], device=self.device)
        elif self.config.direction == MotionDirection.RIGHT:
            return torch.tensor([1.0, 0.0], device=self.device)
        elif self.config.direction == MotionDirection.UP:
            return torch.tensor([0.0, -1.0], device=self.device)
        elif self.config.direction == MotionDirection.DOWN:
            return torch.tensor([0.0, 1.0], device=self.device)
        elif self.config.direction == MotionDirection.DIAGONAL:
            return torch.tensor([0.7, 0.7], device=self.device)
        elif self.config.direction == MotionDirection.WIND:
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            return torch.tensor([np.cos(angle), np.sin(angle)], device=self.device)
        else:
            angle = np.random.uniform(0, 2*np.pi)
            return torch.tensor([np.cos(angle), np.sin(angle)], device=self.device)
    
    def _apply_depth_modulation(
        self,
        flow_field: torch.Tensor,
        depth_map: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply depth-based modulation to flow."""
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        depth_weight = 1.0 + (1.0 - depth_norm) * 0.5
        
        depth_weight = F.interpolate(
            depth_weight,
            size=flow_field.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        flow_field = flow_field * depth_weight * mask.float()
        
        return flow_field
    
    def _apply_temporal_coherence(
        self,
        motions: List[ObjectMotion]
    ) -> List[ObjectMotion]:
        """Apply temporal coherence to smooth motion transitions."""
        if not self._flow_history:
            self._flow_history = [
                torch.zeros_like(m.flow_field) for m in motions
            ]
        
        coherence = self.config.temporal_coherence
        
        smoothed_motions = []
        for i, motion in enumerate(motions):
            if i < len(self._flow_history):
                prev_flow = self._flow_history[i]
                new_flow = coherence * prev_flow + (1 - coherence) * motion.flow_field
                motion.flow_field.copy_(new_flow)
            
            self._flow_history[i] = motion.flow_field.clone()
            smoothed_motions.append(motion)
        
        if len(self._flow_history) > len(motions):
            self._flow_history = self._flow_history[:len(motions)]
        
        return smoothed_motions
    
    def combine_motions(
        self,
        motions: List[ObjectMotion],
        background_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Combine multiple motion fields into single flow field.
        
        Args:
            motions: List of object motions
            background_flow: Optional background motion field
            
        Returns:
            Combined flow field [2, H, W]
        """
        if not motions:
            if background_flow is not None:
                return background_flow
            return torch.zeros(2, 512, 512, device=self.device)
        
        H, W = motions[0].flow_field.shape[-2:]
        combined = torch.zeros(2, H, W, device=self.device)
        weight_sum = torch.zeros(H, W, device=self.device)
        
        for motion in motions:
            strength = motion.strength_map.unsqueeze(0)
            combined = combined + motion.flow_field * strength
            weight_sum = weight_sum + strength.squeeze(0)
        
        weight_sum = weight_sum + 1e-8
        combined = combined / weight_sum.unsqueeze(0)
        
        if background_flow is not None:
            bg_weight = 0.3
            combined = combined * (1 - bg_weight) + background_flow * bg_weight
        
        return combined
    
    def apply_to_image(
        self,
        image: torch.Tensor,
        flow_field: torch.Tensor,
        mode: str = "bilinear"
    ) -> torch.Tensor:
        """Apply motion flow to image using warping.
        
        Args:
            image: Image tensor [C, H, W] or [B, C, H, W]
            flow_field: Flow field [2, H, W]
            mode: Interpolation mode
            
        Returns:
            Warped image
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        grid_x = grid_x + flow_field[0]
        grid_y = grid_y + flow_field[1]
        
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        warped = F.grid_sample(
            image,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True
        )
        
        return warped.squeeze(0)