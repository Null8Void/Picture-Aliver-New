"""Furry-specific motion types and animations.

Motion patterns for:
- Tail wagging/swishing
- Ear movements (perking, flattening)
- Breathing motions
- Fur rustling
- Wing flapping
- Body movements (sitting, standing, walking)
- Expression changes (blinking, mouth movements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class FurryMotionType(Enum):
    """Types of furry character motions."""
    TAIL_WAG = "tail_wag"
    TAIL_SWISH = "tail_swish"
    TAIL_CURL = "tail_curl"
    
    EARS_PERK = "ears_perk"
    EARS_FLATTEN = "ears_flatten"
    EARS_TWITCH = "ears_twitch"
    EARS_SWIVEL = "ears_swivel"
    
    BREATHING = "breathing"
    BREATHING_DEEP = "breathing_deep"
    
    FUR_RUSTLE = "fur_rustle"
    FUR_FLOAT = "fur_float"
    
    WING_FLAP = "wing_flap"
    WING_EXTEND = "wing_extend"
    
    BLINK = "blink"
    BLINK_DOUBLE = "blink_double"
    EYE_LOOK = "eye_look"
    
    MOUTH_MOVE = "mouth_move"
    MOUTH_SMILE = "mouth_smile"
    
    BODY_SHIFT = "body_shift"
    BODY_LEAN = "body_lean"
    BODY_BOUNCE = "body_bounce"
    
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"


class FurryRegion(Enum):
    """Regions of a furry character."""
    HEAD = "head"
    EARS_LEFT = "ears_left"
    EARS_RIGHT = "ears_right"
    FACE = "face"
    EYES = "eyes"
    MOUTH = "mouth"
    BODY = "body"
    CHEST = "chest"
    ARMS = "arms"
    LEGS = "legs"
    TAIL = "tail"
    WINGS = "wings"
    FUR = "fur"
    Paws = "paws"


@dataclass
class FurryMotionConfig:
    """Configuration for furry motion generation."""
    motion_type: FurryMotionType = FurryMotionType.TAIL_WAG
    strength: float = 0.5
    speed: float = 1.0
    frequency: float = 2.0
    phase_offset: float = 0.0
    
    affected_regions: List[FurryRegion] = field(default_factory=list)
    
    natural_variation: float = 0.2
    physics_enabled: bool = True
    symmetry: bool = False
    
    blend_with_base: float = 0.3


class FurryMotionGenerator:
    """Generates natural furry character motions.
    
    Creates realistic motions for:
    - Tail movements (wagging, swishing, curling)
    - Ear animations (perking, flattening, twitching)
    - Breathing (natural chest expansion)
    - Fur physics (rustling, floating)
    - Wing movements (flapping, extending)
    - Body dynamics (shifts, leans, bounces)
    
    These motions can be combined and layered for natural animation.
    
    Args:
        config: Furry motion configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[FurryMotionConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or FurryMotionConfig()
        self.device = device or torch.device("cpu")
        
        self._motion_cache: Dict[FurryMotionType, torch.Tensor] = {}
        self._phase = 0.0
    
    def generate_motion(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        motion_type: Optional[FurryMotionType] = None,
        segmentation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate furry character motion.
        
        Args:
            num_frames: Number of frames
            resolution: (H, W) resolution
            motion_type: Type of motion to generate
            segmentation: Optional segmentation for region-aware motion
            
        Returns:
            Motion flow field [T, 2, H, W]
        """
        motion_type = motion_type or self.config.motion_type
        H, W = resolution
        
        motion = self._generate_base_motion(num_frames, H, W, motion_type)
        
        if segmentation is not None:
            motion = self._apply_region_mask(motion, segmentation, motion_type)
        
        motion = self._add_natural_variation(motion)
        
        return motion
    
    def _generate_base_motion(
        self,
        num_frames: int,
        H: int,
        W: int,
        motion_type: FurryMotionType
    ) -> torch.Tensor:
        """Generate base motion pattern."""
        flows = []
        
        t = torch.linspace(0, 2 * np.pi, num_frames)
        
        base_freq = self.config.frequency
        base_strength = self.config.strength
        phase = self.config.phase_offset
        
        if motion_type == FurryMotionType.TAIL_WAG:
            flow = self._generate_tail_wag_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.TAIL_SWISH:
            flow = self._generate_tail_swish_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.TAIL_CURL:
            flow = self._generate_tail_curl_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.EARS_PERK:
            flow = self._generate_ears_perk_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.EARS_FLATTEN:
            flow = self._generate_ears_flatten_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.EARS_TWITCH:
            flow = self._generate_ears_twitch_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.BREATHING:
            flow = self._generate_breathing_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.FUR_RUSTLE:
            flow = self._generate_fur_rustle_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.WING_FLAP:
            flow = self._generate_wing_flap_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.BLINK:
            flow = self._generate_blink_flow(H, W, t, base_freq, base_strength, phase)
        elif motion_type == FurryMotionType.BODY_BOUNCE:
            flow = self._generate_body_bounce_flow(H, W, t, base_freq, base_strength, phase)
        else:
            flow = self._generate_default_flow(H, W, t, base_freq, base_strength, phase)
        
        for i in range(num_frames):
            flows.append(flow[i])
        
        return torch.stack(flows, dim=0)
    
    def _generate_tail_wag_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate tail wagging motion."""
        flows = []
        
        center_x = W * 0.7
        center_y = H * 0.6
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        tail_region = (x_coords > center_x * 0.8).float()
        tail_region = tail_region.unsqueeze(0).expand(H, -1)
        
        for i, t_val in enumerate(t):
            wag = strength * np.sin(freq * t_val.item() + phase)
            
            distance_y = grid_y - center_y
            distance_x = grid_x - center_x
            distance = torch.sqrt(distance_x**2 + distance_y**2 + 1)
            
            tail_influence = torch.exp(-distance / (W * 0.3)) * tail_region
            
            flow_x = wag * tail_influence * (grid_x - center_x) / (distance + 1)
            flow_y = 0.1 * strength * tail_influence * np.sin(freq * t_val.item() * 2 + phase)
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_tail_swish_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate tail swishing motion."""
        flows = []
        
        center_x = W * 0.7
        center_y = H * 0.5
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        for i, t_val in enumerate(t):
            swish_x = strength * 0.5 * np.sin(freq * t_val.item() + phase)
            swish_y = strength * 0.3 * np.sin(freq * t_val.item() * 1.5 + phase + 0.5)
            
            distance = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2 + 1)
            tail_influence = torch.exp(-distance / (W * 0.4))
            
            flow_x = swish_x * tail_influence * (grid_x - center_x) / distance
            flow_y = swish_y * tail_influence * (grid_y - center_y) / distance
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_tail_curl_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate tail curling motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            curl_amount = strength * 0.3 * np.sin(freq * t_val.item() + phase)
            
            center_x = W * 0.6
            center_y = H * 0.7
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            distance = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2 + 1)
            tail_influence = torch.exp(-distance / (W * 0.3))
            
            flow_x = curl_amount * grid_y / H * tail_influence
            flow_y = -curl_amount * grid_x / W * tail_influence * 0.5
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_ears_perk_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate ears perking motion."""
        flows = []
        
        left_ear_center = (W * 0.35, H * 0.25)
        right_ear_center = (W * 0.65, H * 0.25)
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        for i, t_val in enumerate(t):
            perk = strength * (1 + np.sin(freq * t_val.item() + phase)) * 0.5
            
            left_distance = torch.sqrt(
                (grid_x - left_ear_center[0])**2 + (grid_y - left_ear_center[1])**2
            )
            right_distance = torch.sqrt(
                (grid_x - right_ear_center[0])**2 + (grid_y - right_ear_center[1])**2
            )
            
            left_influence = torch.exp(-left_distance / (W * 0.1)) * (grid_y < H * 0.35).float()
            right_influence = torch.exp(-right_distance / (W * 0.1)) * (grid_y < H * 0.35).float()
            
            flow_x = 0.0
            flow_y = -perk * 0.2 * (left_influence + right_influence)
            
            flow = torch.stack([flow_x * torch.ones_like(grid_x), flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_ears_flatten_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate ears flattening motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            flatten = strength * np.sin(freq * t_val.item() + phase)
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            ear_region = (grid_y < H * 0.4).float()
            
            flow_y = flatten * 0.1 * ear_region
            
            flow = torch.stack([torch.zeros_like(grid_x), flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_ears_twitch_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate ear twitching motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            if i > 0 and np.random.random() < 0.1:
                twitch = strength * 0.3 * np.random.choice([-1, 1])
            else:
                twitch = strength * 0.05 * np.sin(freq * t_val.item() * 3 + phase)
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            left_ear = torch.exp(-((grid_x - W * 0.35)**2 + (grid_y - H * 0.25)**2) / (W * 0.02))
            right_ear = torch.exp(-((grid_x - W * 0.65)**2 + (grid_y - H * 0.25)**2) / (W * 0.02))
            
            flow_x = twitch * 0.2 * (left_ear + right_ear * (-1))
            flow_y = twitch * 0.1 * (left_ear + right_ear)
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_breathing_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate breathing motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            breath = strength * 0.2 * np.sin(freq * t_val.item() + phase)
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            chest_region = ((grid_x > W * 0.3) & (grid_x < W * 0.7) & 
                          (grid_y > H * 0.4) & (grid_y < H * 0.7)).float()
            
            distance_y = torch.abs(grid_y - H * 0.55)
            breath_influence = torch.exp(-distance_y / (H * 0.15))
            
            flow_y = breath * breath_influence * chest_region
            flow_y = flow_y - breath * 0.1 * ((grid_y < H * 0.4).float())
            
            flow = torch.stack([torch.zeros_like(grid_x), flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_fur_rustle_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate fur rustling motion."""
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        flows = []
        
        noise = torch.randn(H, W, device=self.device)
        
        for i, t_val in enumerate(t):
            rustle = strength * 0.15 * np.sin(freq * t_val.item() + phase)
            
            spatial_noise = torch.roll(noise, i * 5, dims=0)
            spatial_noise = torch.roll(spatial_noise, i * 3, dims=1)
            
            flow_x = rustle * spatial_noise
            flow_y = rustle * 0.5 * torch.roll(spatial_noise, 10, dims=0)
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_wing_flap_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate wing flapping motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            flap = strength * np.sin(freq * t_val.item() * 2 + phase)
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            left_wing = ((grid_x < W * 0.3) & (grid_y > H * 0.2) & (grid_y < H * 0.6)).float()
            right_wing = ((grid_x > W * 0.7) & (grid_y > H * 0.2) & (grid_y < H * 0.6)).float()
            
            flow_y = flap * 0.3 * (left_wing - right_wing)
            flow_x = flap * 0.1 * (left_wing - right_wing) * (grid_x - W * 0.5)
            
            flow = torch.stack([flow_x, flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_blink_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate eye blinking motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            blink = strength * 0.1 * np.sin(freq * t_val.item() + phase)
            blink = max(0, blink)
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            eye_region = ((grid_y > H * 0.25) & (grid_y < H * 0.35) & 
                        (grid_x > W * 0.35) & (grid_x < W * 0.65)).float()
            
            flow_y = blink * eye_region
            
            flow = torch.stack([torch.zeros_like(grid_x), flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_body_bounce_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate body bouncing motion."""
        flows = []
        
        for i, t_val in enumerate(t):
            bounce = strength * 0.15 * abs(np.sin(freq * t_val.item() + phase))
            
            y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            body_region = ((grid_x > W * 0.25) & (grid_x < W * 0.75) & 
                          (grid_y > H * 0.25) & (grid_y < H * 0.75)).float()
            
            distance_y = torch.abs(grid_y - H * 0.5)
            bounce_influence = torch.exp(-distance_y / (H * 0.25))
            
            flow_y = -bounce * bounce_influence * body_region
            
            flow = torch.stack([torch.zeros_like(grid_x), flow_y], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _generate_default_flow(
        self,
        H: int,
        W: int,
        t: torch.Tensor,
        freq: float,
        strength: float,
        phase: float
    ) -> torch.Tensor:
        """Generate default subtle motion."""
        flows = []
        
        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        for i, t_val in enumerate(t):
            flow_x = strength * 0.05 * np.sin(freq * t_val.item() + phase)
            flow_y = strength * 0.03 * np.cos(freq * t_val.item() * 0.7 + phase)
            
            flow_x_grid = flow_x * torch.ones_like(grid_x)
            flow_y_grid = flow_y * torch.ones_like(grid_y)
            
            flow = torch.stack([flow_x_grid, flow_y_grid], dim=0)
            flows.append(flow)
        
        return torch.stack(flows, dim=0)
    
    def _apply_region_mask(
        self,
        motion: torch.Tensor,
        segmentation: torch.Tensor,
        motion_type: FurryMotionType
    ) -> torch.Tensor:
        """Apply region-based masking to motion."""
        if segmentation.dim() == 3:
            segmentation = segmentation[0]
        
        region_masks = self._get_region_masks(segmentation)
        
        affected_regions = self._get_affected_regions(motion_type)
        
        masked_motion = motion.clone()
        
        for region in FurryRegion:
            if region not in affected_regions and region != FurryRegion.FUR:
                if region.value in region_masks:
                    mask = region_masks[region.value]
                    if mask is not None:
                        masked_motion = masked_motion * (1 - mask.unsqueeze(0) * 0.8)
        
        return masked_motion
    
    def _get_region_masks(
        self,
        segmentation: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Extract region masks from segmentation."""
        return {
            "head": None,
            "ears_left": None,
            "ears_right": None,
            "tail": None,
            "body": None,
            "fur": segmentation if segmentation is not None else None,
        }
    
    def _get_affected_regions(
        self,
        motion_type: FurryMotionType
    ) -> List[FurryRegion]:
        """Get regions affected by motion type."""
        region_map = {
            FurryMotionType.TAIL_WAG: [FurryRegion.TAIL],
            FurryMotionType.TAIL_SWISH: [FurryRegion.TAIL],
            FurryMotionType.TAIL_CURL: [FurryRegion.TAIL],
            FurryMotionType.EARS_PERK: [FurryRegion.EARS_LEFT, FurryRegion.EARS_RIGHT],
            FurryMotionType.EARS_FLATTEN: [FurryRegion.EARS_LEFT, FurryRegion.EARS_RIGHT],
            FurryMotionType.EARS_TWITCH: [FurryRegion.EARS_LEFT, FurryRegion.EARS_RIGHT],
            FurryMotionType.BREATHING: [FurryRegion.BODY, FurryRegion.CHEST],
            FurryMotionType.FUR_RUSTLE: [FurryRegion.FUR],
            FurryMotionType.WING_FLAP: [FurryRegion.WINGS],
            FurryMotionType.BLINK: [FurryRegion.EYES],
            FurryMotionType.BODY_BOUNCE: [FurryRegion.BODY],
        }
        return region_map.get(motion_type, [FurryRegion.FUR])
    
    def _add_natural_variation(
        self,
        motion: torch.Tensor
    ) -> torch.Tensor:
        """Add natural variation to motion."""
        if self.config.natural_variation <= 0:
            return motion
        
        T, C, H, W = motion.shape
        
        variation = torch.randn(T, C, H, W, device=self.device) * self.config.natural_variation * 0.1
        
        smoothed_variation = F.avg_pool2d(
            variation.view(T * C, 1, H, W),
            kernel_size=3,
            stride=1,
            padding=1
        ).view(T, C, H, W)
        
        return motion + smoothed_variation
    
    def combine_motions(
        self,
        motions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Combine multiple motion patterns.
        
        Args:
            motions: List of motion tensors [T, 2, H, W]
            weights: Optional weights for each motion
            
        Returns:
            Combined motion tensor
        """
        if not motions:
            return torch.zeros(1, 2, 256, 256, device=self.device)
        
        if weights is None:
            weights = [1.0 / len(motions)] * len(motions)
        
        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        combined = torch.zeros_like(motions[0])
        
        for motion, weight in zip(motions, weights_tensor):
            if motion.shape != combined.shape:
                motion = F.interpolate(
                    motion,
                    size=combined.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            combined = combined + motion * weight
        
        return combined
    
    def get_motion_for_pose(
        self,
        pose_type: str,
        num_frames: int,
        resolution: Tuple[int, int]
    ) -> torch.Tensor:
        """Get appropriate motion for a pose type."""
        pose_motion_map = {
            "sitting": self.generate_motion(
                num_frames, resolution, FurryMotionType.BODY_SHIFT
            ),
            "standing": self.generate_motion(
                num_frames, resolution, FurryMotionType.BREATHING
            ),
            "walking": self.generate_motion(
                num_frames, resolution, FurryMotionType.BODY_BOUNCE
            ),
            "running": self.generate_motion(
                num_frames, resolution, FurryMotionType.BODY_BOUNCE
            ),
        }
        
        return pose_motion_map.get(
            pose_type,
            self.generate_motion(num_frames, resolution, FurryMotionType.BREATHING)
        )