"""Environmental motion generation for natural scene effects.

Creates motion for:
- Wind effects
- Lighting shifts (sun movement, clouds)
- Sky motion
- Atmosphere effects
- Weather simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class EnvironmentalType(Enum):
    """Types of environmental motion."""
    WIND = "wind"
    LIGHTING = "lighting"
    SKY = "sky"
    CLOUDS = "clouds"
    ATMOSPHERE = "atmosphere"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    PARTICLES = "particles"


@dataclass
class EnvironmentalEffect:
    """Environmental effect with associated motion."""
    effect_type: EnvironmentalType
    flow_field: torch.Tensor
    intensity: float = 1.0
    temporal_phase: float = 0.0
    spatial_mask: Optional[torch.Tensor] = None
    
    @property
    def has_motion(self) -> bool:
        """Check if effect has motion."""
        return self.flow_field.abs().sum() > 1e-6


@dataclass
class EnvironmentalMotionConfig:
    """Configuration for environmental motion."""
    effect_type: EnvironmentalType = EnvironmentalType.WIND
    intensity: float = 0.5
    direction: float = 0.0
    speed: float = 1.0
    turbulence: float = 0.3
    temporal_frequencies: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    spatial_scales: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    
    wind_config: Optional[Dict] = None
    lighting_config: Optional[Dict] = None
    sky_config: Optional[Dict] = None
    cloud_config: Optional[Dict] = None
    
    def __post_init__(self):
        if self.wind_config is None:
            self.wind_config = {
                "strength": 0.5,
                "gust_frequency": 0.2,
                "gust_strength": 0.3,
            }
        if self.lighting_config is None:
            self.lighting_config = {
                "sun_path_angle": 45.0,
                "cloud_coverage": 0.3,
                "shadow_strength": 0.3,
            }
        if self.sky_config is None:
            self.sky_config = {
                "color_shift_intensity": 0.1,
                "gradient_strength": 0.5,
            }
        if self.cloud_config is None:
            self.cloud_config = {
                "opacity": 0.8,
                "wispy": False,
                "coverage": 0.4,
            }


class EnvironmentalMotionGenerator:
    """Generates environmental motion effects for natural scenes.
    
    Creates:
    - Wind: Directional motion with turbulence
    - Lighting: Sun position, shadow shifts
    - Sky: Color gradients, atmosphere
    - Clouds: Movement across frame
    - Particles: Dust, pollen, etc.
    
    Args:
        config: Environmental motion configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[EnvironmentalMotionConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or EnvironmentalMotionConfig()
        self.device = device or torch.device("cpu")
        
        self._noise_buffer: Optional[torch.Tensor] = None
        self._phase = 0.0
    
    def generate(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        depth_map: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        sky_mask: Optional[torch.Tensor] = None
    ) -> List[EnvironmentalEffect]:
        """Generate environmental motion effects.
        
        Args:
            num_frames: Number of frames
            resolution: (H, W) resolution
            depth_map: Optional depth map
            segmentation: Optional segmentation mask
            sky_mask: Optional sky region mask
            
        Returns:
            List of EnvironmentalEffect objects
        """
        effects = []
        H, W = resolution
        
        if self.config.effect_type == EnvironmentalType.WIND:
            effect = self._generate_wind_effect(H, W, depth_map, segmentation)
            effects.append(effect)
        
        elif self.config.effect_type == EnvironmentalType.LIGHTING:
            effect = self._generate_lighting_effect(H, W, num_frames)
            effects.append(effect)
        
        elif self.config.effect_type == EnvironmentalType.SKY:
            effect = self._generate_sky_effect(H, W, num_frames)
            effects.append(effect)
        
        elif self.config.effect_type == EnvironmentalType.CLOUDS:
            effect = self._generate_cloud_effect(H, W, sky_mask)
            effects.append(effect)
        
        elif self.config.effect_type == EnvironmentalType.FOG:
            effect = self._generate_fog_effect(H, W, depth_map)
            effects.append(effect)
        
        elif self.config.effect_type == EnvironmentalType.PARTICLES:
            effect = self._generate_particle_effect(H, W)
            effects.append(effect)
        
        else:
            effect = self._generate_general_effect(H, W)
            effects.append(effect)
        
        return effects
    
    def _generate_wind_effect(
        self,
        H: int,
        W: int,
        depth_map: Optional[torch.Tensor],
        segmentation: Optional[torch.Tensor]
    ) -> EnvironmentalEffect:
        """Generate wind motion effect."""
        wind_config = self.config.wind_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        direction_rad = np.radians(self.config.direction)
        base_dir = torch.tensor([np.cos(direction_rad), np.sin(direction_rad)], device=self.device)
        
        wind_strength = wind_config.get("strength", 0.5) * self.config.intensity
        
        large_scale = self._generate_fbm_noise(H, W, scale=0.1, octaves=3)
        medium_scale = self._generate_fbm_noise(H, W, scale=0.3, octaves=2)
        small_scale = self._generate_fbm_noise(H, W, scale=0.8, octaves=1)
        
        turbulence = (
            large_scale * 0.5 +
            medium_scale * 0.3 +
            small_scale * 0.2
        )
        
        flow_x = base_dir[0] * wind_strength * (1 + turbulence)
        flow_y = base_dir[1] * wind_strength * (1 + turbulence)
        
        gust_freq = wind_config.get("gust_frequency", 0.2)
        gust_strength = wind_config.get("gust_strength", 0.3)
        
        phase = self._phase * gust_freq * 2 * np.pi
        gust_mod = 1 + gust_strength * np.sin(phase)
        
        flow_x = flow_x * gust_mod
        flow_y = flow_y * gust_mod
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        if depth_map is not None:
            if depth_map.dim() == 2:
                depth_map = depth_map.unsqueeze(0)
            
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_weight = F.interpolate(
                depth_norm.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            
            far_weight = 1.0 + (1.0 - depth_weight) * 0.5
            flow_field = flow_field * far_weight
        
        self._phase += 0.01 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.WIND,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_lighting_effect(
        self,
        H: int,
        W: int,
        num_frames: int
    ) -> EnvironmentalEffect:
        """Generate lighting shift effect."""
        lighting_config = self.config.lighting_config
        
        sun_angle = lighting_config.get("sun_path_angle", 45.0)
        
        phase = self._phase * self.config.speed
        sun_position = sun_angle * np.sin(phase)
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        sun_intensity = lighting_config.get("cloud_coverage", 0.3)
        cloud_modulation = 1.0 - sun_intensity * 0.5 * np.sin(phase * 0.5)
        
        light_gradient = (H - grid_y) / H * cloud_modulation
        
        flow_x = torch.zeros_like(grid_x)
        flow_y = (grid_y - H/2) * 0.01 * np.sin(phase) * light_gradient
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        shadow_strength = lighting_config.get("shadow_strength", 0.3)
        flow_field = flow_field * shadow_strength * self.config.intensity
        
        self._phase += 0.005 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.LIGHTING,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_sky_effect(
        self,
        H: int,
        W: int,
        num_frames: int
    ) -> EnvironmentalEffect:
        """Generate sky/atmosphere motion."""
        sky_config = self.config.sky_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        phase = self._phase * self.config.speed
        
        sky_shift = sky_config.get("color_shift_intensity", 0.1) * self.config.intensity
        
        flow_x = sky_shift * 0.5 * np.sin(grid_y / H * np.pi + phase)
        flow_y = torch.zeros_like(grid_x)
        
        gradient_strength = sky_config.get("gradient_strength", 0.5)
        top_half = (grid_y < H / 2).float()
        flow_x = flow_x * top_half * gradient_strength
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        self._phase += 0.008 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.SKY,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_cloud_effect(
        self,
        H: int,
        W: int,
        sky_mask: Optional[torch.Tensor]
    ) -> EnvironmentalEffect:
        """Generate cloud motion effect."""
        cloud_config = self.config.cloud_config
        
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        phase = self._phase * self.config.speed
        direction_rad = np.radians(self.config.direction)
        
        cloud_opacity = cloud_config.get("opacity", 0.8) * self.config.intensity
        wispy = cloud_config.get("wispy", False)
        
        if wispy:
            cloud_noise = self._generate_fbm_noise(H, W, scale=0.4, octaves=4)
        else:
            cloud_noise = self._generate_fbm_noise(H, W, scale=0.15, octaves=3)
        
        flow_x = np.cos(direction_rad) * cloud_noise * cloud_opacity * 2
        flow_y = np.sin(direction_rad) * cloud_noise * cloud_opacity * 2
        
        flow_x = flow_x + 0.3 * cloud_opacity * np.sin(grid_y / H * 4 + phase)
        flow_y = flow_y + 0.1 * cloud_opacity * np.sin(grid_x / W * 2 + phase * 0.7)
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        if sky_mask is not None:
            flow_field = flow_field * sky_mask.float()
        
        self._phase += 0.003 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.CLOUDS,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase,
            spatial_mask=sky_mask
        )
    
    def _generate_fog_effect(
        self,
        H: int,
        W: int,
        depth_map: Optional[torch.Tensor]
    ) -> EnvironmentalEffect:
        """Generate fog/mist motion."""
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        phase = self._phase * self.config.speed
        
        fog_turbulence = self._generate_fbm_noise(H, W, scale=0.2, octaves=3)
        
        direction_rad = np.radians(self.config.direction)
        flow_x = np.cos(direction_rad) * fog_turbulence * 0.5 * self.config.intensity
        flow_y = np.sin(direction_rad) * fog_turbulence * 0.5 * self.config.intensity
        
        flow_x = flow_x + 0.2 * self.config.intensity * np.sin(grid_x / W * 3 + phase)
        flow_y = flow_y + 0.1 * self.config.intensity * np.sin(grid_y / H * 2 + phase * 0.8)
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        if depth_map is not None:
            if depth_map.dim() == 2:
                depth_map = depth_map.unsqueeze(0)
            
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_weight = F.interpolate(
                depth_norm.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            
            fog_strength = depth_weight * 0.5
            flow_field = flow_field * fog_strength
        
        self._phase += 0.005 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.FOG,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_particle_effect(
        self,
        H: int,
        W: int
    ) -> EnvironmentalEffect:
        """Generate particle/dust motion effect."""
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        phase = self._phase * self.config.speed
        direction_rad = np.radians(self.config.direction)
        
        particle_noise = self._generate_fbm_noise(H, W, scale=0.5, octaves=2)
        
        flow_x = np.cos(direction_rad) * particle_noise * 0.3 * self.config.intensity
        flow_y = np.sin(direction_rad) * particle_noise * 0.3 * self.config.intensity
        
        flow_x = flow_x + 0.1 * self.config.intensity * np.sin(phase + grid_x * 0.1)
        flow_y = flow_y + 0.1 * self.config.intensity * np.cos(phase + grid_y * 0.1)
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        self._phase += 0.02 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=EnvironmentalType.PARTICLES,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_general_effect(
        self,
        H: int,
        W: int
    ) -> EnvironmentalEffect:
        """Generate general environmental effect."""
        noise = self._generate_fbm_noise(H, W, scale=0.2, octaves=3)
        
        direction_rad = np.radians(self.config.direction)
        flow_x = np.cos(direction_rad) * noise * self.config.intensity
        flow_y = np.sin(direction_rad) * noise * self.config.intensity
        
        flow_field = torch.stack([flow_x, flow_y], dim=0)
        
        self._phase += 0.01 * self.config.speed
        
        return EnvironmentalEffect(
            effect_type=self.config.effect_type,
            flow_field=flow_field,
            intensity=self.config.intensity,
            temporal_phase=self._phase
        )
    
    def _generate_fbm_noise(
        self,
        H: int,
        W: int,
        scale: float = 0.1,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> torch.Tensor:
        """Generate fractal Brownian motion noise."""
        if self._noise_buffer is None or self._noise_buffer.shape != (H, W):
            base_noise = torch.randn(H, W, device=self.device)
            base_noise = F.avg_pool2d(
                base_noise.unsqueeze(0).unsqueeze(0),
                kernel_size=int(1/scale),
                stride=1,
                padding=0
            )
            base_noise = F.interpolate(
                base_noise,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze()
            self._noise_buffer = base_noise
        
        noise = self._noise_buffer.clone()
        
        for i in range(octaves):
            freq = scale * (lacunarity ** i)
            amp = persistence ** i
            
            kernel_size = max(3, int(1 / freq))
            if kernel_size > H or kernel_size > W:
                kernel_size = min(H, W)
            
            pooled = F.avg_pool2d(
                noise.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1
            )
            pooled = F.interpolate(
                pooled,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze()
            
            noise = noise + amp * pooled
        
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        noise = noise * 2 - 1
        
        return noise
    
    def combine_effects(
        self,
        effects: List[EnvironmentalEffect]
    ) -> torch.Tensor:
        """Combine multiple environmental effects.
        
        Args:
            effects: List of environmental effects
            
        Returns:
            Combined flow field [2, H, W]
        """
        if not effects:
            return torch.zeros(2, 512, 512, device=self.device)
        
        H, W = effects[0].flow_field.shape[-2:]
        combined = torch.zeros(2, H, W, device=self.device)
        
        for effect in effects:
            weight = effect.intensity
            
            if effect.spatial_mask is not None:
                mask = effect.spatial_mask.float().unsqueeze(0)
                combined = combined + effect.flow_field * mask * weight
            else:
                combined = combined + effect.flow_field * weight
        
        return combined
    
    def get_temporal_variation(
        self,
        effect: EnvironmentalEffect,
        t: float
    ) -> torch.Tensor:
        """Get flow field variation at time t.
        
        Args:
            effect: Base environmental effect
            t: Time value
            
        Returns:
            Modified flow field
        """
        phase_shift = 2 * np.pi * self.config.speed * t + effect.temporal_phase
        
        freq_mod = 1.0 + 0.2 * np.sin(phase_shift)
        
        varied = effect.flow_field * freq_mod
        
        turbulence = self._generate_fbm_noise(
            *effect.flow_field.shape[-2:],
            scale=0.3
        )
        turb_strength = 0.1 * self.config.turbulence
        
        varied = varied + torch.stack([
            turbulence * turb_strength * np.cos(phase_shift),
            turbulence * turb_strength * np.sin(phase_shift)
        ], dim=0)
        
        return varied * effect.intensity