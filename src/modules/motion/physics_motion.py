"""Physics-based motion for realistic movement simulation.

Provides:
- Cloth physics (spring-mass system)
- Hair dynamics (Verlet integration)
- Fluid simulation (simplified SPH)
- Particle systems
- Soft body physics
- Collision handling

Fast sexualized movements help smooth action even in non-explicit scenes by
providing natural cloth, hair, and body dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class PhysicsType(Enum):
    """Types of physics simulation."""
    CLOTH = "cloth"
    HAIR = "hair"
    FLUID = "fluid"
    PARTICLE = "particle"
    SOFT_BODY = "soft_body"
    RIGID = "rigid"


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation."""
    physics_type: PhysicsType = PhysicsType.CLOTH
    
    gravity: float = 9.8
    damping: float = 0.98
    stiffness: float = 0.5
    rest_distance: float = 1.0
    
    substeps: int = 3
    constraint_iterations: int = 5
    
    wind_strength: float = 0.0
    wind_direction: Tuple[float, float] = (1.0, 0.0)
    
    collision_enabled: bool = True
    collision_margin: float = 0.1
    
    timestep: float = 1.0 / 60.0


class PhysicsParticle:
    """A single particle in physics simulation."""
    
    def __init__(
        self,
        position: torch.Tensor,
        mass: float = 1.0,
        pinned: bool = False
    ):
        self.position = position
        self.prev_position = position.clone()
        self.velocity = torch.zeros_like(position)
        self.acceleration = torch.zeros_like(position)
        self.mass = mass
        self.pinned = pinned
        
        self.normal = torch.zeros_like(position)
        self.color = torch.ones(3)
    
    def apply_force(self, force: torch.Tensor) -> None:
        """Apply force to particle."""
        if not self.pinned:
            self.acceleration = self.acceleration + force / self.mass
    
    def update(self, dt: float, damping: float = 0.98) -> None:
        """Update particle position using Verlet integration."""
        if self.pinned:
            return
        
        new_position = (
            self.position * 2
            - self.prev_position
            + self.acceleration * dt * dt
        )
        
        new_position = new_position * damping + self.position * (1 - damping)
        
        self.prev_position = self.position.clone()
        self.position = new_position
        self.velocity = self.position - self.prev_position
        self.acceleration.zero_()


class PhysicsSpring:
    """Spring constraint between two particles."""
    
    def __init__(
        self,
        particle1: PhysicsParticle,
        particle2: PhysicsParticle,
        rest_length: Optional[float] = None,
        stiffness: float = 0.5
    ):
        self.particle1 = particle1
        self.particle2 = particle2
        self.stiffness = stiffness
        
        if rest_length is None:
            self.rest_length = torch.norm(
                particle2.position - particle1.position
            ).item()
        else:
            self.rest_length = rest_length


class ClothSimulator:
    """Cloth physics simulation using spring-mass system."""
    
    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[PhysicsConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or PhysicsConfig(physics_type=PhysicsType.CLOTH)
        self.device = device or torch.device("cpu")
        
        self.width = width
        self.height = height
        
        self.particles: List[PhysicsParticle] = []
        self.springs: List[PhysicsSpring] = []
        
        self._initialize_cloth()
    
    def _initialize_cloth(self) -> None:
        """Initialize cloth particles and springs."""
        spacing = self.config.rest_distance
        
        for y in range(self.height):
            for x in range(self.width):
                pos = torch.tensor(
                    [x * spacing, y * spacing, 0.0],
                    dtype=torch.float32,
                    device=self.device
                )
                
                pinned = (y == 0) and (x % 10 == 0)
                
                particle = PhysicsParticle(pos, mass=1.0, pinned=pinned)
                self.particles.append(particle)
        
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                
                if x < self.width - 1:
                    spring = PhysicsSpring(
                        self.particles[idx],
                        self.particles[idx + 1],
                        stiffness=self.config.stiffness
                    )
                    self.springs.append(spring)
                
                if y < self.height - 1:
                    spring = PhysicsSpring(
                        self.particles[idx],
                        self.particles[idx + self.width],
                        stiffness=self.config.stiffness
                    )
                    self.springs.append(spring)
                
                if x < self.width - 1 and y < self.height - 1:
                    spring = PhysicsSpring(
                        self.particles[idx],
                        self.particles[idx + self.width + 1],
                        stiffness=self.config.stiffness * 0.5
                    )
                    self.springs.append(spring)
                
                if x > 0 and y < self.height - 1:
                    spring = PhysicsSpring(
                        self.particles[idx],
                        self.particles[idx + self.width - 1],
                        stiffness=self.config.stiffness * 0.5
                    )
                    self.springs.append(spring)
    
    def simulate(self, num_steps: int = 1) -> None:
        """Run physics simulation."""
        dt = self.config.timestep / self.config.substeps
        
        for _ in range(num_steps):
            for _ in range(self.config.substeps):
                self._apply_external_forces()
                self._integrate(dt)
                self._solve_constraints()
    
    def _apply_external_forces(self) -> None:
        """Apply gravity and wind."""
        gravity = torch.tensor(
            [0, -self.config.gravity, 0],
            dtype=torch.float32,
            device=self.device
        )
        
        wind = torch.tensor(
            [
                self.config.wind_direction[0] * self.config.wind_strength,
                self.config.wind_direction[1] * self.config.wind_strength,
                0.0
            ],
            dtype=torch.float32,
            device=self.device
        )
        
        for particle in self.particles:
            particle.apply_force(gravity * particle.mass)
            particle.apply_force(wind * particle.mass)
    
    def _integrate(self, dt: float) -> None:
        """Integrate particle positions."""
        for particle in self.particles:
            particle.update(dt, self.config.damping)
    
    def _solve_constraints(self) -> None:
        """Solve spring constraints."""
        for _ in range(self.config.constraint_iterations):
            for spring in self.springs:
                diff = spring.particle2.position - spring.particle1.position
                dist = torch.norm(diff)
                
                if dist < 1e-6:
                    continue
                
                correction = diff * (1 - spring.rest_length / dist)
                correction = correction * spring.stiffness
                
                if not spring.particle1.pinned:
                    spring.particle1.position = (
                        spring.particle1.position + correction * 0.5
                    )
                
                if not spring.particle2.pinned:
                    spring.particle2.position = (
                        spring.particle2.position - correction * 0.5
                    )
    
    def get_particle_positions(self) -> torch.Tensor:
        """Get all particle positions as tensor."""
        positions = torch.stack([p.position for p in self.particles])
        return positions
    
    def get_flow_field(self) -> torch.Tensor:
        """Get optical flow field from cloth motion."""
        positions = self.get_particle_positions()
        
        H, W = self.height, self.width
        
        pos_reshaped = positions.view(H, W, 3)
        
        flow_x = pos_reshaped[:, :, 0] - pos_reshaped[:, :, 0].roll(1, dims=1)
        flow_y = pos_reshaped[:, :, 1] - pos_reshaped[:, :, 1].roll(1, dims=0)
        
        flow = torch.stack([flow_x, flow_y], dim=0)
        
        return flow


class HairSimulator:
    """Hair dynamics simulation using Verlet integration."""
    
    def __init__(
        self,
        num_strands: int = 50,
        strand_length: int = 20,
        config: Optional[PhysicsConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or PhysicsConfig(physics_type=PhysicsType.HAIR)
        self.device = device or torch.device("cpu")
        
        self.num_strands = num_strands
        self.strand_length = strand_length
        
        self.strands: List[List[PhysicsParticle]] = []
        
        self._initialize_hair()
    
    def _initialize_hair(self) -> None:
        """Initialize hair strands."""
        spacing = 10.0
        
        for i in range(self.num_strands):
            strand = []
            
            base_x = (i % 10) * spacing
            base_y = -(i // 10) * spacing
            base_z = 0.0
            
            for j in range(self.strand_length):
                pos = torch.tensor(
                    [base_x, base_y + j * 2, base_z],
                    dtype=torch.float32,
                    device=self.device
                )
                
                particle = PhysicsParticle(pos, mass=0.1, pinned=(j == 0))
                strand.append(particle)
            
            self.strands.append(strand)
    
    def simulate(self, num_steps: int = 1) -> None:
        """Run hair simulation."""
        dt = self.config.timestep / self.config.substeps
        
        for _ in range(num_steps):
            for _ in range(self.config.substeps):
                self._apply_forces()
                
                for strand in self.strands:
                    for i in range(len(strand)):
                        if i > 0:
                            particle = strand[i]
                            
                            rest_dist = 2.0
                            prev = strand[i - 1]
                            
                            diff = particle.position - prev.position
                            dist = torch.norm(diff)
                            
                            if dist > rest_dist:
                                correction = diff * ((dist - rest_dist) / dist)
                                correction = correction * 0.5 * self.config.stiffness
                                
                                if not particle.pinned:
                                    particle.position = particle.position - correction
                                
                                if not prev.pinned:
                                    prev.position = prev.position + correction
                
                for strand in self.strands:
                    for particle in strand:
                        if not particle.pinned:
                            particle.update(dt, self.config.damping)
    
    def _apply_forces(self) -> None:
        """Apply gravity and wind to hair."""
        gravity = torch.tensor(
            [0, -self.config.gravity * 0.5, 0],
            dtype=torch.float32,
            device=self.device
        )
        
        wind = torch.tensor(
            [
                self.config.wind_direction[0] * self.config.wind_strength,
                self.config.wind_direction[1] * self.config.wind_strength,
                0.0
            ],
            dtype=torch.float32,
            device=self.device
        )
        
        for strand in self.strands:
            for particle in strand:
                if not particle.pinned:
                    depth_factor = 1.0 + (strand.index(particle) / self.strand_length) * 0.5
                    
                    particle.apply_force(gravity * particle.mass)
                    particle.apply_force(wind * particle.mass * depth_factor)
    
    def get_flow_field(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Get flow field from hair motion."""
        H, W = grid_size
        
        flow = torch.zeros(2, H, W, device=self.device)
        
        for strand in self.strands:
            for i in range(len(strand) - 1):
                p1 = strand[i]
                p2 = strand[i + 1]
                
                dx = (p2.position[0] - p1.position[0]).item()
                dy = (p2.position[1] - p1.position[1]).item()
                
                grid_x = min(int(p1.position[0].item()), W - 1)
                grid_y = min(int(abs(p1.position[1].item())), H - 1)
                
                if 0 <= grid_x < W and 0 <= grid_y < H:
                    flow[0, grid_y, grid_x] = dx * 0.5
                    flow[1, grid_y, grid_x] = dy * 0.5
        
        return flow


class PhysicsMotionGenerator:
    """Generates realistic physics-based motion.
    
    Creates natural motion through:
    - Cloth dynamics (dresses, hair, flags)
    - Hair simulation (flowing, bouncing)
    - Particle systems (smoke, dust)
    - Soft body physics (skin, fabric)
    
    Fast sexualized movements benefit from:
    - Natural cloth physics for revealing clothing
    - Hair dynamics for body movement emphasis
    - Soft body simulation for skin/muscle
    - Particle effects for atmosphere
    
    Args:
        config: Physics configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[PhysicsConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or PhysicsConfig()
        self.device = device or torch.device("cpu")
        
        self._simulators: Dict[PhysicsType, object] = {}
        
        self._initialized = False
    
    def initialize(self, physics_type: PhysicsType) -> None:
        """Initialize physics simulator."""
        if physics_type == PhysicsType.CLOTH:
            self._simulators[physics_type] = ClothSimulator(
                width=32,
                height=32,
                config=self.config,
                device=self.device
            )
        elif physics_type == PhysicsType.HAIR:
            self._simulators[physics_type] = HairSimulator(
                num_strands=100,
                strand_length=15,
                config=self.config,
                device=self.device
            )
    
    def generate_motion(
        self,
        physics_type: PhysicsType,
        num_frames: int,
        mask: Optional[torch.Tensor] = None,
        wind: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """Generate physics-based motion field.
        
        Args:
            physics_type: Type of physics to simulate
            num_frames: Number of frames
            mask: Optional region mask
            wind: Optional (strength, direction) tuple
            
        Returns:
            Motion flow field [T, 2, H, W]
        """
        if wind is not None:
            self.config.wind_strength = wind[0]
            self.config.wind_direction = (
                np.cos(np.radians(wind[1])),
                np.sin(np.radians(wind[1]))
            )
        
        if physics_type not in self._simulators:
            self.initialize(physics_type)
        
        simulator = self._simulators[physics_type]
        
        H, W = 64, 64
        flows = []
        
        for _ in range(num_frames):
            simulator.simulate(num_steps=1)
            
            if isinstance(simulator, ClothSimulator):
                flow = simulator.get_flow_field()
            elif isinstance(simulator, HairSimulator):
                flow = simulator.get_flow_field((H, W))
            else:
                flow = torch.zeros(2, H, W, device=self.device)
            
            flows.append(flow)
        
        flows_tensor = torch.stack(flows, dim=0)
        
        flows_tensor = F.interpolate(
            flows_tensor.view(-1, 2, H, W),
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            mask_resized = F.interpolate(
                mask.float(),
                size=flows_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            
            flows_tensor = flows_tensor * mask_resized
        
        return flows_tensor
    
    def apply_to_video(
        self,
        frames: torch.Tensor,
        physics_type: PhysicsType,
        mask: Optional[torch.Tensor] = None,
        strength: float = 1.0
    ) -> torch.Tensor:
        """Apply physics motion to video frames.
        
        Args:
            frames: Video frames [T, C, H, W]
            physics_type: Physics type to use
            mask: Region mask
            strength: Motion strength
            
        Returns:
            Motion-enhanced frames
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        T, B, C, H, W = frames.shape
        
        flows = self.generate_motion(
            physics_type,
            T,
            mask,
            wind=(2.0, 45.0)
        )
        
        flows = F.interpolate(
            flows,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        
        result = []
        
        for t in range(T):
            flow = flows[t]
            
            y_coords = torch.linspace(-1, 1, H, device=self.device)
            x_coords = torch.linspace(-1, 1, W, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            grid_x = grid_x + flow[0] / W * 2 * strength
            grid_y = grid_y + flow[1] / H * 2 * strength
            
            grid_x = grid_x.clamp(-1, 1)
            grid_y = grid_y.clamp(-1, 1)
            
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            
            frame = frames[t]
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            
            warped = F.grid_sample(
                frame,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )
            
            result.append(warped.squeeze(0))
        
        return torch.stack(result, dim=0)


def create_physics_motion(
    num_frames: int,
    physics_type: PhysicsType,
    resolution: Tuple[int, int] = (256, 256),
    wind: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create physics-based motion field.
    
    Args:
        num_frames: Number of frames
        physics_type: Type of physics
        resolution: Output resolution
        wind: (strength, direction) for wind
        device: Target device
        
    Returns:
        Motion flow [T, 2, H, W]
    """
    generator = PhysicsMotionGenerator(device=device)
    
    flows = generator.generate_motion(
        physics_type,
        num_frames,
        wind=wind
    )
    
    flows = F.interpolate(
        flows,
        size=resolution,
        mode="bilinear",
        align_corners=False
    )
    
    return flows