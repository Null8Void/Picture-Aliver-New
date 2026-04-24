from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FurryMotionGenerator(nn.Module):
    """
    Furry character motion generator.
    
    Specialized generator for animating furry characters with:
    - Tail wagging/swishing motions
    - Ear movements (twitching, perking)
    - Breathing animations
    - Wing flapping (for winged characters)
    - Fur rustling effects
    
    Attributes:
        device: Compute device
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
    
    def generate(
        self,
        image: torch.Tensor,
        motion_type: str = "tail_wag",
        intensity: float = 0.8,
        speed: float = 1.0,
        num_frames: int = 24
    ) -> List[torch.Tensor]:
        """
        Generate furry character motion.
        
        Args:
            image: Input image tensor (C, H, W)
            motion_type: Type of motion (tail_wag, ears, breathing, wings, fur)
            intensity: Motion strength (0-1)
            speed: Motion speed multiplier
            num_frames: Number of frames
            
        Returns:
            List of flow fields
        """
        _, h, w = image.shape
        
        if motion_type == "tail_wag":
            return self._generate_tail(image, num_frames, intensity, speed)
        elif motion_type == "ears":
            return self._generate_ears(image, num_frames, intensity, speed)
        elif motion_type == "breathing":
            return self._generate_breathing(image, num_frames, intensity, speed)
        elif motion_type == "wings":
            return self._generate_wings(image, num_frames, intensity, speed)
        elif motion_type == "fur":
            return self._generate_fur(image, num_frames, intensity, speed)
        elif motion_type == "combined":
            return self._generate_combined(image, num_frames, intensity, speed)
        else:
            return self._generate_tail(image, num_frames, intensity, speed)
    
    def _generate_tail(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate tail wagging motion."""
        _, h, w = image.shape
        center_x = w * 0.7
        center_y = h * 0.6
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 4 * np.pi * speed
            
            x = torch.arange(w, dtype=torch.float32, device=self.device)
            y = torch.arange(h, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            dist = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            falloff = torch.exp(-dist / (w * 0.3))
            
            amplitude = 15.0 * intensity
            dx = amplitude * torch.sin(phase) * falloff * (xx - center_x) / (dist + 1)
            dy = amplitude * 0.3 * torch.sin(phase * 0.7) * falloff
            
            dx = torch.nan_to_num(dx, nan=0.0)
            dy = torch.nan_to_num(dy, nan=0.0)
            
            flows.append(torch.stack([dx, dy], dim=-1))
        
        return flows
    
    def _generate_ears(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate ear twitching motion."""
        _, h, w = image.shape
        left_ear = (w * 0.35, h * 0.25)
        right_ear = (w * 0.65, h * 0.25)
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 6 * np.pi * speed
            
            x = torch.arange(w, dtype=torch.float32, device=self.device)
            y = torch.arange(h, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            dx = torch.zeros_like(xx)
            dy = torch.zeros_like(xx)
            
            for ear_x, ear_y in [left_ear, right_ear]:
                dist = torch.sqrt((xx - ear_x)**2 + (yy - ear_y)**2)
                falloff = torch.exp(-dist / (w * 0.15))
                dx += 5.0 * intensity * torch.sin(phase) * falloff * (yy - ear_y) / (dist + 1)
                dy += 3.0 * intensity * torch.sin(phase * 1.3) * falloff
            
            dx = torch.nan_to_num(dx, nan=0.0)
            dy = torch.nan_to_num(dy, nan=0.0)
            
            flows.append(torch.stack([dx, dy], dim=-1))
        
        return flows
    
    def _generate_breathing(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate breathing animation."""
        _, h, w = image.shape
        center_x, center_y = w * 0.5, h * 0.5
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 2 * np.pi * speed
            
            x = torch.arange(w, dtype=torch.float32, device=self.device)
            y = torch.arange(h, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            dist = torch.sqrt(((xx - center_x) / w * 2)**2 + ((yy - center_y) / h * 2)**2)
            amplitude = 3.0 * intensity
            dy = amplitude * torch.sin(phase) * (1.0 - torch.tanh(dist * 2))
            
            flows.append(torch.stack([torch.zeros_like(dy), dy], dim=-1))
        
        return flows
    
    def _generate_wings(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate wing flapping motion."""
        _, h, w = image.shape
        left_wing = (w * 0.25, h * 0.4)
        right_wing = (w * 0.75, h * 0.4)
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 4 * np.pi * speed
            
            x = torch.arange(w, dtype=torch.float32, device=self.device)
            y = torch.arange(h, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            dx = torch.zeros_like(xx)
            dy = torch.zeros_like(xx)
            
            for wing_x, wing_y in [left_wing, right_wing]:
                dist = torch.sqrt((xx - wing_x)**2 + (yy - wing_y)**2)
                falloff = torch.exp(-dist / (w * 0.25))
                direction = 1 if wing_x < w / 2 else -1
                dx += 20.0 * intensity * direction * torch.sin(phase) * falloff
                dy += 10.0 * intensity * torch.cos(phase * 0.7) * falloff
            
            flows.append(torch.stack([dx, dy], dim=-1))
        
        return flows
    
    def _generate_fur(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate fur rustling effect."""
        _, h, w = image.shape
        
        flows = []
        for t in range(num_frames):
            phase = t / num_frames * 3 * np.pi * speed
            
            x = torch.arange(w, dtype=torch.float32, device=self.device)
            y = torch.arange(h, dtype=torch.float32, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            
            wave_strength = 5.0 * intensity
            dx = wave_strength * torch.sin(yy / h * 5 * np.pi + phase) * (1 - yy / h)
            dy = wave_strength * 0.5 * torch.sin(xx / w * 4 * np.pi + phase * 1.3)
            
            flows.append(torch.stack([dx, dy], dim=-1))
        
        return flows
    
    def _generate_combined(
        self,
        image: torch.Tensor,
        num_frames: int,
        intensity: float,
        speed: float
    ) -> List[torch.Tensor]:
        """Generate combined furry motions."""
        tail = self._generate_tail(image, num_frames, intensity, speed)
        ears = self._generate_ears(image, num_frames, intensity * 0.6, speed)
        breath = self._generate_breathing(image, num_frames, intensity * 0.3, speed)
        
        combined = []
        for i in range(num_frames):
            combined.append(tail[i] + ears[i] + breath[i])
        
        return combined