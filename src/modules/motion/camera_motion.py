"""Camera motion generation for cinematic video effects.

Generates smooth camera trajectories including:
- Pan (horizontal movement)
- Zoom (in/out)
- Tilt (vertical movement)
- Dolly (forward/backward)
- Orbital (circular paths)
- Parallax (depth-aware movement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class CameraMode(Enum):
    """Camera motion modes."""
    STATIC = "static"
    PAN_LEFT = "pan-left"
    PAN_RIGHT = "pan-right"
    TILT_UP = "tilt-up"
    TILT_DOWN = "tilt-down"
    ZOOM_IN = "zoom-in"
    ZOOM_OUT = "zoom-out"
    DOLLY_FORWARD = "dolly-forward"
    DOLLY_BACKWARD = "dolly-backward"
    ORBITAL = "orbital"
    CINEMATIC = "cinematic"
    SUBTLE = "subtle"
    FREEHAND = "freehand"


@dataclass
class CameraParams:
    """Camera motion parameters at a specific time."""
    timestamp: float
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 1.0
    roll: float = 0.0
    dolly: float = 0.0


@dataclass
class CameraTrajectory:
    """Full camera trajectory for animation."""
    params: List[CameraParams] = field(default_factory=list)
    duration: float = 0.0
    fps: int = 8
    resolution: Tuple[int, int] = (512, 512)
    
    def __len__(self) -> int:
        return len(self.params)
    
    def __getitem__(self, idx: int) -> CameraParams:
        return self.params[idx]
    
    def get_frame_transform(
        self,
        frame_idx: int,
        total_frames: int
    ) -> Tuple[float, float, float]:
        """Get transform parameters (dx, dy, zoom) for a frame.
        
        Returns:
            Tuple of (delta_x, delta_y, zoom_factor)
        """
        if not self.params:
            return (0.0, 0.0, 1.0)
        
        t = frame_idx / max(total_frames - 1, 1)
        
        if len(self.params) == 1:
            p = self.params[0]
            return (p.pan, p.tilt, p.zoom)
        
        idx = int(t * (len(self.params) - 1))
        idx = min(idx, len(self.params) - 2)
        
        p1 = self.params[idx]
        p2 = self.params[idx + 1]
        
        local_t = (t * (len(self.params) - 1)) - idx
        local_t = self._ease_in_out(local_t)
        
        pan = p1.pan + (p2.pan - p1.pan) * local_t
        tilt = p1.tilt + (p2.tilt - p1.tilt) * local_t
        zoom = p1.zoom + (p2.zoom - p1.zoom) * local_t
        
        return (pan, tilt, zoom)
    
    def _ease_in_out(self, t: float) -> float:
        """Ease-in-out interpolation."""
        return t * t * (3 - 2 * t)
    
    def to_transforms(
        self,
        num_frames: int,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert trajectory to transform matrices for batch processing.
        
        Args:
            num_frames: Number of frames to generate
            image_size: (H, W) of images
            
        Returns:
            Transform tensor [num_frames, 3, 3]
        """
        h, w = image_size
        transforms = []
        
        for i in range(num_frames):
            pan, tilt, zoom = self.get_frame_transform(i, num_frames)
            
            dx = pan * w
            dy = tilt * h
            scale = zoom
            
            angle = 0.0
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            
            transform = np.array([
                [scale * cos_a, -scale * sin_a, dx],
                [scale * sin_a, scale * cos_a, dy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            transforms.append(transform)
        
        return torch.from_numpy(np.stack(transforms))
    
    def get_flow_field(
        self,
        num_frames: int,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Generate optical flow fields from camera motion.
        
        Args:
            num_frames: Number of frames
            height: Frame height
            width: Frame width
            
        Returns:
            Flow field [num_frames, 2, H, W]
        """
        flows = []
        prev_transform = None
        
        for i in range(num_frames):
            pan, tilt, zoom = self.get_frame_transform(i, num_frames)
            
            dx = pan * width
            dy = tilt * height
            
            if prev_transform is not None:
                dx = pan * width
                dy = tilt * height
            
            flow = torch.zeros(2, height, width)
            
            y_coords = torch.linspace(-1, 1, height)
            x_coords = torch.linspace(-1, 1, width)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            
            if zoom != 1.0:
                zoom_dx = (grid_x * (zoom - 1)) * width / 2
                zoom_dy = (grid_y * (zoom - 1)) * height / 2
                dx = dx + zoom_dx
                dy = dy + zoom_dy
            
            flow[0] = dx + grid_x * (1 - 1/zoom)
            flow[1] = dy + grid_y * (1 - 1/zoom)
            
            flows.append(flow)
            prev_transform = (pan, tilt, zoom)
        
        return torch.stack(flows, dim=0)


@dataclass
class CameraMotionConfig:
    """Configuration for camera motion generation."""
    mode: CameraMode = CameraMode.CINEMATIC
    strength: float = 0.5
    smoothness: float = 0.8
    amplitude: float = 0.1
    frequency: float = 1.0
    phase_offset: float = 0.0
    enable_roll: bool = False
    enable_dolly: bool = True
    
    min_zoom: float = 0.9
    max_zoom: float = 1.1
    pan_range: float = 0.1
    tilt_range: float = 0.1
    
    loop_compatible: bool = False
    seed: Optional[int] = None


class CameraMotionGenerator:
    """Generates smooth camera motion trajectories.
    
    Supports multiple motion styles:
    - Cinematic: Slow, smooth dolly with subtle pan/tilt
    - Subtle: Minimal movement for realistic video
    - Environmental: Natural wind-like motion
    - Orbital: Circular camera path around center
    - Zoom effects: Smooth zoom in/out
    
    Args:
        config: Camera motion configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[CameraMotionConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or CameraMotionConfig()
        self.device = device or torch.device("cpu")
        
        self._rng = np.random.RandomState(self.config.seed)
    
    def generate(
        self,
        num_frames: int,
        duration: float = 3.0,
        resolution: Tuple[int, int] = (512, 512),
        mode: Optional[CameraMode] = None
    ) -> CameraTrajectory:
        """Generate camera motion trajectory.
        
        Args:
            num_frames: Number of frames in output video
            duration: Duration in seconds
            resolution: Output resolution (H, W)
            mode: Override motion mode
            
        Returns:
            CameraTrajectory with motion parameters
        """
        mode = mode or self.config.mode
        fps = int(num_frames / duration)
        
        if mode == CameraMode.STATIC:
            return self._generate_static(num_frames, resolution, fps)
        elif mode == CameraMode.PAN_LEFT:
            return self._generate_pan(num_frames, resolution, fps, direction=-1)
        elif mode == CameraMode.PAN_RIGHT:
            return self._generate_pan(num_frames, resolution, fps, direction=1)
        elif mode == CameraMode.TILT_UP:
            return self._generate_tilt(num_frames, resolution, fps, direction=-1)
        elif mode == CameraMode.TILT_DOWN:
            return self._generate_tilt(num_frames, resolution, fps, direction=1)
        elif mode == CameraMode.ZOOM_IN:
            return self._generate_zoom(num_frames, resolution, fps, direction=1)
        elif mode == CameraMode.ZOOM_OUT:
            return self._generate_zoom(num_frames, resolution, fps, direction=-1)
        elif mode == CameraMode.DOLLY_FORWARD:
            return self._generate_dolly(num_frames, resolution, fps, direction=1)
        elif mode == CameraMode.DOLLY_BACKWARD:
            return self._generate_dolly(num_frames, resolution, fps, direction=-1)
        elif mode == CameraMode.ORBITAL:
            return self._generate_orbital(num_frames, resolution, fps)
        elif mode == CameraMode.CINEMATIC:
            return self._generate_cinematic(num_frames, resolution, fps)
        elif mode == CameraMode.SUBTLE:
            return self._generate_subtle(num_frames, resolution, fps)
        elif mode == CameraMode.FREEHAND:
            return self._generate_freehand(num_frames, resolution, fps)
        else:
            return self._generate_cinematic(num_frames, resolution, fps)
    
    def _generate_static(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int
    ) -> CameraTrajectory:
        """Generate static (no motion) trajectory."""
        params = [
            CameraParams(timestamp=0.0, pan=0.0, tilt=0.0, zoom=1.0)
        ]
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_pan(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int,
        direction: int = 1
    ) -> CameraTrajectory:
        """Generate horizontal pan motion."""
        params = []
        amplitude = self.config.pan_range * self.config.strength
        
        for i in range(num_frames):
            t = i / num_frames
            eased = self._ease_in_out(t)
            
            pan = direction * amplitude * (2 * eased - 1)
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=pan,
                tilt=0.0,
                zoom=1.0
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_tilt(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int,
        direction: int = 1
    ) -> CameraTrajectory:
        """Generate vertical tilt motion."""
        params = []
        amplitude = self.config.tilt_range * self.config.strength
        
        for i in range(num_frames):
            t = i / num_frames
            eased = self._ease_in_out(t)
            
            tilt = direction * amplitude * (2 * eased - 1)
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=0.0,
                tilt=tilt,
                zoom=1.0
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_zoom(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int,
        direction: int = 1
    ) -> CameraTrajectory:
        """Generate zoom motion."""
        params = []
        min_zoom = self.config.min_zoom
        max_zoom = self.config.max_zoom
        
        for i in range(num_frames):
            t = i / num_frames
            eased = self._ease_in_out(t)
            
            if direction > 0:
                zoom = min_zoom + (max_zoom - min_zoom) * eased
            else:
                zoom = max_zoom + (min_zoom - max_zoom) * eased
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=0.0,
                tilt=0.0,
                zoom=zoom
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_dolly(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int,
        direction: int = 1
    ) -> CameraTrajectory:
        """Generate dolly (forward/backward) motion."""
        params = []
        amplitude = 0.15 * self.config.strength
        
        for i in range(num_frames):
            t = i / num_frames
            eased = self._ease_in_out(t)
            
            dolly = direction * amplitude * (2 * eased - 1)
            zoom = 1.0 + dolly * 0.5
            zoom = np.clip(zoom, 0.8, 1.2)
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=0.0,
                tilt=0.0,
                zoom=zoom,
                dolly=dolly
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_orbital(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int
    ) -> CameraTrajectory:
        """Generate circular orbital motion."""
        params = []
        radius = 0.08 * self.config.strength
        freq = self.config.frequency
        
        phase = self.config.phase_offset
        
        for i in range(num_frames):
            t = i / num_frames
            angle = 2 * np.pi * freq * t + phase
            
            pan = radius * np.sin(angle)
            tilt = radius * 0.5 * np.cos(angle)
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=pan,
                tilt=tilt,
                zoom=1.0
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_cinematic(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int
    ) -> CameraTrajectory:
        """Generate cinematic motion with slow dolly and subtle pan."""
        params = []
        amplitude = 0.12 * self.config.strength
        freq = 0.5
        
        for i in range(num_frames):
            t = i / num_frames
            
            pan = amplitude * np.sin(2 * np.pi * freq * t + self.config.phase_offset)
            tilt = amplitude * 0.6 * np.sin(2 * np.pi * freq * t * 0.7 + 1)
            
            zoom = 1.0 + 0.05 * self.config.strength * np.sin(2 * np.pi * freq * t * 0.5)
            zoom = np.clip(zoom, self.config.min_zoom, self.config.max_zoom)
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=pan,
                tilt=tilt,
                zoom=zoom
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_subtle(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int
    ) -> CameraTrajectory:
        """Generate subtle natural motion."""
        params = []
        amplitude = 0.03 * self.config.strength
        
        prev_pan = 0.0
        prev_tilt = 0.0
        
        for i in range(num_frames):
            t = i / num_frames
            
            if i > 0:
                pan = prev_pan + self._rng.uniform(-0.01, 0.01) * amplitude
                tilt = prev_tilt + self._rng.uniform(-0.008, 0.008) * amplitude
            else:
                pan = self._rng.uniform(-amplitude, amplitude)
                tilt = self._rng.uniform(-amplitude * 0.5, amplitude * 0.5)
            
            pan = np.clip(pan, -amplitude, amplitude)
            tilt = np.clip(tilt, -amplitude * 0.5, amplitude * 0.5)
            
            prev_pan = pan * 0.95 + 0.05 * pan
            prev_tilt = tilt * 0.95 + 0.05 * tilt
            
            params.append(CameraParams(
                timestamp=i / fps,
                pan=pan,
                tilt=tilt,
                zoom=1.0 + self._rng.uniform(-0.005, 0.005)
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_freehand(
        self,
        num_frames: int,
        resolution: Tuple[int, int],
        fps: int
    ) -> CameraTrajectory:
        """Generate freehand natural motion."""
        params = []
        amplitude = 0.08 * self.config.strength
        
        pan_noise = self._generate_1d_noise(num_frames, amplitude * 0.5)
        tilt_noise = self._generate_1d_noise(num_frames, amplitude * 0.3)
        zoom_noise = self._generate_1d_noise(num_frames, 0.02)
        
        for i in range(num_frames):
            params.append(CameraParams(
                timestamp=i / fps,
                pan=pan_noise[i],
                tilt=tilt_noise[i],
                zoom=1.0 + zoom_noise[i]
            ))
        
        return CameraTrajectory(
            params=params,
            duration=num_frames / fps,
            fps=fps,
            resolution=resolution
        )
    
    def _generate_1d_noise(self, length: int, amplitude: float) -> np.ndarray:
        """Generate smooth 1D noise using filtered random walk."""
        noise = np.cumsum(self._rng.randn(length))
        noise = noise - noise[0]
        
        kernel_size = max(3, length // 8)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(noise, kernel, mode='same')
        
        smoothed = smoothed / (np.abs(smoothed).max() + 1e-8) * amplitude
        
        return smoothed
    
    def _ease_in_out(self, t: float) -> float:
        """Ease-in-out cubic interpolation."""
        return t * t * (3 - 2 * t)
    
    def apply_to_frames(
        self,
        frames: torch.Tensor,
        trajectory: CameraTrajectory
    ) -> torch.Tensor:
        """Apply camera motion to video frames.
        
        Args:
            frames: Video tensor [T, C, H, W]
            trajectory: Camera trajectory
            
        Returns:
            Transformed video tensor
        """
        T, C, H, W = frames.shape
        result = []
        
        for i, frame in enumerate(frames):
            pan, tilt, zoom = trajectory.get_frame_transform(i, T)
            
            transformed = self._apply_transform(
                frame, pan, tilt, zoom, H, W
            )
            result.append(transformed)
        
        return torch.stack(result, dim=0)
    
    def _apply_transform(
        self,
        frame: torch.Tensor,
        pan: float,
        tilt: float,
        zoom: float,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Apply camera transform to single frame."""
        dx = int(pan * width)
        dy = int(tilt * height)
        
        if zoom != 1.0:
            scaled_h = int(height * zoom)
            scaled_w = int(width * zoom)
            
            scaled = F.interpolate(
                frame.unsqueeze(0),
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ).squeeze(0)
            
            start_y = max(0, (scaled_h - height) // 2 - dy)
            start_x = max(0, (scaled_w - width) // 2 - dx)
            
            end_y = start_y + height
            end_x = start_x + width
            
            if end_y <= scaled_h and end_x <= scaled_w:
                frame = scaled[:, start_y:end_y, start_x:end_x]
            else:
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
        else:
            pass
        
        return frame
    
    def create_loop_compatible(
        self,
        trajectory: CameraTrajectory,
        blend_frames: int = 4
    ) -> CameraTrajectory:
        """Modify trajectory to be loop-compatible."""
        T = len(trajectory.params)
        
        if T < blend_frames * 2:
            return trajectory
        
        blended_params = trajectory.params.copy()
        
        for i in range(blend_frames):
            alpha = (i + 1) / (blend_frames + 1)
            start_p = trajectory.params[i]
            end_p = trajectory.params[T - 1 - i]
            
            blended_params[i] = CameraParams(
                timestamp=start_p.timestamp,
                pan=start_p.pan * (1 - alpha) + end_p.pan * alpha,
                tilt=start_p.tilt * (1 - alpha) + end_p.tilt * alpha,
                zoom=start_p.zoom * (1 - alpha) + end_p.zoom * alpha,
                roll=start_p.roll * (1 - alpha) + end_p.roll * alpha,
                dolly=start_p.dolly * (1 - alpha) + end_p.dolly * alpha,
            )
            
            blended_params[T - 1 - i] = blended_params[i]
        
        trajectory.params = blended_params
        return trajectory