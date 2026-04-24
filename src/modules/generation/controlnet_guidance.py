"""ControlNet guidance for structural consistency.

Uses ControlNet conditioning to guide video generation, reducing:
- Face distortion (pose/landmark guidance)
- Structural inconsistency (edge preservation)
- Warping (depth/segmentation guidance)
- Flickering (temporal consistency)

ControlNet Types:
- Depth (MiDaS, ZoeDepth)
- Canny Edge
- Pose/Keypoints
- Normal Map
- Segmentation
- Scribble
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlNetType(Enum):
    """Types of ControlNet conditioning."""
    DEPTH = "depth"
    CANNY_EDGE = "canny_edge"
    POSE = "pose"
    NORMAL = "normal"
    SEGMENTATION = "segmentation"
    SCRIBBLE = "scribble"
    SOFT_EDGE = "soft_edge"
    LINEART = "lineart"


@dataclass
class ControlNetConfig:
    """Configuration for ControlNet guidance."""
    controlnet_type: ControlNetType = ControlNetType.DEPTH
    strength: float = 0.8
    guidance_scale: float = 1.0
    start_step: int = 0
    end_step: int = 100
    
    preprocessor: str = "none"
    threshold_low: float = 100
    threshold_high: float = 200
    
    scale_factor: float = 1.0
    use_attention: bool = True
    attention_strength: float = 0.5
    
    temporal_consistency: float = 0.7
    preserve_structure: bool = True


class ControlNetGuidance:
    """ControlNet-based guidance for artifact reduction.
    
    Uses ControlNet conditioning to maintain structural consistency:
    1. Depth guidance: Preserves 3D geometry
    2. Edge guidance: Maintains structure boundaries
    3. Pose guidance: Preserves character pose
    4. Normal guidance: Maintains surface orientation
    
    Tradeoffs:
    - Quality vs Speed: Full ControlNet is high quality but slow
    - SoftEdge: Faster, less precise
    - Depth: Good balance of quality and speed
    
    Args:
        config: ControlNet configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[ControlNetConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or ControlNetConfig()
        self.device = device or torch.device("cpu")
        
        self._models: Dict[ControlNetType, nn.Module] = {}
        self._initialized = False
        self._cache: Dict[str, torch.Tensor] = {}
    
    def initialize(self) -> None:
        """Initialize ControlNet models."""
        if self._initialized:
            return
        
        if self.config.controlnet_type == ControlNetType.DEPTH:
            self._init_depth_controlnet()
        elif self.config.controlnet_type == ControlNetType.CANNY_EDGE:
            self._init_canny_controlnet()
        elif self.config.controlnet_type == ControlNetType.POSE:
            self._init_pose_controlnet()
        elif self.config.controlnet_type == ControlNetType.NORMAL:
            self._init_normal_controlnet()
        elif self.config.controlnet_type == ControlNetType.SEGMENTATION:
            self._init_segmentation_controlnet()
        
        self._initialized = True
    
    def _init_depth_controlnet(self) -> None:
        """Initialize depth ControlNet."""
        self._models[ControlNetType.DEPTH] = self._create_controlnet_wrapper(
            input_channels=3,
            output_channels=1
        )
    
    def _init_canny_controlnet(self) -> None:
        """Initialize canny edge ControlNet."""
        self._models[ControlNetType.CANNY_EDGE] = self._create_controlnet_wrapper(
            input_channels=3,
            output_channels=1
        )
    
    def _init_pose_controlnet(self) -> None:
        """Initialize pose ControlNet."""
        self._models[ControlNetType.POSE] = self._create_controlnet_wrapper(
            input_channels=3,
            output_channels=17 * 3
        )
    
    def _init_normal_controlnet(self) -> None:
        """Initialize normal map ControlNet."""
        self._models[ControlNetType.NORMAL] = self._create_controlnet_wrapper(
            input_channels=3,
            output_channels=3
        )
    
    def _init_segmentation_controlnet(self) -> None:
        """Initialize segmentation ControlNet."""
        self._models[ControlNetType.SEGMENTATION] = self._create_controlnet_wrapper(
            input_channels=3,
            output_channels=1
        )
    
    def _create_controlnet_wrapper(self, input_channels: int, output_channels: int) -> nn.Module:
        """Create ControlNet wrapper model."""
        class ControlNetWrapper(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.control_proj = nn.Conv2d(out_ch, 256, 1)
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, out_ch, 1),
                )
            
            def forward(self, x, control_hint=None):
                features = self.encoder(x)
                if control_hint is not None:
                    control = self.control_proj(control_hint)
                    features = features + control
                return self.decoder(features)
        
        return ControlNetWrapper(input_channels, output_channels)
    
    def preprocess(
        self,
        image: torch.Tensor,
        control_type: Optional[ControlNetType] = None
    ) -> torch.Tensor:
        """Preprocess image to create control condition.
        
        Args:
            image: Input image [C, H, W] or [B, C, H, W]
            control_type: Type of control to generate
            
        Returns:
            Control condition tensor
        """
        control_type = control_type or self.config.controlnet_type
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        if control_type == ControlNetType.DEPTH:
            return self._preprocess_depth(image)
        elif control_type == ControlNetType.CANNY_EDGE:
            return self._preprocess_canny(image)
        elif control_type == ControlNetType.POSE:
            return self._preprocess_pose(image)
        elif control_type == ControlNetType.NORMAL:
            return self._preprocess_normal(image)
        elif control_type == ControlNetType.SEGMENTATION:
            return self._preprocess_segmentation(image)
        else:
            return self._preprocess_soft_edge(image)
    
    def _preprocess_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Generate depth map as control."""
        from ..depth.depth_estimator import DepthEstimator
        
        depth_estimator = DepthEstimator(device=self.device)
        depth_estimator.initialize()
        
        depth = depth_estimator.estimate(image.squeeze(0))
        
        if hasattr(depth, 'depth'):
            depth_tensor = depth.depth
        else:
            depth_tensor = depth
        
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        if image.shape[-2:] != depth_tensor.shape[-2:]:
            depth_tensor = F.interpolate(
                depth_tensor.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        
        if depth_tensor.max() > 1:
            depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
        
        return depth_tensor.unsqueeze(0)
    
    def _preprocess_canny(self, image: torch.Tensor) -> torch.Tensor:
        """Generate canny edge map as control."""
        if image.dim() == 4:
            image = image.squeeze(0)
        
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image
        
        edges_x = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        edges_y = F.conv2d(
            gray.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        
        magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        
        threshold_low = self.config.threshold_low / 255.0
        threshold_high = self.config.threshold_high / 255.0
        
        edges = torch.zeros_like(magnitude)
        edges[magnitude > threshold_high] = 1.0
        edges[(magnitude > threshold_low) & (magnitude <= threshold_high)] = 0.5
        
        edges = F.interpolate(
            edges,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        return edges
    
    def _preprocess_pose(self, image: torch.Tensor) -> torch.Tensor:
        """Generate pose keypoints as control."""
        try:
            from controlnet_aux import OpenposeDetector
            
            detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet",
                device=self.device
            )
            
            if image.dim() == 4:
                image = image.squeeze(0)
            
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            pose_map = detector(image_np)
            
            pose_tensor = torch.from_numpy(pose_map).float().to(self.device)
            if pose_tensor.dim() == 3:
                pose_tensor = pose_tensor.permute(2, 0, 1)
            
            pose_tensor = F.interpolate(
                pose_tensor.unsqueeze(0),
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            
            return pose_tensor.unsqueeze(0)
            
        except ImportError:
            return torch.zeros(1, 17 * 3, *image.shape[-2:], device=self.device)
    
    def _preprocess_normal(self, image: torch.Tensor) -> torch.Tensor:
        """Generate normal map as control."""
        if image.dim() == 4:
            image = image.squeeze(0)
        
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device)
        
        dx = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x.view(1, 1, 3, 3), padding=1)
        dy = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y.view(1, 1, 3, 3), padding=1)
        
        normal_x = -dx
        normal_y = -dy
        normal_z = torch.ones_like(dx)
        
        normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
        normals = F.normalize(normals, p=2, dim=1)
        
        normals = (normals + 1) / 2
        
        normals = F.interpolate(
            normals,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        return normals
    
    def _preprocess_segmentation(self, image: torch.Tensor) -> torch.Tensor:
        """Generate segmentation map as control."""
        from ..segmentation.segmentor import Segmentor
        
        segmentor = Segmentor(device=self.device)
        segmentor.initialize()
        
        if image.dim() == 4:
            image = image.squeeze(0)
        
        seg_result = segmentor.segment_with_prompts(image)
        
        combined = seg_result.combined_mask
        
        if isinstance(combined, torch.Tensor):
            combined_tensor = combined.float()
        else:
            combined_tensor = torch.from_numpy(combined).float()
        
        if combined_tensor.dim() == 2:
            combined_tensor = combined_tensor.unsqueeze(0)
        
        combined_tensor = F.interpolate(
            combined_tensor.unsqueeze(0),
            size=image.shape[-2:],
            mode="nearest"
        ).squeeze(0)
        
        return combined_tensor.unsqueeze(0)
    
    def _preprocess_soft_edge(self, image: torch.Tensor) -> torch.Tensor:
        """Generate soft edge map as control."""
        if image.dim() == 4:
            image = image.squeeze(0)
        
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image
        
        blurred = F.avg_pool2d(
            gray.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze()
        
        grad_x = F.conv2d(
            blurred.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        grad_y = F.conv2d(
            blurred.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32, device=self.device),
            padding=1
        )
        
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        edges = F.sigmoid(edges * 5)
        
        edges = F.interpolate(
            edges,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        return edges
    
    def apply_guidance(
        self,
        latents: torch.Tensor,
        control: torch.Tensor,
        step: int,
        total_steps: int
    ) -> torch.Tensor:
        """Apply ControlNet guidance to latents.
        
        Args:
            latents: Current latent state
            control: Control conditioning
            step: Current diffusion step
            total_steps: Total diffusion steps
            
        Returns:
            Guided latents
        """
        if not self._initialized:
            self.initialize()
        
        progress = step / total_steps
        
        if progress < self.config.start_step / 100:
            return latents
        
        if progress > self.config.end_step / 100:
            return latents
        
        strength = self.config.strength * self.config.guidance_scale
        
        step_factor = 1.0 - abs(progress - 0.5) * 2
        strength = strength * (0.5 + 0.5 * step_factor)
        
        control_resized = F.interpolate(
            control,
            size=latents.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        if control_resized.shape[1] == 1:
            control_resized = control_resized.expand(-1, latents.shape[1], -1, -1)
        
        guided = latents + control_resized * strength
        
        if self.config.use_attention:
            guided = self._apply_attention_guidance(guided, control_resized)
        
        return guided
    
    def _apply_attention_guidance(
        self,
        latents: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention-based guidance."""
        if control.shape[1] >= 3:
            control_norm = F.normalize(control[:, :3], dim=1)
        else:
            control_norm = control.repeat(1, 3, 1, 1) / (control.max() + 1e-8)
        
        B, C, H, W = latents.shape
        
        latents_flat = latents.view(B, C, H * W)
        control_flat = control_norm.view(B, 3, H * W)
        
        similarity = torch.bmm(control_flat.transpose(1, 2), latents_flat)
        
        attention = F.softmax(similarity, dim=1)
        
        attended = torch.bmm(control_flat, attention)
        attended = attended.view(B, C, H, W)
        
        strength = self.config.attention_strength
        guided = (1 - strength) * latents + strength * attended
        
        return guided
    
    def temporal_smooth(
        self,
        controls: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply temporal smoothing to control sequence.
        
        Reduces flickering in ControlNet guidance.
        
        Args:
            controls: List of control tensors
            
        Returns:
            Smoothed control sequence
        """
        if len(controls) <= 1:
            return controls
        
        smoothed = [controls[0]]
        
        alpha = self.config.temporal_consistency
        
        for i in range(1, len(controls)):
            curr = controls[i]
            prev = smoothed[i - 1]
            
            if curr.shape != prev.shape:
                curr = F.interpolate(
                    curr.unsqueeze(0),
                    size=prev.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
            
            new_control = alpha * prev + (1 - alpha) * curr
            smoothed.append(new_control)
        
        return smoothed
    
    def create_control_sequence(
        self,
        image: torch.Tensor,
        num_frames: int,
        control_type: Optional[ControlNetType] = None
    ) -> List[torch.Tensor]:
        """Create control sequence for video generation.
        
        Args:
            image: Source image
            num_frames: Number of frames
            control_type: Control type override
            
        Returns:
            List of control tensors
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        control_type = control_type or self.config.controlnet_type
        
        base_control = self.preprocess(image, control_type)
        
        if image.shape[0] > 1:
            base_control = base_control.repeat(image.shape[0], 1, 1, 1)
        
        controls = []
        for i in range(num_frames):
            t = i / num_frames
            alpha = 0.1 * np.sin(t * 2 * np.pi) * 0.1 * self.config.scale_factor
            
            shifted = base_control + torch.randn_like(base_control) * alpha
            shifted = shifted / (shifted.max() + 1e-8)
            
            controls.append(shifted)
        
        controls = self.temporal_smooth(controls)
        
        return controls


class ControlNetFusion:
    """Fuses multiple ControlNet conditions for robust guidance."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.guidances: List[ControlNetGuidance] = []
        self.fusion_weights: List[float] = []
    
    def add_guidance(
        self,
        guidance: ControlNetGuidance,
        weight: float = 1.0
    ) -> None:
        """Add a ControlNet guidance to fuse."""
        self.guidances.append(guidance)
        self.fusion_weights.append(weight)
    
    def fuse(
        self,
        latents: torch.Tensor,
        controls: List[List[torch.Tensor]],
        step: int,
        total_steps: int
    ) -> torch.Tensor:
        """Fuse multiple guidance signals.
        
        Args:
            latents: Current latents
            controls: List of control sequences per guidance
            step: Current step
            total_steps: Total steps
            
        Returns:
            Fused latents
        """
        if not self.guidances:
            return latents
        
        total_weight = sum(self.fusion_weights)
        
        fused = None
        
        for guidance, controls_i, weight in zip(
            self.guidances, controls, self.fusion_weights
        ):
            if not controls_i:
                continue
            
            control = controls_i[min(step, len(controls_i) - 1)]
            
            guided = guidance.apply_guidance(latents, control, step, total_steps)
            
            if fused is None:
                fused = guided * (weight / total_weight)
            else:
                fused = fused + guided * (weight / total_weight)
        
        return fused if fused is not None else latents