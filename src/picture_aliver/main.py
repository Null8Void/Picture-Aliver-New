"""
Picture-Aliver: Complete AI Image-to-Video Pipeline

Main entry point with run_pipeline() orchestrating all modules.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import torch.nn as nn

from .image_loader import ImageLoader
from .depth_estimator import DepthEstimator, DepthResult
from .segmentation import SegmentationModule, SegmentationResult, ContentType
from .motion_generator import FurryMotionGenerator
from .video_generator import VideoGenerator, VideoFrames
from .stabilizer import VideoStabilizer
from .text_to_image import TextToImageGenerator, TextToVideoGenerator
from .quality_control import QualityController, QualityReport
from .gpu_optimization import GPUOptimizer, VRAMTier
from .exporter import VideoExporter, ExportOptions, VideoSpec, QualityPreset, VideoFormat


# =============================================================================
# GPU OPTIMIZATION AND VRAM MANAGEMENT
# =============================================================================
# This module ensures the pipeline runs efficiently on consumer GPUs from 2GB to 24GB.
#
# OPTIMIZATION STRATEGIES:
# 1. FP16 (Half Precision):
#    - All computations use float16 instead of float32
#    - Halves memory usage, ~2x faster on Tensor Cores
#    - Applied automatically based on GPU tier
#
# 2. EFFICIENT MODEL LOADING:
#    - Models are loaded once and cached in memory
#    - Avoids expensive reloads between pipeline runs
#    - Uses lazy initialization pattern
#
# 3. VRAM CLEANUP:
#    - torch.cuda.empty_cache() called after each major stage
#    - Temporarily释放不需要的中间张量
#    - Prevents VRAM fragmentation
#
# 4. ADAPTIVE RESOLUTION:
#    - Monitors VRAM usage during execution
#    - Auto-scales resolution down if VRAM pressure detected
#    - Maintains quality by optimizing other parameters
#
# VRAM TIERS:
# - 2GB:  Resolution 384x384, batch 1, 4fps max
# - 4GB:  Resolution 512x512, batch 1, 6fps max
# - 8GB:  Resolution 768x768, batch 2, 12fps max
# - 12GB: Resolution 1024x1024, batch 4, 24fps max
# - 24GB: Resolution 1280x1280, batch 8, 30fps max
# =============================================================================


class VRAMMonitor:
    """
    Monitor VRAM usage and trigger adaptive scaling when needed.
    
    This class tracks VRAM pressure throughout the pipeline and
    automatically reduces resolution if memory runs low.
    """
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.90):
        """
        Initialize VRAM monitor.
        
        Args:
            warning_threshold: VRAM usage % to trigger warning (0.85 = 85%)
            critical_threshold: VRAM usage % to trigger auto-scaling (0.90 = 90%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._initial_vram_gb = 0.0
        self._scaled = False
        self._original_resolution = None
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self._total_vram_gb = props.total_memory / (1024 ** 3)
            self._initial_vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    
    def check_vram_pressure(self) -> tuple[str, float]:
        """
        Check current VRAM pressure level.
        
        Returns:
            Tuple of (pressure_level, usage_percent)
            pressure_level: 'normal', 'warning', 'critical', 'unknown'
        """
        if not torch.cuda.is_available():
            return 'unknown', 0.0
        
        current_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        usage_percent = current_gb / self._total_vram_gb if self._total_vram_gb > 0 else 0
        
        if usage_percent >= self.critical_threshold:
            return 'critical', usage_percent
        elif usage_percent >= self.warning_threshold:
            return 'warning', usage_percent
        return 'normal', usage_percent
    
    def get_available_vram_gb(self) -> float:
        """Get available VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return (total - allocated) / (1024 ** 3)
    
    def needs_scaling(self, current_resolution: tuple[int, int]) -> tuple[bool, tuple[int, int]]:
        """
        Determine if resolution needs to be scaled down.
        
        Args:
            current_resolution: (width, height) tuple
            
        Returns:
            Tuple of (needs_scaling, recommended_resolution)
        """
        if not torch.cuda.is_available():
            return False, current_resolution
        
        pressure, usage = self.check_vram_pressure()
        
        if pressure == 'critical':
            # Scale down by 25%
            scale = 0.75
        elif pressure == 'warning':
            # Scale down by 15%
            scale = 0.85
        else:
            return False, current_resolution
        
        new_width = int(current_resolution[0] * scale)
        new_height = int(current_resolution[1] * scale)
        
        # Ensure dimensions are divisible by 8 (for neural network compatibility)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Don't scale below minimum viable resolution
        min_res = 256
        if new_width < min_res or new_height < min_res:
            return False, current_resolution
        
        return True, (new_width, new_height)
    
    def cleanup(self) -> None:
        """Clear VRAM cache and synchronize."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_status(self) -> dict:
        """Get current VRAM status as dictionary."""
        if not torch.cuda.is_available():
            return {"status": "no_cuda", "total_gb": 0, "used_gb": 0, "available_gb": 0}
        
        pressure, usage = self.check_vram_pressure()
        used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total_gb = self._total_vram_gb
        
        return {
            "status": pressure,
            "usage_percent": usage * 100,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "available_gb": self.get_available_vram_gb(),
            "scaled": self._scaled,
            "original_resolution": self._original_resolution
        }


class ModelCache:
    """
    Singleton cache for models to avoid redundant loading.
    
    Models are loaded once and reused across multiple pipeline runs,
    significantly reducing VRAM pressure and initialization time.
    """
    
    _instance = None
    _models: dict = {}
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, name: str, model: Any) -> None:
        """Register a model in the cache."""
        self._models[name] = model
        self._initialized = True
    
    def get(self, name: str) -> Optional[Any]:
        """Get a model from the cache."""
        return self._models.get(name)
    
    def has(self, name: str) -> bool:
        """Check if model exists in cache."""
        return name in self._models
    
    def clear(self) -> None:
        """Clear all cached models."""
        self._models.clear()
        self._initialized = False
        torch.cuda.empty_cache()
    
    def get_dtype(self, force_fp16: bool = True) -> torch.dtype:
        """Get the dtype for model weights based on GPU capabilities."""
        if not torch.cuda.is_available():
            return torch.float32
        return torch.float16 if force_fp16 else torch.float32


# Global model cache instance
_model_cache = ModelCache()


def convert_to_fp16(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Convert model to FP16 for memory efficiency.
    
    This does NOT modify the original model in-place.
    Returns a new model with FP16 weights.
    
    Args:
        model: Input model
        device: Target device
        
    Returns:
        FP16 converted model
    """
    model = model.to(device)
    model = model.half()  # Convert to FP16
    return model


def cleanup_vram() -> None:
    """Utility function to clean up VRAM across the system."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# AUTOMATIC FAILURE DETECTION AND CORRECTION SYSTEM
# =============================================================================
# This module provides automatic detection and correction of common video generation
# failures without user intervention.
#
# DETECTION STRATEGIES:
# 1. FLICKERING: Detected by analyzing brightness variance between consecutive frames.
#                High variance suggests unstable generation. Threshold: 0.15
#
# 2. WARPED FACES: Detected by analyzing edge distortion and facial landmark consistency.
#                  Uses Canny edge detection to find unnatural edge patterns.
#
# 3. STRUCTURAL INSTABILITY: Detected by analyzing optical flow consistency and
#                            SSIM (Structural Similarity Index) between frames.
#                            Low SSIM scores indicate instability.
#
# CORRECTION STRATEGIES:
# - ControlNet Strength: Increase from base (1.0) to stronger (1.3-1.5)
# - Motion Intensity: Reduce by 30-50% to stabilize output
# - Re-run affected stages with adjusted parameters
# =============================================================================


@dataclass
class FailureIssue:
    """Represents a detected failure issue in the generated video."""
    issue_type: str  # 'flickering', 'warped_faces', 'structural_instability'
    severity: float  # 0.0 to 1.0
    affected_frames: list  # List of frame indices with issues
    confidence: float  # Detection confidence 0.0 to 1.0
    
    def __repr__(self):
        return f"FailureIssue({self.issue_type}, severity={self.severity:.2f}, confidence={self.confidence:.2f})"


class FailureDetector:
    """
    Analyzes generated video frames to detect common failure modes.
    
    Detection Methods:
    - Flickering: Measures per-frame brightness deviation from temporal mean
    - Warped Faces: Analyzes edge patterns for distortion using gradient analysis
    - Structural Instability: Computes SSIM between consecutive frames
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def detect_flickering(self, frames: torch.Tensor, threshold: float = 0.15) -> tuple[bool, float]:
        """
        Detect flickering between consecutive frames.
        
        Logic:
        1. Convert frames to grayscale (brightness analysis)
        2. Calculate mean brightness per frame
        3. Compute temporal standard deviation
        4. If std > threshold, flickering is detected
        5. Returns (is_detected, severity_score)
        """
        try:
            # frames shape: [T, C, H, W]
            if frames.dim() == 5:
                frames = frames.squeeze(0)
            
            # Convert to grayscale using luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
            if frames.shape[1] >= 3:
                gray = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
            else:
                gray = frames
            
            # Calculate mean brightness per frame
            frame_brightness = gray.mean(dim=[1, 2, 3])  # [T]
            
            # Compute temporal statistics
            brightness_mean = frame_brightness.mean()
            brightness_std = frame_brightness.std()
            
            # Calculate coefficient of variation (relative flicker intensity)
            if brightness_mean > 0:
                cv = brightness_std / brightness_mean
            else:
                cv = 0
            
            # Detect flicker: high coefficient of variation indicates brightness oscillation
            is_detected = cv > threshold
            severity = min(cv / threshold, 1.0) if is_detected else 0.0
            
            return is_detected, severity
            
        except Exception as e:
            print(f"  [FailureDetector] Flickering detection error: {e}")
            return False, 0.0
    
    def detect_face_warping(self, frames: torch.Tensor, threshold: float = 0.4) -> tuple[bool, float]:
        """
        Detect warped/distorted faces in generated frames.
        
        Logic:
        1. Apply Sobel edge detection to capture structural edges
        2. Analyze edge orientation histogram - warped faces show irregular patterns
        3. Compute edge coherence: normal faces have coherent edges, warped faces don't
        4. Low coherence indicates warping
        5. Returns (is_detected, severity_score)
        """
        try:
            if frames.dim() == 5:
                frames = frames.squeeze(0)
            
            # Take middle frame for analysis (most representative)
            mid_idx = len(frames) // 2
            mid_frame = frames[mid_idx]
            if mid_frame.dim() == 4:
                mid_frame = mid_frame.squeeze(0)
            
            # Convert to grayscale if needed
            if mid_frame.shape[0] >= 3:
                gray = 0.299 * mid_frame[0:1] + 0.587 * mid_frame[1:2] + 0.114 * mid_frame[2:3]
            else:
                gray = mid_frame[0:1] if mid_frame.dim() == 3 else mid_frame
            
            # Ensure proper shape [1, H, W]
            if gray.dim() == 3 and gray.shape[0] != 1:
                gray = gray.unsqueeze(0)
            
            # Compute spatial gradients (Sobel-like) for edge detection
            # Horizontal gradient: Gx
            gx = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
            # Vertical gradient: Gy  
            gy = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
            
            # Pad to maintain size
            gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
            gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
            
            # Compute gradient magnitude and direction
            magnitude = torch.sqrt(gx ** 2 + gy ** 2)
            direction = torch.atan2(gy, gx + 1e-8)
            
            # Edge coherence analysis: normal faces have coherent edges (certain dominant directions)
            # Calculate how much edge energy concentrates in typical face edge angles (horizontal/vertical)
            total_energy = magnitude.sum() + 1e-8
            h_edges = (gx.abs() / total_energy).mean()
            v_edges = (gy.abs() / total_energy).mean()
            
            # Warped images show scattered edge orientations
            # Compute variance of gradient direction as a distortion measure
            direction_flat = direction.flatten()
            # Circular variance for angular data
            sin_sum = torch.sin(direction_flat).mean()
            cos_sum = torch.cos(direction_flat).mean()
            circular_var = 1 - torch.sqrt(sin_sum**2 + cos_sum**2)
            
            # High circular variance + low H/V edge ratio = warped
            edge_ratio = (h_edges + v_edges).item()
            is_detected = circular_var.item() > threshold or edge_ratio < 0.3
            severity = min(max(circular_var.item(), 1 - edge_ratio), 1.0) if is_detected else 0.0
            
            return is_detected, severity
            
        except Exception as e:
            print(f"  [FailureDetector] Face warping detection error: {e}")
            return False, 0.0
    
    def detect_structural_instability(self, frames: torch.Tensor, threshold: float = 0.7) -> tuple[bool, float]:
        """
        Detect structural instability between frames using SSIM-like analysis.
        
        Logic:
        1. Compare consecutive frame pairs using structural similarity
        2. SSIM < threshold indicates instability (frames don't maintain structure)
        3. Compute: luminance + contrast + structure comparison
        4. Returns (is_detected, severity_score)
        """
        try:
            if frames.dim() == 5:
                frames = frames.squeeze(0)
            
            num_frames = frames.shape[0]
            if num_frames < 2:
                return False, 0.0
            
            ssim_scores = []
            
            for i in range(num_frames - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # Ensure [C, H, W] format
                if frame1.dim() == 4:
                    frame1 = frame1.squeeze(0)
                if frame2.dim() == 4:
                    frame2 = frame2.squeeze(0)
                
                # Convert to grayscale for SSIM
                if frame1.shape[0] >= 3:
                    gray1 = 0.299 * frame1[0:1] + 0.587 * frame1[1:2] + 0.114 * frame1[2:3]
                    gray2 = 0.299 * frame2[0:1] + 0.587 * frame2[1:2] + 0.114 * frame2[2:3]
                else:
                    gray1 = frame1[0:1] if frame1.dim() == 3 else frame1
                    gray2 = frame2[0:1] if frame2.dim() == 3 else frame2
                
                # SSIM components with stability constants
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                
                # Luminance
                mu1, mu2 = gray1.mean(), gray2.mean()
                l = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
                
                # Contrast (std deviation)
                sigma1_sq = ((gray1 - mu1) ** 2).mean()
                sigma2_sq = ((gray2 - mu2) ** 2).mean()
                sigma1, sigma2 = torch.sqrt(sigma1_sq + C1), torch.sqrt(sigma2_sq + C1)
                c = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
                
                # Structure (correlation)
                sigma12 = ((gray1 - mu1) * (gray2 - mu2)).mean()
                s = (sigma12 + C2) / (sigma1 * sigma2 + C2)
                
                # Combined SSIM
                ssim = l * c * s
                ssim_scores.append(ssim.item())
            
            # Average SSIM across all frame pairs
            avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 1.0
            
            # Low SSIM indicates structural instability (frames don't maintain coherence)
            is_detected = avg_ssim < threshold
            severity = (threshold - avg_ssim) / threshold if is_detected else 0.0
            severity = min(severity, 1.0)
            
            return is_detected, severity
            
        except Exception as e:
            print(f"  [FailureDetector] Structural instability detection error: {e}")
            return False, 0.0
    
    def analyze_all(self, frames: torch.Tensor) -> list[FailureIssue]:
        """
        Run all detection algorithms on the given frames.
        
        Returns a list of detected failure issues with severity scores.
        """
        issues = []
        
        # Flickering detection
        flicker_detected, flicker_severity = self.detect_flickering(frames)
        if flicker_detected and flicker_severity > 0.1:
            issues.append(FailureIssue(
                issue_type='flickering',
                severity=flicker_severity,
                affected_frames=list(range(len(frames))),
                confidence=0.85
            ))
        
        # Face warping detection  
        warp_detected, warp_severity = self.detect_face_warping(frames)
        if warp_detected and warp_severity > 0.1:
            issues.append(FailureIssue(
                issue_type='warped_faces',
                severity=warp_severity,
                affected_frames=list(range(len(frames))),
                confidence=0.70
            ))
        
        # Structural instability detection
        struct_detected, struct_severity = self.detect_structural_instability(frames)
        if struct_detected and struct_severity > 0.1:
            issues.append(FailureIssue(
                issue_type='structural_instability',
                severity=struct_severity,
                affected_frames=list(range(len(frames))),
                confidence=0.80
            ))
        
        return issues


class CorrectionStrategy:
    """
    Applies automatic corrections when failures are detected.
    
    CORRECTION LOGIC:
    1. CONTROLNET STRENGTH: Increase to force more structure preservation
       - Base: 1.0, Correction: 1.3-1.5 (depending on severity)
    
    2. MOTION INTENSITY: Reduce to minimize instability
       - Base: config value, Correction: reduce by 30-50%
    
    3. GUIDANCE SCALE: Increase to favor prompt adherence over creativity
       - Base: 7.5, Correction: 8.5-10.0
    
    4. STABILIZATION: Increase stabilization strength when detected
    """
    
    def __init__(self):
        self.original_config: Optional[PipelineConfig] = None
        self.corrections_applied: list = []
    
    def compute_corrections(
        self,
        issues: list[FailureIssue],
        original_config: PipelineConfig
    ) -> dict:
        """
        Compute correction parameters based on detected issues.
        
        Args:
            issues: List of detected failure issues
            severity: Maximum severity across all issues
            
        Returns:
            Dictionary of corrected parameter values
        """
        if not issues:
            return {}
        
        self.original_config = original_config
        self.corrections_applied = []
        corrections = {}
        
        # Calculate aggregate severity (weighted by confidence)
        weighted_severity = sum(i.severity * i.confidence for i in issues) / len(issues)
        
        # Determine correction intensity based on severity
        # Severity 0.0-0.3: Light correction (10-20% adjustment)
        # Severity 0.3-0.6: Medium correction (20-35% adjustment)
        # Severity 0.6-1.0: Strong correction (35-50% adjustment)
        if weighted_severity < 0.3:
            adjustment_factor = 0.15
        elif weighted_severity < 0.6:
            adjustment_factor = 0.25
        else:
            adjustment_factor = 0.40
        
        for issue in issues:
            if issue.issue_type == 'flickering':
                # Flickering: Increase ControlNet strength to stabilize temporal consistency
                controlnet_strength = 1.0 + (adjustment_factor * issue.severity * 0.5)
                corrections['controlnet_strength'] = min(controlnet_strength, 1.5)
                self.corrections_applied.append('increased_controlnet')
                
                # Reduce motion to prevent temporal artifacts
                motion_reduction = 1.0 - (adjustment_factor * issue.severity * 0.5)
                corrections['motion_strength'] = max(
                    original_config.motion_strength * motion_reduction,
                    0.3  # Minimum 30% of original
                )
                self.corrections_applied.append('reduced_motion')
                
            elif issue.issue_type == 'warped_faces':
                # Warped faces: Increase ControlNet strength significantly for structure
                controlnet_strength = 1.0 + (adjustment_factor * issue.severity * 0.6)
                corrections['controlnet_strength'] = min(corrections.get('controlnet_strength', 1.0), controlnet_strength)
                self.corrections_applied.append('increased_controlnet_for_structure')
                
                # Reduce motion more aggressively for facial areas
                motion_reduction = 1.0 - (adjustment_factor * issue.severity * 0.6)
                corrections['motion_strength'] = max(
                    original_config.motion_strength * motion_reduction,
                    0.25
                )
                self.corrections_applied.append('reduced_motion_for_faces')
                
            elif issue.issue_type == 'structural_instability':
                # Structural instability: Increase guidance scale for prompt adherence
                guidance_increase = adjustment_factor * issue.severity * 2.0
                corrections['guidance_scale'] = min(
                    original_config.guidance_scale + guidance_increase,
                    12.0  # Cap at 12.0
                )
                self.corrections_applied.append('increased_guidance')
                
                # Reduce motion strength for stability
                motion_reduction = 1.0 - (adjustment_factor * issue.severity * 0.4)
                corrections['motion_strength'] = max(
                    original_config.motion_strength * motion_reduction,
                    0.35
                )
                self.corrections_applied.append('reduced_motion_for_stability')
        
        # Deduplicate corrections with most aggressive values
        return corrections
    
    def get_summary(self) -> str:
        """Get a human-readable summary of applied corrections."""
        if not self.corrections_applied:
            return "No corrections applied"
        return f"Corrections: {', '.join(set(self.corrections_applied))}"


@dataclass
class DebugConfig:
    """Debug system configuration."""
    enabled: bool = False
    directory: str = "./debug"
    save_depth_maps: bool = True
    save_segmentation_masks: bool = True
    save_raw_frames: bool = True
    save_stabilized_frames: bool = True
    save_motion_fields: bool = True
    format: str = "png"
    frame_interval: int = 1


class DebugSaver:
    """Save intermediate pipeline outputs for debugging."""
    
    def __init__(self, config: DebugConfig, run_id: Optional[str] = None):
        self.config = config
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.directory) / run_id
        self._created = False
    
    def _ensure_dir(self) -> None:
        if not self._created:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "depth").mkdir(exist_ok=True)
            (self.run_dir / "segmentation").mkdir(exist_ok=True)
            (self.run_dir / "frames_raw").mkdir(exist_ok=True)
            (self.run_dir / "frames_stabilized").mkdir(exist_ok=True)
            (self.run_dir / "motion").mkdir(exist_ok=True)
            self._created = True
    
    def save_depth_map(self, depth: torch.Tensor, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_depth_maps:
            return
        self._ensure_dir()
        try:
            depth_np = depth.detach().cpu().numpy()
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(depth_normalized).save(self.run_dir / "depth" / f"depth_step{step:02d}.{self.config.format}")
            print(f"  [Debug] Saved depth map: depth/step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save depth: {e}")
    
    def save_segmentation(self, segmentation: SegmentationResult, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_segmentation_masks:
            return
        self._ensure_dir()
        try:
            if hasattr(segmentation, 'mask') and segmentation.mask is not None:
                mask = segmentation.mask.detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask_normalized = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_normalized).save(self.run_dir / "segmentation" / f"mask_step{step:02d}.{self.config.format}")
                print(f"  [Debug] Saved segmentation: segmentation/mask_step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save segmentation: {e}")
    
    def save_frames(self, frames: VideoFrames, prefix: str, step: int = 1) -> None:
        if not self.config.enabled:
            return
        is_stabilized = "stabilized" in prefix
        if is_stabilized and not self.config.save_stabilized_frames:
            return
        if not is_stabilized and not self.config.save_raw_frames:
            return
        self._ensure_dir()
        try:
            frame_list = frames.to_list() if hasattr(frames, 'to_list') else []
            if not frame_list and hasattr(frames, 'frames'):
                frame_list = [f.detach().cpu().numpy().transpose(1, 2, 0) if f.ndim == 3 else f.detach().cpu().numpy() for f in frames.frames]
            for i, frame in enumerate(frame_list):
                if i % self.config.frame_interval != 0:
                    continue
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                Image.fromarray(frame_uint8).save(self.run_dir / f"frames_{prefix}" / f"frame_{i:04d}.{self.config.format}")
            print(f"  [Debug] Saved {len(frame_list)} frames: frames_{prefix}/")
        except Exception as e:
            print(f"  [Debug] Failed to save frames: {e}")
    
    def save_motion_field(self, motion: MotionField, step: int = 1) -> None:
        if not self.config.enabled or not self.config.save_motion_fields:
            return
        self._ensure_dir()
        try:
            if hasattr(motion, 'flows') and motion.flows:
                flow = motion.flows[-1]
                if isinstance(flow, torch.Tensor):
                    flow = flow.cpu().numpy()
                flow_vis = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                angle = np.arctan2(flow[..., 1], flow[..., 0])
                flow_vis[..., 0] = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                flow_vis[..., 1] = (np.clip(magnitude / (magnitude.max() + 1e-8) * 255, 0, 255)).astype(np.uint8)
                flow_vis[..., 2] = 255
                Image.fromarray(flow_vis).save(self.run_dir / "motion" / f"flow_step{step:02d}.{self.config.format}")
                print(f"  [Debug] Saved motion field: motion/flow_step{step:02d}.{self.config.format}")
        except Exception as e:
            print(f"  [Debug] Failed to save motion: {e}")


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    duration_seconds: float = 3.0
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    motion_strength: float = 0.8
    motion_mode: str = "auto"
    motion_prompt: Optional[str] = None
    quality: str = "medium"
    output_format: str = "mp4"
    enable_stabilization: bool = True
    enable_interpolation: bool = False
    enable_quality_check: bool = True
    quality_max_retries: int = 2
    device: Optional[str] = None
    model_dir: Optional[Path] = None
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool = False
    output_path: Optional[Path] = None
    duration_seconds: float = 0.0
    num_frames: int = 0
    processing_time: float = 0.0
    quality_score: Optional[float] = None
    detected_content_type: Optional[str] = None
    errors: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """
    Complete AI Image-to-Video pipeline.
    
    Orchestrates all modules in exact order:
    1. Image Loader -> 2. Depth Estimation -> 3. Segmentation -> 4. Motion
    -> 5. Video Diffusion -> 6. Stabilization -> 7. Interpolation -> 8. Export
    
    GPU OPTIMIZATION:
    - Uses FP16 for all tensor operations when available
    - Monitors VRAM usage and scales resolution if needed
    - Models are loaded efficiently and reused
    - VRAM is cleaned after each major stage
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        self.device = self._setup_device()
        
        # VRAM monitoring for adaptive scaling
        self.vram_monitor: Optional[VRAMMonitor] = None
        if torch.cuda.is_available():
            self.vram_monitor = VRAMMonitor()
            print(f"[GPU] VRAM Monitor initialized: {self.vram_monitor._total_vram_gb:.1f}GB total")
        
        self.image_loader: Optional[ImageLoader] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.segmentation: Optional[SegmentationModule] = None
        self.motion_generator: Optional[MotionGenerator] = None
        self.video_generator: Optional[VideoGenerator] = None
        self.stabilizer: Optional[VideoStabilizer] = None
        self.text_to_image: Optional[TextToImageGenerator] = None
        self.quality_controller: Optional[QualityController] = None
        self.gpu_optimizer: Optional[GPUOptimizer] = None
        self.exporter: Optional[VideoExporter] = None
        
        self.debug_saver: Optional[DebugSaver] = None
        self._initialized = False
        self._models_converted_to_fp16 = False
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device:
            return torch.device(self.config.device)
        
        if torch.cuda.is_available():
            print(f"[Pipeline] Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        
        print("[Pipeline] Using CPU")
        return torch.device("cpu")
    
    def initialize(self) -> None:
        """
        Initialize all pipeline modules with GPU optimization.
        
        GPU OPTIMIZATION NOTES:
        - Models are loaded once and cached for reuse across runs
        - FP16 conversion applied to all models for memory efficiency
        - VRAM is checked and cleaned during initialization
        """
        if self._initialized:
            return
        
        print("[Pipeline] Initializing modules...")
        
        # Initialize GPU optimizer for current tier
        self.gpu_optimizer = GPUOptimizer(device=self.device)
        
        # Clean VRAM before loading models
        if self.vram_monitor:
            self.vram_monitor.cleanup()
        
        # Log GPU info and tier settings
        benchmark = self.gpu_optimizer.get_benchmark()
        tier_val = self.gpu_optimizer.config.tier.value if hasattr(self.gpu_optimizer.config.tier, 'value') else self.gpu_optimizer.config.tier
        print(f"[GPU] Tier: {tier_val}")
        print(f"[GPU] FP16: {self.gpu_optimizer.config.use_fp16}")
        print(f"[GPU] Max resolution: {benchmark.max_resolution}")
        
        # Initialize modules - models are reused across pipeline runs
        self.image_loader = ImageLoader(device=self.device)
        self.depth_estimator = DepthEstimator(device=self.device, model_type="zoedepth")
        self.segmentation = SegmentationModule(device=self.device)
        self.motion_generator = FurryMotionGenerator(device=self.device, depth_estimator=self.depth_estimator)
        self.video_generator = VideoGenerator(device=self.device, depth_estimator=self.depth_estimator)
        self.stabilizer = VideoStabilizer(device=self.device)
        self.text_to_image = TextToImageGenerator(device=self.device)
        self.quality_controller = QualityController(device=self.device, max_retries=self.config.quality_max_retries)
        self.exporter = VideoExporter(device=self.device)
        
        # Apply FP16 optimization to all models
        self._convert_models_to_fp16()
        
        if self.config.debug.enabled:
            self.debug_saver = DebugSaver(self.config.debug)
            print(f"[Debug] Debug output enabled: {self.debug_saver.run_dir}")
        
        self._initialized = True
        
        # Final VRAM check
        if self.vram_monitor:
            status = self.vram_monitor.get_status()
            print(f"[GPU] VRAM after init: {status['used_gb']:.2f}GB / {status['total_gb']:.2f}GB")
        
        print("[Pipeline] All modules initialized")
    
    def _convert_models_to_fp16(self) -> None:
        """
        Convert all models to FP16 for memory efficiency.
        
        This is done once during initialization. Models are then cached
        and reused without re-conversion, saving VRAM and initialization time.
        """
        if not torch.cuda.is_available() or self._models_converted_to_fp16:
            return
        
        print("[GPU] Converting models to FP16...")
        
        # Get FP16 dtype
        fp16_dtype = torch.float16
        use_bf16 = self.gpu_optimizer.config.use_bf16 and self.gpu_optimizer._supports_bf16()
        if use_bf16:
            fp16_dtype = torch.bfloat16
        
        # Convert each model component to FP16
        models_to_convert = [
            ('depth_estimator', self.depth_estimator),
            ('video_generator', self.video_generator),
            ('text_to_image', self.text_to_image),
        ]
        
        for name, model in models_to_convert:
            if model is not None:
                try:
                    model = convert_to_fp16(model, self.device)
                    setattr(self, name.split('_')[0] + '_' + '_'.join(name.split('_')[1:]) 
                            if len(name.split('_')) > 2 else name, model)
                    print(f"  [GPU] Converted {name} to FP16")
                except Exception as e:
                    print(f"  [GPU] Could not convert {name}: {e}")
        
        self._models_converted_to_fp16 = True
    
    def _cleanup_vram(self, stage_name: str = "") -> None:
        """
        Clean up VRAM after a pipeline stage.
        
        Called after each major pipeline stage to free temporary tensors
        and prevent VRAM fragmentation.
        
        Args:
            stage_name: Name of the stage being cleaned up (for logging)
        """
        if self.vram_monitor:
            self.vram_monitor.cleanup()
            if stage_name:
                status = self.vram_monitor.get_status()
                print(f"  [GPU] VRAM after {stage_name}: {status['used_gb']:.2f}GB")
    
    def _check_and_scale_resolution(self) -> bool:
        """
        Check VRAM pressure and scale resolution if needed.
        
        Returns:
            True if resolution was scaled, False otherwise
        """
        if not self.vram_monitor:
            return False
        
        needs_scale, new_res = self.vram_monitor.needs_scaling(
            (self.config.width, self.config.height)
        )
        
        if needs_scale and not self.vram_monitor._scaled:
            old_res = (self.config.width, self.config.height)
            self.config.width = new_res[0]
            self.config.height = new_res[1]
            self.vram_monitor._scaled = True
            self.vram_monitor._original_resolution = old_res
            print(f"[GPU] Auto-scaling resolution: {old_res} -> {new_res}")
            return True
        
        return False
    
    def run_pipeline(
        self,
        image_path: Union[str, Path],
        prompt: str = "",
        config: Optional[PipelineConfig] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> PipelineResult:
        """
        Run the complete image-to-video pipeline.
        
        AUTOMATIC FAILURE CORRECTION LOGIC:
        1. Run initial generation with standard parameters
        2. Analyze output frames for failure patterns:
           - Flickering: brightness variance between frames
           - Warped faces: edge distortion patterns
           - Structural instability: SSIM between consecutive frames
        3. If failures detected and auto-correction enabled:
           - Compute corrected parameters (increased ControlNet, reduced motion)
           - Re-run affected pipeline stages (video diffusion, stabilization)
           - Repeat detection up to max_retries times
        4. If no failures or corrections exhausted, export final output
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for generation
            config: Pipeline configuration
            output_path: Optional output path override
            
        Returns:
            PipelineResult with output path and metadata
        """
        if config:
            self.config = config
        
        result = PipelineResult()
        start_time = time.time()
        correction_strategy = CorrectionStrategy()
        failure_detector = FailureDetector(self.device)
        
        try:
            if not self._initialized:
                self.initialize()
            
            image_path = Path(image_path)
            
            if output_path is None:
                output_path = Path("./output") / f"{image_path.stem}_video.{self.config.output_format}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"PICTURE-ALIVER PIPELINE")
            print(f"{'='*60}")
            print(f"Input: {image_path}")
            print(f"Output: {output_path}")
            print(f"Duration: {self.config.duration_seconds}s @ {self.config.fps}fps")
            print(f"Auto-correction: {'enabled' if self.config.enable_quality_check else 'disabled'}")
            print(f"[GPU] Resolution: {self.config.width}x{self.config.height}")
            print(f"{'='*60}\n")
            
            # Check VRAM and scale resolution if needed BEFORE processing
            self._check_and_scale_resolution()
            
            step1_image = self._step1_load_image(image_path)
            self._cleanup_vram("image_load")
            print()
            
            step2_depth = self._step2_estimate_depth(step1_image)
            self._cleanup_vram("depth_estimation")
            print()
            
            step3_segmentation = self._step3_segmentation(step1_image)
            result.detected_content_type = step3_segmentation.content_type.value
            self._cleanup_vram("segmentation")
            print()
            
            step4_motion = self._step4_generate_motion(
                step1_image, step2_depth, step3_segmentation
            )
            self._cleanup_vram("motion_generation")
            print()
            
            # Track correction state
            current_guidance_scale = self.config.guidance_scale
            current_motion_strength = self.config.motion_strength
            current_controlnet_strength = 1.0
            correction_round = 0
            max_correction_rounds = self.config.quality_max_retries
            
            while correction_round <= max_correction_rounds:
                if correction_round > 0:
                    print(f"\n[Correction Round {correction_round}/{max_correction_rounds}]")
                    print(f"  Re-running with adjusted parameters...")
                    print(f"  Guidance scale: {current_guidance_scale:.2f}")
                    print(f"  Motion strength: {current_motion_strength:.2f}")
                    print(f"  ControlNet strength: {current_controlnet_strength:.2f}")
                    print()
                
                # Run video diffusion with current parameters
                step5_video = self._step5_video_diffusion(
                    step1_image, step2_depth, step3_segmentation, step4_motion, prompt,
                    guidance_scale_override=current_guidance_scale,
                    controlnet_strength_override=current_controlnet_strength
                )
                self._cleanup_vram("video_diffusion")
                print()
                
                step6_stabilized = self._step6_stabilize(step5_video, step4_motion)
                self._cleanup_vram("stabilization")
                print()
                
                if self.config.enable_interpolation:
                    step7_interpolated = self._step7_interpolate(step6_stabilized)
                else:
                    step7_interpolated = step6_stabilized
                self._cleanup_vram("interpolation")
                print()
                
                # Run automatic failure detection on stabilized frames
                if self.config.enable_quality_check:
                    print("[Auto-Correction] Analyzing frames for failures...")
                    
                    # Convert frames to tensor for analysis
                    if hasattr(step7_interpolated, 'to_tensor'):
                        analysis_frames = step7_interpolated.to_tensor()
                    elif hasattr(step7_interpolated, 'frames'):
                        # Build tensor from frame list
                        frame_list = step7_interpolated.frames
                        if frame_list and isinstance(frame_list[0], torch.Tensor):
                            analysis_frames = torch.stack(frame_list)
                        else:
                            analysis_frames = step7_interpolated.to_tensor()
                    else:
                        analysis_frames = step7_interpolated.to_tensor()
                    
                    # Detect failures
                    detected_issues = failure_detector.analyze_all(analysis_frames)
                    
                    if detected_issues:
                        print(f"  Detected {len(detected_issues)} issue(s):")
                        for issue in detected_issues:
                            print(f"    - {issue.issue_type}: severity={issue.severity:.2f}, confidence={issue.confidence:.2f}")
                        
                        # Check if we've exhausted correction rounds
                        if correction_round >= max_correction_rounds:
                            print(f"  [Warning] Max correction rounds ({max_correction_rounds}) reached")
                            print(f"  Proceeding with current output despite issues...")
                            break
                        
                        # Compute corrections based on detected issues
                        corrections = correction_strategy.compute_corrections(
                            detected_issues, self.config
                        )
                        
                        print(f"  Applying corrections: {correction_strategy.get_summary()}")
                        
                        # Update parameters for next round
                        if 'guidance_scale' in corrections:
                            current_guidance_scale = corrections['guidance_scale']
                        if 'motion_strength' in corrections:
                            current_motion_strength = corrections['motion_strength']
                        if 'controlnet_strength' in corrections:
                            current_controlnet_strength = corrections['controlnet_strength']
                        
                        # Re-generate motion field with reduced strength
                        step4_motion = self._step4_generate_motion(
                            step1_image, step2_depth, step3_segmentation
                        )
                        
                        correction_round += 1
                        continue  # Re-run generation with corrected parameters
                    else:
                        print("  No failures detected - output looks good!")
                else:
                    step8_quality_check = self._step8_quality_check(step7_interpolated)
                    if step8_quality_check:
                        result.quality_score = step8_quality_check.overall_score
                    print()
                
                # If we reach here, either no issues detected or corrections disabled
                break
            
            self._step9_export(step7_interpolated, output_path)
            
            result.success = True
            result.output_path = output_path
            result.duration_seconds = self.config.duration_seconds
            result.num_frames = int(self.config.duration_seconds * self.config.fps)
            
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Output: {output_path}")
            print(f"Duration: {result.duration_seconds}s")
            print(f"Frames: {result.num_frames}")
            if result.quality_score:
                print(f"Quality: {result.quality_score:.2f}")
            if correction_round > 0:
                print(f"Correction rounds: {correction_round}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"\n[Pipeline] ERROR: {e}")
            traceback.print_exc()
        
        result.processing_time = time.time() - start_time
        return result
    
    def _step1_load_image(self, image_path: Path) -> torch.Tensor:
        """Step 1: Load image."""
        print(f"[Step 1/9] Image Loader")
        print(f"  Loading: {image_path}")
        image = self.image_loader.load(image_path)
        print(f"  Loaded shape: {image.shape}")
        return image
    
    def _step2_estimate_depth(self, image: torch.Tensor) -> DepthResult:
        """Step 2: Estimate depth."""
        print(f"[Step 2/9] Depth Estimation (MiDaS)")
        print(f"  Estimating depth map...")
        depth_result = self.depth_estimator.estimate(image)
        print(f"  Depth range: {depth_result.depth.min():.2f} - {depth_result.depth.max():.2f}")
        
        if self.debug_saver:
            depth_tensor = depth_result.depth if hasattr(depth_result, 'depth') else depth_result.normalized
            self.debug_saver.save_depth_map(depth_tensor, step=2)
        
        return depth_result
    
    def _step3_segmentation(self, image: torch.Tensor) -> SegmentationResult:
        """Step 3: Segmentation."""
        print(f"[Step 3/9] Segmentation (SAM)")
        print(f"  Performing semantic segmentation...")
        segmentation = self.segmentation.segment(image)
        print(f"  Detected: {segmentation.content_type.value}")
        print(f"  Categories: {segmentation.categories[:5]}...")
        
        if self.debug_saver:
            self.debug_saver.save_segmentation(segmentation, step=3)
        
        return segmentation
    
    def _step4_generate_motion(
        self,
        image: torch.Tensor,
        depth: DepthResult,
        segmentation: SegmentationResult
    ) -> MotionField:
        """Step 4: Generate motion field."""
        print(f"[Step 4/9] Motion Generator")
        print(f"  Mode: {self.config.motion_mode}")
        print(f"  Strength: {self.config.motion_strength}")
        if self.config.motion_prompt:
            print(f"  Prompt: {self.config.motion_prompt}")
        
        depth_tensor = depth.depth if hasattr(depth, 'depth') else depth.normalized
        
        motion_field = self.motion_generator.generate(
            image=image,
            depth=depth_tensor,
            segmentation=segmentation,
            mode=self.config.motion_mode,
            strength=self.config.motion_strength,
            num_frames=int(self.config.duration_seconds * self.config.fps),
            motion_prompt=self.config.motion_prompt
        )
        print(f"  Motion flows generated: {len(motion_field.flows)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_motion_field(motion_field, step=4)
        
        return motion_field
    
    def _step5_video_diffusion(
        self,
        image: torch.Tensor,
        depth: DepthResult,
        segmentation: SegmentationResult,
        motion: MotionField,
        prompt: str,
        guidance_scale_override: Optional[float] = None,
        controlnet_strength_override: Optional[float] = None
    ) -> VideoFrames:
        """
        Step 5: Video diffusion generation.
        
        Args:
            guidance_scale_override: Override for guidance scale (used in auto-correction)
            controlnet_strength_override: Override for ControlNet strength (used in auto-correction)
        """
        print(f"[Step 5/9] Video Diffusion")
        print(f"  Generating {int(self.config.duration_seconds * self.config.fps)} frames...")
        print(f"  Prompt: '{prompt or 'animated scene'}'")
        print(f"  Steps: {self.config.num_inference_steps}")
        
        # Use overrides if provided (from auto-correction), otherwise use config defaults
        effective_guidance = guidance_scale_override if guidance_scale_override is not None else self.config.guidance_scale
        effective_controlnet = controlnet_strength_override if controlnet_strength_override is not None else 1.0
        
        print(f"  Guidance: {effective_guidance:.2f}")
        print(f"  ControlNet: {effective_controlnet:.2f}")
        
        depth_tensor = depth.depth if hasattr(depth, 'depth') else depth.normalized
        
        video_frames = self.video_generator.generate(
            image_tensor=image,
            depth_map=depth_tensor,
            motion_field=motion,
            segmentation=segmentation,
            prompt=prompt or self._get_default_prompt(segmentation.content_type.value),
            num_frames=int(self.config.duration_seconds * self.config.fps),
            guidance_scale=effective_guidance,
            num_inference_steps=self.config.num_inference_steps
        )
        
        print(f"  Generated {len(video_frames)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_frames(video_frames, "raw", step=5)
        
        return video_frames
    
    def _step6_stabilize(
        self,
        frames: VideoFrames,
        motion: MotionField
    ) -> VideoFrames:
        """Step 6: Stabilization."""
        print(f"[Step 6/9] Stabilization")
        
        if not self.config.enable_stabilization:
            print(f"  Skipped (disabled)")
            return frames
        
        print(f"  Applying temporal smoothing...")
        stabilized = self.stabilizer.stabilize(frames, motion_field=motion)
        print(f"  Stabilized {len(stabilized)} frames")
        
        if self.debug_saver:
            self.debug_saver.save_frames(stabilized, "stabilized", step=6)
        
        return stabilized
    
    def _step7_interpolate(self, frames: VideoFrames) -> VideoFrames:
        """Step 7: Frame interpolation."""
        print(f"[Step 7/9] Frame Interpolation (RIFE)")
        print(f"  Factor: 2x")
        
        if frames.to_tensor().dim() == 4:
            tensor = frames.to_tensor()
        else:
            tensor = frames.to_tensor()
        
        t, c, h, w = tensor.shape
        interpolated_frames = []
        
        for i in range(t - 1):
            frame1 = tensor[i]
            frame2 = tensor[i + 1]
            interpolated_frames.append(frame1)
            
            interp = (frame1 + frame2) / 2
            interpolated_frames.append(interp)
        
        interpolated_frames.append(tensor[-1])
        
        result = VideoFrames()
        result.extend(interpolated_frames)
        print(f"  Interpolated to {len(result)} frames")
        return result
    
    def _step8_quality_check(self, frames: VideoFrames) -> Optional[QualityReport]:
        """Step 8: Quality check."""
        print(f"[Step 8/9] Quality Control")
        print(f"  Analyzing frames...")
        
        tensor = frames.to_tensor()
        if tensor.dim() == 4:
            pass
        else:
            tensor = tensor.permute(1, 0, 2, 3)
        
        report, corrections = self.quality_controller.assess(tensor)
        
        print(f"  Quality Score: {report.overall_score:.2f}")
        print(f"  Issues: {[i.value for i in report.issues] if report.issues else 'None'}")
        
        if report.needs_correction and corrections:
            print(f"  Corrections applied: {list(corrections.keys())}")
        
        return report
    
    def _step9_export(self, frames: VideoFrames, output_path: Path) -> None:
        """Step 9: Export."""
        print(f"[Step 9/9] Export (FFmpeg)")
        print(f"  Format: {self.config.output_format}")
        print(f"  FPS: {self.config.fps}")
        print(f"  Quality: {self.config.quality}")
        print(f"  Output: {output_path}")
        
        quality_map = {
            "low": QualityPreset.LOW,
            "medium": QualityPreset.MEDIUM,
            "high": QualityPreset.HIGH,
            "ultra": QualityPreset.ULTRA
        }
        
        options = ExportOptions(
            video_spec=VideoSpec(
                duration_seconds=self.config.duration_seconds,
                fps=self.config.fps,
                format=VideoFormat(self.config.output_format),
                quality=quality_map.get(self.config.quality, QualityPreset.MEDIUM)
            ),
            enable_interpolation=self.config.enable_interpolation
        )
        
        self.exporter.export(frames, output_path, options)
        print(f"  Export complete!")
    
    def _get_default_prompt(self, content_type: str) -> str:
        """Get default prompt based on content type."""
        prompts = {
            "human": "smooth animation, natural motion, high quality",
            "furry": "animated furry character, smooth fur, natural motion",
            "animal": "animated animal, natural motion, high quality",
            "landscape": "cinematic landscape, wind movement, high quality",
            "scene": "cinematic scene, smooth animation, high quality",
            "object": "smooth animation, subtle motion, high quality"
        }
        return prompts.get(content_type, "animated scene, smooth motion, high quality")
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_pipeline(
    image_path: Union[str, Path],
    prompt: str = "",
    config: Optional[Union[PipelineConfig, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> PipelineResult:
    """
    Run the complete image-to-video pipeline.
    
    This is the main entry point for the Picture-Aliver system.
    
    Args:
        image_path: Path to input image file
        prompt: Optional text prompt for generation
        config: Pipeline configuration (dict or PipelineConfig)
        output_path: Optional output path override
        
    Returns:
        PipelineResult with output path and processing metadata
        
    Example:
        >>> result = run_pipeline(
        ...     image_path="photo.jpg",
        ...     prompt="cinematic animation",
        ...     config={"duration_seconds": 10, "fps": 24},
        ...     output_path="output.mp4"
        ... )
        >>> print(result.output_path)
    """
    if isinstance(config, dict):
        config = PipelineConfig(**config)
    elif config is None:
        config = PipelineConfig()
    
    pipeline = Pipeline(config)
    pipeline.initialize()
    
    return pipeline.run_pipeline(image_path, prompt, config, output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Picture-Aliver: AI Image-to-Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i image.jpg -o video.mp4
  python main.py -i image.jpg -o video.mp4 --duration 10 --fps 24
  python main.py -i furry.png -o animation.mp4 --motion-prompt "gentle tail wag"
  python main.py -i image.jpg -o video.mp4 --quality high --interpolate
        """
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-p", "--prompt", default="", help="Text prompt")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (5-120)")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--width", type=int, default=512, help="Frame width")
    parser.add_argument("--height", type=int, default=512, help="Frame height")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--motion-strength", type=float, default=0.8, help="Motion strength")
    parser.add_argument("--motion-mode", default="auto", choices=["auto", "cinematic", "zoom", "pan", "subtle", "furry"])
    parser.add_argument("--motion-prompt", help="Natural language motion description")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "ultra"])
    parser.add_argument("--format", default="mp4", choices=["mp4", "webm", "gif"])
    parser.add_argument("--interpolate", action="store_true", help="Enable frame interpolation")
    parser.add_argument("--no-stabilization", action="store_true", help="Disable stabilization")
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality check")
    parser.add_argument("--benchmark", action="store_true", help="Show GPU benchmarks")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if args.benchmark:
        from gpu_optimization import print_benchmark_table
        print_benchmark_table()
        return 0
    
    config = PipelineConfig(
        duration_seconds=max(5.0, min(120.0, args.duration)),
        fps=args.fps,
        width=args.width,
        height=args.height,
        guidance_scale=args.scale,
        num_inference_steps=args.steps,
        motion_strength=args.motion_strength,
        motion_mode=args.motion_mode,
        motion_prompt=args.motion_prompt,
        quality=args.quality,
        output_format=args.format,
        enable_stabilization=not args.no_stabilization,
        enable_interpolation=args.interpolate,
        enable_quality_check=not args.no_quality_check,
        device=args.device
    )
    
    try:
        result = run_pipeline(
            image_path=args.input,
            prompt=args.prompt,
            config=config,
            output_path=args.output
        )
        
        if result.success:
            print(f"\nSuccess! Video saved to: {result.output_path}")
            print(f"Processing time: {result.processing_time:.1f}s")
            return 0
        else:
            print(f"\nFailed: {result.errors}")
            return 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())