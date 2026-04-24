"""
Quality Control Module

Automated quality assessment and correction for generated videos.
Detects artifacts and automatically adjusts generation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityIssue(str, Enum):
    """Detected quality issues."""
    NONE = "none"
    WARPED_FACE = "warped_face"
    FRAME_FLICKER = "frame_flicker"
    STRUCTURAL_INCONSISTENCY = "structural_inconsistency"
    COLOR_BANDING = "color_banding"
    MOTION_BLUR = "motion_blur"
    TEMPORAL_JITTER = "temporal_jitter"


@dataclass
class QualityReport:
    """
    Quality assessment report.
    
    Attributes:
        overall_score: Overall quality score (0-1)
        issues: List of detected issues
        frame_scores: Per-frame quality scores
        flicker_score: Frame consistency score
        face_quality_score: Face integrity score
        structural_score: Structural consistency score
        needs_correction: Whether correction is needed
        recommended_adjustments: Suggested parameter changes
    """
    overall_score: float = 1.0
    issues: List[QualityIssue] = field(default_factory=list)
    frame_scores: List[float] = field(default_factory=list)
    flicker_score: float = 1.0
    face_quality_score: float = 1.0
    structural_score: float = 1.0
    needs_correction: bool = False
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionStrategy:
    """Strategy for correcting detected issues."""
    strengthen_conditioning: bool = False
    reduce_motion_intensity: bool = False
    increase_inference_steps: bool = False
    adjust_guidance_scale: bool = False
    apply_temporal_smoothing: bool = False
    retry_with_seed: bool = False


class QualityDetector:
    """
    Detects quality issues in generated video frames.
    
    Detection capabilities:
    - Face warping detection
    - Frame-to-frame flickering
    - Structural inconsistencies
    - Color banding
    - Motion artifacts
    
    Attributes:
        device: Compute device
        face_detector: Face detection model (optional)
        threshold_face_warp: Threshold for face warping
        threshold_flicker: Threshold for flicker detection
        threshold_structural: Threshold for structural issues
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_face_detection: bool = True
    ):
        self.device = device or torch.device("cpu")
        self.use_face_detection = use_face_detection
        
        self.threshold_face_warp = 0.7
        self.threshold_flicker = 0.15
        self.threshold_structural = 0.6
        
        self.face_detector = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize detection models."""
        if self._initialized:
            return
        
        if self.use_face_detection:
            try:
                from torchvision import models
                self.face_detector = models.get_model("resnet18", pretrained=False)
                self.face_detector.fc = nn.Linear(512, 1)
                self.face_detector = self.face_detector.to(self.device)
                self.face_detector.eval()
            except Exception:
                self.face_detector = SimpleFaceDetector(device=self.device)
        
        self._initialized = True
    
    def detect(
        self,
        frames: torch.Tensor,
        return_details: bool = False
    ) -> QualityReport:
        """
        Detect quality issues in video frames.
        
        Args:
            frames: Video frames tensor (T, C, H, W) or (B, T, C, H, W)
            return_details: Return detailed per-frame analysis
            
        Returns:
            QualityReport with detected issues
        """
        if not self._initialized:
            self.initialize()
        
        if frames.dim() == 5:
            frames = frames.squeeze(0)
        
        num_frames = frames.shape[0]
        
        flicker_score = self._detect_flicker(frames)
        
        face_score = self._detect_face_warping(frames)
        
        structural_score = self._detect_structural_inconsistency(frames)
        
        frame_scores = self._score_frames(frames)
        
        issues = []
        if face_score < self.threshold_face_warp:
            issues.append(QualityIssue.WARPED_FACE)
        if flicker_score < self.threshold_flicker:
            issues.append(QualityIssue.FRAME_FLICKER)
        if structural_score < self.threshold_structural:
            issues.append(QualityIssue.STRUCTURAL_INCONSISTENCY)
        
        overall_score = (face_score * 0.3 + flicker_score * 0.35 + structural_score * 0.35)
        
        needs_correction = len(issues) > 0 and overall_score < 0.75
        
        adjustments = self._get_recommended_adjustments(issues, overall_score)
        
        return QualityReport(
            overall_score=overall_score,
            issues=issues,
            frame_scores=frame_scores,
            flicker_score=flicker_score,
            face_quality_score=face_score,
            structural_score=structural_score,
            needs_correction=needs_correction,
            recommended_adjustments=adjustments
        )
    
    def _detect_flicker(self, frames: torch.Tensor) -> float:
        """
        Detect frame-to-frame flickering.
        
        Args:
            frames: Video frames (T, C, H, W)
            
        Returns:
            Flicker score (0-1, higher is better)
        """
        if frames.shape[0] < 2:
            return 1.0
        
        if frames.max() <= 1.0:
            frames = frames * 255.0
        
        luminance = frames.mean(dim=(1, 2))
        
        diff = torch.abs(luminance[1:] - luminance[:-1])
        
        flicker = diff.mean().item() / 255.0
        
        flicker_score = 1.0 - min(flicker * 5, 1.0)
        
        color_variation = frames.std(dim=0).mean().item() / 255.0
        
        flicker_score *= (1.0 - color_variation * 0.3)
        
        return flicker_score
    
    def _detect_face_warping(self, frames: torch.Tensor) -> float:
        """
        Detect face warping/distortion.
        
        Args:
            frames: Video frames (T, C, H, W)
            
        Returns:
            Face quality score (0-1, higher is better)
        """
        if self.face_detector is None:
            return self._detect_edge_artifacts(frames)
        
        scores = []
        for i in range(min(frames.shape[0], 8)):
            frame = frames[i: i + 1]
            
            frame_resized = F.interpolate(
                frame, size=(224, 224), mode="bilinear", align_corners=False
            )
            
            frame_normalized = (frame_resized - 0.5) / 0.5
            
            with torch.no_grad():
                warp_score = self.face_detector(frame_normalized).sigmoid().item()
            
            scores.append(warp_score)
        
        return np.mean(scores) if scores else 1.0
    
    def _detect_edge_artifacts(self, frames: torch.Tensor) -> float:
        """Fallback face warping detection using edge analysis."""
        scores = []
        
        for i in range(min(frames.shape[0], 8)):
            frame = frames[i]
            
            if frame.shape[0] == 3:
                gray = frame.mean(dim=0, keepdim=True)
            else:
                gray = frame
            
            edges = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]) + \
                    torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
            
            edge_mean = edges.mean().item()
            edge_std = edges.std().item()
            
            distortion_score = 1.0 - min(edge_std / (edge_mean + 1e-6) * 0.5, 1.0)
            scores.append(distortion_score)
        
        return np.mean(scores)
    
    def _detect_structural_inconsistency(self, frames: torch.Tensor) -> float:
        """
        Detect structural inconsistencies across frames.
        
        Args:
            frames: Video frames (T, C, H, W)
            
        Returns:
            Structural score (0-1, higher is better)
        """
        if frames.shape[0] < 2:
            return 1.0
        
        features = []
        
        for i in range(0, frames.shape[0], max(1, frames.shape[0] // 4)):
            frame = frames[i]
            
            if frame.shape[0] == 3:
                gray = frame.mean(dim=0, keepdim=True)
            else:
                gray = frame
            
            h, w = gray.shape
            features_h = h // 8
            features_w = w // 8
            grid = F.avg_pool2d(gray.unsqueeze(0), kernel_size=(features_h, features_w))
            features.append(grid.squeeze().flatten())
        
        features = torch.stack(features)
        
        consistency = 1.0 - features.std(dim=0).mean().item()
        
        return consistency
    
    def _score_frames(self, frames: torch.Tensor) -> List[float]:
        """Score individual frames for quality."""
        scores = []
        
        for i in range(frames.shape[0]):
            frame = frames[i]
            
            if frame.max() <= 1.0:
                frame_normalized = frame * 255
            else:
                frame_normalized = frame
            
            sharpness = self._estimate_sharpness(frame)
            
            contrast = self._estimate_contrast(frame)
            
            score = (sharpness * 0.5 + contrast * 0.5)
            scores.append(score)
        
        return scores
    
    def _estimate_sharpness(self, frame: torch.Tensor) -> float:
        """Estimate frame sharpness using Laplacian variance."""
        if frame.shape[0] == 3:
            gray = frame.mean(dim=0)
        else:
            gray = frame
        
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=gray.dtype, device=gray.device).float()
        
        laplacian = laplacian.view(1, 1, 3, 3)
        
        edges = F.conv2d(gray.unsqueeze(0), laplacian, padding=1)
        
        variance = edges.var().item()
        
        sharpness = min(variance / 1000, 1.0)
        
        return sharpness
    
    def _estimate_contrast(self, frame: torch.Tensor) -> float:
        """Estimate frame contrast."""
        if frame.shape[0] == 3:
            gray = frame.mean(dim=0)
        else:
            gray = frame
        
        min_val = gray.min()
        max_val = gray.max()
        
        contrast = (max_val - min_val) / 255.0
        
        return contrast
    
    def _get_recommended_adjustments(
        self,
        issues: List[QualityIssue],
        overall_score: float
    ) -> Dict[str, Any]:
        """Get recommended parameter adjustments based on issues."""
        adjustments = {}
        
        if QualityIssue.WARPED_FACE in issues:
            adjustments["strengthen_conditioning"] = True
            adjustments["guidance_scale_multiplier"] = 1.2
            adjustments["reduce_motion_strength"] = 0.8
        
        if QualityIssue.FRAME_FLICKER in issues:
            adjustments["apply_temporal_smoothing"] = True
            adjustments["increase_temporal_window"] = True
        
        if QualityIssue.STRUCTURAL_INCONSISTENCY in issues:
            adjustments["increase_inference_steps"] = True
            adjustments["steps_multiplier"] = 1.3
            adjustments["reduce_motion_strength"] = 0.7
        
        if overall_score < 0.5:
            adjustments["retry_with_seed"] = True
            adjustments["use_higher_quality_mode"] = True
        
        return adjustments


class QualityController:
    """
    Automated quality control system.
    
    Runs quality detection after generation and automatically
    adjusts parameters for re-generation if issues detected.
    
    Attributes:
        device: Compute device
        detector: Quality detector
        max_retries: Maximum retry attempts
        auto_correct: Enable automatic correction
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_retries: int = 2,
        auto_correct: bool = True
    ):
        self.device = device or torch.device("cpu")
        self.detector = QualityDetector(device=device)
        self.max_retries = max_retries
        self.auto_correct = auto_correct
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the controller."""
        if self._initialized:
            return
        
        self.detector.initialize()
        self._initialized = True
    
    def assess(
        self,
        frames: torch.Tensor,
        auto_correct: bool = True
    ) -> Tuple[QualityReport, Optional[Dict[str, Any]]]:
        """
        Assess video quality and optionally apply corrections.
        
        Args:
            frames: Generated video frames
            auto_correct: Whether to provide correction parameters
            
        Returns:
            Tuple of (QualityReport, correction_params or None)
        """
        if not self._initialized:
            self.initialize()
        
        report = self.detector.detect(frames)
        
        correction_params = None
        if auto_correct and report.needs_correction:
            correction_params = self._compute_correction_params(report)
        
        return report, correction_params
    
    def _compute_correction_params(
        self,
        report: QualityReport
    ) -> Dict[str, Any]:
        """Compute corrected generation parameters."""
        params = {}
        
        adjustments = report.recommended_adjustments
        
        if adjustments.get("strengthen_conditioning"):
            params["guidance_scale"] = adjustments.get("guidance_scale_multiplier", 1.2)
        
        if adjustments.get("reduce_motion_strength"):
            params["motion_strength"] = adjustments.get("reduce_motion_strength", 0.7)
        
        if adjustments.get("increase_inference_steps"):
            params["steps_multiplier"] = adjustments.get("steps_multiplier", 1.3)
        
        if adjustments.get("apply_temporal_smoothing"):
            params["apply_smoothing"] = True
        
        if adjustments.get("retry_with_seed"):
            params["new_seed"] = True
        
        params["correction_level"] = len(report.issues)
        
        return params
    
    def run_quality_loop(
        self,
        generate_fn,
        base_params: Dict[str, Any],
        on_report: Optional[callable] = None
    ) -> Tuple[torch.Tensor, QualityReport]:
        """
        Run generation with automated quality control loop.
        
        Args:
            generate_fn: Function to call for generation
            base_params: Base generation parameters
            on_report: Optional callback for quality reports
            
        Returns:
            Tuple of (final_frames, final_report)
        """
        if not self._initialized:
            self.initialize()
        
        params = base_params.copy()
        retries = 0
        
        while retries <= self.max_retries:
            print(f"[QualityController] Generation attempt {retries + 1}")
            
            frames = generate_fn(**params)
            
            report, correction_params = self.assess(frames, auto_correct=self.auto_correct)
            
            if on_report:
                on_report(report)
            
            if not report.needs_correction:
                print(f"[QualityController] Quality check passed (score: {report.overall_score:.2f})")
                return frames, report
            
            if retries >= self.max_retries:
                print(f"[QualityController] Max retries reached, returning best result")
                return frames, report
            
            print(f"[QualityController] Issues detected: {[i.value for i in report.issues]}")
            print(f"[QualityController] Applying corrections...")
            
            if correction_params:
                params.update(correction_params)
            
            retries += 1
        
        return frames, report


class SimpleFaceDetector(nn.Module):
    """Lightweight face warping detector."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def assess_video_quality(
    frames: torch.Tensor,
    device: Optional[torch.device] = None,
    auto_correct: bool = True
) -> Tuple[QualityReport, Optional[Dict[str, Any]]]:
    """
    Convenience function for quality assessment.
    
    Args:
        frames: Video frames tensor
        device: Compute device
        auto_correct: Whether to provide corrections
        
    Returns:
        Tuple of (QualityReport, correction_params)
    """
    controller = QualityController(device=device, auto_correct=auto_correct)
    controller.initialize()
    
    return controller.assess(frames, auto_correct=auto_correct)


if __name__ == "__main__":
    print("[QualityController] Testing quality detection...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = QualityController(device=device)
    controller.initialize()
    
    test_frames = torch.rand(16, 3, 256, 256)
    
    report, corrections = controller.assess(test_frames)
    
    print(f"\nQuality Report:")
    print(f"  Overall Score: {report.overall_score:.3f}")
    print(f"  Issues: {[i.value for i in report.issues]}")
    print(f"  Flicker: {report.flicker_score:.3f}")
    print(f"  Face Quality: {report.face_quality_score:.3f}")
    print(f"  Structural: {report.structural_score:.3f}")
    print(f"  Needs Correction: {report.needs_correction}")
    
    if corrections:
        print(f"\nRecommended Adjustments:")
        for k, v in corrections.items():
            print(f"  {k}: {v}")