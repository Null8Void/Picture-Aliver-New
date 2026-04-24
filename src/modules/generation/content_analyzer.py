"""Dynamic content analyzer for automatic pipeline adjustment.

Detects image content and adjusts pipeline accordingly:
- Human/Furry: Pose + subtle motion
- Landscape: Parallax + environmental motion
- Object: Rotation + small movement
- Animal: Natural + breathing motion
- Scene: Mixed motion patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentType(Enum):
    """Types of content detected in images."""
    HUMAN = "human"
    FURRY = "furry"
    ANIMAL = "animal"
    LANDSCAPE = "landscape"
    OBJECT = "object"
    SCENE = "scene"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FurColor(Enum):
    """Fur color patterns."""
    SINGLE = "single"
    TWO_TONE = "two_tone"
    SPOTTED = "spotted"
    STRIPED = "striped"
    RAINBOW = "rainbow"


@dataclass
class ContentAnalysis:
    """Analysis result of image content."""
    content_type: ContentType
    confidence: float
    
    has_face: bool = False
    has_full_body: bool = False
    has_fur: bool = False
    has_wings: bool = False
    has_tail: bool = False
    
    fur_color: Optional[FurColor] = None
    fur_density: float = 0.0
    
    is_portrait: bool = False
    is_landscape: bool = False
    has_depth: bool = False
    
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    
    recommended_motion: str = "subtle"
    recommended_style: str = "cinematic"
    recommended_conditioning: List[str] = field(default_factory=list)
    
    detected_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class DynamicPipelineConfig:
    """Configuration for dynamic pipeline adaptation."""
    auto_detect: bool = True
    detection_threshold: float = 0.5
    
    human_config: Dict = field(default_factory=lambda: {
        "motion_style": "subtle",
        "use_pose": True,
        "use_depth": True,
        "motion_strength": 0.5,
        "preserve_identity": True
    })
    
    furry_config: Dict = field(default_factory=lambda: {
        "motion_style": "furry_natural",
        "use_pose": True,
        "use_depth": True,
        "use_fur_physics": True,
        "motion_strength": 0.6,
        "preserve_identity": True,
        "fur_simulation": True
    })
    
    animal_config: Dict = field(default_factory=lambda: {
        "motion_style": "natural",
        "use_depth": True,
        "motion_strength": 0.7,
        "breathing_motion": True
    })
    
    landscape_config: Dict = field(default_factory=lambda: {
        "motion_style": "parallax",
        "use_depth": True,
        "use_env_motion": True,
        "motion_strength": 0.8,
        "wind_effect": True
    })
    
    object_config: Dict = field(default_factory=lambda: {
        "motion_style": "rotation",
        "use_depth": True,
        "motion_strength": 0.4,
        "rotation_axis": "y"
    })
    
    scene_config: Dict = field(default_factory=lambda: {
        "motion_style": "mixed",
        "use_depth": True,
        "motion_strength": 0.6
    })


class ContentAnalyzer:
    """Analyzes image content for automatic pipeline adjustment.
    
    Detects:
    - Human/Furry characters (pose estimation)
    - Animal features (ears, tails, fur)
    - Landscape elements
    - Object categories
    - Scene complexity
    
    Args:
        config: Analysis configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[DynamicPipelineConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or DynamicPipelineConfig()
        self.device = device or torch.device("cpu")
        
        self._pose_detector = None
        self._animal_detector = None
        self._depth_estimator = None
        self._feature_extractor = None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize detection models."""
        if self._initialized:
            return
        
        self._init_pose_detector()
        self._init_animal_detector()
        self._init_depth_estimator()
        
        self._initialized = True
    
    def _init_pose_detector(self) -> None:
        """Initialize pose estimation model."""
        try:
            from controlnet_aux import OpenposeDetector
            
            self._pose_detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet",
                device=self.device
            )
        except ImportError:
            self._pose_detector = self._create_simple_pose_detector()
    
    def _init_animal_detector(self) -> None:
        """Initialize animal/furry feature detector."""
        self._animal_detector = self._create_animal_detector()
    
    def _init_depth_estimator(self) -> None:
        """Initialize depth estimator."""
        from ..depth.depth_estimator import DepthEstimator
        self._depth_estimator = DepthEstimator(device=self.device)
        self._depth_estimator.initialize()
    
    def _create_simple_pose_detector(self) -> nn.Module:
        """Create simple pose detector fallback."""
        class SimplePoseNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
                self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
                self.pool = nn.MaxPool2d(2, 2)
                
                self.keypoint_head = nn.Conv2d(128, 17, 1)
                self.confidence_head = nn.Conv2d(128, 1, 1)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                
                keypoints = self.keypoint_head(x)
                confidence = torch.sigmoid(self.confidence_head(x))
                
                return keypoints, confidence
                
            def detect_pose(self, image):
                with torch.no_grad():
                    keypoints, conf = self.forward(image)
                    
                    has_pose = conf.mean() > 0.3
                    keypoint_count = (keypoints.abs().max(dim=1)[0] > 0.5).sum().item()
                    
                return has_pose, keypoint_count
        
        return SimplePoseNet().to(self.device)
    
    def _create_animal_detector(self) -> nn.Module:
        """Create animal/furry feature detector."""
        class AnimalDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                self.feature_heads = nn.ModuleDict({
                    "ears": nn.Conv2d(128, 1, 1),
                    "tail": nn.Conv2d(128, 1, 1),
                    "fur": nn.Conv2d(128, 1, 1),
                    "wings": nn.Conv2d(128, 1, 1),
                    "face": nn.Conv2d(128, 1, 1),
                })
            
            def forward(self, x):
                features = self.encoder(x)
                
                results = {}
                for name, head in self.feature_heads.items():
                    results[name] = torch.sigmoid(head(features))
                
                return results
            
            def detect_animal_features(self, image):
                with torch.no_grad():
                    features = self.forward(image)
                    
                    feature_scores = {
                        "has_ears": features["ears"].mean().item() > 0.3,
                        "has_tail": features["tail"].mean().item() > 0.2,
                        "has_fur": features["fur"].mean().item() > 0.4,
                        "has_wings": features["wings"].mean().item() > 0.2,
                        "has_face": features["face"].mean().item() > 0.5,
                    }
                    
                    fur_density = features["fur"].mean().item()
                    
                return feature_scores, fur_density
        
        return AnimalDetector().to(self.device)
    
    def analyze(
        self,
        image: torch.Tensor,
        return_details: bool = False
    ) -> ContentAnalysis:
        """Analyze image content.
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            return_details: Return detailed analysis
            
        Returns:
            ContentAnalysis with detected content type and features
        """
        if not self._initialized:
            self.initialize()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image_np = image.squeeze(0)
        if image_np.shape[0] == 3:
            image_np = image_np.permute(1, 2, 0)
        
        if image_np.max() <= 1:
            image_np = (image_np * 255).cpu().numpy().astype(np.uint8)
        else:
            image_np = image_np.cpu().numpy().astype(np.uint8)
        
        content_type, confidence = self._detect_content_type(image, image_np)
        
        pose_result = self._detect_pose(image)
        animal_result = self._detect_animal_features(image)
        depth_result = self._estimate_scene_depth(image)
        
        analysis = ContentAnalysis(
            content_type=content_type,
            confidence=confidence,
            has_face=pose_result.get("has_face", False),
            has_full_body=pose_result.get("has_body", False),
            has_fur=animal_result.get("has_fur", False),
            has_wings=animal_result.get("has_wings", False),
            has_tail=animal_result.get("has_tail", False),
            fur_density=animal_result.get("fur_density", 0.0),
            is_portrait=depth_result.get("is_portrait", False),
            is_landscape=depth_result.get("is_landscape", False),
            has_depth=depth_result.get("has_depth", True),
            recommended_motion=self._get_recommended_motion(content_type),
            recommended_style=self._get_recommended_style(content_type),
            recommended_conditioning=self._get_recommended_conditioning(content_type),
            detected_features={**pose_result, **animal_result, **depth_result}
        )
        
        if return_details:
            analysis.dominant_colors = self._extract_colors(image_np)
        
        return analysis
    
    def _detect_content_type(
        self,
        image: torch.Tensor,
        image_np: np.ndarray
    ) -> Tuple[ContentType, float]:
        """Detect primary content type."""
        pose_result = self._detect_pose(image)
        animal_result = self._detect_animal_features(image)
        depth_result = self._estimate_scene_depth(image)
        
        scores = {
            ContentType.HUMAN: 0.0,
            ContentType.FURRY: 0.0,
            ContentType.ANIMAL: 0.0,
            ContentType.LANDSCAPE: 0.0,
            ContentType.OBJECT: 0.0,
            ContentType.SCENE: 0.0,
        }
        
        if pose_result.get("has_pose", False):
            scores[ContentType.HUMAN] += 0.7
            if animal_result.get("has_fur", False):
                scores[ContentType.FURRY] += 0.5
        
        if animal_result.get("has_fur", False) or animal_result.get("fur_density", 0) > 0.3:
            scores[ContentType.FURRY] += 0.6
            scores[ContentType.ANIMAL] += 0.4
            
            if animal_result.get("has_ears", False) or animal_result.get("has_tail", False):
                scores[ContentType.FURRY] += 0.3
        
        if animal_result.get("has_wings", False):
            scores[ContentType.FURRY] += 0.2
            scores[ContentType.ANIMAL] += 0.2
        
        if depth_result.get("is_landscape", False) and not animal_result.get("has_fur", False):
            scores[ContentType.LANDSCAPE] += 0.8
        
        if depth_result.get("is_object", False):
            scores[ContentType.OBJECT] += 0.7
        
        if depth_result.get("is_scene", False) and not animal_result.get("has_fur", False):
            scores[ContentType.SCENE] += 0.5
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score < self.config.detection_threshold:
            return ContentType.UNKNOWN, best_score
        
        return best_type, best_score
    
    def _detect_pose(self, image: torch.Tensor) -> Dict:
        """Detect human pose features."""
        result = {
            "has_pose": False,
            "has_face": False,
            "has_body": False,
            "keypoint_count": 0
        }
        
        try:
            if self._pose_detector is not None:
                pose_result = self._pose_detector(image)
                
                if pose_result is not None:
                    result["has_pose"] = True
                    result["has_body"] = True
                    result["keypoint_count"] = 17
        except Exception:
            pass
        
        try:
            if self._animal_detector is not None:
                features = self._animal_detector.detect_animal_features(image)
                if features:
                    result["has_face"] = features.get("has_face", False)
        except Exception:
            pass
        
        return result
    
    def _detect_animal_features(self, image: torch.Tensor) -> Dict:
        """Detect animal/furry features."""
        result = {
            "has_ears": False,
            "has_tail": False,
            "has_fur": False,
            "has_wings": False,
            "fur_density": 0.0
        }
        
        if self._animal_detector is None:
            return result
        
        try:
            features, fur_density = self._animal_detector.detect_animal_features(image)
            result.update(features)
            result["fur_density"] = fur_density
        except Exception:
            pass
        
        return result
    
    def _estimate_scene_depth(
        self,
        image: torch.Tensor
    ) -> Dict:
        """Estimate scene depth and composition."""
        result = {
            "is_portrait": False,
            "is_landscape": False,
            "is_object": False,
            "is_scene": False,
            "has_depth": True,
            "depth_complexity": 0.5
        }
        
        _, C, H, W = image.shape
        
        aspect_ratio = W / H
        
        if aspect_ratio < 0.8:
            result["is_portrait"] = True
        elif aspect_ratio > 1.2:
            result["is_landscape"] = True
        
        try:
            if self._depth_estimator is not None:
                depth = self._depth_estimator.estimate(image.squeeze(0))
                
                if hasattr(depth, 'depth'):
                    depth_values = depth.depth
                else:
                    depth_values = depth
                
                if isinstance(depth_values, torch.Tensor):
                    depth_std = depth_values.std().item()
                    result["depth_complexity"] = min(depth_std * 10, 1.0)
                    
                    near_pixels = (depth_values < depth_values.quantile(0.3)).float().sum()
                    total_pixels = depth_values.numel()
                    near_ratio = near_pixels / total_pixels
                    
                    if near_ratio > 0.5 and result["is_portrait"]:
                        result["is_object"] = True
                    elif near_ratio < 0.3 and result["is_landscape"]:
                        result["is_scene"] = True
        except Exception:
            pass
        
        return result
    
    def _extract_colors(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        from sklearn.cluster import KMeans
        
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        
        return colors
    
    def _get_recommended_motion(self, content_type: ContentType) -> str:
        """Get recommended motion style for content type."""
        motion_map = {
            ContentType.HUMAN: "subtle",
            ContentType.FURRY: "furry_natural",
            ContentType.ANIMAL: "natural",
            ContentType.LANDSCAPE: "parallax",
            ContentType.OBJECT: "rotation",
            ContentType.SCENE: "mixed",
            ContentType.MIXED: "cinematic",
            ContentType.UNKNOWN: "cinematic"
        }
        return motion_map.get(content_type, "cinematic")
    
    def _get_recommended_style(self, content_type: ContentType) -> str:
        """Get recommended video style for content type."""
        style_map = {
            ContentType.HUMAN: "cinematic",
            ContentType.FURRY: "furry_vibe",
            ContentType.ANIMAL: "natural",
            ContentType.LANDSCAPE: "environmental",
            ContentType.OBJECT: "showcase",
            ContentType.SCENE: "cinematic",
            ContentType.MIXED: "cinematic",
            ContentType.UNKNOWN: "cinematic"
        }
        return style_map.get(content_type, "cinematic")
    
    def _get_recommended_conditioning(
        self,
        content_type: ContentType
    ) -> List[str]:
        """Get recommended ControlNet conditioning types."""
        conditioning_map = {
            ContentType.HUMAN: ["pose", "depth", "softedge"],
            ContentType.FURRY: ["pose", "depth", "fur", "softedge"],
            ContentType.ANIMAL: ["depth", "softedge"],
            ContentType.LANDSCAPE: ["depth", "normal"],
            ContentType.OBJECT: ["depth", "canny"],
            ContentType.SCENE: ["depth", "normal", "softedge"],
            ContentType.MIXED: ["depth", "softedge"],
            ContentType.UNKNOWN: ["depth"]
        }
        return conditioning_map.get(content_type, ["depth"])
    
    def get_pipeline_config(
        self,
        analysis: ContentAnalysis
    ) -> Dict:
        """Get optimized pipeline config based on analysis.
        
        Args:
            analysis: Content analysis result
            
        Returns:
            Pipeline configuration dictionary
        """
        content_type = analysis.content_type
        
        if content_type == ContentType.HUMAN:
            return {
                **self.config.human_config,
                "content_type": "human"
            }
        elif content_type == ContentType.FURRY:
            return {
                **self.config.furry_config,
                "content_type": "furry",
                "fur_simulation": analysis.has_fur,
                "tail_motion": analysis.has_tail,
                "wing_motion": analysis.has_wings,
            }
        elif content_type == ContentType.ANIMAL:
            return {
                **self.config.animal_config,
                "content_type": "animal"
            }
        elif content_type == ContentType.LANDSCAPE:
            return {
                **self.config.landscape_config,
                "content_type": "landscape"
            }
        elif content_type == ContentType.OBJECT:
            return {
                **self.config.object_config,
                "content_type": "object"
            }
        else:
            return {
                **self.config.scene_config,
                "content_type": "scene"
            }


class DynamicPipelineAdapter:
    """Dynamically adapts pipeline based on content analysis.
    
    Adjusts:
    - Motion generation strategy
    - ControlNet conditioning
    - Model selection
    - Processing order
    
    Args:
        config: Adapter configuration
        device: Target compute device
    """
    
    def __init__(
        self,
        config: Optional[DynamicPipelineConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or DynamicPipelineConfig()
        self.device = device or torch.device("cpu")
        
        self._analyzer = ContentAnalyzer(config, device)
        self._analyzer.initialize()
    
    def adapt_pipeline(
        self,
        image: torch.Tensor,
        base_config: Dict
    ) -> Dict:
        """Adapt pipeline configuration based on image content.
        
        Args:
            image: Input image
            base_config: Base pipeline configuration
            
        Returns:
            Adapted pipeline configuration
        """
        analysis = self._analyzer.analyze(image)
        
        pipeline_config = self._analyzer.get_pipeline_config(analysis)
        
        adapted = base_config.copy()
        adapted.update(pipeline_config)
        
        adapted["motion_mode"] = analysis.recommended_motion
        adapted["conditioning_types"] = analysis.recommended_conditioning
        
        adapted["use_pose"] = "pose" in analysis.recommended_conditioning
        adapted["use_depth"] = "depth" in analysis.recommended_conditioning
        adapted["use_fur_physics"] = (
            analysis.content_type == ContentType.FURRY and analysis.has_fur
        )
        
        adapted["motion_strength"] = self._adjust_motion_strength(
            analysis, base_config.get("motion_strength", 0.5)
        )
        
        adapted["preserve_identity"] = analysis.has_face or analysis.has_fur
        
        adapted["artifact_reduction"] = self._get_artifact_config(analysis)
        
        adapted["_analysis"] = {
            "content_type": analysis.content_type.value,
            "confidence": analysis.confidence,
            "has_fur": analysis.has_fur,
            "has_tail": analysis.has_tail,
        }
        
        return adapted
    
    def _adjust_motion_strength(
        self,
        analysis: ContentAnalysis,
        base_strength: float
    ) -> float:
        """Adjust motion strength based on content type."""
        strength_map = {
            ContentType.HUMAN: 0.5,
            ContentType.FURRY: 0.6,
            ContentType.ANIMAL: 0.7,
            ContentType.LANDSCAPE: 0.8,
            ContentType.OBJECT: 0.4,
            ContentType.SCENE: 0.6,
        }
        
        content_strength = strength_map.get(analysis.content_type, 0.6)
        
        if analysis.has_face:
            content_strength *= 0.8
        
        if analysis.has_fur:
            content_strength *= 0.9
        
        return min(base_strength * content_strength * 1.2, 1.0)
    
    def _get_artifact_config(
        self,
        analysis: ContentAnalysis
    ) -> Dict:
        """Get artifact reduction config based on content."""
        if analysis.has_fur:
            return {
                "enable_depth_conditioning": True,
                "enable_controlnet": True,
                "enable_latent_consistency": True,
                "enable_flow_stabilization": True,
                "depth_strength": 0.9,
                "latent_strength": 0.8,
            }
        elif analysis.has_face:
            return {
                "enable_depth_conditioning": True,
                "enable_controlnet": True,
                "enable_latent_consistency": True,
                "depth_strength": 0.8,
                "latent_strength": 0.8,
            }
        else:
            return {
                "enable_depth_conditioning": True,
                "enable_controlnet": False,
                "enable_latent_consistency": True,
                "depth_strength": 0.7,
            }
    
    def get_model_recommendations(
        self,
        analysis: ContentAnalysis,
        vram_mb: int,
        rating: str = "nsfw"
    ) -> Dict[str, str]:
        """Get recommended models based on content type."""
        recommendations = {
            "i2v": "Open-SVD",
            "depth": "MiDaS v3.1",
            "segmentation": "SAM-ViT-Base",
            "interpolation": "RIFE-v4",
        }
        
        if analysis.content_type == ContentType.FURRY:
            recommendations["i2v"] = "Yiffymix V2" if vram_mb >= 8000 else "Fluffyrock"
            recommendations["controlnet"] = "depth_pose"
        elif analysis.content_type == ContentType.HUMAN:
            recommendations["i2v"] = "Open-SVD"
            recommendations["controlnet"] = "depth_pose"
        elif analysis.content_type == ContentType.LANDSCAPE:
            recommendations["i2v"] = "ZeroScope-Unrestricted"
            recommendations["controlnet"] = "depth_normal"
        elif analysis.content_type == ContentType.OBJECT:
            recommendations["i2v"] = "OpenGIF-Unrestricted"
            recommendations["controlnet"] = "depth_canny"
        
        return recommendations