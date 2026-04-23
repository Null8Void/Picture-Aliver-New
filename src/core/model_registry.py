"""Model registry with content rating support for unrestricted generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Type
from typing_extensions import Literal

import torch


class ContentRating(Enum):
    """Content rating levels for models."""
    SAFE = "safe"
    MATURE = "mature"  # Adult content with some restrictions
    NSFW = "nsfw"      # No restrictions


class ModelCategory(Enum):
    """Categories of models in the pipeline."""
    I2V = "i2v"                    # Image-to-Video generation
    DEPTH = "depth"                # Depth estimation
    SEGMENTATION = "segmentation"  # Semantic/instance segmentation
    MOTION = "motion"              # Motion conditioning
    INTERPOLATION = "interpolation" # Frame interpolation


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    repo_id: str
    model_path: str
    category: ModelCategory
    rating: ContentRating
    vram_mb: int
    resolution: Optional[tuple[int, int]] = None
    max_frames: Optional[int] = None
    base_model: Optional[str] = None  # For motion adapters
    variants: Dict[str, str] = field(default_factory=dict)
    requires_base: bool = False
    quantization: Optional[str] = None  # fp16, int8, int4
    is_torchscript: bool = False
    is_onnx: bool = False
    license: Optional[str] = None
    
    @property
    def full_repo_path(self) -> str:
        """Get full HuggingFace path."""
        return self.repo_id


@dataclass 
class ModelRegistry:
    """Registry of available models organized by category and content rating."""
    
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    
    def __post_init__(self):
        self._populate_registry()
    
    def _populate_registry(self):
        """Populate registry with all available models."""
        
        # =============================================
        # IMAGE-TO-VIDEO MODELS - UNRESTRICTED
        # =============================================
        
        self.register(ModelInfo(
            name="Open-SVD",
            repo_id="camenduru/open-svd",
            model_path="svd",
            category=ModelCategory.I2V,
            rating=ContentRating.NSFW,
            vram_mb=12000,
            resolution=(576, 1024),
            max_frames=25,
            variants={
                "fp16": "camenduru/open-svd",
                "int8": "camenduru/open-svd-int8",
            }
        ))
        
        self.register(ModelInfo(
            name="ZeroScope-v2",
            repo_id="cerspense/zeroscope_v2_576w",
            model_path="zeroscope",
            category=ModelCategory.I2V,
            rating=ContentRating.MATURE,
            vram_mb=6000,
            resolution=(320, 576),
            max_frames=16,
            variants={
                "xl": "cerspense/zeroscope_v2_576w",
                "xl_dw": "cerspense/zeroscope_v2_576w_dw",
            }
        ))
        
        self.register(ModelInfo(
            name="I2VGen-XL-Unrestricted",
            repo_id="ali-vilab/i2vgen-xl",
            model_path="i2vgen-xl",
            category=ModelCategory.I2V,
            rating=ContentRating.MATURE,
            vram_mb=10000,
            resolution=(512, 512),
            max_frames=16,
        ))
        
        self.register(ModelInfo(
            name="AnimateDiff-v3",
            repo_id="camenduru/animatediff",
            model_path="animatediff",
            category=ModelCategory.I2V,
            rating=ContentRating.NSFW,
            vram_mb=8000,
            resolution=(512, 512),
            max_frames=16,
            requires_base=True,
            base_model="stabilityai/stable-diffusion-v1-5",
        ))
        
        self.register(ModelInfo(
            name="OpenGIF-Unrestricted",
            repo_id=" caut的英雄/openGIF",
            model_path="opengif",
            category=ModelCategory.I2V,
            rating=ContentRating.NSFW,
            vram_mb=10000,
            resolution=(512, 512),
            max_frames=32,
        ))
        
        self.register(ModelInfo(
            name="ModelScope-DiT",
            repo_id="ali-vilab/modelscope-dit-text-to-video",
            model_path="modelscope-dit",
            category=ModelCategory.I2V,
            rating=ContentRating.MATURE,
            vram_mb=24000,
            resolution=(256, 256),
            max_frames=16,
        ))
        
        # =============================================
        # IMAGE-TO-VIDEO MODELS - GENERAL/SAFE
        # =============================================
        
        self.register(ModelInfo(
            name="SVD (Stable Video Diffusion)",
            repo_id="stabilityai/stable-video-diffusion",
            model_path="svd",
            category=ModelCategory.I2V,
            rating=ContentRating.SAFE,
            vram_mb=12000,
            resolution=(576, 1024),
            max_frames=25,
            variants={
                "svd": "stabilityai/stable-video-diffusion",
                "svd_xt": "stabilityai/stable-video-diffusion-img2vid-xt",
            }
        ))
        
        self.register(ModelInfo(
            name="AnimateDiff-SDXL",
            repo_id="guoyww/animatediff-motion-adapter-sdxl-beta",
            model_path="animatediff-sdxl",
            category=ModelCategory.I2V,
            rating=ContentRating.SAFE,
            vram_mb=12000,
            resolution=(1024, 1024),
            max_frames=16,
            requires_base=True,
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
        ))
        
        self.register(ModelInfo(
            name="AnimateDiff-SD15",
            repo_id="guoyww/animatediff-motion-adapter",
            model_path="animatediff-sd15",
            category=ModelCategory.I2V,
            rating=ContentRating.SAFE,
            vram_mb=8000,
            resolution=(512, 512),
            max_frames=16,
            requires_base=True,
            base_model="runwayml/stable-diffusion-v1-5",
        ))
        
        # =============================================
        # DEPTH ESTIMATION MODELS
        # =============================================
        
        # Unrestricted
        self.register(ModelInfo(
            name="MiDaS v3.1",
            repo_id="Intel/dpt-hybrid-midas",
            model_path="midas",
            category=ModelCategory.DEPTH,
            rating=ContentRating.MATURE,
            vram_mb=2000,
            resolution=(384, 384),
        ))
        
        self.register(ModelInfo(
            name="Depth-Anything",
            repo_id="LiheYoung/depth_anything_vitl14",
            model_path="depth-anything",
            category=ModelCategory.DEPTH,
            rating=ContentRating.MATURE,
            vram_mb=3000,
            resolution=(518, 518),
        ))
        
        self.register(ModelInfo(
            name="LeRes",
            repo_id="liujh04/leres",
            model_path="leres",
            category=ModelCategory.DEPTH,
            rating=ContentRating.MATURE,
            vram_mb=2000,
            resolution=(448, 448),
        ))
        
        # General
        self.register(ModelInfo(
            name="ZoeDepth",
            repo_id="lllyasviel/ldm",
            model_path="zoedepth",
            category=ModelCategory.DEPTH,
            rating=ContentRating.SAFE,
            vram_mb=2000,
            resolution=(384, 384),
        ))
        
        self.register(ModelInfo(
            name="Marigold",
            repo_id="prs-eth/marigold-depth",
            model_path="marigold",
            category=ModelCategory.DEPTH,
            rating=ContentRating.SAFE,
            vram_mb=6000,
            resolution=(768, 768),
            quantization="fp16",
        ))
        
        # =============================================
        # SEGMENTATION MODELS
        # =============================================
        
        # Unrestricted
        self.register(ModelInfo(
            name="SAM-ViT-Huge",
            repo_id="facebook/sam-vit-huge",
            model_path="sam",
            category=ModelCategory.SEGMENTATION,
            rating=ContentRating.MATURE,
            vram_mb=2500,
            resolution=(1024, 1024),
        ))
        
        self.register(ModelInfo(
            name="SAM-ViT-Base",
            repo_id="facebook/sam-vit-base",
            model_path="sam",
            category=ModelCategory.SEGMENTATION,
            rating=ContentRating.MATURE,
            vram_mb=400,
            resolution=(1024, 1024),
        ))
        
        self.register(ModelInfo(
            name="MobileSAM",
            repo_id="dh2811-group/mobile-sam",
            model_path="mobile-sam",
            category=ModelCategory.SEGMENTATION,
            rating=ContentRating.MATURE,
            vram_mb=300,
            resolution=(1024, 1024),
            quantization="int8",
        ))
        
        # General
        self.register(ModelInfo(
            name="SAM-ViT-Large",
            repo_id="facebook/sam-vit-large",
            model_path="sam",
            category=ModelCategory.SEGMENTATION,
            rating=ContentRating.SAFE,
            vram_mb=1000,
            resolution=(1024, 1024),
        ))
        
        self.register(ModelInfo(
            name="DeepLabV3+",
            repo_id="google/deeplabv3_resnet101",
            model_path="deeplabv3",
            category=ModelCategory.SEGMENTATION,
            rating=ContentRating.SAFE,
            vram_mb=1000,
            resolution=(512, 512),
        ))
        
        # =============================================
        # MOTION CONDITIONING MODELS
        # =============================================
        
        self.register(ModelInfo(
            name="AnimateDiff-Motion-Adapter",
            repo_id="camenduru/animatediff-motion-adapter",
            model_path="motion-adapter",
            category=ModelCategory.MOTION,
            rating=ContentRating.NSFW,
            vram_mb=1000,
        ))
        
        self.register(ModelInfo(
            name="AnimateDiff-Motion-Adapter-SDXL",
            repo_id="guoyww/animatediff-motion-adapter-sdxl-beta",
            model_path="motion-adapter-sdxl",
            category=ModelCategory.MOTION,
            rating=ContentRating.SAFE,
            vram_mb=2000,
        ))
        
        self.register(ModelInfo(
            name="MotionDirector",
            repo_id="MotionDirector/MotionDirector",
            model_path="motion-director",
            category=ModelCategory.MOTION,
            rating=ContentRating.NSFW,
            vram_mb=4000,
        ))
        
        # =============================================
        # FRAME INTERPOLATION MODELS
        # =============================================
        
        self.register(ModelInfo(
            name="RIFE-v4",
            repo_id="hzwer/rife",
            model_path="rife",
            category=ModelCategory.INTERPOLATION,
            rating=ContentRating.MATURE,
            vram_mb=2000,
            resolution=(480, 480),
        ))
        
        self.register(ModelInfo(
            name="RIFE-Pro",
            repo_id="hzwer/rife-ncnn",
            model_path="rife-ncnn",
            category=ModelCategory.INTERPOLATION,
            rating=ContentRating.MATURE,
            vram_mb=500,
            resolution=(480, 480),
            is_torchscript=True,
        ))
        
        self.register(ModelInfo(
            name="AMT-S",
            repo_id="google-research/amt",
            model_path="amt",
            category=ModelCategory.INTERPOLATION,
            rating=ContentRating.MATURE,
            vram_mb=3000,
            resolution=(512, 512),
        ))
        
        self.register(ModelInfo(
            name="CAIN",
            repo_id="myungsub/cain",
            model_path="cain",
            category=ModelCategory.INTERPOLATION,
            rating=ContentRating.MATURE,
            vram_mb=1000,
            resolution=(256, 256),
        ))
    
    def register(self, model: ModelInfo):
        """Register a model in the registry."""
        self.models[model.name] = model
    
    def get(
        self,
        name: str,
        raise_not_found: bool = True
    ) -> Optional[ModelInfo]:
        """Get a model by name."""
        model = self.models.get(name)
        if model is None and raise_not_found:
            available = list(self.models.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return model
    
    def get_by_category(
        self,
        category: ModelCategory,
        rating: Optional[ContentRating] = None,
        min_vram: int = 0,
        max_vram: int = 999999
    ) -> List[ModelInfo]:
        """Get models filtered by category, rating, and VRAM."""
        results = []
        for model in self.models.values():
            if model.category != category:
                continue
            if rating and model.rating != rating:
                continue
            if model.vram_mb < min_vram or model.vram_mb > max_vram:
                continue
            results.append(model)
        return results
    
    def get_i2v_models(
        self,
        safe_only: bool = False,
        max_vram: int = 999999
    ) -> List[ModelInfo]:
        """Get image-to-video models."""
        rating = None if not safe_only else ContentRating.SAFE
        return self.get_by_category(
            ModelCategory.I2V,
            rating=rating,
            max_vram=max_vram
        )
    
    def get_best_for_vram(
        self,
        category: ModelCategory,
        rating: ContentRating,
        available_vram_mb: int
    ) -> Optional[ModelInfo]:
        """Get the best model that fits in available VRAM."""
        candidates = self.get_by_category(
            category,
            rating=rating,
            max_vram=available_vram_mb
        )
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.vram_mb)
    
    def get_model_recommendations(
        self,
        rating: ContentRating,
        vram_mb: int,
        high_quality: bool = True
    ) -> Dict[str, ModelInfo]:
        """Get recommended model combination for a given setup."""
        recommendations = {}
        
        # I2V Model
        i2v_candidates = self.get_by_category(
            ModelCategory.I2V,
            rating=rating,
            max_vram=vram_mb // 2  # Reserve space for other models
        )
        if i2v_candidates:
            if high_quality:
                recommendations["i2v"] = max(i2v_candidates, key=lambda m: m.vram_mb)
            else:
                recommendations["i2v"] = min(i2v_candidates, key=lambda m: m.vram_mb)
        
        # Depth Model
        depth = self.get_best_for_vram(
            ModelCategory.DEPTH, rating, vram_mb // 4
        )
        if depth:
            recommendations["depth"] = depth
        
        # Segmentation Model  
        seg = self.get_best_for_vram(
            ModelCategory.SEGMENTATION, rating, vram_mb // 4
        )
        if seg:
            recommendations["segmentation"] = seg
        
        # Interpolation Model
        if vram_mb > 4000:
            interp = self.get_best_for_vram(
                ModelCategory.INTERPOLATION, rating, 2000
            )
            if interp:
                recommendations["interpolation"] = interp
        
        return recommendations


# Global registry instance
MODEL_REGISTRY = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return MODEL_REGISTRY


def get_models_by_rating(rating: ContentRating) -> List[ModelInfo]:
    """Get all models of a specific content rating."""
    return [m for m in MODEL_REGISTRY.models.values() if m.rating == rating]


def get_nsfw_models() -> List[ModelInfo]:
    """Get all unrestricted/NSFW models."""
    return get_models_by_rating(ContentRating.NSFW)


def get_safe_models() -> List[ModelInfo]:
    """Get all safe/general models."""
    return get_models_by_rating(ContentRating.SAFE)


def get_mature_models() -> List[ModelInfo]:
    """Get all mature-rated models."""
    return get_models_by_rating(ContentRating.MATURE)


def print_registry_summary():
    """Print a summary of the model registry."""
    print("\n=== MODEL REGISTRY SUMMARY ===\n")
    
    for rating in ContentRating:
        models = get_models_by_rating(rating)
        print(f"\n{rating.value.upper()} ({len(models)} models):")
        
        for category in ModelCategory:
            category_models = [m for m in models if m.category == category]
            if category_models:
                print(f"  {category.value}: {len(category_models)} models")
                for m in category_models:
                    print(f"    - {m.name} ({m.vram_mb}MB VRAM)")


if __name__ == "__main__":
    print_registry_summary()