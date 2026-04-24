"""Furry-specific image-to-video models and assets.

Models optimized for anthropomorphic/feral animal content:
- Yiffymix: Specialized for furry art style
- Fluffyrock: High-quality furry generation
- Dreamshaper: Character-focused with furry support
- Compass Mix: Balanced furry/anime mix
- PawPunk: Furry punk aesthetic
- FurryForge: Community fine-tunes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


class FurryStyle(Enum):
    """Furry art styles."""
    ANTHROPOMORPHIC = "anthropomorphic"
    FERAL = "feral"
    HYBRID = "hybrid"
    KEMONO = "kemono"
    PROTAGONIST = "protagonist"
    CREATURE = "creature"


class FurPattern(Enum):
    """Fur texture patterns."""
    SOLID = "solid"
    SPOTTED = "spotted"
    STRIPED = "striped"
    GRADIENT = "gradient"
    MULTICOLOR = "multicolor"
    SMOOTH = "smooth"


@dataclass
class FurryModelInfo:
    """Information about a Furry-specific model."""
    name: str
    repo_id: str
    model_type: str
    style: FurryStyle
    vram_mb: int
    resolution: Tuple[int, int]
    max_frames: int
    base_model: Optional[str] = None
    special_features: List[str] = field(default_factory=list)
    rating: str = "nsfw"


FURRY_MODELS = {
    "yiffymix": FurryModelInfo(
        name="Yiffymix",
        repo_id="stablediffusion XL/yiffymix",
        model_type="sdxl",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["fur_texture", "character_consistency", "expression_preservation"],
        rating="nsfw"
    ),
    "yiffymix_v2": FurryModelInfo(
        name="Yiffymix V2",
        repo_id="stablediffusion XL/yiffymix-v2",
        model_type="sdxl",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=25,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["enhanced_fur", "improved_anatomy", "better_hands"],
        rating="nsfw"
    ),
    "fluffyrock": FurryModelInfo(
        name="Fluffyrock",
        repo_id="stablediffusion/fluffyrock",
        model_type="sd15",
        style=FurryStyle.HYBRID,
        vram_mb=4000,
        resolution=(512, 768),
        max_frames=16,
        base_model="runwayml/stable-diffusion-v1-5",
        special_features=["fluffy_fur", "soft_shading", "cute_aesthetic"],
        rating="nsfw"
    ),
    "fluffyrock_unbound": FurryModelInfo(
        name="Fluffyrock Unbound",
        repo_id="stablediffusion/fluffyrock-unbound",
        model_type="sdxl",
        style=FurryStyle.HYBRID,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["unrestricted", "high_detail", "all_furry_styles"],
        rating="nsfw"
    ),
    "dreamshaper": FurryModelInfo(
        name="Dreamshaper",
        repo_id="lykon/dreamshaper",
        model_type="sd15",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=4000,
        resolution=(512, 768),
        max_frames=16,
        base_model="runwayml/stable-diffusion-v1-5",
        special_features=["character_focus", "good_for_furry", "expressive"],
        rating="mature"
    ),
    "dreamshaper_xl": FurryModelInfo(
        name="Dreamshaper XL",
        repo_id="lykon/dreamshaper-xl",
        model_type="sdxl",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["high_resolution", "better_prompts", "improved_hands"],
        rating="mature"
    ),
    "compass_mix": FurryModelInfo(
        name="Compass Mix",
        repo_id="stablediffusion/compass-mix",
        model_type="sd15",
        style=FurryStyle.HYBRID,
        vram_mb=4000,
        resolution=(512, 512),
        max_frames=16,
        base_model="runwayml/stable-diffusion-v1-5",
        special_features=["anime_furry_blend", "versatile", "good_colors"],
        rating="mature"
    ),
    "compass_mix_xl": FurryModelInfo(
        name="Compass Mix XL",
        repo_id="stablediffusion/compass-mix-xl",
        model_type="sdxl",
        style=FurryStyle.HYBRID,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["xl_quality", "anime_furry_mix", "versatile"],
        rating="mature"
    ),
    "pawpunk": FurryModelInfo(
        name="PawPunk",
        repo_id="furry/pawpunk",
        model_type="sdxl",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["punk_aesthetic", "bold_colors", "edgy_style"],
        rating="nsfw"
    ),
    "furryforge": FurryModelInfo(
        name="FurryForge",
        repo_id="community/furryforge",
        model_type="sdxl",
        style=FurryStyle.ANTHROPOMORPHIC,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["community_trained", "varied_styles", "detail_oriented"],
        rating="nsfw"
    ),
    "feralcraft": FurryModelInfo(
        name="FeralCraft",
        repo_id="furry/feralcraft",
        model_type="sd15",
        style=FurryStyle.FERAL,
        vram_mb=4000,
        resolution=(512, 768),
        max_frames=16,
        base_model="runwayml/stable-diffusion-v1-5",
        special_features=["feral_animals", "nature_scenes", "wildlife"],
        rating="mature"
    ),
    "kemonomimi": FurryModelInfo(
        name="Kemonomimi Mix",
        repo_id="furry/kemonomimi",
        model_type="sdxl",
        style=FurryStyle.KEMONO,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["anime_hybrid", "ears_tails", "cute_expressions"],
        rating="mature"
    ),
    "creaturecraft": FurryModelInfo(
        name="CreatureCraft",
        repo_id="furry/creaturecraft",
        model_type="sdxl",
        style=FurryStyle.CREATURE,
        vram_mb=8000,
        resolution=(1024, 1024),
        max_frames=24,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        special_features=["monsters", "creatures", "fantasy_beings"],
        rating="nsfw"
    ),
}


def get_furry_models(
    style: Optional[FurryStyle] = None,
    min_vram: int = 0,
    max_vram: int = 99999
) -> List[FurryModelInfo]:
    """Get Furry models filtered by style and VRAM."""
    models = []
    
    for model in FURRY_MODELS.values():
        if style is not None and model.style != style:
            continue
        if model.vram_mb < min_vram or model.vram_mb > max_vram:
            continue
        models.append(model)
    
    return sorted(models, key=lambda m: m.vram_mb)


def get_recommended_furry_model(
    vram_mb: int,
    style: FurryStyle = FurryStyle.ANTHROPOMORPHIC,
    rating: str = "nsfw"
) -> Optional[FurryModelInfo]:
    """Get recommended Furry model for hardware."""
    candidates = []
    
    for model in FURRY_MODELS.values():
        if model.vram_mb <= vram_mb and model.style == style:
            if rating == "nsfw" or model.rating == rating:
                candidates.append(model)
    
    if not candidates:
        for model in FURRY_MODELS.values():
            if model.vram_mb <= vram_mb:
                candidates.append(model)
    
    if candidates:
        return max(candidates, key=lambda m: m.vram_mb)
    
    return None


def get_all_furry_model_names() -> List[str]:
    """Get list of all Furry model names."""
    return list(FURRY_MODELS.keys())


from enum import Enum