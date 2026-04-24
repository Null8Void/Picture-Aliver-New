"""
Motion Prompt Parser Module

Parses natural language motion prompts and maps them to motion parameters.
Enables users to describe desired motion in plain English.

Examples:
- "cinematic camera pan" -> CameraMotion with pan trajectory
- "subtle breathing motion" -> FurryMotion with breath cycle
- "wind blowing" -> EnvironmentalMotion with wind effect
- "dramatic zoom" -> CameraMotion with zoom trajectory
- "gentle tail wag" -> FurryMotion with tail movement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import re


class MotionCategory(str, Enum):
    """Broad motion categories."""
    CAMERA = "camera"
    OBJECT = "object"
    ENVIRONMENT = "environment"
    CHARACTER = "character"
    FURRY = "furry"
    COMBINED = "combined"


class MotionIntensity(str, Enum):
    """Motion intensity levels."""
    SUBTLE = "subtle"
    GENTLE = "gentle"
    MODERATE = "moderate"
    DRAMATIC = "dramatic"
    INTENSE = "intense"


@dataclass
class MotionParameters:
    """
    Parsed motion parameters from user prompt.
    
    Attributes:
        category: Motion category (camera, object, etc.)
        intensity: Motion intensity level
        primary_motion: Primary motion type
        secondary_motions: Additional motion types
        parameters: Raw parameters extracted
        raw_prompt: Original user prompt
    """
    category: MotionCategory = MotionCategory.CAMERA
    intensity: MotionIntensity = MotionIntensity.MODERATE
    primary_motion: str = "auto"
    secondary_motions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_prompt: str = ""
    speed: float = 1.0
    duration_factor: float = 1.0
    strength: float = 0.7


@dataclass
class MotionPreset:
    """Pre-defined motion preset with parameters."""
    name: str
    category: MotionCategory
    keywords: List[str]
    intensity: MotionIntensity
    camera_type: Optional[str] = None
    furry_motion: Optional[str] = None
    env_effect: Optional[str] = None
    object_motion: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class MotionPromptParser:
    """
    Parser for natural language motion prompts.
    
    Extracts motion type, intensity, speed, and other parameters
    from user-provided text prompts.
    
    Usage:
        parser = MotionPromptParser()
        params = parser.parse("dramatic zoom with gentle wind")
    """
    
    PRESETS: List[MotionPreset] = [
        MotionPreset(
            name="cinematic_pan",
            category=MotionCategory.CAMERA,
            keywords=["cinematic pan", "smooth pan", "horizontal pan", "side pan"],
            intensity=MotionIntensity.DRAMATIC,
            camera_type="pan"
        ),
        MotionPreset(
            name="cinematic_zoom",
            category=MotionCategory.CAMERA,
            keywords=["cinematic zoom", "dramatic zoom", "zoom in", "dolly zoom"],
            intensity=MotionIntensity.DRAMATIC,
            camera_type="zoom"
        ),
        MotionPreset(
            name="orbital",
            category=MotionCategory.CAMERA,
            keywords=["orbital", "circular orbit", "360 spin", "rotate around"],
            intensity=MotionIntensity.DRAMATIC,
            camera_type="orbital"
        ),
        MotionPreset(
            name="subtle_breathing",
            category=MotionCategory.FURRY,
            keywords=["breathing", "breath", "chest rise", "subtle breath"],
            intensity=MotionIntensity.SUBTLE,
            furry_motion="breathing"
        ),
        MotionPreset(
            name="gentle_tail_wag",
            category=MotionCategory.FURRY,
            keywords=["gentle tail wag", "tail wagging", "happy tail", "soft wag"],
            intensity=MotionIntensity.GENTLE,
            furry_motion="tail_wag"
        ),
        MotionPreset(
            name="energetic_tail",
            category=MotionCategory.FURRY,
            keywords=["energetic tail", "fast tail", "excited wag", "vigorous tail"],
            intensity=MotionIntensity.INTENSE,
            furry_motion="tail_wag",
            parameters={"speed": 2.0, "amplitude": 1.5}
        ),
        MotionPreset(
            name="ear_twitch",
            category=MotionCategory.FURRY,
            keywords=["ear twitch", "twitching ears", "alert ears", "perked ears"],
            intensity=MotionIntensity.GENTLE,
            furry_motion="ears"
        ),
        MotionPreset(
            name="wind_blowing",
            category=MotionCategory.ENVIRONMENT,
            keywords=["wind blowing", "windy", "gusting wind", "breeze"],
            intensity=MotionIntensity.MODERATE,
            env_effect="wind"
        ),
        MotionPreset(
            name="gentle_wind",
            category=MotionCategory.ENVIRONMENT,
            keywords=["gentle wind", "light breeze", "soft wind", "subtle breeze"],
            intensity=MotionIntensity.SUBTLE,
            env_effect="wind",
            parameters={"strength": 0.3}
        ),
        MotionPreset(
            name="cloud_movement",
            category=MotionCategory.ENVIRONMENT,
            keywords=["clouds moving", "cloud drift", "sky clouds", "cloudy"],
            intensity=MotionIntensity.GENTLE,
            env_effect="clouds"
        ),
        MotionPreset(
            name="water_ripple",
            category=MotionCategory.OBJECT,
            keywords=["water ripple", "rippling water", "waves", "water surface"],
            intensity=MotionIntensity.MODERATE,
            object_motion="water"
        ),
        MotionPreset(
            name="hair_flow",
            category=MotionCategory.OBJECT,
            keywords=["hair flowing", "flowing hair", "wind in hair", "hair movement"],
            intensity=MotionIntensity.MODERATE,
            object_motion="hair"
        ),
        MotionPreset(
            name="foliage_sway",
            category=MotionCategory.OBJECT,
            keywords=["leaves swaying", "foliage", "trees swaying", "plants moving"],
            intensity=MotionIntensity.GENTLE,
            object_motion="foliage"
        ),
        MotionPreset(
            name="parallax",
            category=MotionCategory.CAMERA,
            keywords=["parallax", "depth effect", "3d motion", "layered depth"],
            intensity=MotionIntensity.SUBTLE,
            camera_type="parallax"
        ),
        MotionPreset(
            name="subtle_shake",
            category=MotionCategory.CAMERA,
            keywords=["subtle shake", "gentle shake", "camera shake", "shaky cam"],
            intensity=MotionIntensity.SUBTLE,
            camera_type="shake"
        ),
        MotionPreset(
            name="tilt",
            category=MotionCategory.CAMERA,
            keywords=["tilt", "camera tilt", "vertical pan", "upward tilt"],
            intensity=MotionIntensity.MODERATE,
            camera_type="tilt"
        ),
    ]
    
    INTENSITY_MAP: Dict[str, MotionIntensity] = {
        "subtle": MotionIntensity.SUBTLE,
        "gentle": MotionIntensity.GENTLE,
        "light": MotionIntensity.SUBTLE,
        "soft": MotionIntensity.GENTLE,
        "mild": MotionIntensity.GENTLE,
        "moderate": MotionIntensity.MODERATE,
        "normal": MotionIntensity.MODERATE,
        "steady": MotionIntensity.MODERATE,
        "dramatic": MotionIntensity.DRAMATIC,
        "intense": MotionIntensity.INTENSE,
        "strong": MotionIntensity.INTENSE,
        "vigorous": MotionIntensity.INTENSE,
        "fast": MotionIntensity.INTENSE,
        "energetic": MotionIntensity.INTENSE,
    }
    
    def __init__(self):
        self.presets = self.PRESETS.copy()
    
    def parse(self, prompt: str) -> MotionParameters:
        """
        Parse a motion prompt into parameters.
        
        Args:
            prompt: User's motion description
            
        Returns:
            MotionParameters with extracted motion settings
        """
        prompt_lower = prompt.lower().strip()
        
        params = MotionParameters(raw_prompt=prompt)
        
        params.intensity = self._extract_intensity(prompt_lower)
        params.strength = self._intensity_to_strength(params.intensity)
        
        matched_preset = self._find_matching_preset(prompt_lower)
        
        if matched_preset:
            params.category = matched_preset.category
            params.primary_motion = matched_preset.name
            params.parameters = matched_preset.parameters.copy()
            
            if matched_preset.camera_type:
                params.parameters["camera_type"] = matched_preset.camera_type
            if matched_preset.furry_motion:
                params.parameters["furry_motion"] = matched_preset.furry_motion
            if matched_preset.env_effect:
                params.parameters["env_effect"] = matched_preset.env_effect
            if matched_preset.object_motion:
                params.parameters["object_motion"] = matched_preset.object_motion
        else:
            params = self._parse_fallback(prompt_lower, params)
        
        params.speed = self._extract_speed(prompt_lower)
        params.duration_factor = self._extract_duration(prompt_lower)
        
        params.secondary_motions = self._extract_secondary_motions(prompt_lower)
        
        return params
    
    def _extract_intensity(self, prompt: str) -> MotionIntensity:
        """Extract intensity level from prompt."""
        for word, intensity in self.INTENSITY_MAP.items():
            if word in prompt:
                return intensity
        
        if any(w in prompt for w in ["very", "extremely", "super"]):
            return MotionIntensity.INTENSE
        elif any(w in prompt for w in ["slightly", "barely", "minimal"]):
            return MotionIntensity.SUBTLE
        
        return MotionIntensity.MODERATE
    
    def _intensity_to_strength(self, intensity: MotionIntensity) -> float:
        """Convert intensity enum to numeric strength."""
        strength_map = {
            MotionIntensity.SUBTLE: 0.3,
            MotionIntensity.GENTLE: 0.5,
            MotionIntensity.MODERATE: 0.7,
            MotionIntensity.DRAMATIC: 0.85,
            MotionIntensity.INTENSE: 1.0,
        }
        return strength_map.get(intensity, 0.7)
    
    def _find_matching_preset(self, prompt: str) -> Optional[MotionPreset]:
        """Find best matching preset."""
        best_match = None
        best_score = 0
        
        for preset in self.presets:
            score = 0
            for keyword in preset.keywords:
                if keyword in prompt:
                    score = max(score, len(keyword))
            
            if score > best_score:
                best_score = score
                best_match = preset
        
        return best_match
    
    def _parse_fallback(self, prompt: str, params: MotionParameters) -> MotionParameters:
        """Parse unknown prompts using keyword extraction."""
        if any(w in prompt for w in ["zoom", "zoom in", "zoom out"]):
            params.category = MotionCategory.CAMERA
            params.primary_motion = "zoom"
            params.parameters["camera_type"] = "zoom"
        elif any(w in prompt for w in ["pan", "sweep"]):
            params.category = MotionCategory.CAMERA
            params.primary_motion = "pan"
            params.parameters["camera_type"] = "pan"
        elif any(w in prompt for w in ["orbit", "spin", "rotate"]):
            params.category = MotionCategory.CAMERA
            params.primary_motion = "orbital"
            params.parameters["camera_type"] = "orbital"
        elif any(w in prompt for w in ["tail", "wag"]):
            params.category = MotionCategory.FURRY
            params.primary_motion = "tail"
            params.parameters["furry_motion"] = "tail_wag"
        elif any(w in prompt for w in ["ear", "alert", "perk"]):
            params.category = MotionCategory.FURRY
            params.primary_motion = "ears"
            params.parameters["furry_motion"] = "ears"
        elif any(w in prompt for w in ["wind", "breeze", "gust"]):
            params.category = MotionCategory.ENVIRONMENT
            params.primary_motion = "wind"
            params.parameters["env_effect"] = "wind"
        elif any(w in prompt for w in ["breath", "breathe"]):
            params.category = MotionCategory.FURRY
            params.primary_motion = "breathing"
            params.parameters["furry_motion"] = "breathing"
        elif any(w in prompt for w in ["water", "wave", "ripple"]):
            params.category = MotionCategory.OBJECT
            params.primary_motion = "water"
            params.parameters["object_motion"] = "water"
        elif any(w in prompt for w in ["hair", "flow"]):
            params.category = MotionCategory.OBJECT
            params.primary_motion = "hair"
            params.parameters["object_motion"] = "hair"
        elif any(w in prompt for w in ["cloud", "sky"]):
            params.category = MotionCategory.ENVIRONMENT
            params.primary_motion = "clouds"
            params.parameters["env_effect"] = "clouds"
        else:
            params.category = MotionCategory.CAMERA
            params.primary_motion = "auto"
        
        return params
    
    def _extract_speed(self, prompt: str) -> float:
        """Extract speed modifier."""
        speed_map = {
            "very slow": 0.5,
            "slow": 0.7,
            "moderate speed": 1.0,
            "normal speed": 1.0,
            "fast": 1.5,
            "very fast": 2.0,
            "quick": 1.3,
            "rapid": 1.8,
            "gentle": 0.6,
            "deliberate": 0.8,
        }
        
        for phrase, speed in speed_map.items():
            if phrase in prompt:
                return speed
        
        if any(w in prompt for w in ["quick", "rapid", "fast"]):
            return 1.5
        elif any(w in prompt for w in ["slow", "gentle", "soft"]):
            return 0.7
        
        return 1.0
    
    def _extract_duration(self, prompt: str) -> float:
        """Extract duration factor."""
        if "long" in prompt or "extended" in prompt:
            return 1.5
        elif "short" in prompt or "brief" in prompt:
            return 0.7
        return 1.0
    
    def _extract_secondary_motions(self, prompt: str) -> List[str]:
        """Extract additional motion requests."""
        motions = []
        
        if "with" in prompt:
            after_with = prompt.split("with")[-1]
            
            if "wind" in after_with:
                motions.append("wind")
            if "cloud" in after_with:
                motions.append("clouds")
            if "breath" in after_with:
                motions.append("breathing")
            if "shake" in after_with:
                motions.append("shake")
        
        return motions
    
    def add_preset(self, preset: MotionPreset) -> None:
        """Add a custom motion preset."""
        self.presets.append(preset)
    
    def get_available_motions(self) -> List[str]:
        """Get list of all available motion descriptions."""
        motions = []
        for preset in self.presets:
            motions.extend(preset.keywords)
        return sorted(set(motions))


class MotionPromptMapper:
    """
    Maps parsed motion parameters to generator configurations.
    
    Converts MotionParameters into actual generator settings
    for CameraMotionGenerator, FurryMotionGenerator, etc.
    """
    
    CAMERA_TYPE_MAP: Dict[str, str] = {
        "pan": "pan",
        "zoom": "dolly",
        "orbital": "orbital",
        "tilt": "tilt",
        "shake": "shake",
        "parallax": "parallax",
        "dolly": "dolly",
    }
    
    def __init__(self):
        self.parser = MotionPromptParser()
    
    def parse_and_configure(
        self,
        prompt: str
    ) -> Tuple[MotionParameters, Dict[str, Any]]:
        """
        Parse prompt and return configuration for generators.
        
        Args:
            prompt: Motion description
            
        Returns:
            Tuple of (MotionParameters, generator_config)
        """
        params = self.parser.parse(prompt)
        config = self._build_generator_config(params)
        
        return params, config
    
    def _build_generator_config(
        self,
        params: MotionParameters
    ) -> Dict[str, Any]:
        """Build generator configuration from parameters."""
        config = {
            "strength": params.strength,
            "speed": params.speed,
            "duration_factor": params.duration_factor,
            "secondary_motions": params.secondary_motions,
        }
        
        if params.category == MotionCategory.CAMERA:
            camera_type = params.parameters.get("camera_type", "auto")
            config["motion_mode"] = self.CAMERA_TYPE_MAP.get(camera_type, camera_type)
            config["camera_config"] = {
                "trajectory": camera_type,
                "intensity": params.strength,
            }
        
        elif params.category == MotionCategory.FURRY:
            furry_motion = params.parameters.get("furry_motion", "auto")
            config["furry_config"] = {
                "motion_type": furry_motion,
                "intensity": params.strength,
                "speed": params.speed,
            }
        
        elif params.category == MotionCategory.ENVIRONMENT:
            env_effect = params.parameters.get("env_effect", "auto")
            config["env_config"] = {
                "effect": env_effect,
                "strength": params.strength,
            }
        
        elif params.category == MotionCategory.OBJECT:
            object_motion = params.parameters.get("object_motion", "auto")
            config["object_config"] = {
                "motion_type": object_motion,
                "strength": params.strength,
            }
        
        elif params.category == MotionCategory.CHARACTER:
            config["character_config"] = {
                "motion_type": params.primary_motion,
                "intensity": params.strength,
            }
        
        return config
    
    def get_generator_kwargs(
        self,
        params: MotionParameters
    ) -> Dict[str, Any]:
        """Get kwargs for motion generator."""
        kwargs = {
            "strength": params.strength,
        }
        
        if "camera_type" in params.parameters:
            kwargs["trajectory"] = params.parameters["camera_type"]
        
        if "furry_motion" in params.parameters:
            kwargs["motion_type"] = params.parameters["furry_motion"]
        
        if "env_effect" in params.parameters:
            kwargs["effect"] = params.parameters["env_effect"]
        
        if "object_motion" in params.parameters:
            kwargs["motion_type"] = params.parameters["object_motion"]
        
        return kwargs


def describe_motion_influence():
    """
    Documentation: How text prompts influence motion behavior.
    
    Text Prompt Structure:
    -----------------------
    A motion prompt typically contains:
    
    1. INTENSITY (optional prefix)
       Words like "subtle", "dramatic", "gentle" set the motion amplitude.
       - "subtle" -> strength 0.3, minimal displacement
       - "dramatic" -> strength 0.85, large displacement
       - "intense" -> strength 1.0, maximum effect
    
    2. PRIMARY MOTION (core concept)
       The main motion type described in plain language.
       - "zoom" -> CameraMotion with zoom trajectory
       - "tail wag" -> FurryMotion with tail oscillation
       - "wind blowing" -> EnvironmentalMotion with wind field
    
    3. MODIFIERS (optional)
       Additional descriptors that refine behavior.
       - Speed: "slow", "fast", "rapid"
       - Duration: "extended", "brief"
       - Style: "smooth", "jerky", "natural"
    
    4. COMBINATIONS (using "with")
       Multiple motions can be combined.
       - "zoom with wind" -> Camera zoom + wind effect
       - "tail wag with breathing" -> Combined furry motions
    
    Text Influence Mapping:
    -----------------------
    "dramatic zoom" ->
        category: CAMERA
        intensity: DRAMATIC
        camera_type: zoom
        strength: 0.85
        speed: 1.0
    
    "gentle tail wagging" ->
        category: FURRY
        intensity: GENTLE
        furry_motion: tail_wag
        strength: 0.5
        speed: 0.8
    
    "subtle wind blowing with clouds" ->
        category: ENVIRONMENT
        intensity: SUBTLE
        env_effect: wind
        strength: 0.3
        secondary_motions: ["clouds"]
    
    Parameter Effects:
    ------------------
    - strength (0.0-1.0): Controls displacement magnitude
    - speed (0.5-2.0): Controls motion frequency
    - duration_factor: Extends or shortens motion cycle
    
    Categories map to generators:
    - CAMERA -> CameraMotionGenerator (pan, zoom, orbital)
    - FURRY -> FurryMotionGenerator (tail, ears, breathing)
    - ENVIRONMENT -> EnvironmentalMotionGenerator (wind, clouds, fog)
    - OBJECT -> ObjectMotionGenerator (hair, water, foliage)
    """
    pass


if __name__ == "__main__":
    parser = MotionPromptParser()
    
    test_prompts = [
        "cinematic camera pan",
        "subtle breathing motion",
        "wind blowing",
        "dramatic zoom",
        "gentle tail wag",
        "energetic tail wagging with wind",
        "slow orbital rotation",
        "water ripples with gentle breeze",
    ]
    
    print("=== Motion Prompt Parser Demo ===\n")
    
    mapper = MotionPromptMapper()
    
    for prompt in test_prompts:
        params, config = mapper.parse_and_configure(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"  Category: {params.category.value}")
        print(f"  Intensity: {params.intensity.value} (strength={params.strength})")
        print(f"  Speed: {params.speed}x")
        print(f"  Primary: {params.primary_motion}")
        if params.secondary_motions:
            print(f"  Secondary: {params.secondary_motions}")
        print(f"  Config: {config}")
        print()