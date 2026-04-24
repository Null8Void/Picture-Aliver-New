"""Motion generation system module index.

This module provides comprehensive motion generation for coherent video synthesis:

CAMERA MOTION (camera_motion.py):
- CameraTrajectory: Full camera path data structure
- CameraMotionGenerator: Pan, zoom, tilt, dolly, orbital, cinematic motion
- CameraMotionConfig: Configuration for camera motion
- CameraMode: Enum for camera motion types

OBJECT MOTION (object_motion.py):
- ObjectMotion: Motion data for object regions
- MotionRegion: Defined regions with motion properties
- ObjectMotionGenerator: Hair, cloth, foliage, water, ripple effects
- ObjectMotionType: Types of object motion

ENVIRONMENTAL MOTION (env_motion.py):
- EnvironmentalEffect: Environmental effect with flow field
- EnvironmentalMotionGenerator: Wind, lighting, sky, clouds, fog, particles
- EnvironmentalType: Types of environmental effects

DEPTH PARALLAX (parallax.py):
- ParallaxConfig: Configuration for depth-based parallax
- DepthLayer: Represents a depth layer in parallax system
- DepthParallaxGenerator: Creates parallax from depth maps

MOTION INJECTION (motion_injector.py):
- MotionConditioning: Motion data for diffusion conditioning
- MotionInjector: Injects motion into video diffusion models
- ConditioningType: Types of motion conditioning
- MotionInjectionStrategy: Injection strategies

SUPPORT FLOW ESTIMATION (flow_estimator.py):
- FlowEstimator: Optical flow estimation (RAFT, GMFlow, Farneback)
- FlowField: Flow field data structure
- MotionTrajectory: Tracked point trajectories
- MotionMagnitude: Motion statistics
"""

from .camera_motion import (
    CameraMotionGenerator,
    CameraMotionConfig,
    CameraTrajectory,
    CameraParams,
    CameraMode,
)

from .object_motion import (
    ObjectMotionGenerator,
    ObjectMotionConfig,
    ObjectMotion,
    MotionRegion,
    ObjectMotionType,
    MotionDirection,
)

from .env_motion import (
    EnvironmentalMotionGenerator,
    EnvironmentalMotionConfig,
    EnvironmentalEffect,
    EnvironmentalType,
)

from .parallax import (
    DepthParallaxGenerator,
    ParallaxConfig,
    DepthLayer,
)

from .motion_injector import (
    MotionInjector,
    MotionConditioning,
    ConditioningType,
    MotionInjectionStrategy,
)

from .flow_estimator import FlowEstimator
from .types import FlowField, MotionTrajectory, MotionMagnitude


__all__ = [
    # Camera Motion
    "CameraMotionGenerator",
    "CameraMotionConfig",
    "CameraTrajectory",
    "CameraParams",
    "CameraMode",

    # Object Motion
    "ObjectMotionGenerator",
    "ObjectMotionConfig",
    "ObjectMotion",
    "MotionRegion",
    "ObjectMotionType",
    "MotionDirection",

    # Environmental Motion
    "EnvironmentalMotionGenerator",
    "EnvironmentalMotionConfig",
    "EnvironmentalEffect",
    "EnvironmentalType",

    # Parallax
    "DepthParallaxGenerator",
    "ParallaxConfig",
    "DepthLayer",

    # Motion Injection
    "MotionInjector",
    "MotionConditioning",
    "ConditioningType",
    "MotionInjectionStrategy",

    # Flow Estimation (existing)
    "FlowEstimator",
    "FlowField",
    "MotionTrajectory",
    "MotionMagnitude",
]


# ============================================================================
# MOTION GENERATION SYSTEM OVERVIEW
# ============================================================================
"""
The motion generation system creates coherent, natural-looking motion for
image-to-video conversion through a hierarchical approach:

1. CAMERA MOTION (Primary Motion)
   - Defines global camera movement through the scene
   - Types: Pan, tilt, zoom, dolly, orbital, cinematic, subtle
   - Outputs: CameraTrajectory with per-frame transforms
   - Used for: Global motion that affects entire frame

2. OBJECT MOTION (Secondary Motion)
   - Adds realistic motion to specific regions
   - Types: Hair flowing, cloth movement, foliage rustling
   - Uses depth/segmentation for region identification
   - Used for: Small object movements within scene

3. ENVIRONMENTAL MOTION (Ambient Motion)
   - Adds atmospheric effects
   - Types: Wind, lighting shifts, clouds, fog, particles
   - Uses noise fields for natural variation
   - Used for: Background/atmosphere motion

4. DEPTH PARALLAX (3D Motion)
   - Creates depth-aware motion from depth maps
   - Separates scene into depth layers
   - Applies differential motion based on depth
   - Used for: Realistic 3D camera movement

5. MOTION INJECTION (Integration)
   - Injects motion into diffusion model conditioning
   - Strategies: ControlNet, attention modulation, warping
   - Combines all motion sources
   - Used for: Conditioning video diffusion models

HOW MOTION IS GENERATED:
1. User specifies motion style (cinematic, subtle, etc.)
2. CameraMotionGenerator creates base camera trajectory
3. DepthParallaxGenerator creates depth-aware flow
4. ObjectMotionGenerator adds region-specific motion
5. EnvironmentalMotionGenerator adds ambient effects
6. MotionInjector combines all sources and injects into model

HOW MOTION IS INJECTED:
1. Flow fields generated for each motion type
2. Combined into single motion field per frame
3. Converted to conditioning (ControlNet, attention, etc.)
4. Injected during diffusion sampling
5. Latent warping for smooth motion transfer

AVOIDING UNNATURAL MOVEMENT:
1. Smooth interpolation (ease-in-out curves)
2. Temporal coherence (consistent motion over time)
3. Depth-aware parallax (closer = more motion)
4. Region-aware masking (different motion per object)
5. Physics constraints (realistic hair/cloth behavior)
6. Consistency checks (maintain spatial relationships)
"""