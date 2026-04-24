"""Motion estimation module exports."""

from .flow_estimator import FlowEstimator
from .types import FlowField, MotionTrajectory, MotionMagnitude
from .generation import (
    CameraMotionGenerator,
    CameraTrajectory,
    ObjectMotionGenerator,
    ObjectMotion,
    MotionRegion,
    EnvironmentalMotionGenerator,
    EnvironmentalEffect,
    MotionInjector,
    MotionConditioning,
    DepthParallaxGenerator,
    ParallaxConfig,
)
from .physics_motion import (
    PhysicsMotionGenerator,
    PhysicsConfig,
    PhysicsParticle,
    ClothSimulator,
    HairSimulator,
    PhysicsType,
    create_physics_motion,
)
from .furry_motions import (
    FurryMotionGenerator,
    FurryMotionType,
    FurryCharacterMotion,
    create_furry_motion,
)

__all__ = [
    "FlowEstimator",
    "FlowField",
    "MotionTrajectory",
    "MotionMagnitude",
    "CameraMotionGenerator",
    "CameraTrajectory",
    "ObjectMotionGenerator",
    "ObjectMotion",
    "MotionRegion",
    "EnvironmentalMotionGenerator",
    "EnvironmentalEffect",
    "MotionInjector",
    "MotionConditioning",
    "DepthParallaxGenerator",
    "ParallaxConfig",
    "PhysicsMotionGenerator",
    "PhysicsConfig",
    "PhysicsParticle",
    "ClothSimulator",
    "HairSimulator",
    "PhysicsType",
    "create_physics_motion",
    "FurryMotionGenerator",
    "FurryMotionType",
    "FurryCharacterMotion",
    "create_furry_motion",
]