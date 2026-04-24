"""Artifact reduction system for high-quality video synthesis.

This module provides comprehensive artifact reduction through:
- Depth conditioning (MiDaS, ZoeDepth)
- ControlNet guidance (depth, edges, pose)
- Latent consistency management
- Optical flow stabilization
- Frame interpolation (RIFE/FILM)

No reliance on filtering or censorship - uses architectural solutions.

METHODS EXPLAINED:

1. DEPTH CONDITIONING
   - Reduces: Flickering, warping, face distortion, structural inconsistency
   - How: Uses depth maps (MiDaS/ZoeDepth) to guide latent warping
   - Per-layer motion modulation based on depth
   - Tradeoffs:
     * Quality vs Speed: Ensemble is most accurate but slowest
     * ZoeDepth: Fast, good metric accuracy (RECOMMENDED)
     * MiDaS: General purpose, moderate speed
     * Marigold: Highest quality, slowest
     * Ensemble: Combines all for best quality but 3x slower

2. CONTROLNET GUIDANCE
   - Reduces: Face distortion, structural inconsistency, warping
   - How: Uses ControlNet conditioning (depth/edges/pose) to preserve structure
   - Maintains 3D geometry and object boundaries
   - Tradeoffs:
     * Depth: Good balance of quality and speed
     * Canny Edge: More precise but requires edge detection
     * Pose: Best for character preservation
     * Soft Edge: Faster, less precise

3. LATENT CONSISTENCY
   - Reduces: Flickering, temporal inconsistency
   - How: Cross-frame attention ensures frames attend to neighbors
   - Trajectory smoothing for natural motion
   - Identity preservation using reference latent
   - Tradeoffs:
     * Cross Attention: High quality, moderate speed
     * Simple Smoothing: Fast, lower quality
     * Full Pipeline: Best coherence but slower

4. OPTICAL FLOW STABILIZATION
   - Reduces: Camera shake, motion warping, motion anomalies
   - How: Estimates optical flow, smooths motion trajectory
   - Detects and corrects anomalous motion
   - Tradeoffs:
     * RAFT: High accuracy, moderate speed
     * GMFlow: Transformer-based, best quality
     * Farneback: CPU-friendly, faster but lower accuracy
     * Trajectory smoothing: Best for camera shake

5. FRAME INTERPOLATION
   - Reduces: Motion judder, frame gaps
   - How: Generates intermediate frames between keyframes
   - Uses RIFE/FILM for motion-compensated interpolation
   - Tradeoffs:
     * RIFE: Best balance of quality and speed (RECOMMENDED)
     * FILM: Better for large motions, slower
     * CAIN: Good for occlusion handling
     * Generic: Fast fallback, lower quality
"""

from .depth_conditioning import DepthConditioner, DepthConsistencyLoss
from .controlnet_guidance import ControlNetGuidance, ControlNetType
from .latent_consistency import LatentConsistencyManager, ConsistencyMetric
from .optical_flow_stabilizer import OpticalFlowStabilizer, StabilizationConfig
from .frame_interpolator import FrameInterpolator, InterpolationMethod
from .artifact_reducer import ArtifactReducer, ArtifactConfig

__all__ = [
    "DepthConditioner",
    "DepthConsistencyLoss",
    "ControlNetGuidance",
    "ControlNetType",
    "LatentConsistencyManager",
    "ConsistencyMetric",
    "OpticalFlowStabilizer",
    "StabilizationConfig",
    "FrameInterpolator",
    "InterpolationMethod",
    "ArtifactReducer",
    "ArtifactConfig",
]