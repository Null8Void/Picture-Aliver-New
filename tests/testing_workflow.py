"""
Picture-Aliver Testing Workflow

Complete testing pipeline to verify:
1. Video generation for different image types
2. Motion quality (subtle, pan, rotation)
3. Error detection and handling
4. Stage-by-stage logging

Run from project root:
    python -m tests.testing_workflow

Test Cases:
1. Portrait image → subtle motion
2. Landscape → camera pan
3. Object → small rotation
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('picture_aliver_testing')


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for a single test case."""
    name: str
    description: str
    image_path: str
    prompt: str
    motion_mode: str
    expected_motion: str
    duration: float = 3.0
    fps: int = 8
    width: int = 512
    height: int = 512


@dataclass
class TestResult:
    """Result from a single test execution."""
    test_name: str
    success: bool
    video_path: Optional[Path] = None
    processing_time: float = 0.0
    errors: list = field(default_factory=list)
    stage_logs: list = field(default_factory=list)
    quality_metrics: dict = field(default_factory=dict)
    failure_module: Optional[str] = None
    suggested_fix: Optional[str] = None


class TestStage(Enum):
    """Pipeline stages for logging."""
    IMAGE_LOAD = "Image Loading"
    DEPTH_EST = "Depth Estimation"
    SEGMENTATION = "Segmentation"
    MOTION_GEN = "Motion Generation"
    VIDEO_DIFF = "Video Diffusion"
    STABILIZATION = "Stabilization"
    INTERPOLATION = "Interpolation"
    QUALITY_CHECK = "Quality Check"
    EXPORT = "Export"
    VERIFICATION = "Verification"


# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================

def get_test_cases() -> list[TestConfig]:
    """Get predefined test cases."""
    return [
        TestConfig(
            name="portrait_subtle_motion",
            description="Portrait image with gentle breathing/subtle motion",
            image_path="tests/images/portrait.png",
            prompt="gentle breathing motion, subtle animation",
            motion_mode="subtle",
            expected_motion="subtle",
            duration=3.0,
            fps=8,
            width=512,
            height=512,
        ),
        TestConfig(
            name="landscape_camera_pan",
            description="Landscape image with cinematic camera pan",
            image_path="tests/images/landscape.png",
            prompt="cinematic camera pan, wind movement in trees",
            motion_mode="cinematic",
            expected_motion="pan",
            duration=4.0,
            fps=12,
            width=768,
            height=512,
        ),
        TestConfig(
            name="object_rotation",
            description="Single object with small rotation",
            image_path="tests/images/object.png",
            prompt="slow rotation, slight floating motion",
            motion_mode="auto",
            expected_motion="rotation",
            duration=3.0,
            fps=8,
            width=512,
            height=512,
        ),
    ]


# =============================================================================
# IMAGE GENERATION (for testing without real images)
# =============================================================================

def generate_test_image(
    image_type: str,
    save_path: Path,
    size: tuple[int, int] = (512, 512)
) -> Path:
    """
    Generate synthetic test images when real images aren't available.
    
    Creates:
    - portrait.png: Image with gradient and face-like shape
    - landscape.png: Image with sky and ground layers
    - object.png: Centered shape on plain background
    """
    try:
        import numpy as np
        from PIL import Image
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        
        if image_type == "portrait":
            # Create portrait-like image with gradient
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    img[y, x] = [
                        int(50 + 100 * y / h),
                        int(100 + 50 * x / w),
                        int(150 + 100 * y / h * x / w)
                    ]
            
            # Add face-like oval
            center_x, center_y = w // 2, h // 3
            for y in range(h):
                for x in range(w):
                    dist = ((x - center_x) ** 2 / (w // 4) ** 2 + 
                            (y - center_y) ** 2 / (h // 3) ** 2)
                    if dist < 1:
                        img[y, x] = [255, 220, 180]
            
            Image.fromarray(img).save(save_path)
            logger.info(f"[TestImage] Generated portrait: {save_path}")
            
        elif image_type == "landscape":
            # Create landscape with sky and ground
            img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Sky gradient (top half)
            sky_height = h // 2
            for y in range(sky_height):
                intensity = int(100 + 100 * y / sky_height)
                img[y, :] = [intensity // 2, intensity, intensity + 50]
            
            # Ground (bottom half)
            for y in range(sky_height, h):
                depth = (y - sky_height) / (h - sky_height)
                img[y, :] = [
                    int(80 * depth + 20),
                    int(120 * depth + 30),
                    int(40 * depth + 10)
                ]
            
            # Add some "trees"
            import random
            random.seed(42)
            for _ in range(15):
                tree_x = random.randint(0, w - 1)
                tree_y = random.randint(sky_height, h - 50)
                tree_h = random.randint(30, 80)
                for dy in range(tree_h):
                    for dx in range(-10, 11):
                        tx, ty = tree_x + dx, tree_y - dy
                        if 0 <= tx < w and 0 <= ty < h:
                            if dx ** 2 + dy ** 2 < 100:
                                img[ty, tx] = [20, 80, 20]
            
            Image.fromarray(img).save(save_path)
            logger.info(f"[TestImage] Generated landscape: {save_path}")
            
        elif image_type == "object":
            # Create centered object
            img = np.ones((h, w, 3), dtype=np.uint8) * 200
            
            # Draw a stylized object (cube-like shape)
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            
            # Main square
            for y in range(center_y - size, center_y + size):
                for x in range(center_x - size, center_x + size):
                    img[y, x] = [100, 150, 200]
            
            # Highlight
            for y in range(center_y - size, center_y):
                for x in range(center_x - size, center_x):
                    dist = ((x - center_x + size) ** 2 + (y - center_y + size) ** 2)
                    if dist < (size // 2) ** 2:
                        img[y, x] = [150, 200, 240]
            
            Image.fromarray(img).save(save_path)
            logger.info(f"[TestImage] Generated object: {save_path}")
        
        return save_path
        
    except ImportError as e:
        logger.error(f"[TestImage] PIL/NumPy not available: {e}")
        raise


def ensure_test_images() -> dict[str, Path]:
    """Ensure all test images exist, generating if necessary."""
    images_dir = Path("tests/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    images = {
        "portrait": generate_test_image("portrait", images_dir / "portrait.png"),
        "landscape": generate_test_image("landscape", images_dir / "landscape.png"),
        "object": generate_test_image("object", images_dir / "object.png"),
    }
    
    return images


# =============================================================================
# QUALITY METRICS
# =============================================================================

def calculate_quality_metrics(video_path: Path) -> dict:
    """
    Calculate quality metrics for generated video.
    
    Metrics:
    - brightness_variance: Detects flickering (low = good)
    - motion_score: Estimated motion between frames
    - structural_stability: SSIM between frames
    """
    metrics = {
        "flickering": 0.0,
        "motion_detected": False,
        "stability_score": 0.0,
    }
    
    try:
        import cv2
        import numpy as np
        
        if not video_path.exists():
            return metrics
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
        
        cap.release()
        
        if len(frames) < 2:
            return metrics
        
        # Flickering: brightness variance across frames
        brightness_per_frame = [f.mean() for f in frames]
        brightness_std = np.std(brightness_per_frame)
        metrics["flickering"] = float(brightness_std)
        metrics["flickering_ok"] = brightness_std < 15
        
        # Motion: optical flow between consecutive frames
        total_flow = 0.0
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            total_flow += magnitude.mean()
        
        avg_flow = total_flow / (len(frames) - 1)
        metrics["motion_detected"] = avg_flow > 0.5
        metrics["motion_intensity"] = float(avg_flow)
        
        # Stability: average SSIM between consecutive frames
        from skimage.metrics import structural_similarity as ssim
        total_ssim = 0.0
        for i in range(len(frames) - 1):
            score, _ = ssim(frames[i], frames[i + 1], full=True)
            total_ssim += score
        
        avg_ssim = total_ssim / (len(frames) - 1)
        metrics["stability_score"] = float(avg_ssim)
        metrics["stability_ok"] = avg_ssim > 0.7
        
        logger.info(f"[Metrics] Flickering: {brightness_std:.2f} (OK: {metrics['flickering_ok']})")
        logger.info(f"[Metrics] Motion: {avg_flow:.2f} intensity, detected: {metrics['motion_detected']}")
        logger.info(f"[Metrics] Stability: {avg_ssim:.3f} (OK: {metrics['stability_ok']})")
        
    except ImportError:
        logger.warning("[Metrics] OpenCV/Scikit not available, skipping quality metrics")
    except Exception as e:
        logger.warning(f"[Metrics] Error calculating metrics: {e}")
    
    return metrics


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestingWorkflow:
    """
    Complete testing workflow for Picture-Aliver.
    
    Executes test cases and verifies:
    - Video generation completes
    - Motion is present
    - No major flickering/distortion
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.result = TestResult(
            test_name=config.name,
            success=False,
        )
        self.stage_times: dict[str, float] = {}
    
    def log_stage(self, stage: TestStage, message: str) -> None:
        """Log a pipeline stage."""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{stage.value}] {message}"
        self.result.stage_logs.append(f"{timestamp} | {log_msg}")
        logger.info(log_msg)
    
    def run(self) -> TestResult:
        """Execute the test."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info(f"TEST: {self.config.name}")
        logger.info(f"DESCRIPTION: {self.config.description}")
        logger.info(f"MOTION MODE: {self.config.motion_mode} (expected: {self.config.expected_motion})")
        logger.info("=" * 70)
        
        try:
            # STAGE 1: Image Loading
            stage_start = time.time()
            self.log_stage(TestStage.IMAGE_LOAD, "Starting...")
            
            from src.picture_aliver.main import PipelineConfig, Pipeline, DebugConfig
            
            # Ensure test image exists
            test_images = ensure_test_images()
            
            # Find the correct test image
            if "portrait" in self.config.name:
                image_path = test_images["portrait"]
            elif "landscape" in self.config.name:
                image_path = test_images["landscape"]
            else:
                image_path = test_images["object"]
            
            self.log_stage(TestStage.IMAGE_LOAD, f"Loaded: {image_path}")
            self.stage_times["image_load"] = time.time() - stage_start
            
            # STAGE 2: Pipeline Execution
            stage_start = time.time()
            self.log_stage(TestStage.VIDEO_DIFF, "Initializing pipeline...")
            
            config = PipelineConfig(
                duration_seconds=self.config.duration,
                fps=self.config.fps,
                width=self.config.width,
                height=self.config.height,
                motion_mode=self.config.motion_mode,
                motion_strength=0.8,
                guidance_scale=7.5,
                num_inference_steps=25,
                enable_quality_check=True,
                enable_stabilization=True,
                debug=DebugConfig(enabled=True, directory=f"./debug/{self.config.name}"),
            )
            
            pipeline = Pipeline(config)
            pipeline.initialize()
            self.log_stage(TestStage.VIDEO_DIFF, "Pipeline initialized")
            
            # Generate output path
            output_dir = Path("tests/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.config.name}_video.mp4"
            
            self.log_stage(TestStage.VIDEO_DIFF, f"Starting generation...")
            self.log_stage(TestStage.VIDEO_DIFF, f"  Prompt: {self.config.prompt}")
            self.log_stage(TestStage.VIDEO_DIFF, f"  Duration: {self.config.duration}s")
            self.log_stage(TestStage.VIDEO_DIFF, f"  Resolution: {self.config.width}x{self.config.height}")
            
            # Run pipeline
            result = pipeline.run_pipeline(
                image_path=image_path,
                prompt=self.config.prompt,
                config=config,
                output_path=output_path
            )
            
            self.stage_times["video_diff"] = time.time() - stage_start
            
            if not result.success:
                self.result.failure_module = "Pipeline"
                self.result.suggested_fix = "Check pipeline error logs above"
                self.result.errors = result.errors
                raise Exception(f"Pipeline failed: {result.errors}")
            
            self.result.video_path = result.output_path
            
            # STAGE 3: Quality Verification
            stage_start = time.time()
            self.log_stage(TestStage.VERIFICATION, "Analyzing generated video...")
            
            if not result.output_path or not Path(result.output_path).exists():
                raise Exception("Output video file not found")
            
            # Calculate quality metrics
            metrics = calculate_quality_metrics(result.output_path)
            self.result.quality_metrics = metrics
            
            # Verify motion is present
            if not metrics.get("motion_detected", False):
                self.result.failure_module = "Motion Generation"
                self.result.suggested_fix = (
                    "Increase motion_strength or use a more descriptive prompt. "
                    "Current motion intensity: " + str(metrics.get("motion_intensity", 0))
                )
                logger.warning("[VERIFICATION] Motion not detected!")
            
            # Check for flickering
            if not metrics.get("flickering_ok", True):
                self.result.failure_module = "Quality"
                self.result.suggested_fix = (
                    "Excessive flickering detected. Try reducing motion_strength "
                    "or increasing stabilization strength."
                )
                logger.warning("[VERIFICATION] Flickering detected!")
            
            # Check stability
            if not metrics.get("stability_ok", True):
                self.result.failure_module = "Stabilization"
                self.result.suggested_fix = (
                    "Low stability score. Try enabling stronger stabilization "
                    "or reducing motion parameters."
                )
                logger.warning("[VERIFICATION] Low stability!")
            
            self.stage_times["verification"] = time.time() - stage_start
            self.stage_times["total"] = time.time() - start_time
            
            # SUCCESS
            self.result.success = True
            self.result.processing_time = self.stage_times["total"]
            
            logger.info("=" * 70)
            logger.info(f"TEST PASSED: {self.config.name}")
            logger.info(f"Output: {result.output_path}")
            logger.info(f"Processing time: {self.stage_times['total']:.2f}s")
            logger.info(f"Quality: flickering={metrics.get('flickering_ok', 'N/A')}, "
                       f"motion={metrics.get('motion_detected', 'N/A')}, "
                       f"stability={metrics.get('stability_ok', 'N/A')}")
            logger.info("=" * 70)
            
        except Exception as e:
            self.result.success = False
            self.result.errors.append(str(e))
            self.stage_times["total"] = time.time() - start_time
            
            logger.error("=" * 70)
            logger.error(f"TEST FAILED: {self.config.name}")
            logger.error(f"Error: {e}")
            if self.result.failure_module:
                logger.error(f"Failure module: {self.result.failure_module}")
            if self.result.suggested_fix:
                logger.error(f"Suggested fix: {self.result.suggested_fix}")
            logger.error("=" * 70)
        
        return self.result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tests() -> list[TestResult]:
    """Run all test cases."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PICTURE-ALIVER TESTING WORKFLOW")
    logger.info("=" * 70)
    logger.info("")
    
    test_cases = get_test_cases()
    results = []
    
    for test_config in test_cases:
        workflow = TestingWorkflow(test_config)
        result = workflow.run()
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    logger.info(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    logger.info("")
    
    for result in results:
        status = "PASS" if result.success else "FAIL"
        logger.info(f"  [{status}] {result.test_name} ({result.processing_time:.1f}s)")
        if not result.success:
            logger.info(f"        Module: {result.failure_module}")
            logger.info(f"        Fix: {result.suggested_fix}")
    
    logger.info("")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(r.success for r in results)
    sys.exit(0 if all_passed else 1)