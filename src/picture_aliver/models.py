"""
Picture-Aliver Unified Model Interface

Provides a unified interface for video generation models:
- Wan 2.1 (via Diffusers)
- LightX2V (lightweight inference framework)
- Legacy (original pipeline)

Usage:
    from src.picture_aliver.models import VideoModel, create_model
    
    # Create model instance
    model = create_model("wan21")
    
    # Generate video
    result = model.generate(
        image="input.jpg",
        prompt="gentle wave animation"
    )
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("picture_aliver.models")


class ModelType(Enum):
    """Supported model types."""
    WAN21 = "wan21"
    WAN22 = "wan22"
    LIGHTX2V = "lightx2v"
    HUNYUAN = "hunyuan"
    LTX = "ltx"
    COGVIDEO = "cogvideo"
    SVD = "svd"
    ZEROSCOPE = "zeroscope"
    LEGACY = "legacy"


class GenerationStatus(Enum):
    """Generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Request for video generation."""
    image: Union[str, Path, Image.Image]
    prompt: str
    negative_prompt: str = ""
    duration: float = 3.0
    fps: int = 8
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    motion_strength: float = 0.8
    motion_mode: str = "auto"
    enable_quality_check: bool = True
    seed: int = -1


@dataclass
class GenerationResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    status: GenerationStatus = GenerationStatus.PENDING
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class ModelConfig:
    """Configuration for model."""
    model_type: ModelType = ModelType.WAN21
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    num_frames: int = 81
    guidance_scale: float = 5.0
    num_inference_steps: int = 40
    fps: int = 16
    height: int = 480
    width: int = 832
    enable_offload: bool = True
    torch_dtype: str = "bfloat16"
    negative_prompt: str = ""
    output_dir: str = "./outputs"
    device: str = "cuda"


class VideoModel:
    """
    Unified video generation model interface.
    
    Supports multiple backends with a common API.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipeline = None
        self._model = None
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._logger = logging.getLogger(f"picture_aliver.models.{config.model_type.value}")
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device
    
    def load(self) -> bool:
        """
        Load the model into memory.
        
        Returns:
            True if successful, False otherwise
        """
        if self._loaded:
            return True
            
        self._logger.info(f"Loading {self.config.model_type.value} model...")
        start_time = time.time()
        
        try:
            if self.config.model_type == ModelType.WAN21:
                success = self._load_wan21()
            elif self.config.model_type == ModelType.WAN22:
                success = self._load_wan22()
            elif self.config.model_type == ModelType.LIGHTX2V:
                success = self._load_lightx2v()
            elif self.config.model_type == ModelType.HUNYUAN:
                success = self._load_hunyuan()
            elif self.config.model_type == ModelType.LTX:
                success = self._load_ltx()
            elif self.config.model_type == ModelType.COGVIDEO:
                success = self._load_cogvideo()
            elif self.config.model_type == ModelType.SVD:
                success = self._load_svd()
            elif self.config.model_type == ModelType.ZEROSCOPE:
                success = self._load_zeroscope()
            elif self.config.model_type == ModelType.LEGACY:
                success = self._load_legacy()
            else:
                self._logger.error(f"Unknown model type: {self.config.model_type}")
                return False
                
            self._loaded = success
            elapsed = time.time() - start_time
            self._logger.info(f"Model loaded in {elapsed:.1f}s")
            return success
            
        except Exception as e:
            self._logger.exception(f"Failed to load model: {e}")
            return False
    
    def _load_wan21(self) -> bool:
        """Load Wan 2.1 model via Diffusers."""
        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.utils import export_to_video, load_image
            from transformers import CLIPVisionModel
            
            dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
            
            self._logger.info(f"Loading Wan 2.1: {self.config.model_id}")
            
            # Load pipeline
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to(self._device)
            
            # Store utilities for generate()
            self._pipeline_utils = {
                "load_image": load_image,
                "export_to_video": export_to_video,
            }
            
            self._logger.info("Wan 2.1 loaded successfully")
            return True
            
        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            return False
        except Exception as e:
            self._logger.exception(f"Wan 2.1 load failed: {e}")
            return False
    
    def _load_lightx2v(self) -> bool:
        """Load LightX2V framework."""
        try:
            from lightx2v import LightX2VPipeline
            
            self._logger.info(f"Loading LightX2V: {self.config.model_id}")
            
            # Create pipeline
            self._pipeline = LightX2VPipeline(
                model_path=self.config.model_id,
                model_cls="wan2.2_moe",  # or wan2.1
                task="i2v",
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_offload(
                    cpu_offload=True,
                    offload_granularity="block",
                    text_encoder_offload=True,
                )
            
            self._logger.info("LightX2V loaded successfully")
            return True
            
        except ImportError as e:
            self._logger.error(f"Missing dependency: {e}")
            return False
        except Exception as e:
            self._logger.exception(f"LightX2V load failed: {e}")
            return False
    
    def _load_wan22(self) -> bool:
        """Load Wan 2.2 model via Diffusers."""
        try:
            from diffusers import WanImageToVideoPipeline
            from diffusers.utils import export_to_video, load_image
            
            dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
            self._logger.info(f"Loading Wan 2.2: {self.config.model_id}")
            
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to(self._device)
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("Wan 2.2 loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"Wan 2.2 load failed: {e}")
            return False
    
    def _load_hunyuan(self) -> bool:
        """Load HunyuanVideo via Diffusers."""
        try:
            from diffusers import HunyuanVideoPipeline
            from diffusers.utils import export_to_video, load_image
            
            self._logger.info(f"Loading HunyuanVideo: {self.config.model_id}")
            
            self._pipeline = HunyuanVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("HunyuanVideo loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"HunyuanVideo load failed: {e}")
            return False
    
    def _load_ltx(self) -> bool:
        """Load LTX-Video via Diffusers."""
        try:
            from diffusers import LTXVideoPipeline
            from diffusers.utils import export_to_video, load_image
            
            self._logger.info(f"Loading LTX-Video: {self.config.model_id}")
            
            self._pipeline = LTXVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("LTX-Video loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"LTX-Video load failed: {e}")
            return False
    
    def _load_cogvideo(self) -> bool:
        """Load CogVideo via Diffusers."""
        try:
            from diffusers import CogVideoPipeline
            from diffusers.utils import export_to_video, load_image
            
            self._logger.info(f"Loading CogVideo: {self.config.model_id}")
            
            self._pipeline = CogVideoPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("CogVideo loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"CogVideo load failed: {e}")
            return False
    
    def _load_svd(self) -> bool:
        """Load SVD via Diffusers."""
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import export_to_video, load_image
            
            self._logger.info(f"Loading SVD: {self.config.model_id}")
            
            self._pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("SVD loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"SVD load failed: {e}")
            return False
    
    def _load_zeroscope(self) -> bool:
        """Load ZeroScope via Diffusers."""
        try:
            from diffusers import DiffusionPipeline
            from diffusers.utils import export_to_video, load_image
            
            self._logger.info(f"Loading ZeroScope: {self.config.model_id}")
            
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16,
            )
            
            if self.config.enable_offload:
                self._pipeline.enable_model_cpu_offload()
            
            self._pipeline_utils = {"load_image": load_image, "export_to_video": export_to_video}
            self._logger.info("ZeroScope loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"ZeroScope load failed: {e}")
            return False
    
    def _load_legacy(self) -> bool:
        """Load legacy pipeline."""
        try:
            # Add project root to path
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.picture_aliver.main import Pipeline, PipelineConfig
            
            # Create config
            config = PipelineConfig(
                duration_seconds=self.config.num_frames / self.config.fps,
                fps=self.config.fps,
                width=self.config.width,
                height=self.config.height,
                guidance_scale=self.config.guidance_scale,
            )
            
            self._pipeline = Pipeline(config)
            self._pipeline.initialize()
            
            self._logger.info("Legacy pipeline loaded successfully")
            return True
            
        except Exception as e:
            self._logger.exception(f"Legacy load failed: {e}")
            return False
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate video from image.
        
        Args:
            image: Input image (path or PIL Image)
            prompt: Animation prompt
            negative_prompt: Negative prompt (optional)
            num_frames: Number of frames (optional)
            guidance_scale: CFG scale (optional)
            num_inference_steps: Inference steps (optional)
            seed: Random seed (optional, -1 for random)
            output_path: Output path (optional)
            
        Returns:
            GenerationResult with video path or error
        """
        if not self._loaded:
            if not self.load():
                return GenerationResult(
                    success=False,
                    error="Failed to load model",
                    status=GenerationStatus.FAILED,
                )
        
        start_time = time.time()
        negative_prompt = negative_prompt or self.config.negative_prompt
        num_frames = num_frames or self.config.num_frames
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        # Generate output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"video_{timestamp}.mp4")
        
        try:
            self._logger.info(f"Generating video: {prompt}")
            self._logger.debug(f"  Image: {image}")
            self._logger.debug(f"  Frames: {num_frames}, Steps: {num_inference_steps}")
            
            if self.config.model_type == ModelType.WAN21:
                result = self._generate_wan21(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    seed, output_path
                )
            elif self.config.model_type == ModelType.WAN22:
                result = self._generate_wan21(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    seed, output_path
                )
            elif self.config.model_type == ModelType.LIGHTX2V:
                result = self._generate_lightx2v(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    seed, output_path
                )
            elif self.config.model_type in (ModelType.HUNYUAN, ModelType.LTX, ModelType.COGVIDEO, ModelType.SVD, ModelType.ZEROSCOPE):
                result = self._generate_diffusers(
                    image, prompt, negative_prompt,
                    num_frames, guidance_scale, num_inference_steps,
                    seed, output_path
                )
            elif self.config.model_type == ModelType.LEGACY:
                result = self._generate_legacy(
                    image, prompt,
                    num_frames / 16,
                    output_path
                )
            else:
                return GenerationResult(
                    success=False,
                    error=f"Unknown model type: {self.config.model_type}",
                    status=GenerationStatus.FAILED,
                )
            
            elapsed = time.time() - start_time
            result.generation_time = elapsed
            self._logger.info(f"Generation complete in {elapsed:.1f}s")
            
            return result
            
        except Exception as e:
            self._logger.exception(f"Generation failed: {e}")
            return GenerationResult(
                success=False,
                error=str(e),
                status=GenerationStatus.FAILED,
            )
    
    def _generate_wan21(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        output_path: str,
    ) -> GenerationResult:
        """Generate using Wan 2.1."""
        from diffusers.utils import load_image
        
        # Load image
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        
        # Resize for model
        aspect_ratio = image.height / image.width
        max_area = self.config.height * self.config.width
        mod_value = 8  # typical patch size
        
        height = int(np.sqrt(max_area * aspect_ratio) // mod_value * mod_value)
        width = int(np.sqrt(max_area / aspect_ratio) // mod_value * mod_value)
        image = image.resize((width, height))
        
        # Generate
        output = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        # Export
        from diffusers.utils import export_to_video
        export_to_video(output, output_path, fps=self.config.fps)
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            status=GenerationStatus.COMPLETED,
        )
    
    def _generate_lightx2v(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        output_path: str,
    ) -> GenerationResult:
        """Generate using LightX2V."""
        # Convert image to path if needed
        image_path = str(image) if isinstance(image, (str, Path)) else None
        
        if image_path and Path(image_path).exists():
            pass  # Use as-is
        elif isinstance(image, Image.Image):
            # Save temporarily
            temp_path = Path(self.config.output_dir) / "temp_input.png"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp_path)
            image_path = str(temp_path)
        
        # Generate
        self._pipeline.generate(
            seed=seed,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=output_path,
        )
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            status=GenerationStatus.COMPLETED,
        )
    
    def _generate_diffusers(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
        output_path: str,
    ) -> GenerationResult:
        """Generate using Diffusers pipeline (generic for Hunyuan, LTX, CogVideo, SVD, ZeroScope)."""
        from diffusers.utils import load_image, export_to_video
        
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
        
        aspect_ratio = image.height / image.width
        max_area = self.config.height * self.config.width
        mod_value = 8
        
        height = int(np.sqrt(max_area * aspect_ratio) // mod_value * mod_value)
        width = int(np.sqrt(max_area / aspect_ratio) // mod_value * mod_value)
        image = image.resize((width, height))
        
        output = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        export_to_video(output, output_path, fps=self.config.fps)
        
        return GenerationResult(
            success=True,
            video_path=output_path,
            status=GenerationStatus.COMPLETED,
        )
    
    def _generate_legacy(
        self,
        image: Union[str, Path],
        prompt: str,
        duration: float,
        output_path: str,
    ) -> GenerationResult:
        """Generate using legacy pipeline."""
        from src.picture_aliver.main import Pipeline, PipelineConfig
        
        config = PipelineConfig(
            duration_seconds=duration,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
            guidance_scale=self.config.guidance_scale,
        )
        
        result = self._pipeline.run_pipeline(
            image_path=str(image),
            prompt=prompt,
            config=config,
            output_path=output_path,
        )
        
        return GenerationResult(
            success=result.success,
            video_path=output_path if result.success else None,
            error=", ".join(result.errors) if result.errors else None,
            status=GenerationStatus.COMPLETED if result.success else GenerationStatus.FAILED,
        )
    
    def unload(self):
        """Unload model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        self._logger.info("Model unloaded")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.config.model_type.value,
            "loaded": self._loaded,
            "device": self._device,
            "config": {
                "model_id": self.config.model_id,
                "num_frames": self.config.num_frames,
                "guidance_scale": self.config.guidance_scale,
                "num_inference_steps": self.config.num_inference_steps,
            }
        }
    
    def __del__(self):
        """Cleanup."""
        self.unload()


def create_model(
    model_type: Union[str, ModelType] = "wan21",
    config: Optional[Dict[str, Any]] = None,
) -> VideoModel:
    """
    Create a VideoModel instance.
    
    Args:
        model_type: Model type ("wan21", "lightx2v", "legacy")
        config: Optional configuration overrides
        
    Returns:
        VideoModel instance
    """
    # Parse model type
    if isinstance(model_type, str):
        model_type = ModelType(model_type.lower())
    
    # Load config
    config = config or {}
    
    # Build model config
    model_config = ModelConfig(
        model_type=model_type,
        **config
    )
    
    return VideoModel(model_config)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract model-specific config
    model_type = config.get("primary_model", "wan21")
    model_config = config.get(model_type, {})
    
    return {
        "model_type": model_type,
        **model_config,
        "common": config.get("common", {}),
    }


def create_model_from_config(config_path: Optional[str] = None) -> VideoModel:
    """
    Create VideoModel from config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        VideoModel instance
    """
    config = load_config(config_path)
    
    model_type = config.pop("model_type", "wan21")
    common = config.pop("common", {})
    
    # Merge common settings
    for key in ["output_dir", "device", "torch_dtype"]:
        if key in common and key not in config:
            config[key] = common[key]
    
    return create_model(model_type, config)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model(model_type: str = "wan21") -> Dict[str, Any]:
    """
    Validate model installation and configuration.
    
    Args:
        model_type: Model to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "model_type": model_type,
        "available": False,
        "errors": [],
        "warnings": [],
    }
    
    # Check Python version
    if sys.version_info < (3, 9):
        results["errors"].append(f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check dependencies
    required = {
        "torch": "torch",
        "numpy": "numpy",
        "PIL": "PIL",
    }
    
    for name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            results["errors"].append(f"Missing: {name}")
    
    # Check model-specific dependencies
    if model_type == "wan21":
        try:
            from diffusers import WanImageToVideoPipeline
            from transformers import CLIPVisionModel
            results["available"] = True
        except ImportError as e:
            results["errors"].append(f"Missing diffusers/transformers: {e}")
    
    elif model_type == "lightx2v":
        try:
            from lightx2v import LightX2VPipeline
            results["available"] = True
        except ImportError as e:
            results["errors"].append(f"Missing lightx2v: {e}")
            results["warnings"].append("Install: pip install lightx2v")
    
    # Check GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        results["vram_gb"] = round(vram_gb, 1)
        results["gpu_name"] = props.name
    else:
        results["warnings"].append("No GPU available - will use CPU (slower)")
    
    results["passed"] = len(results["errors"]) == 0
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Picture-Aliver Model Validation")
    parser.add_argument("--model", default="wan21", help="Model type to validate")
    parser.add_argument("--check", action="store_true", help="Run validation check")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    if args.check:
        results = validate_model(args.model)
        
        print(f"\nValidation Results: {args.model}")
        print("=" * 40)
        
        for key, value in results.items():
            if isinstance(value, list):
                if value:
                    print(f"{key}:")
                    for item in value:
                        print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        
        print(f"\nPassed: {results['passed']}")