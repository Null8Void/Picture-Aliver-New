"""
Picture-Aliver Model Manager with Fallback

Provides automatic model selection with fallback handling:
- Tries primary model first
- Falls back to next available model on failure
- Logs all errors for debugging

Usage:
    from src.picture_aliver.model_manager import ModelManager
    
    manager = ModelManager()
    result = manager.generate("input.jpg", "motion prompt")
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field

import torch

logger = logging.getLogger("picture_aliver.model_manager")


@dataclass
class ModelAttempt:
    """Result of a model attempt."""
    model_type: str
    success: bool
    error: Optional[str] = None
    generation_time: Optional[float] = None


class ModelManager:
    """
    Manages video generation with automatic fallback.
    
    Tries models in order until one succeeds.
    """
    
    def __init__(
        self,
        primary: str = "wan21",
        fallback: str = "legacy",
        config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize ModelManager.
        
        Args:
            primary: Primary model type ("wan21", "lightx2v", "legacy")
            fallback: Fallback model type
            config_path: Optional path to config file
            device: Device to use ("cuda" or "cpu")
        """
        self.primary = primary
        self.fallback = fallback
        self.config_path = config_path
        self.device = device
        
        self._current_model = None
        self._model_order: List[str] = []
        self._attempts: List[ModelAttempt] = []
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration and determine model order."""
        # Check available models
        available = self._check_available_models()
        
        # Build model order
        self._model_order = []
        
        if self.primary in available:
            self._model_order.append(self.primary)
        else:
            logger.warning(f"Primary model '{self.primary}' not available")
        
        if self.fallback and self.fallback != "none":
            if self.fallback not in available:
                logger.warning(f"Fallback model '{self.fallback}' not available")
            elif self.fallback not in self._model_order:
                self._model_order.append(self.fallback)
        
        # Add any other available models
        for model in available:
            if model not in self._model_order:
                self._model_order.append(model)
        
        if not self._model_order:
            logger.error("No video generation models available!")
        
        logger.info(f"Model order: {self._model_order}")
    
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which models can be loaded."""
        available = {}
        
        # Check Wan 2.1
        try:
            from diffusers import WanImageToVideoPipeline
            available["wan21"] = True
        except ImportError:
            available["wan21"] = False
        
        # Check Wan 2.2
        try:
            from diffusers import WanImageToVideoPipeline
            available["wan22"] = True
        except ImportError:
            available["wan22"] = False
        
        # Check LightX2V
        try:
            from lightx2v import LightX2VPipeline
            available["lightx2v"] = True
        except ImportError:
            available["lightx2v"] = False
        
        # Check HunyuanVideo
        try:
            from diffusers import HunyuanVideoPipeline
            available["hunyuan"] = True
        except ImportError:
            available["hunyuan"] = False
        
        # Check LTX-Video
        try:
            from diffusers import LTXVideoPipeline
            available["ltx"] = True
        except ImportError:
            available["ltx"] = False
        
        # Check CogVideo (via diffusers)
        try:
            from diffusers import CogVideoPipeline
            available["cogvideo"] = True
        except ImportError:
            available["cogvideo"] = False
        
        # Check SVD (Stable Video Diffusion)
        try:
            from diffusers import StableVideoDiffusionPipeline
            available["svd"] = True
        except ImportError:
            available["svd"] = False
        
        # Check ZeroScope
        try:
            from diffusers import DiffusionPipeline
            available["zeroscope"] = True
        except ImportError:
            available["zeroscope"] = False
        
        # Check legacy
        try:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from src.picture_aliver.main import Pipeline
            available["legacy"] = True
        except Exception:
            available["legacy"] = False
        
        return available
    
    def load_model(self, model_type: Optional[str] = None) -> Any:
        """
        Load a model.
        
        Args:
            model_type: Model type to load (uses first available if None)
            
        Returns:
            Loaded model or None
        """
        if model_type is None:
            model_type = self._model_order[0] if self._model_order else None
        
        if model_type is None:
            return None
        
        # Try to load
        from .models import create_model
        
        config = self._load_model_config(model_type)
        
        try:
            model = create_model(model_type, config)
            if model.load():
                logger.info(f"Loaded model: {model_type}")
                return model
            else:
                logger.error(f"Failed to load: {model_type}")
                return None
        except Exception as e:
            logger.exception(f"Error loading {model_type}: {e}")
            return None
    
    def _load_model_config(self, model_type: str) -> Dict[str, Any]:
        """Load configuration for model type."""
        import yaml
        
        config = {}
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path) as f:
                full_config = yaml.safe_load(f)
            
            model_section = full_config.get(model_type, {})
            common = full_config.get("common", {})
            
            # Merge common into model config
            config = {**common, **model_section}
        
        return config
    
    def generate(
        self,
        image: Union[str, Path],
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: float = 3.0,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video with automatic fallback.
        
        Args:
            image: Input image path
            prompt: Animation prompt
            negative_prompt: Negative prompt
            duration: Video duration
            fps: FPS
            width: Width
            height: Height
            output_path: Output path
            
        Returns:
            Dictionary with result
        """
        self._attempts = []
        start_time = time.time()
        
        # Try each model in order
        for model_type in self._model_order:
            attempt_start = time.time()
            logger.info(f"Trying model: {model_type}")
            
            model = self.load_model(model_type)
            
            if model is None:
                self._attempts.append(ModelAttempt(
                    model_type=model_type,
                    success=False,
                    error="Failed to load model",
                ))
                continue
            
            try:
                # Generate
                result = model.generate(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    output_path=output_path,
                )
                
                generation_time = time.time() - attempt_start
                
                if result.success:
                    logger.info(f"Success with {model_type} in {generation_time:.1f}s")
                    
                    return {
                        "success": True,
                        "video_path": result.video_path,
                        "model_type": model_type,
                        "generation_time": generation_time,
                        "total_time": time.time() - start_time,
                        "attempts": self._attempts,
                    }
                else:
                    logger.warning(f"{model_type} failed: {result.error}")
                    self._attempts.append(ModelAttempt(
                        model_type=model_type,
                        success=False,
                        error=result.error,
                        generation_time=generation_time,
                    ))
                    
                    # Try next model
                    model.unload()
                    continue
                    
            except Exception as e:
                logger.exception(f"Error with {model_type}: {e}")
                self._attempts.append(ModelAttempt(
                    model_type=model_type,
                    success=False,
                    error=str(e),
                    generation_time=time.time() - attempt_start,
                ))
                
                # Try next model
                try:
                    model.unload()
                except:
                    pass
                continue
        
        # All models failed
        logger.error("All models failed")
        
        return {
            "success": False,
            "error": "All models failed",
            "model_type": None,
            "attempts": self._attempts,
            "total_time": time.time() - start_time,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "primary": self.primary,
            "fallback": self.fallback,
            "model_order": self._model_order,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

_default_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Get default model manager."""
    global _default_manager
    
    if _default_manager is None:
        _default_manager = ModelManager()
    
    return _default_manager


def generate_video(
    image: Union[str, Path],
    prompt: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate video.
    
    Usage:
        result = generate_video("input.jpg", "gentle wave")
        if result["success"]:
            print(f"Video: {result['video_path']}")
    """
    manager = get_manager()
    return manager.generate(image, prompt, **kwargs)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Picture-Aliver Model Manager")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Animation prompt")
    parser.add_argument("--primary", default="wan21", help="Primary model")
    parser.add_argument("--fallback", default="legacy", help="Fallback model")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    # Create manager
    manager = ModelManager(primary=args.primary, fallback=args.fallback)
    
    print(f"\nStatus: {manager.get_status()}")
    
    # Generate
    result = manager.generate(args.image, args.prompt)
    
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    
    if result["success"]:
        print(f"  Video: {result['video_path']}")
        print(f"  Model: {result['model_type']}")
        print(f"  Time: {result['generation_time']:.1f}s")
    else:
        print(f"  Error: {result.get('error')}")
    
    print(f"\nAttempts:")
    for attempt in result.get("attempts", []):
        status = "OK" if attempt.success else "FAIL"
        print(f"  [{status}] {attempt.model_type}: {attempt.generation_time:.1f}s - {attempt.error or 'OK'}")