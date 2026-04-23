"""Model loader with content rating support and unrestricted generation."""

from __future__ import annotations

import os
import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from .model_registry import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelRegistry,
    ContentRating,
    ModelCategory,
    get_registry,
    get_nsfw_models,
    get_safe_models,
)


class ModelLoader:
    """Centralized model loader with content rating support.
    
    This loader manages model downloading, caching, and loading with support for:
    - Safe vs unrestricted model selection
    - VRAM-aware model selection
    - Model quantization options
    - Hot-swapping between models
    - Memory management
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        default_rating: ContentRating = ContentRating.SAFE,
        trust_remote_code: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_rating = default_rating
        self.trust_remote_code = trust_remote_code
        
        self._loaded_models: Dict[str, nn.Module] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._model_locks: Dict[str, bool] = {}  # Prevent duplicate loading
        
        self._session_vram_mb = self._get_available_vram_mb()
    
    def _get_default_cache_dir(self) -> Path:
        """Get default model cache directory."""
        if "HF_HOME" in os.environ:
            return Path(os.environ["HF_HOME"]) / "models"
        if "MODEL_CACHE" in os.environ:
            return Path(os.environ["MODEL_CACHE"])
        return Path.home() / ".cache" / "picture_aliver" / "models"
    
    def _get_available_vram_mb(self) -> int:
        """Get available VRAM in MB."""
        if not torch.cuda.is_available():
            return 8000  # Assume 8GB for CPU/other
        return int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
    
    @property
    def available_vram_mb(self) -> int:
        """Get available VRAM for model selection."""
        return self._session_vram_mb
    
    def get_vram_usage_mb(self) -> int:
        """Get current VRAM usage."""
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 ** 2))
        return 0
    
    def get_vram_reserved_mb(self) -> int:
        """Get reserved VRAM."""
        if torch.cuda.is_available():
            return int(torch.cuda.memory_reserved() / (1024 ** 2))
        return 0
    
    def can_load_model(self, model_info: ModelInfo) -> bool:
        """Check if model can be loaded with available VRAM."""
        available = self._session_vram_mb - self.get_vram_usage_mb() - 500  # 500MB buffer
        return model_info.vram_mb <= available
    
    def _should_use_quantization(self, model_info: ModelInfo) -> Optional[str]:
        """Determine if quantization should be used."""
        available = self._session_vram_mb - self.get_vram_usage_mb()
        
        if model_info.vram_mb > available:
            if model_info.vram_mb // 2 <= available:
                return "fp16"
            if model_info.vram_mb // 4 <= available:
                return "int8"
        return None
    
    def load_model(
        self,
        model_info: ModelInfo,
        force_reload: bool = False,
        variant: Optional[str] = None,
        rating: Optional[ContentRating] = None,
        **kwargs
    ) -> nn.Module:
        """Load a model from the registry.
        
        Args:
            model_info: Model information from registry
            force_reload: Force reload even if already loaded
            variant: Model variant to load (e.g., "fp16", "int8")
            rating: Override content rating
            **kwargs: Additional model-specific arguments
            
        Returns:
            Loaded model
        """
        cache_key = self._get_cache_key(model_info, variant)
        
        # Check if already loaded
        if cache_key in self._loaded_models and not force_reload:
            return self._loaded_models[cache_key]
        
        # Check if loading is in progress
        if cache_key in self._model_locks:
            import time
            while cache_key in self._model_locks:
                time.sleep(0.1)
            if cache_key in self._loaded_models:
                return self._loaded_models[cache_key]
        
        self._model_locks[cache_key] = True
        
        try:
            # Determine variant
            actual_variant = variant or self._select_variant(model_info)
            repo_id = model_info.variants.get(actual_variant, model_info.repo_id)
            
            # Check VRAM and apply quantization if needed
            quantization = kwargs.pop("quantization", None)
            if quantization is None:
                quantization = self._should_use_quantization(model_info)
            
            # Load the model
            model = self._load_from_source(
                repo_id=repo_id,
                model_info=model_info,
                quantization=quantization,
                **kwargs
            )
            
            model = model.to(self.device)
            model.eval()
            
            self._loaded_models[cache_key] = model
            self._model_configs[cache_key] = {
                "repo_id": repo_id,
                "variant": actual_variant,
                "quantization": quantization,
                "rating": rating or model_info.rating,
            }
            
            return model
            
        finally:
            self._model_locks.pop(cache_key, None)
    
    def _select_variant(self, model_info: ModelInfo) -> str:
        """Select best variant based on available VRAM."""
        available = self._session_vram_mb - self.get_vram_usage_mb()
        
        if "int8" in model_info.variants and available >= model_info.vram_mb // 4 * 3:
            return "int8"
        if "fp16" in model_info.variants and available >= model_info.vram_mb // 2:
            return "fp16"
        if "fp16" in model_info.variants and torch.cuda.is_bf16_supported():
            return "fp16"
        return "default"
    
    def _load_from_source(
        self,
        repo_id: str,
        model_info: ModelInfo,
        quantization: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load model from HuggingFace or local cache."""
        
        if model_info.category == ModelCategory.I2V:
            return self._load_i2v_model(repo_id, model_info, quantization, **kwargs)
        elif model_info.category == ModelCategory.DEPTH:
            return self._load_depth_model(repo_id, model_info, quantization, **kwargs)
        elif model_info.category == ModelCategory.SEGMENTATION:
            return self._load_segmentation_model(repo_id, model_info, quantization, **kwargs)
        elif model_info.category == ModelCategory.INTERPOLATION:
            return self._load_interpolation_model(repo_id, model_info, quantization, **kwargs)
        else:
            raise ValueError(f"Unknown model category: {model_info.category}")
    
    def _load_i2v_model(
        self,
        repo_id: str,
        model_info: ModelInfo,
        quantization: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load image-to-video model."""
        
        dtype = torch.float16 if quantization == "fp16" else torch.float32
        
        try:
            if "svd" in repo_id.lower() or "video-diffusion" in repo_id.lower():
                return self._load_svd_model(repo_id, dtype, **kwargs)
            elif "animatediff" in repo_id.lower():
                return self._load_animatediff_model(repo_id, dtype, **kwargs)
            elif "zeroscope" in repo_id.lower():
                return self._load_zeroscope_model(repo_id, dtype, **kwargs)
            elif "i2vgen" in repo_id.lower():
                return self._load_i2vgen_model(repo_id, dtype, **kwargs)
            elif "opengif" in repo_id.lower() or "open-gif" in repo_id.lower():
                return self._load_opengif_model(repo_id, dtype, **kwargs)
            else:
                return self._load_generic_i2v_model(repo_id, dtype, **kwargs)
                
        except ImportError as e:
            warnings.warn(f"Could not load model {repo_id}: {e}")
            return self._create_fallback_i2v_model(model_info)
    
    def _load_svd_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load Stable Video Diffusion model."""
        from diffusers import StableVideoDiffusionPipeline
        
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code,
        )
        
        if torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
        
        return pipeline
    
    def _load_animatediff_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load AnimateDiff model."""
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        
        adapter_path = repo_id if "motion-adapter" in repo_id else None
        
        if adapter_path:
            try:
                motion_adapter = MotionAdapter.from_pretrained(
                    adapter_path,
                    torch_dtype=dtype,
                    cache_dir=str(self.cache_dir),
                )
            except Exception:
                motion_adapter = None
            
            base_model = kwargs.pop("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            pipeline = AnimateDiffPipeline.from_pretrained(
                base_model,
                motion_adapter=motion_adapter,
                torch_dtype=dtype,
                cache_dir=str(self.cache_dir),
                trust_remote_code=self.trust_remote_code,
            )
        else:
            pipeline = AnimateDiffPipeline.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                cache_dir=str(self.cache_dir),
                trust_remote_code=self.trust_remote_code,
            )
        
        if torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()
        
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        return pipeline
    
    def _load_zeroscope_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load ZeroScope model."""
        from diffusers import DiffusionPipeline
        
        pipeline = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code,
        )
        
        if torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    def _load_i2vgen_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load I2VGen-XL model."""
        from diffusers import I2VGenXLPipeline
        
        pipeline = I2VGenXLPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code,
        )
        
        if torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()
        
        return pipeline
    
    def _load_opengif_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load OpenGIF model."""
        # OpenGIF uses a custom architecture
        from diffusers import DiffusionPipeline
        
        pipeline = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code,
        )
        
        return pipeline
    
    def _load_generic_i2v_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        **kwargs
    ) -> nn.Module:
        """Load generic I2V model."""
        from diffusers import DiffusionPipeline
        
        pipeline = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=self.trust_remote_code,
        )
        
        return pipeline
    
    def _create_fallback_i2v_model(self, model_info: ModelInfo) -> nn.Module:
        """Create a simple fallback I2V model."""
        
        class SimpleI2VModel(nn.Module):
            """Simple image-to-video fallback model."""
            
            def __init__(self, max_frames=16, resolution=(512, 512)):
                super().__init__()
                self.max_frames = max_frames
                self.resolution = resolution
                
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                
                self.temporal = nn.LSTM(256, 256, batch_first=True)
                
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 7, padding=3),
                    nn.Sigmoid(),
                )
            
            def forward(self, x, num_frames=None):
                if num_frames is None:
                    num_frames = self.max_frames
                
                B, C, H, W = x.shape
                features = self.encoder(x)
                
                C_out, H_out, W_out = features.shape[1:]
                features = features.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
                features = features.reshape(B * num_frames, C_out, H_out, W_out)
                
                features_flat = features.reshape(B * num_frames, C_out, -1).transpose(1, 2)
                _, (hidden, _) = self.temporal(features_flat)
                
                hidden = hidden.transpose(0, 1).reshape(B, num_frames, C_out, H_out, W_out)
                hidden = hidden.reshape(B * num_frames, C_out, H_out, W_out)
                
                output = self.decoder(hidden)
                
                return output.view(B, num_frames, C, H, W)
        
        return SimpleI2VModel(
            max_frames=model_info.max_frames or 16,
            resolution=model_info.resolution or (512, 512)
        )
    
    def _load_depth_model(
        self,
        repo_id: str,
        model_info: ModelInfo,
        quantization: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load depth estimation model."""
        
        dtype = torch.float16 if quantization == "fp16" else torch.float32
        
        if "zoedepth" in repo_id.lower() or "ldm" in repo_id.lower():
            return self._load_zoedepth_model(repo_id, dtype, **kwargs)
        elif "midas" in repo_id.lower():
            return self._load_midas_model(repo_id, dtype, **kwargs)
        elif "marigold" in repo_id.lower():
            return self._load_marigold_model(repo_id, dtype, **kwargs)
        elif "depth-anything" in repo_id.lower():
            return self._load_depth_anything_model(repo_id, dtype, **kwargs)
        else:
            return self._load_generic_depth_model(repo_id, dtype, **kwargs)
    
    def _load_zoedepth_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load ZoeDepth model."""
        try:
            from zoedepth.models import ZoeDepth
            
            model = ZoeDepth.from_pretrained("Zoedepth/zoedepth-nyu", map_location=self.device)
            return model
        except Exception:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            
            processor = DPTImageProcessor.from_pretrained(repo_id)
            model = DPTForDepthEstimation.from_pretrained(repo_id)
            
            class DepthModelWrapper(nn.Module):
                def __init__(self, model, processor):
                    super().__init__()
                    self.model = model
                    self.processor = processor
                
                def forward(self, x):
                    return self.model(x).predicted_depth
                
                def infer(self, x):
                    with torch.no_grad():
                        return self.model(x).predicted_depth
            
            return DepthModelWrapper(model, processor)
    
    def _load_midas_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load MiDaS model."""
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        
        try:
            processor = DPTImageProcessor.from_pretrained(repo_id)
            model = DPTForDepthEstimation.from_pretrained(repo_id)
        except Exception:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            processor = AutoImageProcessor.from_pretrained(repo_id)
            model = AutoModelForDepthEstimation.from_pretrained(repo_id)
        
        class MidasWrapper(nn.Module):
            def __init__(self, model, processor):
                super().__init__()
                self.model = model
                self.processor = processor
            
            def forward(self, x):
                return self.model(x).predicted_depth
            
            def infer(self, x):
                with torch.no_grad():
                    return self.model(x).predicted_depth
        
        return MidasWrapper(model, processor)
    
    def _load_marigold_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load Marigold model."""
        from diffusers import DiffusionPipeline
        
        pipeline = DiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            cache_dir=str(self.cache_dir),
        )
        
        return pipeline
    
    def _load_depth_anything_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load Depth Anything model."""
        from transformers import AutoImageProcessor, AutoModel
        
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id)
        
        class DepthAnythingWrapper(nn.Module):
            def __init__(self, model, processor):
                super().__init__()
                self.model = model
                self.processor = processor
            
            def forward(self, x):
                return self.model(x).predicted_depth
        
        return DepthAnythingWrapper(model, processor)
    
    def _load_generic_depth_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load generic depth model."""
        from transformers import AutoModelForDepthEstimation
        
        model = AutoModelForDepthEstimation.from_pretrained(repo_id)
        return model
    
    def _load_segmentation_model(
        self,
        repo_id: str,
        model_info: ModelInfo,
        quantization: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load segmentation model."""
        
        dtype = torch.float16 if quantization == "fp16" else torch.float32
        
        if "sam" in repo_id.lower():
            return self._load_sam_model(repo_id, dtype, model_info, **kwargs)
        elif "deeplab" in repo_id.lower():
            return self._load_deeplab_model(repo_id, dtype, **kwargs)
        else:
            return self._load_generic_segmentation_model(repo_id, dtype, **kwargs)
    
    def _load_sam_model(
        self,
        repo_id: str,
        dtype: torch.dtype,
        model_info: ModelInfo,
        **kwargs
    ) -> nn.Module:
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if "vit-h" in repo_id.lower() or "huge" in repo_id.lower():
                model_type = "vit_h"
            elif "vit-l" in repo_id.lower() or "large" in repo_id.lower():
                model_type = "vit_l"
            elif "mobile" in repo_id.lower():
                model_type = "vit_b"  # MobileSAM uses vit_b architecture
            else:
                model_type = "vit_b"
            
            model = sam_model_registry[model_type]()
            
            predictor = SamPredictor(model)
            
            class SAMWrapper(nn.Module):
                def __init__(self, predictor):
                    super().__init__()
                    self.predictor = predictor
                
                def set_image(self, image):
                    self.predictor.set_image(image)
                
                def predict(self, points=None, labels=None, multimask=True):
                    if points is not None and labels is not None:
                        masks, scores, _ = self.predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            multimask_output=multimask
                        )
                        return masks, scores
                    else:
                        masks, scores, _ = self.predictor.predict(
                            multimask_output=multimask
                        )
                        return masks, scores
            
            return SAMWrapper(predictor)
            
        except ImportError:
            raise ImportError("segment_anything not installed. Install with: pip install segment_anything")
    
    def _load_deeplab_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load DeepLabV3 model."""
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(repo_id)
        
        class DeepLabWrapper(nn.Module):
            def __init__(self, model, processor):
                super().__init__()
                self.model = model
                self.processor = processor
            
            def forward(self, x):
                outputs = self.model(x)
                return outputs.logits
            
            def segment(self, x):
                with torch.no_grad():
                    return self.model(x)
        
        return DeepLabWrapper(model, processor)
    
    def _load_generic_segmentation_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load generic segmentation model."""
        from transformers import AutoModelForSemanticSegmentation
        
        model = AutoModelForSemanticSegmentation.from_pretrained(repo_id)
        return model
    
    def _load_interpolation_model(
        self,
        repo_id: str,
        model_info: ModelInfo,
        quantization: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """Load frame interpolation model."""
        
        dtype = torch.float16 if quantization == "fp16" else torch.float32
        
        if "rife" in repo_id.lower():
            return self._load_rife_model(repo_id, dtype, **kwargs)
        elif "amt" in repo_id.lower():
            return self._load_amt_model(repo_id, dtype, **kwargs)
        else:
            return self._load_generic_interpolation_model(repo_id, dtype, **kwargs)
    
    def _load_rife_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load RIFE model."""
        try:
            import rifenet
            
            model = rifenet.RIFENet()
            
            try:
                state_dict = torch.load(
                    self.cache_dir / "rife" / "rife.pth",
                    map_location=self.device
                )
                model.load_state_dict(state_dict)
            except FileNotFoundError:
                pass
            
            class RIFEWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.model.eval()
                
                def forward(self, frame1, frame2):
                    return self.model(frame1, frame2)
                
                def interpolate(self, frame1, frame2, num_frames=2):
                    outputs = [frame1]
                    for i in range(num_frames):
                        t = (i + 1) / (num_frames + 1)
                        mid = self.model(frame1, frame2)
                        if i < num_frames - 1:
                            outputs.append(mid)
                        frame1 = mid
                    outputs.append(frame2)
                    return outputs
            
            return RIFEWrapper(model)
            
        except ImportError:
            warnings.warn("RIFE not installed. Install from: https://github.com/hzwer/RIFE")
            raise ImportError("RIFE not available")
    
    def _load_amt_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load AMT model."""
        # AMT from Google Research
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "amt"))
        
        from amt.models import AMT
        
        model = AMT()
        return model
    
    def _load_generic_interpolation_model(self, repo_id: str, dtype: torch.dtype, **kwargs) -> nn.Module:
        """Load generic interpolation model."""
        raise NotImplementedError("Generic interpolation not yet implemented")
    
    def _get_cache_key(self, model_info: ModelInfo, variant: Optional[str] = None) -> str:
        """Get cache key for a model."""
        return f"{model_info.name}_{variant or 'default'}"
    
    def unload_model(self, model_name: str, variant: Optional[str] = None):
        """Unload a model from memory."""
        cache_key = f"{model_name}_{variant or 'default'}"
        
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
        
        if cache_key in self._model_configs:
            del self._model_configs[cache_key]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_all(self):
        """Unload all models from memory."""
        self._loaded_models.clear()
        self._model_configs.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def is_loaded(self, model_name: str, variant: Optional[str] = None) -> bool:
        """Check if a model is loaded."""
        cache_key = f"{model_name}_{variant or 'default'}"
        return cache_key in self._loaded_models
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._loaded_models.keys())
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a loaded model."""
        return self._model_configs.get(model_name)
    
    def get_vram_info(self) -> Dict[str, int]:
        """Get VRAM information."""
        return {
            "total_mb": self._session_vram_mb,
            "used_mb": self.get_vram_usage_mb(),
            "reserved_mb": self.get_vram_reserved_mb(),
            "available_mb": self._session_vram_mb - self.get_vram_usage_mb(),
        }


from pathlib import Path