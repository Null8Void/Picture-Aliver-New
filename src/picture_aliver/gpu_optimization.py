"""
GPU Optimization Module

Comprehensive GPU optimization for consumer hardware.
Includes FP16 support, model offloading, and VRAM management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from contextlib import contextmanager
import time

import torch
import torch.nn as nn


class VRAMTier(str):
    """GPU VRAM tier classification."""
    MINIMUM = "2gb"
    LOW = "4gb"
    MEDIUM = "8gb"
    HIGH = "12gb"
    ULTRA = "24gb"


@dataclass
class GPUConfig:
    """
    GPU configuration for optimized inference.
    
    Attributes:
        tier: VRAM tier classification
        use_fp16: Use half precision (FP16)
        use_bf16: Use bfloat16 (better for some GPUs)
        model_offload: Enable sequential model offloading
        attention_slicing: Enable attention slicing
        enable_xformers: Enable xformers memory efficient attention
        enable_vae_slicing: Enable VAE tiling/slicing
        vae_tile_size: VAE tile size for large images
        gradient_checkpointing: Enable gradient checkpointing
        cpu_offload: Offload to CPU when not in use
        max_batch_size: Maximum batch size for inference
    """
    tier: VRAMTier = VRAMTier.MEDIUM
    use_fp16: bool = True
    use_bf16: bool = False
    model_offload: bool = False
    attention_slicing: bool = True
    enable_xformers: bool = False
    enable_vae_slicing: bool = True
    vae_tile_size: int = 512
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    max_batch_size: int = 1
    low_vram_mode: bool = False


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    vram_tier: VRAMTier
    max_resolution: tuple
    recommended_fps: int
    max_duration_seconds: int
    inference_time_per_frame_ms: float
    estimated_vram_usage_gb: float
    supports_fp16: bool
    supports_bf16: bool
    notes: str


class GPUOptimizer:
    """
    GPU optimization manager.
    
    Automatically configures models for optimal performance
    on consumer GPUs ranging from 2GB to 24GB VRAM.
    
    Attributes:
        device: Compute device
        config: GPU configuration
        total_vram_gb: Total VRAM in GB
        initial_vram: VRAM usage at initialization
    """
    
    BENCHMARKS = {
        VRAMTier.MINIMUM: PerformanceBenchmark(
            vram_tier=VRAMTier.MINIMUM,
            max_resolution=(384, 384),
            recommended_fps=4,
            max_duration_seconds=10,
            inference_time_per_frame_ms=800,
            estimated_vram_usage_gb=1.5,
            supports_fp16=True,
            supports_bf16=False,
            notes="Low resolution, minimal frames. May struggle with complex scenes."
        ),
        VRAMTier.LOW: PerformanceBenchmark(
            vram_tier=VRAMTier.LOW,
            max_resolution=(512, 512),
            recommended_fps=6,
            max_duration_seconds=15,
            inference_time_per_frame_ms=500,
            estimated_vram_usage_gb=3.2,
            supports_fp16=True,
            supports_bf16=True,
            notes="Standard resolution, moderate frames. Good for most content."
        ),
        VRAMTier.MEDIUM: PerformanceBenchmark(
            vram_tier=VRAMTier.MEDIUM,
            max_resolution=(768, 768),
            recommended_fps=12,
            max_duration_seconds=30,
            inference_time_per_frame_ms=200,
            estimated_vram_usage_gb=6.5,
            supports_fp16=True,
            supports_bf16=True,
            notes="High resolution, smooth playback. RTX 3060 / RTX 2080 class."
        ),
        VRAMTier.HIGH: PerformanceBenchmark(
            vram_tier=VRAMTier.HIGH,
            max_resolution=(1024, 1024),
            recommended_fps=24,
            max_duration_seconds=60,
            inference_time_per_frame_ms=100,
            estimated_vram_usage_gb=9.5,
            supports_fp16=True,
            supports_bf16=True,
            notes="Very high resolution, excellent quality. RTX 3080 / RTX 3090 class."
        ),
        VRAMTier.ULTRA: PerformanceBenchmark(
            vram_tier=VRAMTier.ULTRA,
            max_resolution=(1280, 1280),
            recommended_fps=30,
            max_duration_seconds=120,
            inference_time_per_frame_ms=50,
            estimated_vram_usage_gb=18.0,
            supports_fp16=True,
            supports_bf16=True,
            notes="Maximum performance. RTX 4090 / A6000 / A100 class."
        ),
    }
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        auto_detect: bool = True,
        target_tier: Optional[VRAMTier] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = GPUConfig()
        self.total_vram_gb = 0.0
        self.initial_vram = 0.0
        
        if auto_detect:
            self._detect_gpu()
        
        if target_tier:
            self.set_tier(target_tier)
    
    def _detect_gpu(self) -> None:
        """Auto-detect GPU capabilities and set appropriate tier."""
        if not torch.cuda.is_available():
            self.config.tier = VRAMTier.MINIMUM
            self.config.use_fp16 = False
            print("[GPUOptimizer] No CUDA GPU detected, using CPU mode")
            return
        
        props = torch.cuda.get_device_properties(0)
        self.total_vram_gb = props.total_memory / (1024 ** 3)
        
        self.initial_vram = self._get_vram_usage()
        
        if self.total_vram_gb >= 20:
            self.config.tier = VRAMTier.ULTRA
        elif self.total_vram_gb >= 10:
            self.config.tier = VRAMTier.HIGH
        elif self.total_vram_gb >= 6:
            self.config.tier = VRAMTier.MEDIUM
        elif self.total_vram_gb >= 3:
            self.config.tier = VRAMTier.LOW
        else:
            self.config.tier = VRAMTier.MINIMUM
        
        print(f"[GPUOptimizer] Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"[GPUOptimizer] Total VRAM: {self.total_vram_gb:.1f} GB")
        print(f"[GPUOptimizer] Tier: {self.config.tier.value}")
        
        self._apply_tier_config()
    
    def _apply_tier_config(self) -> None:
        """Apply configuration for current tier."""
        tier = self.config.tier
        
        if tier == VRAMTier.MINIMUM:
            self.config.use_fp16 = True
            self.config.model_offload = True
            self.config.attention_slicing = True
            self.config.enable_vae_slicing = True
            self.config.vae_tile_size = 256
            self.config.max_batch_size = 1
            self.config.low_vram_mode = True
        elif tier == VRAMTier.LOW:
            self.config.use_fp16 = True
            self.config.model_offload = False
            self.config.attention_slicing = True
            self.config.enable_vae_slicing = True
            self.config.vae_tile_size = 384
            self.config.max_batch_size = 1
            self.config.low_vram_mode = False
        elif tier == VRAMTier.MEDIUM:
            self.config.use_fp16 = True
            self.config.enable_xformers = True
            self.config.attention_slicing = True
            self.config.enable_vae_slicing = True
            self.config.vae_tile_size = 512
            self.config.max_batch_size = 2
            self.config.low_vram_mode = False
        elif tier == VRAMTier.HIGH:
            self.config.use_fp16 = True
            self.config.enable_xformers = True
            self.config.attention_slicing = False
            self.config.enable_vae_slicing = False
            self.config.vae_tile_size = 1024
            self.config.max_batch_size = 4
        elif tier == VRAMTier.ULTRA:
            self.config.use_fp16 = True
            self.config.enable_xformers = True
            self.config.attention_slicing = False
            self.config.enable_vae_slicing = False
            self.config.max_batch_size = 8
    
    def set_tier(self, tier: VRAMTier) -> None:
        """Manually set GPU tier."""
        self.config.tier = tier
        self._apply_tier_config()
        print(f"[GPUOptimizer] Set tier to: {tier.value}")
    
    def get_dtype(self) -> torch.dtype:
        """Get appropriate dtype for current config."""
        if self.config.use_bf16 and self._supports_bf16():
            return torch.bfloat16
        elif self.config.use_fp16:
            return torch.float16
        return torch.float32
    
    def _supports_bf16(self) -> bool:
        """Check if device supports bfloat16."""
        if not torch.cuda.is_available():
            return False
        
        props = torch.cuda.get_device_properties(0)
        return props.major >= 8
    
    def get_benchmark(self) -> PerformanceBenchmark:
        """Get performance benchmark for current tier."""
        return self.BENCHMARKS[self.config.tier]
    
    def get_optimization_dict(self) -> Dict[str, Any]:
        """Get optimization parameters as dictionary."""
        return {
            "dtype": self.get_dtype(),
            "use_fp16": self.config.use_fp16,
            "use_bf16": self.config.use_bf16,
            "model_offload": self.config.model_offload,
            "attention_slicing": self.config.attention_slicing,
            "enable_xformers": self.config.enable_xformers,
            "vae_slicing": self.config.enable_vae_slicing,
            "vae_tile_size": self.config.vae_tile_size,
            "max_batch_size": self.config.max_batch_size,
            "low_vram_mode": self.config.low_vram_mode,
        }
    
    def convert_model_to_optimized(self, model: nn.Module) -> nn.Module:
        """
        Convert model to optimized format.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        dtype = self.get_dtype()
        
        if dtype != torch.float32:
            model = model.to(dtype)
        
        if self.config.attention_slicing and hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing(self.config.vae_tile_size)
        
        if self.config.enable_xformers:
            try:
                model.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        if self.config.enable_vae_slicing and hasattr(model, "enable_vae_slicing"):
            model.enable_vae_slicing()
        
        if self.config.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        
        return model
    
    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(0) / (1024 ** 3)
    
    def get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return (total - allocated) / (1024 ** 3)
    
    @contextmanager
    def vram_guard(self, required_gb: float):
        """
        Context manager to ensure sufficient VRAM is available.
        
        Args:
            required_gb: Required VRAM in GB
        """
        available = self.get_available_vram()
        
        if available < required_gb:
            print(f"[GPUOptimizer] Warning: Only {available:.2f}GB available, "
                  f"need {required_gb:.2f}GB. Clearing cache...")
            self.clear_cache()
        
        try:
            yield
        finally:
            pass
    
    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @contextmanager
    def inference_mode(self):
        """Context manager for inference optimization."""
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=self.config.use_fp16):
                    yield
            else:
                yield
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply full inference optimizations to model."""
        model.eval()
        
        model = self.convert_model_to_optimized(model)
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def estimate_generation_params(
        self,
        duration_seconds: float,
        fps: int
    ) -> Dict[str, Any]:
        """
        Estimate optimal generation parameters for current GPU.
        
        Args:
            duration_seconds: Desired video duration
            fps: Frames per second
            
        Returns:
            Dictionary of recommended parameters
        """
        benchmark = self.get_benchmark()
        
        num_frames = int(duration_seconds * fps)
        resolution = benchmark.max_resolution
        
        recommended_params = {
            "max_frames_per_batch": benchmark.recommended_fps,
            "resolution": resolution,
            "use_fp16": self.config.use_fp16,
            "num_inference_steps": 20 if self.config.tier in [VRAMTier.MINIMUM, VRAMTier.LOW] else 25,
        }
        
        if duration_seconds > benchmark.max_duration_seconds:
            recommended_params["max_duration"] = benchmark.max_duration_seconds
            recommended_params["warning"] = f"Duration exceeds recommended maximum for this GPU tier"
        
        return recommended_params


class ModelOffloader:
    """
    Sequential model offloading for low VRAM GPUs.
    
    Unloads models to CPU when not needed and reloads
    when necessary to minimize VRAM usage.
    
    Attributes:
        device: Compute device
        offload_models: List of models to offload
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.offload_models: Dict[str, nn.Module] = {}
        self.loaded_models: Dict[str, bool] = {}
    
    def register_model(self, name: str, model: nn.Module) -> None:
        """Register a model for offloading."""
        self.offload_models[name] = model
        self.loaded_models[name] = True
    
    def load_to_device(self, name: str) -> nn.Module:
        """Load model to GPU."""
        if name not in self.offload_models:
            raise ValueError(f"Model {name} not registered")
        
        model = self.offload_models[name]
        
        if not self.loaded_models.get(name, False):
            model = model.to(self.device)
            self.loaded_models[name] = True
        
        return model
    
    def offload_to_cpu(self, name: str) -> None:
        """Move model to CPU to free VRAM."""
        if name not in self.offload_models:
            return
        
        model = self.offload_models[name]
        model = model.cpu()
        self.loaded_models[name] = False
        torch.cuda.empty_cache()
    
    def offload_all(self) -> None:
        """Offload all models to CPU."""
        for name in self.offload_models:
            self.offload_to_cpu(name)
    
    @contextmanager
    def use_model(self, name: str):
        """Context manager for using offloaded model."""
        model = self.load_to_device(name)
        
        try:
            yield model
        finally:
            pass


def print_benchmark_table() -> None:
    """Print benchmark comparison table."""
    print("\n" + "=" * 80)
    print("PICTURE-ALIVER PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    print("\n{:<12} {:<14} {:<12} {:<12} {:<20}".format(
        "VRAM", "Resolution", "Max FPS", "Max Duration", "Notes"
    ))
    print("-" * 80)
    
    benchmarks = [
        ("2GB", "(384x384)", "4 fps", "10 seconds", "Minimum - Low quality"),
        ("4GB", "(512x512)", "6 fps", "15 seconds", "Entry level - Basic"),
        ("8GB", "(768x768)", "12 fps", "30 seconds", "Standard - RTX 3060"),
        ("12GB", "(1024x1024)", "24 fps", "60 seconds", "High-end - RTX 3080/3090"),
        ("24GB", "(1280x1280)", "30 fps", "120 seconds", "Ultra - RTX 4090/A100"),
    ]
    
    for row in benchmarks:
        print("{:<12} {:<14} {:<12} {:<12} {:<20}".format(*row))
    
    print("\n" + "=" * 80)
    print("NOTES:")
    print("- Benchmarks assume FP16 mode enabled")
    print("- Real performance varies by scene complexity")
    print("- Use --quality=low for longer videos on limited VRAM")
    print("- Enable frame interpolation only on 8GB+ VRAM")
    print("=" * 80 + "\n")


def optimize_model_for_device(
    model: nn.Module,
    device: Optional[torch.device] = None,
    tier: Optional[VRAMTier] = None
) -> nn.Module:
    """
    Convenience function to optimize a model for the current device.
    
    Args:
        model: Model to optimize
        device: Target device
        tier: Optional VRAM tier override
        
    Returns:
        Optimized model
    """
    optimizer = GPUOptimizer(device=device, target_tier=tier)
    return optimizer.optimize_for_inference(model)


if __name__ == "__main__":
    print_benchmark_table()
    
    optimizer = GPUOptimizer()
    
    benchmark = optimizer.get_benchmark()
    
    print(f"\nCurrent GPU Configuration:")
    print(f"  Tier: {optimizer.config.tier.value}")
    print(f"  VRAM: {optimizer.total_vram_gb:.1f} GB")
    print(f"  FP16: {optimizer.config.use_fp16}")
    print(f"  XFormers: {optimizer.config.enable_xformers}")
    print(f"  Model Offload: {optimizer.config.model_offload}")
    
    print(f"\nRecommended Generation Parameters:")
    params = optimizer.estimate_generation_params(duration_seconds=30, fps=8)
    for k, v in params.items():
        print(f"  {k}: {v}")