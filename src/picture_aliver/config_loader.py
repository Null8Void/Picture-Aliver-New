"""
Configuration Loader Module

Loads and manages all runtime parameters from config.yaml.
Provides easy access to configuration values throughout the pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field, asdict
import copy

import yaml


@dataclass
class ResolutionConfig:
    """Resolution settings."""
    width: int = 512
    height: int = 512
    auto_adjust: bool = True


@dataclass
class CameraConfig:
    """Camera motion settings."""
    type: str = "pan"
    range: float = 1.0
    path: str = "ease"


@dataclass
class ThresholdsConfig:
    """Quality detection thresholds."""
    face_warp: float = 0.7
    flicker: float = 0.15
    structural: float = 0.6


@dataclass
class AutoCorrectConfig:
    """Automatic correction settings."""
    strengthen_conditioning: bool = True
    reduce_motion: bool = True
    adjust_guidance: bool = True


@dataclass
class ControlnetModelsConfig:
    """ControlNet model settings."""
    models: list = field(default_factory=lambda: ["depth", "canny", "pose"])


@dataclass
class PipelineConfigSection:
    """Pipeline section settings."""
    enable_quality_check: bool = True
    enable_stabilization: bool = True
    enable_interpolation: bool = False
    quality_max_retries: int = 2


@dataclass
class OutputConfigSection:
    """Output section settings."""
    format: str = "mp4"
    quality: str = "medium"
    directory: str = "./output"


@dataclass
class VideoConfigSection:
    """Video section settings."""
    duration_seconds: float = 3.0
    fps: int = 8
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)


@dataclass
class GenerationConfigSection:
    """Generation section settings."""
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None


@dataclass
class MotionConfigSection:
    """Motion section settings."""
    mode: str = "auto"
    strength: float = 0.8
    prompt: Optional[str] = None
    camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass
class ArtifactReductionConfigSection:
    """Artifact reduction section settings."""
    depth_conditioning: bool = True
    controlnet: bool = True
    latent_consistency: bool = True
    optical_flow_stabilization: bool = True
    frame_interpolation: bool = False


@dataclass
class QualityConfigSection:
    """Quality control section settings."""
    enabled: bool = True
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    auto_correct: AutoCorrectConfig = field(default_factory=AutoCorrectConfig)


@dataclass
class GPUConfigSection:
    """GPU section settings."""
    device: str = "auto"
    tier_override: Optional[str] = None
    precision: str = "fp16"
    model_offload: bool = False
    attention_slicing: bool = True
    enable_xformers: bool = True


@dataclass
class ModelsConfigSection:
    """Models section settings."""
    base_dir: str = "./models"
    depth_model: str = "zoedepth"
    segmentation_model: str = "sam"
    diffusion_model: str = "sd-v1-5"
    controlnet_models: list = field(default_factory=lambda: ["depth", "canny", "pose"])


@dataclass
class ContentConfigSection:
    """Content section settings."""
    type_hint: Optional[str] = None
    enable_furry_support: bool = True
    enable_nsfw: bool = False
    filter_level: str = "none"


@dataclass
class RemoteConfigSection:
    """Remote compute section settings."""
    enabled: bool = False
    server_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: int = 300


@dataclass
class LoggingConfigSection:
    """Logging section settings."""
    level: str = "info"
    log_file: Optional[str] = None
    verbose: bool = True


@dataclass
class Config:
    """
    Complete configuration container.
    
    All runtime parameters are loaded from config.yaml
    and accessible through this class.
    """
    pipeline: PipelineConfigSection = field(default_factory=PipelineConfigSection)
    output: OutputConfigSection = field(default_factory=OutputConfigSection)
    video: VideoConfigSection = field(default_factory=VideoConfigSection)
    generation: GenerationConfigSection = field(default_factory=GenerationConfigSection)
    motion: MotionConfigSection = field(default_factory=MotionConfigSection)
    artifact_reduction: ArtifactReductionConfigSection = field(default_factory=ArtifactReductionConfigSection)
    quality: QualityConfigSection = field(default_factory=QualityConfigSection)
    gpu: GPUConfigSection = field(default_factory=GPUConfigSection)
    models: ModelsConfigSection = field(default_factory=ModelsConfigSection)
    content: ContentConfigSection = field(default_factory=ContentConfigSection)
    remote: RemoteConfigSection = field(default_factory=RemoteConfigSection)
    logging: LoggingConfigSection = field(default_factory=LoggingConfigSection)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        
        if not path.exists():
            print(f"[Config] Warning: Config file not found at {path}")
            print("[Config] Using default configuration")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return cls()
        
        config = cls()
        
        config._load_nested(data, config)
        
        return config
    
    def _load_nested(self, data: Dict[str, Any], target: Any) -> None:
        """Recursively load nested configuration."""
        for key, value in data.items():
            if hasattr(target, key):
                attr = getattr(target, key)
                
                if isinstance(value, dict) and isinstance(attr, object):
                    if hasattr(attr, '__dataclass_fields__'):
                        nested = attr.__class__()
                        self._load_nested(value, nested)
                        setattr(target, key, nested)
                    else:
                        setattr(target, key, value)
                else:
                    setattr(target, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self
        
        for k in keys:
            if isinstance(value, object) and hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value


class ConfigLoader:
    """
    Configuration loader and manager.
    
    Handles loading config.yaml and providing
    configuration to all pipeline modules.
    
    Usage:
        loader = ConfigLoader()
        config = loader.load("config.yaml")
        value = config.get("video.fps")
    """
    
    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Config] = None
    
    def __new__(cls, config_path: Optional[Union[str, Path]] = None):
        """Singleton pattern for config loader."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if self._initialized:
            return
        
        self._config = None
        self._config_path = None
        
        if config_path:
            self.load(config_path)
        
        self._initialized = True
    
    def load(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Loaded Config object
        """
        self._config_path = Path(config_path)
        self._config = Config.from_yaml(self._config_path)
        return self._config
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        save_path = Path(path) if path else self._config_path
        if save_path is None:
            save_path = Path("config.yaml")
        
        self._config.to_yaml(save_path)
    
    @property
    def config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            self._config = Config()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default)
    
    def reload(self) -> Config:
        """Reload configuration from file."""
        if self._config_path:
            return self.load(self._config_path)
        return self.config
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None
        cls._config = None


def load_config(
    config_path: Union[str, Path] = "config.yaml",
    create_default: bool = True
) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config.yaml
        create_default: Create default config if file not found
        
    Returns:
        Config object
    """
    path = Path(config_path)
    
    if path.exists():
        return Config.from_yaml(path)
    elif create_default:
        print(f"[ConfigLoader] Config not found at {path}, using defaults")
        config = Config()
        config.to_yaml(path)
        print(f"[ConfigLoader] Created default config at {path}")
        return config
    else:
        return Config()


def get_default_config_path() -> Path:
    """Get the default config file path."""
    possible_paths = [
        Path("config.yaml"),
        Path("./src/picture_aliver/config.yaml"),
        Path(__file__).parent / "config.yaml",
    ]
    
    for p in possible_paths:
        if p.exists():
            return p
    
    return possible_paths[0]


def load_or_default(
    config_path: Optional[Union[str, Path]] = None
) -> Config:
    """
    Load config from path or find default.
    
    Args:
        config_path: Optional explicit config path
        
    Returns:
        Config object
    """
    if config_path:
        return Config.from_yaml(config_path)
    
    path = get_default_config_path()
    return load_config(path, create_default=True)


if __name__ == "__main__":
    print("[ConfigLoader] Testing configuration loader...")
    
    config = load_config("config.yaml", create_default=False)
    
    print(f"\nLoaded Configuration:")
    print(f"  Video: {config.video.duration_seconds}s @ {config.video.fps}fps")
    print(f"  Resolution: {config.video.resolution.width}x{config.video.resolution.height}")
    print(f"  Motion: {config.motion.mode} (strength: {config.motion.strength})")
    print(f"  Quality: {config.output.quality}")
    print(f"  GPU: {config.gpu.device} ({config.gpu.precision})")
    
    print(f"\nExample get:")
    print(f"  video.fps = {config.get('video.fps')}")
    print(f"  motion.camera.type = {config.get('motion.camera.type')}")
    print(f"  gpu.precision = {config.get('gpu.precision')}")