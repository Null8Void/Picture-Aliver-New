# Picture-Aliver: Image-to-Video AI System

Production-ready AI system for converting images to animated videos. Supports both **safe** and **unrestricted** content generation modes.

## Features

- **Multi-Mode Support**: Safe, Mature, and Unrestricted (NSFW) generation
- **Depth Estimation**: ZoeDepth, MiDaS, Marigold
- **Semantic Segmentation**: SAM, DeepLabV3
- **Optical Flow**: RAFT, GMFlow, Farneback
- **Video Generation**: SVD, AnimateDiff, Open-SVD, OpenGIF
- **Frame Interpolation**: RIFE, AMT
- **Temporal Consistency**: Gaussian smoothing, flicker reduction
- **GPU Accelerated**: CUDA/MPS support

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from picture_aliver import PictureAliver

# Safe mode (default)
app = PictureAliver(mode="safe")
result = app.convert("photo.jpg", "video.mp4")

# Unrestricted mode
app = PictureAliver(mode="nsfw")
result = app.convert("photo.jpg", "video.mp4")
```

### CLI

```bash
# Safe mode
python -m src.bin.cli -i photo.jpg -o video.mp4

# Unrestricted mode
python -m src.bin.cli -i photo.jpg -o video.mp4 --mode nsfw

# Show available models
python -m src.bin.cli --list-models
```

## Content Modes

| Mode | Description | Available Models |
|------|-------------|-------------------|
| `safe` | General content only | SVD, ZoeDepth, SAM |
| `mature` | Some restrictions | ZeroScope, MiDaS, SAM |
| `nsfw` / `unrestricted` | No restrictions | Open-SVD, OpenGIF, all models |

## Model Categories

### Image-to-Video Models

**Safe/General:**
- SVD (Stable Video Diffusion) - Highest quality
- AnimateDiff SDXL - High resolution

**Mature:**
- ZeroScope v2 - Fast, good quality
- I2VGen-XL - Image preservation

**Unrestricted:**
- Open-SVD - High quality, no filters
- OpenGIF - Extended sequences
- AnimateDiff v3 - Motion adapters

### Supporting Models

| Category | Safe | Unrestricted |
|----------|------|--------------|
| Depth | ZoeDepth | MiDaS v3.1 |
| Segmentation | SAM Vit-L | SAM Vit-B, MobileSAM |
| Interpolation | RIFE v4 | RIFE v4 |

## Configuration

### Python API

```python
from picture_aliver import PictureAliverBuilder

app = PictureAliverBuilder() \
    .mode("nsfw") \
    .vram(12000) \
    .frames(24) \
    .resolution(512) \
    .motion("cinematic") \
    .i2v_model("Open-SVD") \
    .build()

result = app.convert("photo.jpg", "video.mp4")
```

### CLI Options

```bash
python -m src.bin.cli \
    --input photo.jpg \
    --output video.mp4 \
    --mode nsfw \
    --i2v-model "Open-SVD" \
    --num-frames 24 \
    --fps 8 \
    --motion-mode cinematic
```

## Model Selection

### By Hardware Tier

**High-End (12GB+ VRAM):**
```python
app = PictureAliver(mode="nsfw", vram_mb=16000)
# Uses: Open-SVD, ZoeDepth, SAM Vit-H
```

**Mid-Range (8GB VRAM):**
```python
app = PictureAliver(mode="nsfw", vram_mb=8000)
# Uses: OpenGIF, MiDaS, MobileSAM
```

**Low-End (4GB VRAM):**
```python
app = PictureAliver(mode="mature", vram_mb=4000)
# Uses: ZeroScope, MiDaS, MobileSAM
```

### List Available Models

```bash
# All models
python -m src.bin.cli --list-models

# NSFW/Unrestricted models only
python -m src.bin.cli --list-models --mode nsfw

# Show VRAM requirements
python -m src.bin.cli --show-vram
```

## Architecture

```
Input Image
    │
    ├── Content Rating Check (Safe/Mature/Unrestricted)
    │
    ▼
Scene Understanding
    ├── Depth Estimation (context-aware model selection)
    ├── Segmentation (context-aware model selection)
    └── Scene Classification
    │
    ▼
Motion Generation
    ├── Camera Trajectory
    └── Flow Fields
    │
    ▼
Video Synthesis (model selected by rating)
    ├── Safe: SVD, AnimateDiff-SDXL
    ├── Mature: ZeroScope, I2VGen
    └── Unrestricted: Open-SVD, OpenGIF
    │
    ▼
Temporal Stabilization
    │
    ▼
Export
```

## VRAM Requirements

| Model | VRAM | Quality |
|-------|------|---------|
| Open-SVD | 12GB | Excellent |
| OpenGIF | 10GB | Excellent |
| SVD | 12GB | Excellent |
| AnimateDiff SDXL | 12GB | High |
| ZeroScope | 6GB | Good |
| MobileSAM | 300MB | Medium |

## Motion Styles

- `cinematic`: Slow dolly/pan
- `subtle`: Minimal movement
- `environmental`: Natural elements
- `orbital`: Circular camera path
- `zoom-in` / `zoom-out`: Zoom effects
- `pan-left` / `pan-right`: Horizontal pan

## API Reference

### PictureAliver

```python
class PictureAliver:
    def convert(image, output_path, prompt="", **kwargs) -> PipelineResult
    def convert_with_motion(image, output_path, motion_style, **kwargs) -> PipelineResult
    def set_mode(mode) -> PictureAliver
    def set_model(category, model_name) -> PictureAliver
    def get_available_models() -> Dict
    def clear_cache()
```

### Model Registry

```python
from picture_aliver import MODEL_REGISTRY, ContentRating

# Get models by rating
nsfw_models = MODEL_REGISTRY.get_by_category(
    ModelCategory.I2V,
    rating=ContentRating.NSFW
)

# Get recommendations
recommendations = MODEL_REGISTRY.get_model_recommendations(
    rating=ContentRating.NSFW,
    vram_mb=12000,
    high_quality=True
)
```

## License

Apache 2.0

**Note**: Using unrestricted/NSFW models may have legal implications depending on your jurisdiction. Ensure compliance with local laws and platform terms of service.