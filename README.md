# Picture-Aliver

**AI Image-to-Video Generation Pipeline** - Convert static images into animated videos using deep learning.

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Pipeline** | 9-stage pipeline: Image → Depth → Segmentation → Motion → Video → Stabilization → Export |
| **Auto-Correction** | Detects flickering, warped faces, structural instability and auto-corrects |
| **GPU Optimization** | FP16, VRAM monitoring, adaptive scaling for 2GB-24GB GPUs |
| **Debug System** | Saves intermediate outputs: depth maps, masks, frames, motion fields |
| **REST API** | FastAPI backend for mobile/web integrations |
| **Mobile App** | React Native app with image selection and video preview |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| VRAM | 2GB | 8GB+ |
| RAM | 8GB | 16GB+ |
| Storage | 10GB | 20GB+ |
| Python | 3.9+ | 3.10+ |

### GPU Tiers

| VRAM | Resolution | FPS | Quality |
|------|------------|-----|---------|
| 2GB | 384x384 | 4 | Minimum |
| 4GB | 512x512 | 6 | Basic |
| 8GB | 768x768 | 12 | Standard |
| 12GB | 1024x1024 | 24 | High |
| 24GB | 1280x1280 | 30 | Ultra |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Null8Void/Picture-Aliver.git
cd Picture-Aliver
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from src.picture_aliver.main import Pipeline; print('Pipeline OK')"
```

---

## How to Run

### Backend Server (PC)

```bash
# Start API server on all interfaces
python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000

# Development mode with auto-reload
uvicorn src.picture_aliver.api:app --reload --host 0.0.0.0 --port 8000
```

Verify server:
```bash
curl http://localhost:8000/health
```

### CLI Usage

```bash
# Basic usage
python main.py --image photo.jpg --prompt "gentle wave animation"

# With options
python main.py --image photo.jpg --prompt "cinematic pan" --duration 5 --fps 12

# Use custom config
python main.py --image photo.jpg --prompt "motion" --config configs/default.yaml
```

### Mobile App

```bash
cd mobile_app

# Install dependencies
npm install

# Configure API URL in lib/services/api.ts (update physical device IP)

# Start development
npm start

# Run on device
npx expo run:android  # Android
npx expo run:ios       # iOS
```

---

## Testing

### Run Test Suite

```bash
# Run all test cases
python -m tests.testing_workflow

# Test cases:
# - portrait_subtle_motion
# - landscape_pan
# - object_rotation
```

### Test Output

```
======================================================================
TEST SUMMARY
======================================================================
Total: 3 | Passed: 3 | Failed: 0

  [PASS] portrait_subtle_motion (45.2s)
  [PASS] landscape_pan (52.1s)
  [PASS] object_rotation (45.8s)
======================================================================
```

### Quality Metrics

Each test verifies:
- **Flickering**: Brightness variance < 15 (lower = better)
- **Motion**: Optical flow intensity > 0.5
- **Stability**: SSIM > 0.7 between frames

---

## Project Structure

```
Picture-Aliver/
├── main.py                    # CLI entry point
├── src/
│   └── picture_aliver/
│       ├── main.py           # Pipeline orchestration
│       ├── config.yaml       # Default configuration
│       ├── api.py            # FastAPI server
│       ├── gpu_optimization.py
│       ├── config_loader.py
│       └── debug_saver.py
├── mobile_app/               # React Native app
│   ├── App.tsx
│   ├── lib/
│   │   ├── screens/         # HomeScreen, SettingsScreen
│   │   ├── services/       # API service
│   │   └── models/         # TypeScript types
│   └── package.json
├── tests/
│   ├── testing_workflow.py  # Test runner
│   └── README.md
├── configs/
│   └── default.yaml         # Pipeline config
├── models/                   # Downloaded models
├── outputs/                  # Generated videos
├── debug/                    # Debug outputs
├── requirements.txt
├── setup.py
└── README.md
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Start video generation |
| GET | `/tasks/{id}` | Get task status |
| GET | `/tasks` | List all tasks |
| GET | `/download/{id}` | Download video |
| GET | `/health` | Server health check |

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -F "image=@photo.jpg" \
  -F "prompt=gentle animation" \
  -F "duration=3" \
  -F "fps=8"
```

### Response

```json
{
  "task_id": "abc12345",
  "status": "pending",
  "message": "Generation started"
}
```

---

## Configuration

### Pipeline Config (`src/picture_aliver/config.yaml`)

```yaml
debug:
  enabled: false
  directory: "./debug"
  save:
    depth_maps: true
    segmentation_masks: true
    raw_frames: true
    stabilized_frames: true
    motion_fields: true

video:
  duration_seconds: 3.0
  fps: 8
  resolution:
    width: 512
    height: 512

generation:
  num_inference_steps: 25
  guidance_scale: 7.5

gpu:
  device: "auto"
  precision: "fp16"

motion:
  mode: "auto"
  strength: 0.8
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Memory settings |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| GPU not detected | Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |
| Out of memory | Reduce resolution or use `--device cpu` |
| Connection refused | Ensure server running, check firewall |
| App won't connect | Update API URL in mobile app settings |

### Common Fixes

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check GPU info
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

---

## Development

### Adding New Modules

1. Create module in `src/picture_aliver/`
2. Add imports to `main.py`
3. Add pipeline step in `Pipeline` class
4. Update `DebugConfig` if debug output needed
5. Add tests in `tests/`

### Code Style

```bash
# Format code
black src/

# Type check
mypy src/
```

---

## License

Apache 2.0

---

## Credits

Built with:
- PyTorch / TorchVision
- Diffusers (HuggingFace)
- OpenCV
- FastAPI
- React Native / Expo