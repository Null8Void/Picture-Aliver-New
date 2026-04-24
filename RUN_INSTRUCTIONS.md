# Picture-Aliver - Running the Full Stack

Complete instructions for running both the backend server and mobile app.

---

## Part 1: Backend Server (PC)

### Prerequisites
- Python 3.9+
- GPU recommended (2GB+ VRAM for minimum)
- pip installed

### Setup

```bash
# Navigate to project
cd E:/Picture-Aliver

# Install dependencies
pip install -r requirements.txt

# Verify Python environment
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Starting the Server

**Option A: Quick Start (Default settings)**
```bash
python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000
```

**Option B: Development mode (auto-reload)**
```bash
uvicorn src.picture_aliver.api:app --reload --host 0.0.0.0 --port 8000
```

**Option C: Custom port**
```bash
uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8080
```

### Verify Server is Running

```bash
# Health check (in another terminal)
curl http://localhost:8000/health

# Should return JSON like:
# {"status": "healthy", "gpu": {...}}
```

### Server Logs

Expected startup output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

## Part 2: Mobile App

### Prerequisites
- Node.js 18+
- npm or yarn
- Expo CLI: `npm install -g expo-cli`
- Android Studio (for Android) OR Xcode (for iOS)

### Setup

```bash
# Navigate to mobile app
cd E:/Picture-Aliver/mobile_app

# Install dependencies
npm install
```

### Configuration

Edit `lib/services/api.ts` to set your server IP:

```typescript
// Find this section and update with your PC's IP address
export const API_CONFIG = {
  DEFAULT_URLS: {
    // For Android Emulator: use 10.0.2.2
    android: 'http://10.0.2.2:8000',
    
    // For iOS Simulator: use localhost
    ios: 'http://localhost:8000',
    
    // For Physical Device: use your PC's local IP
    physical: 'http://192.168.1.100:8000',  // <-- UPDATE THIS
  },
  // ...
};
```

**Finding your PC's IP:**

Windows:
```cmd
ipconfig | findstr "IPv4"
```

Mac/Linux:
```bash
ifconfig | grep "inet "
```

### Run on Device

**Android Emulator:**
```bash
npm run android
# Or: npx expo run:android
```

**iOS Simulator:**
```bash
npm run ios
# Or: npx expo run:ios
```

**Physical Device (requires same WiFi):**

1. Start Metro bundler:
```bash
npm start
```

2. Open Expo Go app on your phone

3. Scan the QR code shown in terminal

### Mobile App Screens

1. **Home Screen**
   - Tap to select image (gallery/camera)
   - Enter prompt text
   - Adjust duration/FPS
   - Tap "Generate Video"
   - Watch progress
   - Preview result

2. **Settings Screen**
   - Configure API URL
   - Test connection
   - View server status

---

## Part 3: Testing the Full Stack

### Test 1: API Health Check

```bash
# From PC terminal
curl http://localhost:8000/health
```

### Test 2: Generate via API

```bash
# Prepare test image
curl -o test.png "https://picsum.photos/512/512"

# Generate video
curl -X POST http://localhost:8000/generate \
  -F "image=@test.png" \
  -F "prompt=gentle wave animation" \
  -F "duration=3" \
  -F "fps=8"

# Response:
# {"task_id":"abc12345","status":"pending","message":"..."}
```

### Test 3: Check Task Status

```bash
# Replace TASK_ID with actual ID from previous step
curl http://localhost:8000/tasks/TASK_ID
```

### Test 4: Download Video

```bash
# Replace TASK_ID
curl -O http://localhost:8000/download/TASK_ID -o output.mp4
```

---

## Part 4: Running the Testing Workflow

### Run All Tests

```bash
cd E:/Picture-Aliver
python -m tests.testing_workflow
```

### Expected Output

```
======================================================================
PICTURE-ALIVER TESTING WORKFLOW
======================================================================

14:32:01 | INFO     | TEST: portrait_subtle_motion
14:32:01 | INFO     | [IMAGE_LOAD] Starting...
14:32:01 | INFO     | [IMAGE_LOAD] Loaded: tests/images/portrait.png
14:32:02 | INFO     | [VIDEO_DIFF] Initializing pipeline...
14:32:45 | INFO     | [VERIFICATION] Analyzing...
14:32:46 | INFO     | [Metrics] Motion: 2.15 intensity
14:32:46 | INFO     | [PASS] portrait_subtle_motion (45.2s)

14:32:48 | INFO     | TEST: landscape_pan
...
14:32:98 | INFO     | [PASS] landscape_pan (52.1s)

14:33:00 | INFO     | TEST: object_rotation
...
14:33:45 | INFO     | [PASS] object_rotation (45.8s)

======================================================================
TEST SUMMARY
======================================================================
Total: 3 | Passed: 3 | Failed: 0
======================================================================
```

---

## Troubleshooting

### Server Issues

| Problem | Solution |
|---------|----------|
| "Address already in use" | Change port: `--port 8080` |
| "Module not found" | Run `pip install -r requirements.txt` |
| GPU not detected | Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"` |

### Mobile App Issues

| Problem | Solution |
|---------|----------|
| "Connection refused" | Verify server is running, check IP address |
| "Network Error" | Ensure phone and PC on same WiFi |
| App won't load | Run `npm start`, check Metro bundler |
| Image upload fails | Check file size < 10MB |

### Network Setup (Physical Device)

1. Disable firewall for port 8000:
   ```bash
   # Windows (Admin)
   netsh advfirewall firewall add rule name="PictureAliver" dir=in action=allow port=8000
   ```

2. Or add exception in Windows Firewall

3. Verify connection:
   ```bash
   # From phone's browser
   http://192.168.1.100:8000/health
   ```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000` | Start server |
| `python -m tests.testing_workflow` | Run tests |
| `npm start` | Start mobile dev server |
| `curl http://localhost:8000/health` | Health check |

---

## Architecture Overview

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│   Mobile App    │ ─────────────▶│  FastAPI Server  │
│  (Expo/React)   │  POST /generate│  (Uvicorn)       │
└─────────────────┘               └────────┬─────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
              │  Pipeline │        │   GPU     │        │  Output   │
              │  Stages   │───────▶│ Optimizer │───────▶│  Videos   │
              └───────────┘        └───────────┘        └───────────┘
```

## File Locations

| Component | Path |
|-----------|------|
| Backend API | `src/picture_aliver/api.py` |
| Pipeline | `src/picture_aliver/main.py` |
| Mobile App | `mobile_app/` |
| Tests | `tests/` |
| Outputs | `outputs/` |
| Debug | `debug/` |