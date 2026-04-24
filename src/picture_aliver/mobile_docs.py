"""
Picture-Aliver Mobile Integration Documentation

This document describes the mobile integration architecture for Picture-Aliver,
enabling image and video generation on mobile devices.
"""

# =============================================================================
# MOBILE INTEGRATION ARCHITECTURE
# =============================================================================

MOBILE_ARCHITECTURE = """
================================================================================
PICTURE-ALIVER MOBILE INTEGRATION
================================================================================

OVERVIEW
--------
Picture-Aliver provides two deployment modes for mobile:
1. On-Device Mode: Lightweight models running locally
2. Remote Compute Mode: Send to server for full pipeline

RECOMMENDED FRAMEWORK: React Native (with native modules)
- Cross-platform (iOS + Android) from single codebase
- Easier integration with native ML frameworks
- Good performance with native code bridges
- Large ecosystem and community support

ALTERNATIVE: Flutter
- Better UI performance but harder ML integration
- Consider if UI/UX is primary concern

================================================================================
"""

# =============================================================================
# API DESIGN
# =============================================================================

API_ENDPOINTS = """
================================================================================
API ENDPOINTS
================================================================================

BASE URL (Local Server Mode):
    http://[SERVER_IP]:8080/api/v1

--------------------------------------------------------------------------------
IMAGE ENDPOINTS
--------------------------------------------------------------------------------

POST /image/generate
    Generate image from text prompt
    
    Request:
    {
        "prompt": "a furry wolf character sitting in a forest",
        "negative_prompt": "blurry, distorted, deformed",
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": null
    }
    
    Response:
    {
        "success": true,
        "job_id": "uuid-string",
        "status": "completed",
        "image_url": "/api/v1/image/download/uuid-string",
        "processing_time_ms": 5420,
        "metadata": {
            "model": "stable-diffusion-1-5",
            "seed": 42
        }
    }

POST /image/generate-batch
    Generate multiple images from prompts
    
    Request:
    {
        "prompts": ["prompt1", "prompt2", "prompt3"],
        "same_params": true,
        "params": { ... }
    }
    
    Response:
    {
        "success": true,
        "job_id": "uuid-string",
        "status": "completed",
        "images": ["/api/v1/image/download/uuid1", ...],
        "processing_time_ms": 15420
    }

GET /image/download/{job_id}
    Download generated image
    
    Response: Binary image data (JPEG/PNG)

--------------------------------------------------------------------------------
VIDEO ENDPOINTS
--------------------------------------------------------------------------------

POST /video/generate
    Generate video from image or text
    
    Request (from image):
    {
        "image_id": "uploaded_image_id",
        "prompt": "furry character breathing naturally",
        "motion_prompt": "gentle breathing with subtle tail movement",
        "num_frames": 24,
        "fps": 8,
        "motion_strength": 0.8,
        "apply_stabilization": true
    }
    
    Request (from text):
    {
        "prompt": "a cat stretching",
        "text_to_video": true,
        "num_frames": 16,
        "fps": 8,
        "motion_prompt": "stretching motion"
    }
    
    Response:
    {
        "success": true,
        "job_id": "uuid-string",
        "status": "processing",
        "estimated_time_ms": 30000,
        "status_url": "/api/v1/job/status/uuid-string"
    }

GET /video/download/{job_id}
    Download generated video
    
    Response: Binary video data (MP4/WebM)

GET /job/status/{job_id}
    Get job processing status
    
    Response:
    {
        "job_id": "uuid-string",
        "status": "processing|completed|failed",
        "progress_percent": 45,
        "current_step": "video_generation",
        "error": null
    }

--------------------------------------------------------------------------------
UPLOAD ENDPOINTS
--------------------------------------------------------------------------------

POST /upload/image
    Upload image for processing
    
    Request: multipart/form-data
        - file: Image file (JPEG, PNG, WebP)
        - metadata: JSON string with optional metadata
    
    Response:
    {
        "success": true,
        "image_id": "uuid-string",
        "url": "/api/v1/image/download/uuid-string",
        "properties": {
            "width": 1024,
            "height": 768,
            "format": "png",
            "size_bytes": 245000
        }
    }

================================================================================
"""

# =============================================================================
# FILE HANDLING SPECIFICATIONS
# =============================================================================

FILE_HANDLING = """
================================================================================
FILE HANDLING SPECIFICATIONS
================================================================================

IMAGE UPLOAD
-----------
Supported formats: JPEG, PNG, WebP
Max file size: 50MB
Recommended size: < 10MB for fast upload
Preprocessing:
    - Auto-resize if > 2048px on any dimension
    - Convert RGBA to RGB
    - Strip EXIF metadata for privacy

UPLOAD FLOW:
    1. Client compresses image if needed
    2. Client uploads with progress callback
    3. Server validates format and size
    4. Server stores with unique ID
    5. Server returns image_id for referencing

VIDEO DOWNLOAD
--------------
Formats: MP4 (H.264), WebM (VP9)
Max duration: 30 seconds (on-device), 2 minutes (remote)
Default: 24 frames at 8fps = 3 seconds
Quality presets:
    - Low: 480p, 24fps
    - Medium: 720p, 24fps
    - High: 1080p, 24fps

DOWNLOAD OPTIONS:
    - Direct download (blocking)
    - Progressive download (chunked)
    - Streaming (for preview)

CHUNKED UPLOAD (Large files):
    POST /upload/chunked/init
    POST /upload/chunked/{upload_id}/chunk/{chunk_num}
    POST /upload/chunked/{upload_id}/complete

================================================================================
"""

# =============================================================================
# LATENCY EXPECTATIONS
# =============================================================================

LATENCY = """
================================================================================
LATENCY EXPECTATIONS
================================================================================

ON-DEVICE MODE
--------------
Image Generation (512x512):
    - Cold start: 5-15 seconds
    - Subsequent: 3-8 seconds
    
Video Generation (24 frames):
    - Generation: 10-30 seconds
    - Total (gen + stabilization + export): 20-60 seconds
    
Factors affecting latency:
    - Device RAM (min 4GB recommended)
    - Device thermal state
    - Background processes

REMOTE COMPUTE MODE
-------------------
Network round-trip (RTT):
    - Local network: 10-50ms
    - Internet (via tunnel): 100-300ms

Image Generation (512x512):
    - Network upload: 1-5 seconds (取决于 image size)
    - Server processing: 3-10 seconds
    - Network download: 1-3 seconds
    - Total: 5-18 seconds

Video Generation (24 frames):
    - Network upload: 1-5 seconds
    - Server processing: 10-45 seconds
    - Network download: 3-10 seconds
    - Total: 14-60 seconds

Progress Updates:
    - WebSocket for real-time progress
    - Polling fallback every 2 seconds

================================================================================
"""

# =============================================================================
# ON-DEVICE MODEL SPECIFICATIONS
# =============================================================================

ON_DEVICE_MODELS = """
================================================================================
ON-DEVICE MODEL SPECIFICATIONS
================================================================================

TEXT-TO-IMAGE MODELS
--------------------
Recommended: Stable Diffusion ONNX (Quantized)

Options:
1. StableDiffusion-ONNX (Recommended)
   - Size: ~1.5GB (int8 quantized)
   - Resolution: 512x512
   - Speed: 3-8 seconds on modern device

2. LoraLite (Distilled)
   - Size: ~500MB
   - Resolution: 512x512
   - Speed: 2-4 seconds
   - Quality tradeoff: 15-20% lower FID

3. TensorFlow Lite (Alternative)
   - Size: ~800MB
   - Resolution: 512x512
   - Speed: 4-10 seconds

FRAMEWORK: ONNX Runtime Mobile
    - Optimized for mobile CPUs/GPUs
    - Supports NNAPI on Android
    - Supports Metal on iOS

VIDEO GENERATION MODELS
------------------------
Recommended: Distilled I2V + Motion Templates

1. I2V Model (Distilled)
   - Size: ~800MB
   - Input: 512x512 image
   - Output: 8-16 frames
   - Speed: 5-15 seconds

2. Motion Templates (No ML)
   - Size: < 50MB
   - Pre-computed motion fields
   - Speed: 1-3 seconds
   - Quality: Good for specific motions

3. LoRA + Base Model
   - Size: ~600MB
   - Uses pre-trained weights
   - Speed: 8-20 seconds

MODEL QUANTIZATION
------------------
INT8 Quantization:
    - 4x smaller than FP32
    - ~10% accuracy loss (acceptable)
    - 2x faster inference

INT4 Quantization:
    - 8x smaller than FP32
    - ~20% accuracy loss
    - 4x faster inference
    - Use for older devices

DISTILLATION
------------
Knowledge Distillation for smaller models:
    - Teacher: Full SD model
    - Student: 50% size
    - Training: MSE + perceptual loss
    - Result: 50% size, 85% quality

================================================================================
"""

# =============================================================================
# SERVER ARCHITECTURE
# =============================================================================

SERVER_ARCHITECTURE = """
================================================================================
REMOTE COMPUTE SERVER ARCHITECTURE
================================================================================

SERVER REQUIREMENTS
------------------
- GPU: NVIDIA RTX 3080+ (10GB VRAM minimum)
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ for models
- Network: 1Gbps for fast transfers

DOCKER SETUP
------------
services:
  picture-aliver-api:
    image: picture-aliver/server:latest
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - /models:/app/models
      - /output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

API SERVER (FastAPI)
--------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/api/v1/image/generate")
async def generate_image(request: ImageRequest):
    # Process request
    pass

@app.post("/api/v1/video/generate")
async def generate_video(request: VideoRequest):
    # Queue job
    pass

@app.websocket("/ws/job/{job_id}")
async def job_progress(websocket, job_id):
    # Stream progress
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

JOB QUEUE (Redis + Celery)
--------------------------
- Job submitted -> Queue -> Worker -> Result storage
- Redis for queue management
- S3/MinIO for file storage
- Status polling or WebSocket updates

SCALING
-------
- Horizontal scaling: Multiple GPU workers
- Load balancer: Round-robin or least connections
- Auto-scaling: Based on queue depth
- Rate limiting: Per-user limits

================================================================================
"""

# =============================================================================
# MOBILE APP ARCHITECTURE
# =============================================================================

MOBILE_APP_ARCHITECTURE = """
================================================================================
MOBILE APP ARCHITECTURE
================================================================================

RECOMMENDED: React Native (Expo + Native Modules)

PROJECT STRUCTURE
-----------------
/PictureAliverApp
  /src
    /api
      client.ts           # API client
      endpoints.ts       # Endpoint definitions
      types.ts           # TypeScript types
    /screens
      HomeScreen.tsx
      ImageGenScreen.tsx
      VideoGenScreen.tsx
      SettingsScreen.tsx
    /components
      PromptInput.tsx
      ImagePreview.tsx
      VideoPlayer.tsx
      ProcessingIndicator.tsx
    /hooks
      useImageGeneration.ts
      useVideoGeneration.ts
      useUpload.ts
    /store
      settingsStore.ts
      historyStore.ts
    /native
      ImageProcessor.ts  # Native module
      ModelLoader.ts     # ONNX Runtime bridge
    /utils
      imageUtils.ts
      networkUtils.ts

STATE MANAGEMENT
----------------
Zustand for global state:
- Generation settings
- History
- Cached results
- User preferences

SCREENS
--------
1. HomeScreen
   - Quick actions (text2img, img2vid)
   - Recent generations
   - Mode toggle (on-device/remote)

2. ImageGenScreen
   - Text input for prompt
   - Negative prompt
   - Advanced settings (steps, scale, size)
   - Generate button
   - Preview with download/share

3. VideoGenScreen
   - Image upload / camera
   - Motion prompt input
   - Preview settings
   - Generate button
   - Video player with controls

4. SettingsScreen
   - Server connection (remote mode)
   - Model selection (on-device mode)
   - Quality presets
   - Clear cache

NATIVE MODULES
---------------
Android (Kotlin):
- ONNX Runtime integration
- Image preprocessing
- GPU acceleration

iOS (Swift):
- Core ML fallback
- Image preprocessing
- Metal acceleration

================================================================================
"""

# =============================================================================
# IMPLEMENTATION EXAMPLE
# =============================================================================

EXAMPLE_IMPLEMENTATION = """
================================================================================
IMPLEMENTATION EXAMPLES
================================================================================

API CLIENT (TypeScript)
-----------------------
// src/api/client.ts
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8080/api/v1',
  timeout: 120000,
});

export const imageApi = {
  generate: async (prompt: string, options?: Partial<ImageOptions>) => {
    const response = await api.post('/image/generate', {
      prompt,
      ...options,
    });
    return response.data;
  },
  
  uploadImage: async (file: File, onProgress?: (p: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => onProgress?.(e.loaded / e.total),
    });
    return response.data;
  },
};

export const videoApi = {
  generate: async (imageId: string, motionPrompt: string) => {
    const response = await api.post('/video/generate', {
      image_id: imageId,
      motion_prompt: motionPrompt,
      num_frames: 24,
      fps: 8,
    });
    return response.data;
  },
  
  getStatus: async (jobId: string) => {
    const response = await api.get(`/job/status/${jobId}`);
    return response.data;
  },
};

VIDEO PLAYER (React Native)
----------------------------
// src/components/VideoPlayer.tsx
import React from 'react';
import { Video, ResizeMode } from 'expo-av';
import { ActivityIndicator, View, StyleSheet } from 'expo';

interface Props {
  uri: string;
  onLoad?: () => void;
}

export const VideoPlayer: React.FC<Props> = ({ uri, onLoad }) => {
  return (
    <Video
      source={{ uri }}
      style={styles.video}
      resizeMode={ResizeMode.CONTAIN}
      useNativeControls
      onLoad={onLoad}
    />
  );
};

const styles = StyleSheet.create({
  video: {
    width: '100%',
    aspectRatio: 16/9,
  },
});

ON-DEVICE GENERATION (Android)
--------------------------------
// Native module for ONNX Runtime
public class OnnxGenerator {
    private OrtEnvironment env;
    private OrtSession session;
    
    public void loadModel(String modelPath) {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }
    
    public float[][][][] generate(String prompt, int width, int height) {
        // Tokenize prompt
        long[] inputTokens = tokenize(prompt);
        
        // Run inference
        float[] inputData = preprocess(inputTokens, width, height);
        float[] output = runInference(inputData);
        
        // Post-process to image
        return postprocess(output, width, height);
    }
}

================================================================================
"""


def get_mobile_docs():
    """Return the complete mobile integration documentation."""
    return "\n".join([
        MOBILE_ARCHITECTURE,
        API_ENDPOINTS,
        FILE_HANDLING,
        LATENCY,
        ON_DEVICE_MODELS,
        SERVER_ARCHITECTURE,
        MOBILE_APP_ARCHITECTURE,
        EXAMPLE_IMPLEMENTATION,
    ])


if __name__ == "__main__":
    print(get_mobile_docs())