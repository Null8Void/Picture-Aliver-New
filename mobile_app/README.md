# Picture-Aliver Mobile App

React Native (Expo) mobile application for AI image-to-video generation.

## Features

- **Image Selection**: Pick from gallery or take photo with camera
- **Text Prompt**: Describe the desired animation/motion
- **Quick Settings**: Duration and FPS controls
- **Advanced Options**: Resolution, motion mode, strength
- **Real-time Progress**: Track generation status
- **Video Preview**: Watch generated video in-app
- **Async Processing**: Non-blocking generation with polling

## Requirements

- Node.js 18+
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- Android Studio (for Android) or Xcode (for iOS)

## Setup

### 1. Install Dependencies

```bash
cd mobile_app
npm install
```

### 2. Configure API URL

Edit `lib/services/api.ts` to set your backend server URL:

```typescript
// For Android emulator (uses 10.0.2.2 for localhost)
const API_BASE_URL = 'http://10.0.2.2:8000';

// For iOS simulator
const API_BASE_URL = 'http://localhost:8000';

// For physical device or production
const API_BASE_URL = 'https://your-server.com:8000';
```

### 3. Start Development Server

```bash
# Start Metro bundler
npm start

# Or with Expo
npx expo start
```

### 4. Run on Device/Emulator

```bash
# Android
npm run android
# or npx expo run:android

# iOS
npm run ios
# or npx expo run:ios
```

## API Integration

The app connects to the `src/picture_aliver/api.py` FastAPI backend.

### Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Submit image + prompt, returns task_id |
| GET | `/tasks/{task_id}` | Poll for task status |
| GET | `/download/{task_id}` | Download generated video |
| GET | `/health` | Check API health |

### Request Flow

```
1. User selects image + enters prompt
2. App uploads to /generate (multipart/form-data)
3. API returns task_id immediately (async mode)
4. App polls /tasks/{task_id} every 2 seconds
5. On completion, app plays video from /download/{task_id}
```

## Project Structure

```
mobile_app/
├── App.tsx                    # Main app entry
├── app/
│   └── index.tsx             # Expo Router layout
├── lib/
│   ├── screens/
│   │   └── HomeScreen.tsx    # Main generation UI
│   ├── services/
│   │   └── api.ts            # Backend API service
│   ├── models/
│   │   └── types.ts          # TypeScript types
│   └── widgets/              # Reusable components
├── assets/                   # Images and icons
├── package.json
└── app.json                  # Expo configuration
```

## UI Screens

### Home Screen

- Image picker (tap to select/gallery/camera)
- Prompt text input
- Duration and FPS quick controls
- Advanced settings (expandable)
- Generate/Cancel/New Generation buttons
- Status progress indicator
- Video preview modal

## Permissions

### Android

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

### iOS

```xml
<!-- ios/Info.plist -->
<key>NSCameraUsageDescription</key>
<string>Take photos for video generation</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Select images for video generation</string>
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

## Troubleshooting

### Connection Refused

If the app can't connect to the API:
1. Ensure the backend server is running (`uvicorn src.picture_aliver.api:app`)
2. Check the API URL in `api.ts` matches your server
3. For Android emulator, use `10.0.2.2:8000` instead of `localhost:8000`
4. For physical devices, use your computer's local IP address

### Slow Generation

Video generation is computationally intensive. On mobile:
- Use lower resolutions (256 or 512)
- Reduce duration (1-3 seconds)
- Enable async mode to avoid timeout

### Video Not Playing

Ensure video format compatibility:
- App outputs MP4 (H.264 codec)
- Works on Android 5+ and iOS 12+