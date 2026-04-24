/**
 * Mobile App Type Definitions
 */

// Generation state machine
export type GenerationState = 
  | 'idle'           // Initial state, ready to submit
  | 'uploading'      // Image being uploaded
  | 'processing'     // Generation in progress
  | 'completed'      // Video ready
  | 'failed'         // Generation failed
  | 'cancelled';    // User cancelled

// Generation request form data
export interface GenerationFormData {
  imageUri: string | null;
  imageBase64?: string;
  prompt: string;
  duration: number;        // 1-60 seconds
  fps: number;             // 1-60
  width: number;           // 256-2048
  height: number;         // 256-2048
  guidanceScale: number;   // 1-20
  motionStrength: number;  // 0-1
  motionMode: 'auto' | 'cinematic' | 'zoom' | 'pan' | 'subtle' | 'furry';
  enableQualityCheck: boolean;
}

// Default form values
export const DEFAULT_FORM_DATA: GenerationFormData = {
  imageUri: null,
  prompt: '',
  duration: 3.0,
  fps: 8,
  width: 512,
  height: 512,
  guidanceScale: 7.5,
  motionStrength: 0.8,
  motionMode: 'auto',
  enableQualityCheck: true,
};

// Task tracking info
export interface TaskInfo {
  taskId: string;
  status: GenerationState;
  progress: number;
  message: string;
  videoUrl?: string;
  error?: string;
  startedAt: Date;
  completedAt?: Date;
}

// App settings
export interface AppSettings {
  apiBaseUrl: string;
  autoQualityCheck: boolean;
  defaultDuration: number;
  defaultFps: number;
}

// Default settings
export const DEFAULT_SETTINGS: AppSettings = {
  apiBaseUrl: 'http://10.0.2.2:8000', // Android emulator
  autoQualityCheck: true,
  defaultDuration: 3.0,
  defaultFps: 8,
};

// Motion mode options for UI
export const MOTION_MODE_OPTIONS = [
  { label: 'Auto', value: 'auto', description: 'Automatically detect best motion' },
  { label: 'Cinematic', value: 'cinematic', description: 'Film-like camera movement' },
  { label: 'Zoom', value: 'zoom', description: 'Zoom in/out effect' },
  { label: 'Pan', value: 'pan', description: 'Horizontal panning' },
  { label: 'Subtle', value: 'subtle', description: 'Gentle, minimal motion' },
  { label: 'Furry', value: 'furry', description: 'Optimized for furry art' },
] as const;

// Validation helpers
export const validateFormData = (data: GenerationFormData): string[] => {
  const errors: string[] = [];

  if (!data.imageUri) {
    errors.push('Please select an image');
  }

  if (!data.prompt.trim()) {
    errors.push('Please enter a prompt');
  }

  if (data.duration < 1 || data.duration > 60) {
    errors.push('Duration must be between 1 and 60 seconds');
  }

  if (data.fps < 1 || data.fps > 60) {
    errors.push('FPS must be between 1 and 60');
  }

  if (data.width < 256 || data.width > 2048) {
    errors.push('Width must be between 256 and 2048');
  }

  if (data.height < 256 || data.height > 2048) {
    errors.push('Height must be between 256 and 2048');
  }

  if (data.guidanceScale < 1 || data.guidanceScale > 20) {
    errors.push('Guidance scale must be between 1 and 20');
  }

  if (data.motionStrength < 0 || data.motionStrength > 1) {
    errors.push('Motion strength must be between 0 and 1');
  }

  return errors;
};