/**
 * Picture-Aliver API Service
 * 
 * Handles all communication with the backend API server.
 * Supports:
 * - Health checks
 * - Video generation requests
 * - Task status polling
 * - Video file downloads
 */

import axios, { AxiosInstance, AxiosProgressEvent } from 'axios';

// API Configuration
const API_BASE_URL = 'http://10.0.2.2:8000'; // Android emulator localhost
// const API_BASE_URL = 'http://localhost:8000'; // iOS simulator
// const API_BASE_URL = 'https://your-server.com'; // Production server

// Types
export interface GenerationParams {
  prompt: string;
  duration?: number;
  fps?: number;
  width?: number;
  height?: number;
  guidanceScale?: number;
  motionStrength?: number;
  motionMode?: 'auto' | 'cinematic' | 'zoom' | 'pan' | 'subtle' | 'furry';
  enableQualityCheck?: boolean;
  sync?: boolean;
}

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  video_path?: string;
  error?: string;
  processing_time?: number;
}

export interface GenerationResponse {
  task_id: string;
  status: string;
  message: string;
  video_url?: string;
}

export interface HealthStatus {
  status: string;
  gpu: {
    cuda_available: boolean;
    device_count: number;
    device_name?: string;
    total_memory_gb?: number;
    memory_allocated_gb?: number;
  };
  timestamp: string;
}

// API Service Class
class ApiService {
  private client: AxiosInstance;
  private pollingIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000, // 60 second timeout
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[API] Response ${response.status} from ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('[API] Response error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Update the API base URL
   */
  setBaseUrl(url: string): void {
    this.client.defaults.baseURL = url;
  }

  /**
   * Check API health and GPU status
   */
  async checkHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  /**
   * Upload image and start video generation
   * 
   * @param imageUri - Local URI of the selected image
   * @param params - Generation parameters
   * @param onUploadProgress - Optional progress callback
   * @returns Generation response with task_id
   */
  async startGeneration(
    imageUri: string,
    params: GenerationParams,
    onUploadProgress?: (progress: number) => void
  ): Promise<GenerationResponse> {
    // Create form data
    const formData = new FormData();

    // Add image file
    const imageFile = {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'image.jpg',
    };
    formData.append('image', imageFile as any);

    // Add parameters
    formData.append('prompt', params.prompt);
    formData.append('duration', String(params.duration ?? 3.0));
    formData.append('fps', String(params.fps ?? 8));
    formData.append('width', String(params.width ?? 512));
    formData.append('height', String(params.height ?? 512));
    formData.append('guidance_scale', String(params.guidanceScale ?? 7.5));
    formData.append('motion_strength', String(params.motionStrength ?? 0.8));
    formData.append('motion_mode', params.motionMode ?? 'auto');
    formData.append(
      'enable_quality_check',
      String(params.enableQualityCheck ?? true)
    );
    formData.append('sync', String(params.sync ?? false));

    const response = await this.client.post<GenerationResponse>(
      '/generate',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (onUploadProgress && progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            onUploadProgress(progress);
          }
        },
      }
    );

    return response.data;
  }

  /**
   * Get status of a generation task
   */
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const response = await this.client.get<TaskStatus>(`/tasks/${taskId}`);
    return response.data;
  }

  /**
   * Get list of all tasks
   */
  async listTasks(): Promise<{ count: number; tasks: any[] }> {
    const response = await this.client.get('/tasks');
    return response.data;
  }

  /**
   * Get URL for downloading video
   */
  getVideoUrl(taskId: string): string {
    return `${this.client.defaults.baseURL}/download/${taskId}`;
  }

  /**
   * Start polling for task status
   * Calls callback with status updates until completed/failed or stopped
   */
  pollTaskStatus(
    taskId: string,
    onStatusUpdate: (status: TaskStatus) => void,
    intervalMs: number = 2000
  ): void {
    // Stop any existing polling for this task
    this.stopPolling(taskId);

    const poll = async () => {
      try {
        const status = await this.getTaskStatus(taskId);
        onStatusUpdate(status);

        // Stop polling if task is complete
        if (status.status === 'completed' || status.status === 'failed') {
          this.stopPolling(taskId);
        }
      } catch (error) {
        console.error(`[API] Polling error for task ${taskId}:`, error);
      }
    };

    // Start polling
    const intervalId = setInterval(poll, intervalMs);
    this.pollingIntervals.set(taskId, intervalId);

    // Initial poll
    poll();
  }

  /**
   * Stop polling for a specific task
   */
  stopPolling(taskId: string): void {
    const intervalId = this.pollingIntervals.get(taskId);
    if (intervalId) {
      clearInterval(intervalId);
      this.pollingIntervals.delete(taskId);
    }
  }

  /**
   * Stop all polling
   */
  stopAllPolling(): void {
    this.pollingIntervals.forEach((intervalId) => {
      clearInterval(intervalId);
    });
    this.pollingIntervals.clear();
  }

  /**
   * Check if a task is still being polled
   */
  isPolling(taskId: string): boolean {
    return this.pollingIntervals.has(taskId);
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;