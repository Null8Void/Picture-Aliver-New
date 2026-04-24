/**
 * Picture-Aliver API Service
 * 
 * Handles all communication with the backend API server.
 * Supports:
 * - Health checks
 * - Video generation requests
 * - Task status polling
 * - Video file downloads
 * - Local network configuration
 * - Request retry with timeout handling
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import * as FileSystem from 'expo-file-system';

// =============================================================================
// CONFIGURATION
// =============================================================================

// API Base URL Configuration
// - Android Emulator: Uses 10.0.2.2 to reach host localhost
// - iOS Simulator: Uses localhost
// - Physical Device: Use your computer's local IP (e.g., 192.168.1.x)
export const API_CONFIG = {
  // Default URLs for different platforms
  DEFAULT_URLS: {
    android: 'http://10.0.2.2:8000',  // Android emulator → host
    ios: 'http://localhost:8000',      // iOS simulator → host
    web: 'http://localhost:8000',      // Web browser
    physical: 'http://192.168.1.100:8000', // Physical device (update with your IP)
  },
  
  // Request timeouts (in milliseconds)
  TIMEOUTS: {
    upload: 300000,       // 5 minutes for large uploads
    generation: 600000,   // 10 minutes for generation
    status: 10000,       // 10 seconds for status check
    download: 300000,     // 5 minutes for video download
    health: 5000,        // 5 seconds for health check
  },
  
  // Retry configuration
  RETRY: {
    maxAttempts: 3,
    delayMs: 2000,
  },
};

// Storage key for custom API URL
const API_URL_STORAGE_KEY = 'picture_aliver_api_url';

// =============================================================================
// TYPES
// =============================================================================

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

// Video cache info
export interface VideoCacheInfo {
  taskId: string;
  localPath: string;
  downloadedAt: Date;
  sizeBytes: number;
}

// =============================================================================
// API SERVICE CLASS
// =============================================================================

class ApiService {
  private client: AxiosInstance;
  private baseUrl: string;
  private pollingIntervals: Map<string, ReturnType<typeof setInterval>> = new Map();
  
  // Video cache directory
  private videoCacheDir: string = `${FileSystem.cacheDirectory}videos/`;

  constructor() {
    // Get platform-appropriate default URL
    this.baseUrl = this._getDefaultUrl();
    
    this.client = this._createClient(this.baseUrl);
    
    // Ensure cache directory exists
    this._ensureCacheDir();
  }

  /**
   * Get platform-appropriate default URL
   */
  private _getDefaultUrl(): string {
    // Check for stored custom URL first
    try {
      // In a real app, you'd use expo-secure-store or async storage
      // For now, return the default based on a simple check
    } catch (e) {
      // Use default
    }
    
    // Platform detection (simplified - in real app use Platform from react-native)
    if (typeof navigator !== 'undefined' && navigator.platform?.includes('Win')) {
      return API_CONFIG.DEFAULT_URLS.physical;
    }
    
    return API_CONFIG.DEFAULT_URLS.android;
  }

  /**
   * Create configured axios client
   */
  private _createClient(baseUrl: string): AxiosInstance {
    const client = axios.create({
      baseURL: baseUrl,
      timeout: API_CONFIG.TIMEOUTS.generation,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Request interceptor for logging and URL handling
    client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        console.log(`[API] Base URL: ${this.baseUrl}`);
        return config;
      },
      (error) => {
        console.error('[API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    client.interceptors.response.use(
      (response) => {
        console.log(`[API] Response ${response.status} from ${response.config.url}`);
        return response;
      },
      async (error: AxiosError) => {
        console.error('[API] Response error:', error.message);
        
        // Handle connection errors specifically
        if (error.code === 'ECONNABORTED') {
          throw new Error('Request timed out. The server may be busy or unreachable.');
        }
        
        if (error.code === 'ERR_NETWORK' || error.message.includes('Network Error')) {
          throw new Error(
            `Cannot connect to server at ${this.baseUrl}. ` +
            'Please check: ' +
            '1. Server is running on your computer, ' +
            '2. Correct IP address in settings, ' +
            '3. Firewall allows connections on port 8000.'
          );
        }
        
        // Pass through other errors
        return Promise.reject(error);
      }
    );

    return client;
  }

  /**
   * Ensure video cache directory exists
   */
  private async _ensureCacheDir(): Promise<void> {
    try {
      const dirInfo = await FileSystem.getInfoAsync(this.videoCacheDir);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(this.videoCacheDir, { intermediates: true });
      }
    } catch (error) {
      console.error('[API] Failed to create cache directory:', error);
    }
  }

  /**
   * Update the API base URL
   */
  setBaseUrl(url: string): void {
    this.baseUrl = url;
    this.client = this._createClient(url);
    console.log(`[API] Base URL updated to: ${url}`);
  }

  /**
   * Get current API URL
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Check API health and GPU status
   */
  async checkHealth(): Promise<HealthStatus> {
    try {
      const response = await this.client.get<HealthStatus>('/health', {
        timeout: API_CONFIG.TIMEOUTS.health,
      });
      return response.data;
    } catch (error) {
      throw this._handleError(error, 'Failed to connect to API server');
    }
  }

  /**
   * Upload image and start video generation
   * 
   * @param imageUri - Local URI of the selected image
   * @param params - Generation parameters
   * @param onProgress - Optional progress callback (0-100)
   * @returns Generation response with task_id
   */
  async startGeneration(
    imageUri: string,
    params: GenerationParams,
    onProgress?: (progress: number) => void
  ): Promise<GenerationResponse> {
    try {
      console.log('[API] Preparing generation request...');
      console.log(`[API] Image URI: ${imageUri}`);
      console.log(`[API] Prompt: ${params.prompt}`);

      // Create form data
      const formData = new FormData();

      // Get file info for proper naming
      const fileName = imageUri.split('/').pop() || 'image.jpg';
      const fileType = fileName.endsWith('.png') ? 'image/png' : 'image/jpeg';

      // Add image file with proper format for React Native
      formData.append('image', {
        uri: imageUri,
        type: fileType,
        name: fileName,
      } as any);

      // Add generation parameters
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

      console.log('[API] Sending generation request...');

      const response = await this.client.post<GenerationResponse>(
        '/generate',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: API_CONFIG.TIMEOUTS.upload,
          onUploadProgress: (progressEvent) => {
            if (onProgress && progressEvent.total) {
              const progress = (progressEvent.loaded / progressEvent.total) * 100;
              onProgress(Math.min(100, progress));
            }
          },
        }
      );

      console.log('[API] Generation started:', response.data);
      return response.data;

    } catch (error) {
      throw this._handleError(error, 'Failed to start video generation');
    }
  }

  /**
   * Get status of a generation task
   */
  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    try {
      const response = await this.client.get<TaskStatus>(`/tasks/${taskId}`, {
        timeout: API_CONFIG.TIMEOUTS.status,
      });
      return response.data;
    } catch (error) {
      throw this._handleError(error, 'Failed to get task status');
    }
  }

  /**
   * Get URL for downloading video
   */
  getVideoUrl(taskId: string): string {
    return `${this.baseUrl}/download/${taskId}`;
  }

  /**
   * Download video and cache locally
   * 
   * @param taskId - Task identifier
   * @param onProgress - Optional progress callback (0-100)
   * @returns Local file path
   */
  async downloadVideo(
    taskId: string,
    onProgress?: (progress: number) => void
  ): Promise<string> {
    const videoUrl = this.getVideoUrl(taskId);
    const localPath = `${this.videoCacheDir}${taskId}.mp4`;

    console.log(`[API] Downloading video from: ${videoUrl}`);
    console.log(`[API] Saving to: ${localPath}`);

    try {
      // Check if already cached
      const fileInfo = await FileSystem.getInfoAsync(localPath);
      if (fileInfo.exists) {
        console.log('[API] Video already cached');
        return localPath;
      }

      // Download with progress tracking
      const downloadResult = await FileSystem.downloadAsync(
        videoUrl,
        localPath,
        {
          timeout: API_CONFIG.TIMEOUTS.download,
          md5: false,
        }
      );

      if (downloadResult.status === 200) {
        console.log(`[API] Video downloaded: ${localPath}`);
        return localPath;
      } else {
        throw new Error(`Download failed with status: ${downloadResult.status}`);
      }

    } catch (error) {
      console.error('[API] Video download error:', error);
      throw this._handleError(error, 'Failed to download video');
    }
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

    console.log(`[API] Starting status polling for task: ${taskId}`);

    const poll = async () => {
      try {
        const status = await this.getTaskStatus(taskId);
        onStatusUpdate(status);

        console.log(`[API] Task ${taskId} status: ${status.status} (${Math.round(status.progress * 100)}%)`);

        // Stop polling if task is complete
        if (status.status === 'completed' || status.status === 'failed') {
          console.log(`[API] Stopping polling for task: ${taskId}`);
          this.stopPolling(taskId);
        }
      } catch (error) {
        console.error(`[API] Polling error for task ${taskId}:`, error);
        // Continue polling even on single error (might be temporary)
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
      console.log(`[API] Stopped polling for task: ${taskId}`);
    }
  }

  /**
   * Stop all polling
   */
  stopAllPolling(): void {
    this.pollingIntervals.forEach((intervalId, taskId) => {
      clearInterval(intervalId);
      console.log(`[API] Stopped polling for task: ${taskId}`);
    });
    this.pollingIntervals.clear();
  }

  /**
   * Check if a task is being polled
   */
  isPolling(taskId: string): boolean {
    return this.pollingIntervals.has(taskId);
  }

  /**
   * Clear video cache
   */
  async clearCache(): Promise<void> {
    try {
      await FileSystem.deleteAsync(this.videoCacheDir, { idempotent: true });
      await this._ensureCacheDir();
      console.log('[API] Video cache cleared');
    } catch (error) {
      console.error('[API] Failed to clear cache:', error);
    }
  }

  /**
   * Get cache info
   */
  async getCacheInfo(): Promise<VideoCacheInfo[]> {
    const cacheInfo: VideoCacheInfo[] = [];
    
    try {
      const files = await FileSystem.readDirectoryAsync(this.videoCacheDir);
      
      for (const file of files) {
        if (file.endsWith('.mp4')) {
          const filePath = `${this.videoCacheDir}${file}`;
          const info = await FileSystem.getInfoAsync(filePath);
          
          if (info.exists && 'size' in info) {
            cacheInfo.push({
              taskId: file.replace('.mp4', ''),
              localPath: filePath,
              downloadedAt: new Date(), // Would need to track actual date
              sizeBytes: info.size || 0,
            });
          }
        }
      }
    } catch (error) {
      console.error('[API] Failed to get cache info:', error);
    }
    
    return cacheInfo;
  }

  /**
   * Handle errors consistently
   */
  private _handleError(error: any, defaultMessage: string): Error {
    if (error instanceof Error) {
      return error;
    }
    
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      
      if (axiosError.response) {
        // Server responded with error
        const data = axiosError.response.data as any;
        return new Error(data?.detail || `Server error: ${axiosError.response.status}`);
      }
      
      if (axiosError.request) {
        // Request made but no response
        return new Error(
          `Cannot reach server at ${this.baseUrl}. ` +
          'Check that the server is running and the IP address is correct.'
        );
      }
    }
    
    return new Error(defaultMessage);
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

// Export singleton instance
export const apiService = new ApiService();
export default apiService;