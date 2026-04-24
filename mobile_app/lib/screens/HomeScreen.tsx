/**
 * Home Screen - Main generation interface
 * 
 * Features:
 * - Image selection/upload
 * - Prompt text input
 * - Advanced parameter sliders
 * - Generate button with progress
 * - Video preview on completion
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Image,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  Dimensions,
  Modal,
  Platform,
} from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { apiService, GenerationParams, TaskStatus } from '../services/api';
import {
  GenerationFormData,
  DEFAULT_FORM_DATA,
  TaskInfo,
  MOTION_MODE_OPTIONS,
  validateFormData,
} from '../models/types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function HomeScreen() {
  // Form state
  const [formData, setFormData] = useState<GenerationFormData>(DEFAULT_FORM_DATA);
  
  // Generation state
  const [taskInfo, setTaskInfo] = useState<TaskInfo | null>(null);
  const [generationState, setGenerationState] = useState<GenerationState>('idle');
  
  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewVideo, setPreviewVideo] = useState<string | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  
  // Video player ref
  const videoRef = useRef<Video>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (taskInfo?.taskId) {
        apiService.stopPolling(taskInfo.taskId);
      }
    };
  }, [taskInfo?.taskId]);

  /**
   * Pick image from gallery
   */
  const pickImage = useCallback(async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (!permissionResult.granted) {
        Alert.alert(
          'Permission Required',
          'Please grant access to your photo library to select images.'
        );
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
      });

      if (!result.canceled && result.assets[0]) {
        setFormData(prev => ({
          ...prev,
          imageUri: result.assets[0].uri,
        }));
        // Reset video when new image selected
        setPreviewVideo(null);
        setTaskInfo(null);
        setGenerationState('idle');
      }
    } catch (error) {
      console.error('Image picker error:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  }, []);

  /**
   * Take photo with camera
   */
  const takePhoto = useCallback(async () => {
    try {
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      
      if (!permissionResult.granted) {
        Alert.alert(
          'Permission Required',
          'Please grant camera access to take photos.'
        );
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.9,
      });

      if (!result.canceled && result.assets[0]) {
        setFormData(prev => ({
          ...prev,
          imageUri: result.assets[0].uri,
        }));
        setPreviewVideo(null);
        setTaskInfo(null);
        setGenerationState('idle');
      }
    } catch (error) {
      console.error('Camera error:', error);
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    }
  }, []);

  /**
   * Update form field
   */
  const updateField = useCallback(<K extends keyof GenerationFormData>(
    field: K,
    value: GenerationFormData[K]
  ) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  }, []);

  /**
   * Start video generation
   */
  const startGeneration = useCallback(async () => {
    // Validate
    const errors = validateFormData(formData);
    if (errors.length > 0) {
      Alert.alert('Validation Error', errors.join('\n'));
      return;
    }

    if (!formData.imageUri) {
      Alert.alert('Error', 'Please select an image first.');
      return;
    }

    setGenerationState('uploading');
    setUploadProgress(0);

    try {
      // Start generation
      const params: GenerationParams = {
        prompt: formData.prompt,
        duration: formData.duration,
        fps: formData.fps,
        width: formData.width,
        height: formData.height,
        guidanceScale: formData.guidanceScale,
        motionStrength: formData.motionStrength,
        motionMode: formData.motionMode,
        enableQualityCheck: formData.enableQualityCheck,
        sync: false, // Use async mode for mobile
      };

      const response = await apiService.startGeneration(
        formData.imageUri,
        params,
        (progress) => setUploadProgress(progress)
      );

      // Create task info
      const newTaskInfo: TaskInfo = {
        taskId: response.task_id,
        status: 'processing',
        progress: 0,
        message: 'Generation started',
        startedAt: new Date(),
      };
      setTaskInfo(newTaskInfo);
      setGenerationState('processing');

      // Start polling for status
      apiService.pollTaskStatus(
        response.task_id,
        (status: TaskStatus) => {
          setTaskInfo(prev => {
            if (!prev) return null;
            return {
              ...prev,
              status: status.status as GenerationState,
              progress: status.progress,
              message: status.message,
              videoUrl: status.video_path ? apiService.getVideoUrl(status.task_id) : prev.videoUrl,
              error: status.error,
              completedAt: status.status === 'completed' ? new Date() : undefined,
            };
          });

          // Update generation state based on task status
          if (status.status === 'completed') {
            setGenerationState('completed');
            setPreviewVideo(apiService.getVideoUrl(response.task_id));
          } else if (status.status === 'failed') {
            setGenerationState('failed');
          }
        }
      );

    } catch (error: any) {
      console.error('Generation error:', error);
      setGenerationState('failed');
      setTaskInfo(prev => prev ? {
        ...prev,
        status: 'failed',
        error: error.response?.data?.detail || error.message || 'Generation failed',
      } : null);
      Alert.alert('Error', 'Failed to start generation. Please check your connection.');
    }
  }, [formData]);

  /**
   * Cancel current generation
   */
  const cancelGeneration = useCallback(() => {
    if (taskInfo?.taskId) {
      apiService.stopPolling(taskInfo.taskId);
    }
    setGenerationState('cancelled');
    setTaskInfo(null);
  }, [taskInfo?.taskId]);

  /**
   * Reset for new generation
   */
  const resetGeneration = useCallback(() => {
    setFormData(DEFAULT_FORM_DATA);
    setTaskInfo(null);
    setPreviewVideo(null);
    setGenerationState('idle');
    setUploadProgress(0);
  }, []);

  /**
   * Show image picker options
   */
  const showImageOptions = useCallback(() => {
    Alert.alert(
      'Select Image',
      'Choose how you want to add an image',
      [
        { text: 'Take Photo', onPress: takePhoto },
        { text: 'Choose from Gallery', onPress: pickImage },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  }, [pickImage, takePhoto]);

  // Render status indicator
  const renderStatusIndicator = () => {
    switch (generationState) {
      case 'uploading':
        return (
          <View style={styles.statusContainer}>
            <ActivityIndicator size="large" color="#6366F1" />
            <Text style={styles.statusText}>
              Uploading image... {Math.round(uploadProgress)}%
            </Text>
          </View>
        );
      case 'processing':
        return (
          <View style={styles.statusContainer}>
            <ActivityIndicator size="large" color="#6366F1" />
            <Text style={styles.statusText}>
              {taskInfo?.message || 'Processing...'}
            </Text>
            {taskInfo?.progress !== undefined && (
              <View style={styles.progressBar}>
                <View style={[styles.progressFill, { width: `${taskInfo.progress * 100}%` }]} />
              </View>
            )}
          </View>
        );
      case 'completed':
        return (
          <View style={styles.statusContainer}>
            <Text style={styles.successText}>Video Ready!</Text>
            <TouchableOpacity
              style={styles.previewButton}
              onPress={() => setShowPreviewModal(true)}
            >
              <Text style={styles.previewButtonText}>Watch Preview</Text>
            </TouchableOpacity>
          </View>
        );
      case 'failed':
        return (
          <View style={styles.statusContainer}>
            <Text style={styles.errorText}>Generation Failed</Text>
            <Text style={styles.errorDetail}>{taskInfo?.error}</Text>
          </View>
        );
      default:
        return null;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#F9FAFB" />
      
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Picture-Aliver</Text>
          <Text style={styles.subtitle}>AI Image-to-Video</Text>
        </View>

        {/* Image Selection */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Source Image</Text>
          <TouchableOpacity
            style={styles.imagePickerContainer}
            onPress={showImageOptions}
            disabled={generationState === 'processing' || generationState === 'uploading'}
          >
            {formData.imageUri ? (
              <Image
                source={{ uri: formData.imageUri }}
                style={styles.selectedImage}
                resizeMode="cover"
              />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.imagePlaceholderIcon}>+</Text>
                <Text style={styles.imagePlaceholderText}>
                  Tap to add image
                </Text>
              </View>
            )}
          </TouchableOpacity>
          {formData.imageUri && generationState === 'idle' && (
            <TouchableOpacity
              style={styles.changeImageButton}
              onPress={showImageOptions}
            >
              <Text style={styles.changeImageText}>Change Image</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Prompt Input */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Animation Prompt</Text>
          <TextInput
            style={styles.promptInput}
            placeholder="Describe the motion you want (e.g., gentle wave, wind blowing)"
            placeholderTextColor="#9CA3AF"
            value={formData.prompt}
            onChangeText={(text) => updateField('prompt', text)}
            multiline
            numberOfLines={3}
            textAlignVertical="top"
            editable={generationState === 'idle' || generationState === 'cancelled'}
          />
        </View>

        {/* Quick Settings */}
        <View style={styles.section}>
          <View style={styles.quickSettingsRow}>
            <View style={styles.quickSettingItem}>
              <Text style={styles.quickSettingLabel}>Duration</Text>
              <View style={styles.quickSettingValue}>
                <TouchableOpacity
                  style={styles.stepButton}
                  onPress={() => updateField('duration', Math.max(1, formData.duration - 1))}
                  disabled={generationState !== 'idle'}
                >
                  <Text>-</Text>
                </TouchableOpacity>
                <Text style={styles.stepValue}>{formData.duration}s</Text>
                <TouchableOpacity
                  style={styles.stepButton}
                  onPress={() => updateField('duration', Math.min(60, formData.duration + 1))}
                  disabled={generationState !== 'idle'}
                >
                  <Text>+</Text>
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.quickSettingItem}>
              <Text style={styles.quickSettingLabel}>FPS</Text>
              <View style={styles.quickSettingValue}>
                <TouchableOpacity
                  style={styles.stepButton}
                  onPress={() => updateField('fps', Math.max(1, formData.fps - 1))}
                  disabled={generationState !== 'idle'}
                >
                  <Text>-</Text>
                </TouchableOpacity>
                <Text style={styles.stepValue}>{formData.fps}</Text>
                <TouchableOpacity
                  style={styles.stepButton}
                  onPress={() => updateField('fps', Math.min(60, formData.fps + 1))}
                  disabled={generationState !== 'idle'}
                >
                  <Text>+</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        {/* Advanced Toggle */}
        <TouchableOpacity
          style={styles.advancedToggle}
          onPress={() => setShowAdvanced(!showAdvanced)}
        >
          <Text style={styles.advancedToggleText}>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </Text>
          <Text style={styles.advancedToggleIcon}>
            {showAdvanced ? '▲' : '▼'}
          </Text>
        </TouchableOpacity>

        {/* Advanced Settings */}
        {showAdvanced && (
          <View style={styles.advancedSection}>
            {/* Motion Mode */}
            <View style={styles.advancedItem}>
              <Text style={styles.advancedLabel}>Motion Mode</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                {MOTION_MODE_OPTIONS.map((option) => (
                  <TouchableOpacity
                    key={option.value}
                    style={[
                      styles.modeChip,
                      formData.motionMode === option.value && styles.modeChipActive,
                    ]}
                    onPress={() => updateField('motionMode', option.value as any)}
                    disabled={generationState !== 'idle'}
                  >
                    <Text
                      style={[
                        styles.modeChipText,
                        formData.motionMode === option.value && styles.modeChipTextActive,
                      ]}
                    >
                      {option.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            {/* Resolution */}
            <View style={styles.advancedItem}>
              <Text style={styles.advancedLabel}>
                Resolution: {formData.width} x {formData.height}
              </Text>
              <View style={styles.resolutionOptions}>
                {[
                  { label: '256', w: 256, h: 256 },
                  { label: '512', w: 512, h: 512 },
                  { label: '768', w: 768, h: 768 },
                  { label: '1024', w: 1024, h: 1024 },
                ].map((res) => (
                  <TouchableOpacity
                    key={res.label}
                    style={[
                      styles.resolutionButton,
                      formData.width === res.w && styles.resolutionButtonActive,
                    ]}
                    onPress={() => {
                      updateField('width', res.w);
                      updateField('height', res.h);
                    }}
                    disabled={generationState !== 'idle'}
                  >
                    <Text
                      style={[
                        styles.resolutionText,
                        formData.width === res.w && styles.resolutionTextActive,
                      ]}
                    >
                      {res.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Motion Strength */}
            <View style={styles.advancedItem}>
              <Text style={styles.advancedLabel}>
                Motion Strength: {formData.motionStrength.toFixed(1)}
              </Text>
              <View style={styles.sliderRow}>
                <Text style={styles.sliderLabel}>0</Text>
                <View style={styles.sliderContainer}>
                  {/* Simple slider representation */}
                  <View style={styles.sliderTrack}>
                    <View
                      style={[
                        styles.sliderFill,
                        { width: `${formData.motionStrength * 100}%` },
                      ]}
                    />
                    <TouchableOpacity
                      style={[
                        styles.sliderThumb,
                        { left: `${formData.motionStrength * 100}%` },
                      ]}
                      disabled={generationState !== 'idle'}
                    />
                  </View>
                </View>
                <Text style={styles.sliderLabel}>1</Text>
              </View>
            </View>
          </View>
        )}

        {/* Status Indicator */}
        {renderStatusIndicator()}

        {/* Action Buttons */}
        <View style={styles.buttonContainer}>
          {(generationState === 'idle' || generationState === 'cancelled') && (
            <TouchableOpacity
              style={styles.generateButton}
              onPress={startGeneration}
            >
              <Text style={styles.generateButtonText}>Generate Video</Text>
            </TouchableOpacity>
          )}

          {(generationState === 'processing' || generationState === 'uploading') && (
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={cancelGeneration}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          )}

          {(generationState === 'completed' || generationState === 'failed') && (
            <TouchableOpacity
              style={styles.resetButton}
              onPress={resetGeneration}
            >
              <Text style={styles.resetButtonText}>New Generation</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>

      {/* Video Preview Modal */}
      <Modal
        visible={showPreviewModal}
        animationType="slide"
        onRequestClose={() => setShowPreviewModal(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Video Preview</Text>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setShowPreviewModal(false)}
            >
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.videoContainer}>
            {previewVideo && (
              <Video
                ref={videoRef}
                source={{ uri: previewVideo }}
                style={styles.videoPlayer}
                useNativeControls
                resizeMode={ResizeMode.CONTAIN}
                isLooping
                shouldPlay
              />
            )}
          </View>
          <View style={styles.modalActions}>
            <TouchableOpacity
              style={styles.downloadButton}
              onPress={() => {
                // In a real app, you'd download the video here
                Alert.alert('Download', 'Video download would start here');
              }}
            >
              <Text style={styles.downloadButtonText}>Download Video</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  subtitle: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 4,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 12,
  },
  imagePickerContainer: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#E5E7EB',
    borderWidth: 2,
    borderColor: '#D1D5DB',
    borderStyle: 'dashed',
  },
  selectedImage: {
    width: '100%',
    height: '100%',
  },
  imagePlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  imagePlaceholderIcon: {
    fontSize: 48,
    color: '#9CA3AF',
  },
  imagePlaceholderText: {
    fontSize: 16,
    color: '#9CA3AF',
    marginTop: 8,
  },
  changeImageButton: {
    alignSelf: 'center',
    marginTop: 12,
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  changeImageText: {
    color: '#6366F1',
    fontSize: 14,
    fontWeight: '500',
  },
  promptInput: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    padding: 16,
    fontSize: 16,
    color: '#1F2937',
    minHeight: 100,
  },
  quickSettingsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  quickSettingItem: {
    alignItems: 'center',
  },
  quickSettingLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 8,
  },
  quickSettingValue: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  stepButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#E5E7EB',
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginHorizontal: 16,
    minWidth: 50,
    textAlign: 'center',
  },
  advancedToggle: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
  },
  advancedToggleText: {
    color: '#6366F1',
    fontSize: 14,
    fontWeight: '500',
  },
  advancedToggleIcon: {
    color: '#6366F1',
    marginLeft: 8,
  },
  advancedSection: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  advancedItem: {
    marginBottom: 16,
  },
  advancedLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 8,
  },
  modeChip: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#E5E7EB',
    marginRight: 8,
  },
  modeChipActive: {
    backgroundColor: '#6366F1',
  },
  modeChipText: {
    fontSize: 14,
    color: '#374151',
  },
  modeChipTextActive: {
    color: '#FFFFFF',
  },
  resolutionOptions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  resolutionButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: '#E5E7EB',
    flex: 1,
    marginHorizontal: 4,
    alignItems: 'center',
  },
  resolutionButtonActive: {
    backgroundColor: '#6366F1',
  },
  resolutionText: {
    fontSize: 14,
    color: '#374151',
  },
  resolutionTextActive: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  sliderLabel: {
    fontSize: 12,
    color: '#9CA3AF',
    width: 24,
  },
  sliderContainer: {
    flex: 1,
    marginHorizontal: 8,
  },
  sliderTrack: {
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
    position: 'relative',
  },
  sliderFill: {
    height: '100%',
    backgroundColor: '#6366F1',
    borderRadius: 3,
  },
  sliderThumb: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#6366F1',
    top: -7,
    marginLeft: -10,
  },
  statusContainer: {
    alignItems: 'center',
    paddingVertical: 24,
  },
  statusText: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 12,
  },
  progressBar: {
    width: '80%',
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
    marginTop: 12,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#6366F1',
    borderRadius: 3,
  },
  successText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#10B981',
  },
  previewButton: {
    marginTop: 16,
    paddingHorizontal: 24,
    paddingVertical: 12,
    backgroundColor: '#6366F1',
    borderRadius: 8,
  },
  previewButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  errorText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#EF4444',
  },
  errorDetail: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 8,
    textAlign: 'center',
  },
  buttonContainer: {
    marginTop: 24,
  },
  generateButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  generateButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  cancelButton: {
    backgroundColor: '#EF4444',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  resetButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#000000',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  closeButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  closeButtonText: {
    color: '#6366F1',
    fontSize: 16,
  },
  videoContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  videoPlayer: {
    width: '100%',
    height: '100%',
  },
  modalActions: {
    padding: 20,
  },
  downloadButton: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  downloadButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});

type GenerationState = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed' | 'cancelled';