/**
 * Settings Screen
 * 
 * Configure API connection and app settings.
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { apiService, HealthStatus } from '../services/api';

export default function SettingsScreen() {
  const [apiUrl, setApiUrl] = useState('http://10.0.2.2:8000');
  const [testing, setTesting] = useState(false);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setApiUrl(apiService.getBaseUrl());
  }, []);

  const testConnection = async () => {
    setTesting(true);
    setError(null);
    setHealthStatus(null);

    try {
      // Update API URL first
      apiService.setBaseUrl(apiUrl);
      
      // Test health endpoint
      const status = await apiService.checkHealth();
      setHealthStatus(status);
      Alert.alert('Success', 'Connected to API server!');
    } catch (err: any) {
      setError(err.message);
      Alert.alert('Connection Failed', err.message);
    } finally {
      setTesting(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>API Configuration</Text>
        
        <View style={styles.section}>
          <Text style={styles.label}>Server URL</Text>
          <TextInput
            style={styles.input}
            value={apiUrl}
            onChangeText={setApiUrl}
            placeholder="http://192.168.1.100:8000"
            placeholderTextColor="#9CA3AF"
            autoCapitalize="none"
            autoCorrect={false}
          />
          <Text style={styles.hint}>
            Use your computer's local IP for WiFi connection
          </Text>
        </View>

        <TouchableOpacity
          style={[styles.button, testing && styles.buttonDisabled]}
          onPress={testConnection}
          disabled={testing}
        >
          {testing ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.buttonText}>Test Connection</Text>
          )}
        </TouchableOpacity>

        {healthStatus && (
          <View style={styles.statusContainer}>
            <Text style={styles.statusTitle}>Server Status</Text>
            <Text style={styles.statusItem}>
              Status: {healthStatus.status}
            </Text>
            <Text style={styles.statusItem}>
              CUDA Available: {healthStatus.gpu.cuda_available ? 'Yes' : 'No'}
            </Text>
            {healthStatus.gpu.device_name && (
              <Text style={styles.statusItem}>
                GPU: {healthStatus.gpu.device_name}
              </Text>
            )}
            {healthStatus.gpu.total_memory_gb && (
              <Text style={styles.statusItem}>
                VRAM: {healthStatus.gpu.total_memory_gb.toFixed(1)} GB
              </Text>
            )}
          </View>
        )}

        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorTitle}>Connection Error</Text>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        <View style={styles.instructions}>
          <Text style={styles.instructionsTitle}>Quick Setup</Text>
          
          <View style={styles.step}>
            <Text style={styles.stepNumber}>1</Text>
            <View style={styles.stepContent}>
              <Text style={styles.stepTitle}>Find Your Computer's IP</Text>
              <Text style={styles.stepText}>
                Windows: ipconfig | findstr "IPv4 Address"{'\n'}
                Mac/Linux: ifconfig | grep "inet "
              </Text>
            </View>
          </View>

          <View style={styles.step}>
            <Text style={styles.stepNumber}>2</Text>
            <View style={styles.stepContent}>
              <Text style={styles.stepTitle}>Start Backend Server</Text>
              <Text style={styles.stepText}>
                python -m uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000
              </Text>
            </View>
          </View>

          <View style={styles.step}>
            <Text style={styles.stepNumber}>3</Text>
            <View style={styles.stepContent}>
              <Text style={styles.stepTitle}>Enter IP Above</Text>
              <Text style={styles.stepText}>
                Example: http://192.168.1.100:8000
              </Text>
            </View>
          </View>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 24,
  },
  section: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    padding: 16,
    fontSize: 16,
    color: '#1F2937',
  },
  hint: {
    fontSize: 12,
    color: '#9CA3AF',
    marginTop: 8,
  },
  button: {
    backgroundColor: '#6366F1',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  statusContainer: {
    backgroundColor: '#ECFDF5',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  statusTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#065F46',
    marginBottom: 8,
  },
  statusItem: {
    fontSize: 14,
    color: '#047857',
    marginBottom: 4,
  },
  errorContainer: {
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#991B1B',
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#DC2626',
  },
  instructions: {
    backgroundColor: '#F3F4F6',
    borderRadius: 12,
    padding: 16,
  },
  instructionsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  step: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#6366F1',
    color: '#FFFFFF',
    textAlign: 'center',
    lineHeight: 28,
    fontSize: 14,
    fontWeight: '600',
    marginRight: 12,
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#374151',
    marginBottom: 4,
  },
  stepText: {
    fontSize: 12,
    color: '#6B7280',
  },
});