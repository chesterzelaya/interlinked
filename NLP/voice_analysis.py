"""
Voice and diction analysis module for detecting changes in speech patterns,
prosody, and vocal characteristics that might indicate external influence.
"""

import librosa
import numpy as np
from scipy import stats
import json
from datetime import datetime


class VoiceAnalyzer:
    def __init__(self):
        self.features = [
            'pitch_mean', 'pitch_std', 'pitch_range',
            'speaking_rate', 'pause_frequency', 'pause_duration_mean',
            'spectral_centroid_mean', 'spectral_rolloff_mean',
            'mfcc_coefficients', 'zero_crossing_rate',
            'energy_mean', 'energy_std', 'jitter', 'shimmer'
        ]
    
    def analyze_audio(self, audio_path):
        """
        Extract comprehensive voice features from audio file.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Basic audio properties
            duration = len(y) / sr
            
            # Pitch analysis using fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            pitches = pitches[pitches > 0]
            
            # Speaking rate and pause analysis
            speaking_rate, pause_metrics = self._analyze_speech_timing(y, sr)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(y, sr)
            
            # MFCC coefficients (voice timbre)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Voice quality metrics
            jitter, shimmer = self._calculate_voice_quality(y, sr)
            
            # Energy analysis
            energy = librosa.feature.rms(y=y)[0]
            
            voice_features = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': audio_path.split('/')[-1],
                'duration': duration,
                'pitch_mean': float(np.mean(pitches)) if len(pitches) > 0 else 0,
                'pitch_std': float(np.std(pitches)) if len(pitches) > 0 else 0,
                'pitch_range': float(np.ptp(pitches)) if len(pitches) > 0 else 0,
                'speaking_rate': speaking_rate,
                'pause_frequency': pause_metrics['frequency'],
                'pause_duration_mean': pause_metrics['duration_mean'],
                'spectral_centroid_mean': spectral_features['centroid_mean'],
                'spectral_rolloff_mean': spectral_features['rolloff_mean'],
                'mfcc_coefficients': mfcc_means.tolist(),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'jitter': jitter,
                'shimmer': shimmer
            }
            
            return voice_features
            
        except Exception as e:
            print(f"Voice analysis error for {audio_path}: {str(e)}")
            return None
    
    def _analyze_speech_timing(self, y, sr):
        """
        Analyze speaking rate and pause patterns.
        """
        # Simple energy-based voice activity detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energy_threshold = np.percentile(energy, 20)  # Bottom 20% as silence
        
        # Voice activity detection
        voice_activity = energy > energy_threshold
        
        # Calculate speaking rate (syllables per second approximation)
        # Using energy peaks as syllable approximation
        peaks = librosa.util.peak_pick(energy, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
        speaking_rate = len(peaks) / (len(y) / sr)
        
        # Pause analysis
        silent_frames = ~voice_activity
        pause_segments = self._get_segments(silent_frames)
        
        if len(pause_segments) > 0:
            pause_durations = [(end - start) * hop_length / sr for start, end in pause_segments]
            pause_frequency = len(pause_segments) / (len(y) / sr)
            pause_duration_mean = np.mean(pause_durations)
        else:
            pause_frequency = 0
            pause_duration_mean = 0
        
        return speaking_rate, {
            'frequency': pause_frequency,
            'duration_mean': pause_duration_mean
        }
    
    def _extract_spectral_features(self, y, sr):
        """
        Extract spectral characteristics of the voice.
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            'centroid_mean': float(np.mean(spectral_centroids)),
            'rolloff_mean': float(np.mean(spectral_rolloff))
        }
    
    def _calculate_voice_quality(self, y, sr):
        """
        Calculate jitter and shimmer as indicators of voice quality/stress.
        """
        # Simplified jitter calculation (pitch period variation)
        # This is a basic implementation - professional tools use more sophisticated methods
        try:
            # Get pitch periods
            autocorr = librosa.autocorrelate(y)
            peaks = librosa.util.peak_pick(autocorr, pre_max=10, post_max=10, pre_avg=10, post_avg=10, delta=0.1, wait=10)
            
            if len(peaks) > 1:
                periods = np.diff(peaks)
                jitter = float(np.std(periods) / np.mean(periods)) if np.mean(periods) > 0 else 0
            else:
                jitter = 0
            
            # Simplified shimmer calculation (amplitude variation)
            energy = librosa.feature.rms(y=y)[0]
            if len(energy) > 1:
                shimmer = float(np.std(energy) / np.mean(energy)) if np.mean(energy) > 0 else 0
            else:
                shimmer = 0
            
            return jitter, shimmer
            
        except:
            return 0, 0
    
    def _get_segments(self, binary_array):
        """
        Get start and end indices of True segments in binary array.
        """
        if not np.any(binary_array):
            return []
        
        # Find transitions
        diff = np.diff(binary_array.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if binary_array[0]:
            starts = np.concatenate([[0], starts])
        if binary_array[-1]:
            ends = np.concatenate([ends, [len(binary_array)]])
        
        return list(zip(starts, ends))
    
    def compare_baseline(self, current_features, baseline_features):
        """
        Compare current voice features against baseline to detect anomalies.
        """
        anomaly_scores = {}
        
        for feature in self.features:
            if feature == 'mfcc_coefficients':
                # Compare MFCC vectors using cosine similarity
                if len(current_features[feature]) == len(baseline_features[feature]):
                    current_mfcc = np.array(current_features[feature])
                    baseline_mfcc = np.array(baseline_features[feature])
                    similarity = np.dot(current_mfcc, baseline_mfcc) / (np.linalg.norm(current_mfcc) * np.linalg.norm(baseline_mfcc))
                    anomaly_scores[feature] = 1 - similarity
                else:
                    anomaly_scores[feature] = 0
            else:
                # Normalized difference for scalar features
                current_val = current_features.get(feature, 0)
                baseline_val = baseline_features.get(feature, 0)
                
                if baseline_val != 0:
                    anomaly_scores[feature] = abs(current_val - baseline_val) / abs(baseline_val)
                else:
                    anomaly_scores[feature] = abs(current_val)
        
        return anomaly_scores


if __name__ == "__main__":
    # Test the voice analyzer
    analyzer = VoiceAnalyzer()
    
    # Example usage
    audio_file = "../backend/datastorage/example.wav"
    if os.path.exists(audio_file):
        result = analyzer.analyze_audio(audio_file)
        print("Voice features extracted:", len(result) if result else "Failed")