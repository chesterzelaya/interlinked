"""
Anomaly detection module for identifying unusual patterns in voice, sentiment,
and semantic features that might indicate external influence or manipulation.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
        
        # Define feature categories and their expected stability
        self.feature_categories = {
            'voice_stability': ['pitch_mean', 'pitch_std', 'speaking_rate', 'energy_mean', 'jitter', 'shimmer'],
            'semantic_coherence': ['overall_semantic_similarity', 'coherence_change', 'concept_diversity_change'],
            'emotional_patterns': ['sentiment_vader_compound_change', 'emotional_volatility', 'manipulation_markers'],
            'linguistic_complexity': ['avg_sentence_length', 'avg_word_length', 'lexical_diversity']
        }
        
        # Thresholds for different types of anomalies
        self.anomaly_thresholds = {
            'voice_change': 2.0,          # Z-score threshold
            'semantic_shift': 0.3,        # Cosine similarity threshold  
            'sentiment_volatility': 1.5,   # Standard deviations from baseline
            'manipulation_increase': 0.15, # Ratio increase threshold
            'overall_anomaly': -0.1        # Isolation forest threshold
        }
    
    def detect_anomalies(self, current_analysis, baseline_profile, historical_analyses=None):
        """
        Comprehensive anomaly detection across all analysis dimensions.
        """
        try:
            anomaly_report = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': current_analysis.get('voice_analysis', {}).get('audio_file', ''),
                'anomaly_detected': False,
                'confidence_score': 0,
                'anomaly_types': [],
                'detailed_scores': {},
                'risk_level': 'low',
                'recommendations': []
            }
            
            if not baseline_profile:
                anomaly_report['message'] = 'No baseline profile available for comparison'
                return anomaly_report
            
            # Extract features for analysis
            features = self._extract_features(current_analysis, baseline_profile)
            
            if not features:
                anomaly_report['message'] = 'Insufficient data for anomaly detection'
                return anomaly_report
            
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(features, baseline_profile)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(current_analysis, baseline_profile)
            
            # Machine learning-based anomaly detection
            if historical_analyses and len(historical_analyses) >= 10:
                ml_anomalies = self._detect_ml_anomalies(features, historical_analyses)
            else:
                ml_anomalies = {'anomaly_score': 0, 'is_anomaly': False}
            
            # Combine all anomaly signals
            anomaly_report = self._combine_anomaly_signals(
                statistical_anomalies, 
                pattern_anomalies, 
                ml_anomalies, 
                anomaly_report
            )
            
            # Generate risk assessment and recommendations
            anomaly_report = self._assess_risk_and_recommend(anomaly_report, features)
            
            return anomaly_report
            
        except Exception as e:
            print(f"Anomaly detection error: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'anomaly_detected': False,
                'error': str(e)
            }
    
    def _extract_features(self, current_analysis, baseline_profile):
        """
        Extract numerical features from all analysis modules for anomaly detection.
        """
        features = {}
        
        # Voice features
        voice_data = current_analysis.get('voice_analysis', {})
        voice_baseline = baseline_profile.get('voice_profile', {})
        
        for feature in self.feature_categories['voice_stability']:
            current_val = voice_data.get(feature, 0)
            baseline_val = voice_baseline.get(feature, 0)
            
            # Handle case where baseline_val might be a dict with 'mean' key
            if isinstance(baseline_val, dict):
                baseline_val = baseline_val.get('mean', 0)
            
            # Ensure both values are numbers
            if not isinstance(current_val, (int, float)):
                current_val = 0
            if not isinstance(baseline_val, (int, float)):
                baseline_val = 0
            
            if baseline_val != 0:
                features[f'voice_{feature}_change'] = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                features[f'voice_{feature}_change'] = abs(current_val)
        
        # Sentiment features
        sentiment_data = current_analysis.get('sentiment_analysis', {})
        sentiment_baseline = baseline_profile.get('sentiment_profile', {})
        
        # Overall sentiment changes
        current_sentiment = sentiment_data.get('overall_sentiment', {})
        baseline_sentiment = sentiment_baseline.get('overall_sentiment', {})
        
        for metric in ['vader_compound', 'textblob_polarity', 'textblob_subjectivity']:
            current_val = current_sentiment.get(metric, 0)
            baseline_val = baseline_sentiment.get(metric, 0)
            
            # Ensure both values are numbers
            if not isinstance(current_val, (int, float)):
                current_val = 0
            if not isinstance(baseline_val, (int, float)):
                baseline_val = 0
                
            features[f'sentiment_{metric}_change'] = abs(current_val - baseline_val)
        
        # Manipulation markers
        current_manip = sentiment_data.get('manipulation_markers', {})
        baseline_manip = sentiment_baseline.get('manipulation_markers', {})
        
        features['manipulation_overall_change'] = abs(
            current_manip.get('overall_manipulation_ratio', 0) - 
            baseline_manip.get('overall_manipulation_ratio', 0)
        )
        
        # Emotional volatility
        current_vol = sentiment_data.get('emotional_volatility', {})
        baseline_vol = sentiment_baseline.get('emotional_volatility', {})
        
        features['emotional_volatility_change'] = abs(
            current_vol.get('compound_volatility', 0) - 
            baseline_vol.get('compound_volatility', 0)
        )
        
        # Semantic features
        embeddings_data = current_analysis.get('embeddings_analysis', {})
        embeddings_baseline = baseline_profile.get('embeddings_profile', {})
        
        # Semantic similarity to baseline
        current_embedding = np.array(embeddings_data.get('overall_embedding', []))
        baseline_embedding = np.array(embeddings_baseline.get('overall_embedding', []))
        
        if len(current_embedding) == len(baseline_embedding) and len(current_embedding) > 0:
            similarity = np.dot(current_embedding, baseline_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(baseline_embedding)
            )
            features['semantic_similarity'] = float(similarity)
        else:
            features['semantic_similarity'] = 1.0
        
        # Coherence changes
        current_coherence = embeddings_data.get('semantic_coherence', {})
        baseline_coherence = embeddings_baseline.get('semantic_coherence', {})
        
        features['coherence_change'] = abs(
            current_coherence.get('coherence_score', 0) - 
            baseline_coherence.get('coherence_score', 0)
        )
        
        return features
    
    def _detect_statistical_anomalies(self, features, baseline_profile):
        """
        Detect anomalies using statistical methods (Z-scores, percentiles).
        """
        anomalies = {
            'z_score_anomalies': [],
            'extreme_changes': [],
            'max_z_score': 0
        }
        
        # Historical statistics from baseline
        historical_stats = baseline_profile.get('historical_statistics', {})
        
        for feature_name, current_value in features.items():
            if feature_name in historical_stats:
                mean_val = historical_stats[feature_name].get('mean', current_value)
                std_val = historical_stats[feature_name].get('std', 0)
                
                if std_val > 0:
                    z_score = abs(current_value - mean_val) / std_val
                    
                    if z_score > self.anomaly_thresholds['voice_change']:
                        anomalies['z_score_anomalies'].append({
                            'feature': feature_name,
                            'z_score': float(z_score),
                            'current_value': float(current_value),
                            'baseline_mean': float(mean_val)
                        })
                    
                    anomalies['max_z_score'] = max(anomalies['max_z_score'], z_score)
        
        # Check for extreme changes
        extreme_threshold = 3.0  # 3 standard deviations
        extreme_features = [anom for anom in anomalies['z_score_anomalies'] 
                          if anom['z_score'] > extreme_threshold]
        anomalies['extreme_changes'] = extreme_features
        
        return anomalies
    
    def _detect_pattern_anomalies(self, current_analysis, baseline_profile):
        """
        Detect anomalies based on known manipulation patterns.
        """
        patterns = {
            'semantic_drift': False,
            'manipulation_spike': False,
            'emotional_instability': False,
            'voice_stress': False,
            'pattern_score': 0
        }
        
        # Check semantic drift
        embeddings_data = current_analysis.get('embeddings_analysis', {})
        semantic_similarity = embeddings_data.get('overall_embedding', [])
        baseline_embedding = baseline_profile.get('embeddings_profile', {}).get('overall_embedding', [])
        
        if len(semantic_similarity) == len(baseline_embedding) and len(semantic_similarity) > 0:
            similarity = np.dot(semantic_similarity, baseline_embedding) / (
                np.linalg.norm(semantic_similarity) * np.linalg.norm(baseline_embedding)
            )
            if similarity < self.anomaly_thresholds['semantic_shift']:
                patterns['semantic_drift'] = True
                patterns['pattern_score'] += 0.3
        
        # Check manipulation markers spike
        sentiment_data = current_analysis.get('sentiment_analysis', {})
        current_manip = sentiment_data.get('manipulation_markers', {}).get('overall_manipulation_ratio', 0)
        baseline_manip = baseline_profile.get('sentiment_profile', {}).get('manipulation_markers', {}).get('overall_manipulation_ratio', 0)
        
        if current_manip - baseline_manip > self.anomaly_thresholds['manipulation_increase']:
            patterns['manipulation_spike'] = True
            patterns['pattern_score'] += 0.4
        
        # Check emotional volatility
        current_vol = sentiment_data.get('emotional_volatility', {}).get('compound_volatility', 0)
        baseline_vol = baseline_profile.get('sentiment_profile', {}).get('emotional_volatility', {}).get('compound_volatility', 0)
        
        if current_vol > baseline_vol * (1 + self.anomaly_thresholds['sentiment_volatility']):
            patterns['emotional_instability'] = True
            patterns['pattern_score'] += 0.2
        
        # Check voice stress indicators
        voice_data = current_analysis.get('voice_analysis', {})
        jitter = voice_data.get('jitter', 0)
        shimmer = voice_data.get('shimmer', 0)
        
        baseline_voice = baseline_profile.get('voice_profile', {})
        baseline_jitter = baseline_voice.get('jitter', 0)
        baseline_shimmer = baseline_voice.get('shimmer', 0)
        
        if (jitter > baseline_jitter * 1.5) or (shimmer > baseline_shimmer * 1.5):
            patterns['voice_stress'] = True
            patterns['pattern_score'] += 0.1
        
        return patterns
    
    def _detect_ml_anomalies(self, features, historical_analyses):
        """
        Use machine learning models to detect anomalies based on historical patterns.
        """
        try:
            # Prepare historical feature matrix
            historical_features = []
            for analysis in historical_analyses:
                hist_features = self._extract_features_from_analysis(analysis)
                if hist_features:
                    historical_features.append(list(hist_features.values()))
            
            if len(historical_features) < 5:
                return {'anomaly_score': 0, 'is_anomaly': False}
            
            # Current features
            current_features_array = np.array(list(features.values())).reshape(1, -1)
            historical_features_array = np.array(historical_features)
            
            # Normalize features
            all_features = np.vstack([historical_features_array, current_features_array])
            normalized_features = self.scaler.fit_transform(all_features)
            
            # Separate historical and current
            hist_normalized = normalized_features[:-1]
            current_normalized = normalized_features[-1:] 
            
            # Fit models on historical data
            self.isolation_forest.fit(hist_normalized)
            self.elliptic_envelope.fit(hist_normalized)
            
            # Predict anomaly for current sample
            iso_score = self.isolation_forest.decision_function(current_normalized)[0]
            iso_anomaly = self.isolation_forest.predict(current_normalized)[0] == -1
            
            ellip_anomaly = self.elliptic_envelope.predict(current_normalized)[0] == -1
            
            # Combine ML predictions
            ml_anomaly_score = abs(iso_score)
            is_ml_anomaly = iso_anomaly or ellip_anomaly
            
            return {
                'anomaly_score': float(ml_anomaly_score),
                'is_anomaly': bool(is_ml_anomaly),
                'isolation_score': float(iso_score),
                'isolation_anomaly': bool(iso_anomaly),
                'elliptic_anomaly': bool(ellip_anomaly)
            }
            
        except Exception as e:
            print(f"ML anomaly detection error: {str(e)}")
            return {'anomaly_score': 0, 'is_anomaly': False}
    
    def _extract_features_from_analysis(self, analysis):
        """
        Helper to extract features from a historical analysis.
        """
        # This is a simplified version - in practice, you'd want to store
        # the extracted features with each analysis
        voice_data = analysis.get('voice_analysis', {})
        sentiment_data = analysis.get('sentiment_analysis', {})
        
        features = {}
        
        # Basic features
        features['pitch_mean'] = voice_data.get('pitch_mean', 0)
        features['speaking_rate'] = voice_data.get('speaking_rate', 0)
        features['sentiment_compound'] = sentiment_data.get('overall_sentiment', {}).get('vader_compound', 0)
        features['manipulation_ratio'] = sentiment_data.get('manipulation_markers', {}).get('overall_manipulation_ratio', 0)
        
        return features
    
    def _combine_anomaly_signals(self, statistical, pattern, ml, report):
        """
        Combine different anomaly detection signals into overall assessment.
        """
        anomaly_indicators = 0
        confidence_factors = []
        
        # Statistical anomalies
        if statistical['z_score_anomalies']:
            anomaly_indicators += 1
            confidence_factors.append(min(statistical['max_z_score'] / 3.0, 1.0))
            report['anomaly_types'].append('statistical_deviation')
        
        if statistical['extreme_changes']:
            anomaly_indicators += 1
            report['anomaly_types'].append('extreme_change')
        
        # Pattern anomalies
        if pattern['pattern_score'] > 0.3:
            anomaly_indicators += 1
            confidence_factors.append(min(pattern['pattern_score'], 1.0))
            report['anomaly_types'].extend([
                key for key, value in pattern.items() 
                if isinstance(value, bool) and value
            ])
        
        # ML anomalies
        if ml.get('is_anomaly', False):
            anomaly_indicators += 1
            confidence_factors.append(min(ml.get('anomaly_score', 0), 1.0))
            report['anomaly_types'].append('machine_learning_detection')
        
        # Overall assessment
        report['anomaly_detected'] = anomaly_indicators >= 2  # At least 2 different methods detect anomaly
        report['confidence_score'] = np.mean(confidence_factors) if confidence_factors else 0
        
        # Store detailed scores
        report['detailed_scores'] = {
            'statistical': statistical,
            'pattern': pattern,
            'machine_learning': ml,
            'anomaly_indicators': anomaly_indicators
        }
        
        return report
    
    def _assess_risk_and_recommend(self, report, features):
        """
        Assess risk level and provide recommendations based on detected anomalies.
        """
        confidence = report['confidence_score']
        anomaly_types = report['anomaly_types']
        
        # Determine risk level
        if not report['anomaly_detected']:
            report['risk_level'] = 'low'
        elif confidence < 0.4:
            report['risk_level'] = 'low'
        elif confidence < 0.7:
            report['risk_level'] = 'medium'
        else:
            report['risk_level'] = 'high'
        
        # Generate recommendations
        recommendations = []
        
        if 'semantic_drift' in anomaly_types:
            recommendations.append("Significant semantic drift detected. Review recent media consumption and social interactions.")
        
        if 'manipulation_spike' in anomaly_types:
            recommendations.append("Increased use of manipulation-associated language patterns. Consider media detox.")
        
        if 'emotional_instability' in anomaly_types:
            recommendations.append("Emotional volatility detected. Review stress factors and emotional triggers.")
        
        if 'voice_stress' in anomaly_types:
            recommendations.append("Voice stress indicators present. Consider relaxation techniques and stress management.")
        
        if 'extreme_change' in anomaly_types:
            recommendations.append("Extreme changes detected across multiple metrics. Recommend comprehensive review of recent influences.")
        
        if report['risk_level'] == 'high':
            recommendations.append("HIGH RISK: Multiple anomaly indicators suggest possible external influence. Consider professional consultation.")
        
        if not recommendations:
            recommendations.append("No significant anomalies detected. Baseline patterns maintained.")
        
        report['recommendations'] = recommendations
        
        return report


if __name__ == "__main__":
    # Test the anomaly detector
    detector = AnomalyDetector()
    
    # Example usage would require actual analysis data
    print("Anomaly detector initialized successfully")